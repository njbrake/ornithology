/* ornith_sample.c — logits sampler implementation. See ornith_sample.h. */
#include "ornith_sample.h"
#include <stdlib.h>
#include <math.h>

osample_params osample_params_default(void) {
    osample_params p;
    p.temperature    = 0.0f;   /* greedy */
    p.top_k          = 0;
    p.top_p          = 1.0f;
    p.min_p          = 0.0f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n  = 64;
    p.seed           = 0;
    return p;
}

static int32_t argmax_logits(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

typedef struct { int32_t id; float logit; float prob; } osample_cand;

/* Sort descending by logit; break ties by ascending id for cross-platform
 * determinism (qsort is not stable). */
static int cand_cmp(const void *a, const void *b) {
    const osample_cand *x = a, *y = b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return  1;
    if (x->id < y->id) return -1;
    if (x->id > y->id) return  1;
    return 0;
}

/* Apply the repetition penalty to the logits of tokens that appear in the
 * recent window. llama.cpp convention: positive logits are divided, negative
 * logits are multiplied, so the magnitude always shrinks toward 0. */
static void apply_repeat_penalty(float *logits, int n_vocab,
                                 const osample_params *p,
                                 const int32_t *recent, int n_recent) {
    if (!recent || n_recent <= 0) return;
    if (p->repeat_penalty == 1.0f || p->repeat_penalty <= 0.0f) return;
    int window = (p->repeat_last_n > 0 && p->repeat_last_n < n_recent)
                 ? p->repeat_last_n : n_recent;
    int start = n_recent - window;
    for (int i = start; i < n_recent; i++) {
        int32_t t = recent[i];
        if (t < 0 || t >= n_vocab) continue;
        float l = logits[t];
        logits[t] = (l > 0.0f) ? (l / p->repeat_penalty)
                               : (l * p->repeat_penalty);
    }
}

int32_t osample(float *logits, int n_vocab, const osample_params *p,
                ot_rng *rng, const int32_t *recent, int n_recent) {
    if (n_vocab <= 0) return 0;

    apply_repeat_penalty(logits, n_vocab, p, recent, n_recent);

    /* Greedy: argmax of the (possibly penalized) logits. */
    if (p->temperature <= 0.0f) return argmax_logits(logits, n_vocab);

    /* Build candidate list with temperature-scaled logits. */
    osample_cand *c = malloc((size_t)n_vocab * sizeof *c);
    if (!c) return argmax_logits(logits, n_vocab);  /* graceful fallback */
    float inv_t = 1.0f / p->temperature;
    for (int i = 0; i < n_vocab; i++) {
        c[i].id = i;
        c[i].logit = logits[i] * inv_t;
        c[i].prob = 0.0f;
    }

    /* Sort descending by scaled logit (needed for top-k / top-p / min-p). */
    qsort(c, (size_t)n_vocab, sizeof *c, cand_cmp);

    int n = n_vocab;

    /* top-k: keep the k highest. */
    if (p->top_k > 0 && p->top_k < n) n = p->top_k;

    /* Softmax over the surviving prefix (max is c[0] since sorted). */
    float maxl = c[0].logit;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = expf(c[i].logit - maxl);
        c[i].prob = e;
        sum += e;
    }
    float invs = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; i++) c[i].prob *= invs;

    /* top-p (nucleus): smallest prefix whose cumulative prob >= top_p. */
    if (p->top_p < 1.0f && p->top_p > 0.0f) {
        float cum = 0.0f;
        int keep = n;
        for (int i = 0; i < n; i++) {
            cum += c[i].prob;
            if (cum >= p->top_p) { keep = i + 1; break; }
        }
        if (keep < 1) keep = 1;
        n = keep;
    }

    /* min-p: drop tokens below min_p * max_prob (max_prob is c[0].prob). */
    if (p->min_p > 0.0f) {
        float thresh = p->min_p * c[0].prob;
        int keep = 1;
        for (int i = 1; i < n; i++) {
            if (c[i].prob >= thresh) keep = i + 1; else break;
        }
        n = keep;
    }

    /* Renormalize over the survivors. */
    float rsum = 0.0f;
    for (int i = 0; i < n; i++) rsum += c[i].prob;
    float rinv = (rsum > 0.0f) ? 1.0f / rsum : 0.0f;

    /* Sample: inverse-CDF over the survivors using the seeded RNG. */
    float r = ot_rng_uniform(rng, 0.0f, 1.0f);
    float cum = 0.0f;
    int32_t chosen = c[0].id;
    for (int i = 0; i < n; i++) {
        cum += c[i].prob * rinv;
        if (r < cum) { chosen = c[i].id; break; }
        chosen = c[i].id;  /* guard against float rounding leaving r >= cum */
    }

    free(c);
    return chosen;
}
