/* ornith_eval.c — see ornith_eval.h.
 *
 * Perplexity via the public session API. rmodel_session_eval only exposes the
 * NEXT-token logits (the last fed position), so to score every position we feed
 * the corpus one token at a time: after feeding ids[0..i] the session's logits
 * predict ids[i+1], whose NLL we accumulate before advancing. Mean NLL is in
 * nats; perplexity = exp(mean NLL).
 */
#include "ornith_eval.h"
#include "ornith_model.h"
#include "ornith_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double ornith_eval_token_nll(const float *logits, int vocab, int32_t next) {
    if (vocab <= 0 || next < 0 || next >= vocab) return NAN;
    float mx = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < vocab; i++) sum += exp((double)logits[i] - (double)mx);
    double logsumexp = (double)mx + log(sum);
    double nll = logsumexp - (double)logits[next];   /* -log softmax[next] */
    if (nll < 0.0) nll = 0.0;                         /* clamp fp noise     */
    return nll;
}

ornith_status ornith_eval_ids(rmodel *m, const int32_t *ids, int n,
                              double *out_mean_nll, double *out_ppl,
                              long *out_count) {
    if (n < 2) {
        ornith_set_error("perplexity needs at least 2 tokens (got %d)", n);
        return ORNITH_ERR_UNSUPPORTED;
    }
    int V = rmodel_arch(m)->vocab_size;
    rsession *s = rmodel_session_new(m, n + 2);
    if (!s) return ORNITH_ERR_OOM;

    ornith_status st = rmodel_session_eval(s, &ids[0], 1);
    double total = 0.0;
    long c = 0;
    for (int i = 1; st == ORNITH_OK && i < n; i++) {
        const float *lg = rmodel_session_logits(s);
        if (!lg) { st = ORNITH_ERR_UNSUPPORTED; break; }
        double nll = ornith_eval_token_nll(lg, V, ids[i]);
        if (isfinite(nll)) { total += nll; c++; }
        st = rmodel_session_eval(s, &ids[i], 1);
    }
    rmodel_session_free(s);
    if (st != ORNITH_OK) return st;

    double mean = c ? total / (double)c : 0.0;
    if (out_mean_nll) *out_mean_nll = mean;
    if (out_ppl)      *out_ppl = exp(mean);
    if (out_count)    *out_count = c;
    return ORNITH_OK;
}

/* Read an entire text file into a malloc'd NUL-terminated buffer. */
static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    return buf;
}

int ornith_eval_main(int argc, char **argv) {
    const char *model = NULL, *corpus = NULL;
    int max_tokens = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (!model)  model = argv[i];
        else if (!corpus) corpus = argv[i];
    }
    if (!model || !corpus) {
        fprintf(stderr,
            "usage: ornith eval <model.gguf> <corpus.txt> [--max-tokens N]\n");
        return 1;
    }

    fprintf(stderr, "eval: loading %s ...\n", model);
    rmodel *m = NULL;
    ornith_status s = rmodel_load(model, &m);
    if (s != ORNITH_OK) {
        fprintf(stderr, "eval: load failed: %s\n", ornith_last_error());
        return 1;
    }

    char *text = read_text_file(corpus);
    if (!text) {
        fprintf(stderr, "eval: cannot read corpus %s\n", corpus);
        rmodel_free(m);
        return 1;
    }
    int32_t *ids = NULL;
    int n = 0;
    s = otok_encode(rmodel_tokenizer(m), text, &ids, &n);
    free(text);
    if (s != ORNITH_OK || n == 0) {
        fprintf(stderr, "eval: tokenize failed or empty corpus\n");
        free(ids); rmodel_free(m);
        return 1;
    }
    if (max_tokens > 0 && n > max_tokens) n = max_tokens;
    if (n < 2) {
        fprintf(stderr, "eval: corpus too short (%d tokens); need >= 2\n", n);
        free(ids); rmodel_free(m);
        return 1;
    }

    fprintf(stderr, "eval: %d tokens, scoring %d predictions ...\n", n, n - 1);

    double mean_nll = 0, ppl = 0;
    long count = 0;
    s = ornith_eval_ids(m, ids, n, &mean_nll, &ppl, &count);
    free(ids);
    if (s != ORNITH_OK) {
        fprintf(stderr, "eval: failed: %s\n", ornith_last_error());
        rmodel_free(m);
        return 1;
    }

    const char *name = rmodel_name(m);
    printf("\n");
    printf("ornithology eval — %s\n", name ? name : "(unnamed model)");
    printf("  model  : %s\n", model);
    printf("  corpus : %s\n\n", corpus);
    printf("  tokens scored : %ld\n", count);
    printf("  mean NLL      : %.4f nats/token\n", mean_nll);
    printf("  perplexity    : %.4f\n", ppl);
    printf("\n");

    rmodel_free(m);
    return 0;
}
