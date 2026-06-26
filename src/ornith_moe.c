/* ornith_moe.c — top-k router + grouped SwiGLU experts + shared expert. */
#include "ornith_moe.h"
#include "ornith_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void ornith_moe_route(const float *logits, int n_experts, int top_k,
                      int *idx, float *weight) {
    /* Selection sort for the top_k (top_k is tiny: 8/10). Ties -> lower index,
     * which we get for free by scanning ascending and using strict '>'. */
    bool *taken = calloc((size_t)n_experts, sizeof(bool));
    for (int s = 0; s < top_k; s++) {
        int best = -1; float bestv = -INFINITY;
        for (int e = 0; e < n_experts; e++) {
            if (taken[e]) continue;
            if (logits[e] > bestv) { bestv = logits[e]; best = e; }
        }
        idx[s] = best; taken[best] = true;
    }
    free(taken);
    /* softmax over the selected logits only */
    float mx = logits[idx[0]];
    for (int s = 1; s < top_k; s++) if (logits[idx[s]] > mx) mx = logits[idx[s]];
    float sum = 0.0f;
    for (int s = 0; s < top_k; s++) {
        weight[s] = expf(logits[idx[s]] - mx); sum += weight[s];
    }
    for (int s = 0; s < top_k; s++) weight[s] /= sum;
}

void ornith_ffn_swiglu(const float *Wg, const float *Wu, const float *Wd,
                       const float *x, float *out,
                       int hidden, int inter, float *scratch) {
    float *g = scratch;          /* [inter] */
    float *u = scratch + inter;  /* [inter] */
    ot_linear(Wg, x, g, inter, hidden);
    ot_linear(Wu, x, u, inter, hidden);
    ot_swiglu(g, u, g, inter);   /* g = SiLU(g)*u */
    ot_linear(Wd, g, out, hidden, inter);
}

void ornith_moe_forward(const ornith_moe *m, const float *x, float *out,
                        float *scratch) {
    int H = m->hidden, I = m->inter, K = m->top_k;
    float *epart  = scratch;                 /* [hidden] expert output        */
    float *ffn_sc = scratch + H;             /* [2*max(inter,shared_inter)]   */

    memset(out, 0, (size_t)H * sizeof(float));

    /* Routed experts (skipped entirely for a dense model: n_experts/top_k 0). */
    if (m->n_experts > 0 && K > 0 && m->w_router) {
        float *logits = malloc((size_t)m->n_experts * sizeof(float));
        int   *idx    = malloc((size_t)K * sizeof(int));
        float *w      = malloc((size_t)K * sizeof(float));
        ot_linear(m->w_router, x, logits, m->n_experts, H);
        ornith_moe_route(logits, m->n_experts, K, idx, w);
        for (int s = 0; s < K; s++) {
            int e = idx[s];
            const float *Wg = m->w_gate + (size_t)e * I * H;
            const float *Wu = m->w_up   + (size_t)e * I * H;
            const float *Wd = m->w_down + (size_t)e * H * I;
            ornith_ffn_swiglu(Wg, Wu, Wd, x, epart, H, I, ffn_sc);
            ot_addscaled_(out, epart, w[s], H);
        }
        free(logits); free(idx); free(w);
    }

    if (m->shared_inter > 0 && m->sw_gate) {
        ornith_ffn_swiglu(m->sw_gate, m->sw_up, m->sw_down, x, epart,
                          H, m->shared_inter, ffn_sc);
        ot_add_(out, epart, H);
    }
}
