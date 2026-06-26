/* ornith_moe.h — token-choice MoE FFN for qwen3_5_moe.
 *
 * Per token: a router GEMM produces a logit per expert; we pick the top-k by
 * logit, softmax over just those k (the "norm_topk_prob" convention), run each
 * selected expert's SwiGLU FFN, and combine them weighted by the softmax. A
 * single always-on shared expert FFN is added on top. The router, shared expert
 * and combine all run in f32 (DESIGN.md §2.3: a bad route is unrecoverable).
 *
 * Weights, per expert e: gate W_g[inter,hidden], up W_u[inter,hidden],
 * down W_d[hidden,inter]; FFN(x) = W_d ( SiLU(W_g x) * (W_u x) ). The shared
 * expert has the same shape with shared_inter as its intermediate size.
 */
#ifndef ORNITH_MOE_H
#define ORNITH_MOE_H

#include "ornith.h"

/* Weights for one MoE block (one decoder layer). Expert tensors are packed
 * contiguously over experts: gate[e] starts at w_gate + e*inter*hidden, etc. */
typedef struct {
    int hidden, inter, n_experts, top_k, shared_inter;
    const float *w_router;   /* [n_experts, hidden]                          */
    const float *w_gate;     /* [n_experts, inter, hidden]                   */
    const float *w_up;       /* [n_experts, inter, hidden]                   */
    const float *w_down;     /* [n_experts, hidden, inter]                   */
    /* shared expert (NULL w_* if shared_inter == 0)                          */
    const float *sw_gate;    /* [shared_inter, hidden]                       */
    const float *sw_up;      /* [shared_inter, hidden]                       */
    const float *sw_down;    /* [hidden, shared_inter]                       */
} ornith_moe;

/* Select the top-k experts for logits[n_experts]: fills idx[top_k] with expert
 * indices (descending logit) and weight[top_k] with softmax-over-selected
 * weights. Stable: ties broken by lower index. */
void ornith_moe_route(const float *logits, int n_experts, int top_k,
                      int *idx, float *weight);

/* Single-expert SwiGLU FFN: out[hidden] = Wd ( SiLU(Wg x) * (Wu x) ).
 * scratch must hold >= 2*inter floats. */
void ornith_ffn_swiglu(const float *Wg, const float *Wu, const float *Wd,
                       const float *x, float *out,
                       int hidden, int inter, float *scratch);

/* Full MoE forward for one token x[hidden] -> out[hidden] (overwritten).
 * Computes router, top-k combine, and the shared expert. scratch must hold
 * >= 2*max(inter, shared_inter) + hidden floats. */
void ornith_moe_forward(const ornith_moe *m, const float *x, float *out,
                        float *scratch);

#endif /* ORNITH_MOE_H */
