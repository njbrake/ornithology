/* ornith_attn.h — the two attention flavors of qwen3_5_moe.
 *
 *   1. Full GQA causal softmax attention (¼ of layers), with a simple
 *      contiguous f32 KV cache.
 *   2. Gated delta-net "linear attention" (¾ of layers): a constant-size fp32
 *      recurrent state, plus a short causal depthwise conv. It has a step
 *      (decode) path and a chunked-scan (prefill) path that are required to
 *      produce bit-for-bit-equivalent (within fp tolerance) state and output.
 *      That parity is the headline correctness property of milestone M2.
 *
 * Everything here is per-head core math operating on f32 buffers. The layer
 * wiring (projections, gates, head grouping) lives in ornith_forward.c; keeping
 * the recurrence isolated here is what makes the prefill==decode test sharp.
 *
 * ---- gated delta rule (the exact recurrence implemented) ----
 * State S is a [dv, dk] matrix per value head. Per step t, given decay
 * alpha_t in (0,1), write-rate beta_t in (0,1), key k_t (dk), value v_t (dv),
 * query q_t (dk):
 *
 *     S'   = alpha_t * S_{t-1}                 (decay)
 *     u_t  = beta_t * (v_t - S' k_t)           (delta value: error vs. current
 *                                               associative recall of k_t)
 *     S_t  = S' + u_t k_t^T                     (rank-1 write)
 *     o_t  = S_t q_t                            (read, post-write)
 *
 * This is the gated DeltaNet recurrence (Yang et al., "Gated Delta Networks"),
 * S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T, regrouped so
 * the chunked form is easy to derive. The chunked path solves the within-chunk
 * triangular system for the {u_t} in one shot and carries S across chunks; see
 * ornith_attn.c for the derivation. Both paths share one fp32 state.
 */
#ifndef ORNITH_ATTN_H
#define ORNITH_ATTN_H

#include "ornith.h"

/* ---- full GQA causal attention ---------------------------------------- */

/* Contiguous f32 KV cache for one full-attention layer. K and V are stored as
 * [capacity, n_kv_heads, head_dim] row-major; `len` grows as tokens append. */
typedef struct {
    float  *K, *V;
    int     capacity;     /* max positions                                   */
    int     len;          /* positions currently stored                      */
    int     n_kv_heads;
    int     head_dim;
} ornith_kv_cache;

ornith_status ornith_kv_init(ornith_kv_cache *c, int capacity,
                             int n_kv_heads, int head_dim);
void          ornith_kv_free(ornith_kv_cache *c);
void          ornith_kv_reset(ornith_kv_cache *c);

/* Append one token's K,V (each [n_kv_heads*head_dim]) then compute GQA output
 * for `n_heads` query heads q ([n_heads*head_dim]) attending causally over the
 * whole cache. out is [n_heads*head_dim]. RoPE is the caller's responsibility
 * (apply to q and to k before appending). scale defaults to 1/sqrt(head_dim)
 * when <= 0. */
void ornith_gqa_step(ornith_kv_cache *c, const float *q,
                     const float *k_new, const float *v_new,
                     int n_heads, float *out, float scale);

/* Naive O(T^2) reference: full causal GQA over a whole sequence at once.
 * Q is [T, n_heads*head_dim], K,V are [T, n_kv_heads*head_dim] (RoPE already
 * applied to Q,K by the caller). Writes O [T, n_heads*head_dim]. Used both as a
 * prefill path and as the independent reference the step path is tested against. */
void ornith_gqa_reference(const float *Q, const float *K, const float *V,
                          int T, int n_heads, int n_kv_heads, int head_dim,
                          float *O, float scale);

/* ---- gated delta-net linear attention --------------------------------- */

/* Per-value-head recurrent state: an [dv, dk] fp32 matrix. */
typedef struct {
    float *S;     /* [dv*dk] fp32                                            */
    int    dk, dv;
} ornith_delta_state;

ornith_status ornith_delta_init(ornith_delta_state *s, int dk, int dv);
void          ornith_delta_free(ornith_delta_state *s);
void          ornith_delta_reset(ornith_delta_state *s);

/* One decode step (the recurrence above). q,k are [dk], v is [dv], alpha/beta
 * scalars. Writes o [dv] and updates s->S in place. */
void ornith_delta_step(ornith_delta_state *s, const float *q, const float *k,
                       const float *v, float alpha, float beta, float *o);

/* Chunked parallel-scan prefill over T steps for one head, updating s->S in
 * place and writing O [T*dv]. Q,K are [T*dk], V is [T*dv], alpha/beta are [T].
 * `chunk` is the chunk size (>=1; clamped to T). For any chunk size this must
 * produce the same O and final S as calling ornith_delta_step T times. */
void ornith_delta_prefill(ornith_delta_state *s, const float *Q, const float *K,
                          const float *V, const float *alpha, const float *beta,
                          int T, int chunk, float *O);

/* ---- short causal depthwise conv (kernel `K`, per channel) ------------- */

/* Conv state: the last K-1 inputs per channel, oldest first, [(K-1)*C]. */
typedef struct {
    float *buf;   /* [(K-1)*C]                                               */
    int    K, C;
} ornith_conv_state;

ornith_status ornith_conv_init(ornith_conv_state *cs, int K, int C);
void          ornith_conv_free(ornith_conv_state *cs);
void          ornith_conv_reset(ornith_conv_state *cs);

/* One decode step: x_in [C], weight [C*K] (per channel, w[c*K + j], j=K-1 is the
 * current position), bias [C] or NULL. Writes out [C], updates conv state. */
void ornith_conv_step(ornith_conv_state *cs, const float *x_in,
                      const float *weight, const float *bias, float *out);

/* Prefill: X [T*C] -> OUT [T*C], same weight/bias, advancing conv state so a
 * subsequent ornith_conv_step continues seamlessly. Must equal T conv steps. */
void ornith_conv_prefill(ornith_conv_state *cs, const float *X,
                         const float *weight, const float *bias,
                         int T, float *OUT);

#endif /* ORNITH_ATTN_H */
