/* ornith_tensor.h — a minimal f32 tensor type and the core math kernels the
 * CPU reference forward pass is built from.
 *
 * This is deliberately small: no broadcasting engine, no autograd, no fancy
 * memory pool. Just the handful of dense f32 operations a qwen3_5_moe decode
 * needs (matvec/linear, RMSNorm, RoPE, softmax, SiLU/SwiGLU, elementwise), each
 * implemented as a plain naive kernel so it is obviously correct and easy to
 * test against hand-computed values. Speed is M3's problem; this layer exists to
 * be *right*.
 *
 * Convention: weights follow the PyTorch/GGUF layout, a linear layer's weight is
 * [out_features, in_features] row-major and computes y = W x. Vectors are plain
 * contiguous f32. "Batch" variants loop over T tokens (row-major [T, dim]).
 */
#ifndef ORNITH_TENSOR_H
#define ORNITH_TENSOR_H

#include "ornith.h"

/* ---- tiny owned tensor ------------------------------------------------- */

#define OT_MAX_DIMS 4

typedef struct {
    float  *data;
    int32_t ne[OT_MAX_DIMS]; /* ne[0] is the fastest-moving (innermost) dim   */
    int32_t n_dims;
    bool    owns;            /* true if data must be freed by ot_free()        */
} ot_tensor;

/* Allocate a zeroed tensor with the given dims (1..4). Aborts on OOM (this is
 * reference code; allocations are small and a failure is unrecoverable). */
ot_tensor ot_new(int n_dims, int32_t ne0, int32_t ne1, int32_t ne2, int32_t ne3);
void      ot_free(ot_tensor *t);
int64_t   ot_nelem(const ot_tensor *t);

/* ---- deterministic RNG (seeded, for synthetic weights/tests) ----------- */

typedef struct { uint64_t s; } ot_rng;
ot_rng ot_rng_seed(uint64_t seed);
float  ot_rng_uniform(ot_rng *r, float lo, float hi);  /* uniform [lo,hi)     */
void   ot_rng_fill(ot_rng *r, float *dst, int64_t n, float lo, float hi);

/* ---- core math kernels (raw f32 pointers, explicit dims) --------------- */

/* y[out] = W[out,in] @ x[in].  W is row-major [out_features, in_features]. */
void ot_linear(const float *W, const float *x, float *y,
               int out_features, int in_features);
/* Same, plus a bias[out] added (bias may be NULL). */
void ot_linear_bias(const float *W, const float *bias, const float *x, float *y,
                    int out_features, int in_features);
/* Batched: Y[T,out] = X[T,in] @ W^T. */
void ot_linear_batch(const float *W, const float *X, float *Y,
                     int T, int out_features, int in_features);

/* C[m,n] = A[m,k] @ B[k,n], all row-major. Naive triple loop. */
void ot_matmul(const float *A, const float *B, float *C, int m, int k, int n);

/* out[n] = x[n] / rms(x) * weight[n], rms = sqrt(mean(x^2) + eps).
 * weight may be NULL (pure normalization). out may alias x. */
void ot_rmsnorm(const float *x, const float *weight, float *out, int n, float eps);

/* In-place numerically-stable softmax over the first n elements. */
void ot_softmax(float *x, int n);

/* SiLU(x) = x * sigmoid(x), elementwise, in place over n. */
void ot_silu_(float *x, int n);
float ot_silu1(float x);
float ot_sigmoid(float x);
/* SwiGLU: out[i] = SiLU(gate[i]) * up[i]. out may alias gate or up. */
void ot_swiglu(const float *gate, const float *up, float *out, int n);

/* Elementwise helpers (all length n, may alias sensibly). */
void ot_add_(float *acc, const float *x, int n);            /* acc += x        */
void ot_addscaled_(float *acc, const float *x, float s, int n); /* acc += s*x  */
void ot_scale_(float *x, float s, int n);                   /* x  *= s         */
void ot_mul_(float *acc, const float *x, int n);            /* acc *= x (elt)  */
float ot_dot(const float *a, const float *b, int n);
float ot_l2norm(const float *x, int n);
void  ot_l2normalize(const float *x, float *out, int n, float eps); /* x/||x|| */

/* RoPE (rotary position embedding), NeoX/HF "rotate-half" convention.
 * Rotates a single head vector `v` of length head_dim in place, at integer
 * position `pos`, using base frequency `theta`. head_dim must be even. */
void ot_rope_inplace(float *v, int head_dim, int pos, float theta);

#endif /* ORNITH_TENSOR_H */
