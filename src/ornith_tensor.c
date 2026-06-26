/* ornith_tensor.c — implementations of the core f32 kernels.
 *
 * Every kernel here is the naive, obvious version. The point is correctness and
 * legibility; the Metal/CUDA backends (M3+) carry the fast paths. Each function
 * has a unit test in tests/test_tensor.c checking it against a hand-computed or
 * independently-derived value.
 */
#include "ornith_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- tiny owned tensor ------------------------------------------------- */

ot_tensor ot_new(int n_dims, int32_t ne0, int32_t ne1, int32_t ne2, int32_t ne3) {
    ot_tensor t;
    memset(&t, 0, sizeof(t));
    t.n_dims = n_dims;
    t.ne[0] = ne0; t.ne[1] = ne1 ? ne1 : 1;
    t.ne[2] = ne2 ? ne2 : 1; t.ne[3] = ne3 ? ne3 : 1;
    int64_t n = ot_nelem(&t);
    t.data = calloc((size_t)n, sizeof(float));
    if (!t.data) { ornith_set_error("ot_new: oom for %lld floats",
                                    (long long)n); abort(); }
    t.owns = true;
    return t;
}

void ot_free(ot_tensor *t) {
    if (!t) return;
    if (t->owns) free(t->data);
    t->data = NULL; t->owns = false;
}

int64_t ot_nelem(const ot_tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < OT_MAX_DIMS; i++) n *= (t->ne[i] ? t->ne[i] : 1);
    return n;
}

/* ---- deterministic RNG ------------------------------------------------- */
/* splitmix64: small, well-distributed, reproducible across platforms. */

ot_rng ot_rng_seed(uint64_t seed) { ot_rng r; r.s = seed; return r; }

static uint64_t ot_rng_next(ot_rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float ot_rng_uniform(ot_rng *r, float lo, float hi) {
    /* 24-bit mantissa worth of randomness in [0,1). */
    uint32_t u = (uint32_t)(ot_rng_next(r) >> 40);
    float f = (float)u / (float)(1u << 24);
    return lo + (hi - lo) * f;
}

void ot_rng_fill(ot_rng *r, float *dst, int64_t n, float lo, float hi) {
    for (int64_t i = 0; i < n; i++) dst[i] = ot_rng_uniform(r, lo, hi);
}

/* ---- linear / matmul --------------------------------------------------- */

void ot_linear(const float *W, const float *x, float *y,
               int out_features, int in_features) {
    ot_linear_bias(W, NULL, x, y, out_features, in_features);
}

void ot_linear_bias(const float *W, const float *bias, const float *x, float *y,
                    int out_features, int in_features) {
    for (int o = 0; o < out_features; o++) {
        const float *wr = W + (size_t)o * in_features;
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_features; i++) acc += wr[i] * x[i];
        y[o] = acc;
    }
}

void ot_linear_batch(const float *W, const float *X, float *Y,
                     int T, int out_features, int in_features) {
    for (int t = 0; t < T; t++)
        ot_linear(W, X + (size_t)t * in_features,
                  Y + (size_t)t * out_features, out_features, in_features);
}

void ot_matmul(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) acc += A[(size_t)i*k+p] * B[(size_t)p*n+j];
            C[(size_t)i*n+j] = acc;
        }
    }
}

/* ---- norms / activations ----------------------------------------------- */

void ot_rmsnorm(const float *x, const float *weight, float *out, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv * (weight ? weight[i] : 1.0f);
}

void ot_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

float ot_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float ot_silu1(float x)  { return x * ot_sigmoid(x); }

void ot_silu_(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = ot_silu1(x[i]);
}

void ot_swiglu(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = ot_silu1(gate[i]) * up[i];
}

/* ---- elementwise ------------------------------------------------------- */

void ot_add_(float *acc, const float *x, int n) {
    for (int i = 0; i < n; i++) acc[i] += x[i];
}
void ot_addscaled_(float *acc, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) acc[i] += s * x[i];
}
void ot_scale_(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}
void ot_mul_(float *acc, const float *x, int n) {
    for (int i = 0; i < n; i++) acc[i] *= x[i];
}
float ot_dot(const float *a, const float *b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}
float ot_l2norm(const float *x, int n) {
    return sqrtf(ot_dot(x, x, n));
}
void ot_l2normalize(const float *x, float *out, int n, float eps) {
    float inv = 1.0f / (ot_l2norm(x, n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

/* ---- RoPE -------------------------------------------------------------- */
/* NeoX / HF "rotate-half" layout: the head vector is split into two halves and
 * dimension i is paired with dimension i + head_dim/2. Frequency for pair i is
 * theta^(-2i/head_dim); the rotation angle at position p is p * freq. */

void ot_rope_inplace(float *v, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
        float ang  = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}
