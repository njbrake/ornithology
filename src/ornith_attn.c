/* ornith_attn.c — full GQA attention + gated delta-net linear attention. */
#include "ornith_attn.h"
#include "ornith_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================= */
/*  Full GQA causal attention                                              */
/* ======================================================================= */

size_t ornith_kv_bytes_per_token(int n_kv_heads, int head_dim, int quantized) {
    size_t hv = (size_t)n_kv_heads * head_dim;
    if (quantized)
        /* K + V: int8 codes (1 byte each) + one f32 scale per head, x2 (K,V). */
        return 2 * (hv * sizeof(int8_t) + (size_t)n_kv_heads * sizeof(float));
    return 2 * hv * sizeof(float);   /* K + V, fp32 */
}

ornith_status ornith_kv_init_ex(ornith_kv_cache *c, int capacity,
                                int n_kv_heads, int head_dim, int quantized) {
    memset(c, 0, sizeof(*c));
    c->capacity = capacity; c->n_kv_heads = n_kv_heads; c->head_dim = head_dim;
    c->quantized = quantized ? 1 : 0;
    size_t per   = (size_t)capacity * n_kv_heads * head_dim;
    size_t scl   = (size_t)capacity * n_kv_heads;
    if (c->quantized) {
        c->Kq = calloc(per, sizeof(int8_t));
        c->Vq = calloc(per, sizeof(int8_t));
        c->Ks = calloc(scl, sizeof(float));
        c->Vs = calloc(scl, sizeof(float));
        if (!c->Kq || !c->Vq || !c->Ks || !c->Vs) {
            free(c->Kq); free(c->Vq); free(c->Ks); free(c->Vs);
            memset(c, 0, sizeof(*c));
            ornith_set_error("kv cache (q8) oom (capacity %d)", capacity);
            return ORNITH_ERR_OOM;
        }
        return ORNITH_OK;
    }
    c->K = calloc(per, sizeof(float));
    c->V = calloc(per, sizeof(float));
    if (!c->K || !c->V) {
        free(c->K); free(c->V); memset(c, 0, sizeof(*c));
        ornith_set_error("kv cache oom (capacity %d)", capacity);
        return ORNITH_ERR_OOM;
    }
    return ORNITH_OK;
}

ornith_status ornith_kv_init(ornith_kv_cache *c, int capacity,
                             int n_kv_heads, int head_dim) {
    return ornith_kv_init_ex(c, capacity, n_kv_heads, head_dim, 0);
}

void ornith_kv_free(ornith_kv_cache *c) {
    if (!c) return;
    free(c->K); free(c->V); free(c->Kq); free(c->Vq); free(c->Ks); free(c->Vs);
    memset(c, 0, sizeof(*c));
}
void ornith_kv_reset(ornith_kv_cache *c) { c->len = 0; }

/* Symmetric per-vector int8 quantize of `x` [n]: scale = amax/127, code =
 * round(x/scale) clamped to [-127,127]. Returns the scale (0 for an all-zero
 * vector, in which case all codes are 0). */
static float kv_quant_vec(const float *x, int n, int8_t *q) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(q, 0, (size_t)n * sizeof(int8_t)); return 0.0f; }
    float scale = amax / 127.0f, inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(x[i] * inv);
        if (v >  127) v =  127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    return scale;
}

/* dot(q_f32, dequant(code, scale)) = scale * sum(q_i * code_i). */
static float kv_dot_q8(const float *qf, const int8_t *code, float scale, int n) {
    if (scale == 0.0f) return 0.0f;
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += qf[i] * (float)code[i];
    return acc * scale;
}

void ornith_gqa_step(ornith_kv_cache *c, const float *q,
                     const float *k_new, const float *v_new,
                     int n_heads, float *out, float scale) {
    int hd = c->head_dim, nkv = c->n_kv_heads;
    if (scale <= 0.0f) scale = 1.0f / sqrtf((float)hd);
    if (c->len >= c->capacity) return; /* caller guarantees capacity */

    int pos = c->len;
    if (c->quantized) {
        /* append: quantize each head's K,V head-vector to int8 + scale */
        for (int h = 0; h < nkv; h++) {
            size_t off = (size_t)pos * nkv + h;
            c->Ks[off] = kv_quant_vec(k_new + (size_t)h * hd, hd,
                                      c->Kq + off * hd);
            c->Vs[off] = kv_quant_vec(v_new + (size_t)h * hd, hd,
                                      c->Vq + off * hd);
        }
    } else {
        memcpy(c->K + (size_t)pos * nkv * hd, k_new,
               (size_t)nkv * hd * sizeof(float));
        memcpy(c->V + (size_t)pos * nkv * hd, v_new,
               (size_t)nkv * hd * sizeof(float));
    }
    c->len++;

    int T = c->len;
    int group = n_heads / nkv;   /* query heads per kv head */
    float *scores = malloc((size_t)T * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        int kvh = h / group;
        const float *qh = q + (size_t)h * hd;
        if (c->quantized) {
            for (int t = 0; t < T; t++) {
                size_t off = (size_t)t * nkv + kvh;
                scores[t] = kv_dot_q8(qh, c->Kq + off * hd, c->Ks[off], hd) * scale;
            }
        } else {
            for (int t = 0; t < T; t++) {
                const float *kt = c->K + ((size_t)t * nkv + kvh) * hd;
                scores[t] = ot_dot(qh, kt, hd) * scale;
            }
        }
        ot_softmax(scores, T);
        float *oh = out + (size_t)h * hd;
        memset(oh, 0, (size_t)hd * sizeof(float));
        if (c->quantized) {
            for (int t = 0; t < T; t++) {
                size_t off = (size_t)t * nkv + kvh;
                /* oh += scores[t] * dequant(Vq) = (scores[t]*Vs) * code */
                float sv = scores[t] * c->Vs[off];
                if (sv == 0.0f) continue;
                const int8_t *vt = c->Vq + off * hd;
                for (int i = 0; i < hd; i++) oh[i] += sv * (float)vt[i];
            }
        } else {
            for (int t = 0; t < T; t++) {
                const float *vt = c->V + ((size_t)t * nkv + kvh) * hd;
                ot_addscaled_(oh, vt, scores[t], hd);
            }
        }
    }
    free(scores);
}

void ornith_gqa_reference(const float *Q, const float *K, const float *V,
                          int T, int n_heads, int n_kv_heads, int head_dim,
                          float *O, float scale) {
    int hd = head_dim, nkv = n_kv_heads;
    if (scale <= 0.0f) scale = 1.0f / sqrtf((float)hd);
    int group = n_heads / nkv;
    float *scores = malloc((size_t)T * sizeof(float));
    for (int i = 0; i < T; i++) {            /* query position */
        for (int h = 0; h < n_heads; h++) {
            int kvh = h / group;
            const float *qh = Q + ((size_t)i * n_heads + h) * hd;
            for (int j = 0; j <= i; j++) {   /* causal: attend j<=i */
                const float *kj = K + ((size_t)j * nkv + kvh) * hd;
                scores[j] = ot_dot(qh, kj, hd) * scale;
            }
            ot_softmax(scores, i + 1);
            float *oh = O + ((size_t)i * n_heads + h) * hd;
            memset(oh, 0, (size_t)hd * sizeof(float));
            for (int j = 0; j <= i; j++) {
                const float *vj = V + ((size_t)j * nkv + kvh) * hd;
                ot_addscaled_(oh, vj, scores[j], hd);
            }
        }
    }
    free(scores);
}

/* ======================================================================= */
/*  Gated delta-net linear attention                                      */
/* ======================================================================= */

ornith_status ornith_delta_init(ornith_delta_state *s, int dk, int dv) {
    s->dk = dk; s->dv = dv;
    s->S = calloc((size_t)dk * dv, sizeof(float));
    if (!s->S) { ornith_set_error("delta state oom"); return ORNITH_ERR_OOM; }
    return ORNITH_OK;
}
void ornith_delta_free(ornith_delta_state *s) {
    if (!s) return;
    free(s->S); s->S = NULL;
}
void ornith_delta_reset(ornith_delta_state *s) {
    memset(s->S, 0, (size_t)s->dk * s->dv * sizeof(float));
}

/* S is stored [dv, dk] row-major: S[r*dk + c], r in [0,dv), c in [0,dk). */

void ornith_delta_step(ornith_delta_state *s, const float *q, const float *k,
                       const float *v, float alpha, float beta, float *o) {
    int dk = s->dk, dv = s->dv;
    float *S = s->S;
    /* S' = alpha * S */
    for (int i = 0; i < dk * dv; i++) S[i] *= alpha;
    /* u = beta * (v - S' k) ; then S += u k^T ; o = S q */
    for (int r = 0; r < dv; r++) {
        const float *Sr = S + (size_t)r * dk;
        float Sk = ot_dot(Sr, k, dk);
        float u  = beta * (v[r] - Sk);
        float *Sw = S + (size_t)r * dk;
        for (int c = 0; c < dk; c++) Sw[c] += u * k[c];
        o[r] = ot_dot(Sw, q, dk);
    }
}

/* Chunked parallel scan. For a chunk of length C with incoming state S_in:
 *
 *   p_i = prod_{l<=i} alpha_l              (cumulative decay within chunk)
 *   W_i = alpha_i S_{i-1}                  (state just before the i-th write)
 *       = p_i S_in + sum_{j<i} (p_i/p_j) u_j k_j^T
 *   u_i = beta_i (v_i - W_i k_i)
 *       = t_i - sum_{j<i} M_{ij} u_j ,
 *     with t_i   = beta_i v_i - beta_i p_i (S_in k_i)
 *          M_{ij}= beta_i (p_i/p_j)(k_j . k_i)   (j<i, strictly lower-tri)
 *   => (I + M) U = T, solved by forward substitution (the intra-chunk solve;
 *      GPU kernels invert the CxC block instead — same result).
 *   o_i = p_i (S_in q_i) + sum_{j<=i} (p_i/p_j)(k_j . q_i) u_j
 *   S_out = p_{C-1} S_in + sum_j (p_{C-1}/p_j) u_j k_j^T
 *
 * S_in projections (S_in k_i, S_in q_i) are computed against the *incoming*
 * state for the whole chunk before S is mutated, so chunks compose exactly.
 *
 * NUMERICAL STABILITY: the within-chunk decay ratios p_i/p_j (j<=i) are NOT
 * formed by dividing the two cumulative products p_i and p_j. With real-model
 * decays (alpha can be well below 1) p_i underflows to 0 within a chunk of a
 * few dozen steps, and p_i/p_j would become 0/0 = NaN. Instead each ratio
 * p_i/p_j = prod_{l=j+1}^{i} alpha_l is accumulated directly as a bounded
 * sub-product (always in (0,1], underflow at worst yields a harmless 0). The
 * standalone p_i (decay of the *incoming* state into position i) is still
 * accumulated cumulatively; underflowing it to 0 is correct — the carried
 * state has simply decayed away — and never produces a NaN. This is exactly
 * the form a chunked GPU kernel must use (decay computed per-block, not via a
 * global cumulative product), so it doubles as the Metal kernel reference.
 */
static void delta_chunk(ornith_delta_state *s, const float *Q, const float *K,
                        const float *V, const float *alpha, const float *beta,
                        int C, float *O) {
    int dk = s->dk, dv = s->dv;
    float *S = s->S;

    float *p   = malloc((size_t)C * sizeof(float));
    float *Sk  = malloc((size_t)C * dv * sizeof(float)); /* S_in k_i  [C,dv] */
    float *Sq  = malloc((size_t)C * dv * sizeof(float)); /* S_in q_i  [C,dv] */
    float *U   = malloc((size_t)C * dv * sizeof(float)); /* delta vals [C,dv]*/
    float *kk  = malloc((size_t)C * C  * sizeof(float)); /* k_j . k_i */
    float *kq  = malloc((size_t)C * C  * sizeof(float)); /* k_j . q_i */

    /* cumulative decay */
    for (int i = 0; i < C; i++)
        p[i] = (i == 0 ? alpha[0] : p[i-1] * alpha[i]);

    /* S_in projections (against incoming state, before any mutation) and grams */
    for (int i = 0; i < C; i++) {
        const float *ki = K + (size_t)i * dk;
        const float *qi = Q + (size_t)i * dk;
        for (int r = 0; r < dv; r++) {
            const float *Sr = S + (size_t)r * dk;
            Sk[(size_t)i*dv + r] = ot_dot(Sr, ki, dk);
            Sq[(size_t)i*dv + r] = ot_dot(Sr, qi, dk);
        }
        for (int j = 0; j < C; j++) {
            const float *kj = K + (size_t)j * dk;
            kk[(size_t)i*C + j] = ot_dot(kj, ki, dk);  /* k_j . k_i */
            kq[(size_t)i*C + j] = ot_dot(kj, qi, dk);  /* k_j . q_i */
        }
    }

    /* solve (I+M)U = T by forward substitution (row i depends on rows j<i).
     * ratio = p_i/p_j is accumulated downward as prod_{l=j+1}^{i} alpha_l. */
    for (int i = 0; i < C; i++) {
        float bi = beta[i], pi = p[i];
        float *ui = U + (size_t)i * dv;
        const float *vi = V + (size_t)i * dv;
        for (int r = 0; r < dv; r++)
            ui[r] = bi * vi[r] - bi * pi * Sk[(size_t)i*dv + r];   /* t_i */
        float ratio = 1.0f;                       /* ratio = p_i/p_j */
        for (int j = i - 1; j >= 0; j--) {
            ratio *= alpha[j + 1];                /* prod_{l=j+1}^{i} alpha_l */
            float Mij = bi * ratio * kk[(size_t)i*C + j];
            if (Mij != 0.0f) {
                const float *uj = U + (size_t)j * dv;
                for (int r = 0; r < dv; r++) ui[r] -= Mij * uj[r];
            }
        }
    }

    /* outputs */
    for (int i = 0; i < C; i++) {
        float pi = p[i];
        float *oi = O + (size_t)i * dv;
        for (int r = 0; r < dv; r++) oi[r] = pi * Sq[(size_t)i*dv + r];
        /* j == i term (ratio == 1) */
        {
            float w = kq[(size_t)i*C + i];
            const float *ui = U + (size_t)i * dv;
            for (int r = 0; r < dv; r++) oi[r] += w * ui[r];
        }
        float ratio = 1.0f;                       /* ratio = p_i/p_j */
        for (int j = i - 1; j >= 0; j--) {
            ratio *= alpha[j + 1];
            float w = ratio * kq[(size_t)i*C + j];
            const float *uj = U + (size_t)j * dv;
            for (int r = 0; r < dv; r++) oi[r] += w * uj[r];
        }
    }

    /* state update: S_out = p_{C-1} S_in + sum_j (p_{C-1}/p_j) u_j k_j^T.
     * ratio = p_{C-1}/p_j accumulated downward as prod_{l=j+1}^{C-1} alpha_l. */
    float pC = p[C-1];
    for (int i = 0; i < dk * dv; i++) S[i] *= pC;
    float ratio = 1.0f;
    for (int j = C - 1; j >= 0; j--) {
        float wj = ratio;                         /* p_{C-1}/p_j */
        const float *uj = U + (size_t)j * dv;
        const float *kj = K + (size_t)j * dk;
        for (int r = 0; r < dv; r++) {
            float ur = wj * uj[r];
            if (ur == 0.0f) continue;
            float *Sr = S + (size_t)r * dk;
            for (int c = 0; c < dk; c++) Sr[c] += ur * kj[c];
        }
        ratio *= alpha[j];                        /* next: p_{C-1}/p_{j-1} */
    }

    free(p); free(Sk); free(Sq); free(U); free(kk); free(kq);
}

void ornith_delta_prefill(ornith_delta_state *s, const float *Q, const float *K,
                          const float *V, const float *alpha, const float *beta,
                          int T, int chunk, float *O) {
    int dk = s->dk, dv = s->dv;
    if (chunk < 1) chunk = 1;
    if (chunk > T) chunk = T;
    for (int c0 = 0; c0 < T; c0 += chunk) {
        int C = (c0 + chunk <= T) ? chunk : (T - c0);
        delta_chunk(s,
            Q + (size_t)c0 * dk, K + (size_t)c0 * dk, V + (size_t)c0 * dv,
            alpha + c0, beta + c0, C, O + (size_t)c0 * dv);
    }
}

/* ======================================================================= */
/*  Short causal depthwise conv                                           */
/* ======================================================================= */

ornith_status ornith_conv_init(ornith_conv_state *cs, int K, int C) {
    cs->K = K; cs->C = C;
    cs->buf = calloc((size_t)(K > 0 ? K-1 : 0) * C + 1, sizeof(float));
    if (!cs->buf) { ornith_set_error("conv state oom"); return ORNITH_ERR_OOM; }
    return ORNITH_OK;
}
void ornith_conv_free(ornith_conv_state *cs) {
    if (!cs) return;
    free(cs->buf); cs->buf = NULL;
}
void ornith_conv_reset(ornith_conv_state *cs) {
    memset(cs->buf, 0, (size_t)(cs->K > 0 ? cs->K-1 : 0) * cs->C * sizeof(float));
}

/* buf holds the last K-1 inputs, oldest first: buf[j*C + c], j in [0,K-1). */
void ornith_conv_step(ornith_conv_state *cs, const float *x_in,
                      const float *weight, const float *bias, float *out) {
    int K = cs->K, C = cs->C;
    for (int c = 0; c < C; c++) {
        float acc = bias ? bias[c] : 0.0f;
        const float *w = weight + (size_t)c * K;
        /* window: buf[0..K-2] then current x_in; j=K-1 weights the current. */
        for (int j = 0; j < K - 1; j++)
            acc += w[j] * cs->buf[(size_t)j * C + c];
        acc += w[K-1] * x_in[c];
        out[c] = acc;
    }
    /* shift state: drop oldest, append current */
    for (int j = 0; j < K - 2; j++)
        memcpy(cs->buf + (size_t)j * C, cs->buf + (size_t)(j+1) * C,
               (size_t)C * sizeof(float));
    if (K >= 2) memcpy(cs->buf + (size_t)(K-2) * C, x_in,
                       (size_t)C * sizeof(float));
}

void ornith_conv_prefill(ornith_conv_state *cs, const float *X,
                         const float *weight, const float *bias,
                         int T, float *OUT) {
    int C = cs->C;
    for (int t = 0; t < T; t++)
        ornith_conv_step(cs, X + (size_t)t * C, weight, bias,
                         OUT + (size_t)t * C);
}
