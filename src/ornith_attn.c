/* ornith_attn.c — full GQA attention + gated delta-net linear attention. */
#include "ornith_attn.h"
#include "ornith_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================= */
/*  Full GQA causal attention                                              */
/* ======================================================================= */

ornith_status ornith_kv_init(ornith_kv_cache *c, int capacity,
                             int n_kv_heads, int head_dim) {
    memset(c, 0, sizeof(*c));
    c->capacity = capacity; c->n_kv_heads = n_kv_heads; c->head_dim = head_dim;
    size_t per = (size_t)capacity * n_kv_heads * head_dim;
    c->K = calloc(per, sizeof(float));
    c->V = calloc(per, sizeof(float));
    if (!c->K || !c->V) {
        free(c->K); free(c->V); memset(c, 0, sizeof(*c));
        ornith_set_error("kv cache oom (capacity %d)", capacity);
        return ORNITH_ERR_OOM;
    }
    return ORNITH_OK;
}

void ornith_kv_free(ornith_kv_cache *c) {
    if (!c) return;
    free(c->K); free(c->V); memset(c, 0, sizeof(*c));
}
void ornith_kv_reset(ornith_kv_cache *c) { c->len = 0; }

void ornith_gqa_step(ornith_kv_cache *c, const float *q,
                     const float *k_new, const float *v_new,
                     int n_heads, float *out, float scale) {
    int hd = c->head_dim, nkv = c->n_kv_heads;
    if (scale <= 0.0f) scale = 1.0f / sqrtf((float)hd);
    if (c->len >= c->capacity) return; /* caller guarantees capacity */

    /* append new K,V at position len */
    int pos = c->len;
    memcpy(c->K + (size_t)pos * nkv * hd, k_new, (size_t)nkv * hd * sizeof(float));
    memcpy(c->V + (size_t)pos * nkv * hd, v_new, (size_t)nkv * hd * sizeof(float));
    c->len++;

    int T = c->len;
    int group = n_heads / nkv;   /* query heads per kv head */
    float *scores = malloc((size_t)T * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        int kvh = h / group;
        const float *qh = q + (size_t)h * hd;
        for (int t = 0; t < T; t++) {
            const float *kt = c->K + ((size_t)t * nkv + kvh) * hd;
            scores[t] = ot_dot(qh, kt, hd) * scale;
        }
        ot_softmax(scores, T);
        float *oh = out + (size_t)h * hd;
        memset(oh, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *vt = c->V + ((size_t)t * nkv + kvh) * hd;
            ot_addscaled_(oh, vt, scores[t], hd);
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

    /* solve (I+M)U = T by forward substitution (row i depends on rows j<i) */
    for (int i = 0; i < C; i++) {
        float bi = beta[i], pi = p[i];
        float *ui = U + (size_t)i * dv;
        const float *vi = V + (size_t)i * dv;
        for (int r = 0; r < dv; r++)
            ui[r] = bi * vi[r] - bi * pi * Sk[(size_t)i*dv + r];   /* t_i */
        for (int j = 0; j < i; j++) {
            float Mij = bi * (pi / p[j]) * kk[(size_t)i*C + j];
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
        for (int j = 0; j <= i; j++) {
            float w = (pi / p[j]) * kq[(size_t)i*C + j];
            const float *uj = U + (size_t)j * dv;
            for (int r = 0; r < dv; r++) oi[r] += w * uj[r];
        }
    }

    /* state update: S_out = p_{C-1} S_in + sum_j (p_{C-1}/p_j) u_j k_j^T */
    float pC = p[C-1];
    for (int i = 0; i < dk * dv; i++) S[i] *= pC;
    for (int j = 0; j < C; j++) {
        float wj = pC / p[j];
        const float *uj = U + (size_t)j * dv;
        const float *kj = K + (size_t)j * dk;
        for (int r = 0; r < dv; r++) {
            float ur = wj * uj[r];
            if (ur == 0.0f) continue;
            float *Sr = S + (size_t)r * dk;
            for (int c = 0; c < dk; c++) Sr[c] += ur * kj[c];
        }
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
