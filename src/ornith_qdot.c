/* ornith_qdot.c — quant-aware integer dot products.
 *
 * Portable scalar implementations of the ds4 / ggml "quantize activation to
 * Q8_K, integer vec_dot against quantized weights" kernels. Covers the types
 * the 9B uses on the CPU hot path (Q4_K weights everywhere, Q6_K lm_head) plus
 * Q8_0. Anything else returns false from qdot_can() so the caller falls back to
 * the dequant + f32 dot path. See ornith_qdot.h for the rationale.
 *
 * Block byte layouts match ornith_quant.c's decoders exactly:
 *   Q4_K (144B): d:f16 dmin:f16 scales[12] qs[128]
 *   Q6_K (210B): ql[128] qh[64] scales[16]:int8 d:f16
 *   Q8_0 (34B/blk of 32): d:f16 qs[32]:int8
 *
 * Attribution: derived from ggml (MIT) via the ds4 reference; see LICENSE.
 */
#include "ornith_qdot.h"
#include "ornith_quant.h"   /* oq_f16_to_f32, oggml_type enum */
#include <math.h>
#include <string.h>

#define QK_K OQDOT_QK_K

/* Compile-time check that our Q8_K block matches the ggml/ds4 292-byte layout. */
typedef char oq8k_block_size_check[(sizeof(oq8k_block) == 292) ? 1 : -1];

static uint16_t le_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

/* ---- activation -> Q8_K (ds4 ds4_quantize_row_q8_K) --------------------- */

void qdot_quantize_row_q8_K(const float *x, oq8k_block *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += QK_K;
    }
}

/* Unpack the 6-bit scale `d` and 6-bit min `m` for sub-block j (0..7) from a
 * Q4_K scales[12] field (matches ornith_quant.c get_scale_min_k4). */
static void q4k_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);
    }
}

/* ---- Q4_K x Q8_K (ds4 ds4_vec_dot_q4_K_q8_K, scalar path) -------------- */

static void vec_dot_q4_K(int n, float *s, const uint8_t *wr, const oq8k_block *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *blk = wr + (size_t)i * 144;
        const float d  =  y[i].d * oq_f16_to_f32(le_u16(blk));
        const float dm = -y[i].d * oq_f16_to_f32(le_u16(blk + 2));
        const uint8_t *sc = blk + 4;       /* scales[12] */
        const uint8_t *qs = blk + 16;      /* qs[128]    */
        const int8_t  *q8 = y[i].qs;

        int summs = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t scv, mv; q4k_scale_min(j, sc, &scv, &mv);
            summs += mv * ((int)y[i].bsums[j * 2] + (int)y[i].bsums[j * 2 + 1]);
        }
        int isum = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t scv, mv; q4k_scale_min(j, sc, &scv, &mv);
            const int byte_off = (j >> 1) * 32;
            const int shift     = (j & 1) * 4;
            int acc = 0;
            for (int l = 0; l < 32; l++)
                acc += (int)((qs[byte_off + l] >> shift) & 0xF) * (int)q8[j * 32 + l];
            isum += acc * scv;
        }
        sumf += d * (float)isum + dm * (float)summs;
    }
    *s = sumf;
}

/* ---- Q6_K x Q8_K (ggml ggml_vec_dot_q6_K_q8_K, scalar path) ------------ */

static void vec_dot_q6_K(int n, float *s, const uint8_t *wr, const oq8k_block *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *blk = wr + (size_t)i * 210;
        const uint8_t *ql = blk;            /* 128 */
        const uint8_t *qh = blk + 128;      /* 64  */
        const int8_t  *sc = (const int8_t *)(blk + 192); /* 16 */
        const float d = y[i].d * oq_f16_to_f32(le_u16(blk + 208));
        const int8_t *q8 = y[i].qs;

        int isum = 0;
        for (int n2 = 0; n2 < QK_K; n2 += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                isum += (int)q8[l +  0] * q1 * sc[is + 0]
                      + (int)q8[l + 32] * q2 * sc[is + 2]
                      + (int)q8[l + 64] * q3 * sc[is + 4]
                      + (int)q8[l + 96] * q4 * sc[is + 6];
            }
            q8 += 128; ql += 64; qh += 32; sc += 8;
        }
        sumf += d * (float)isum;
    }
    *s = sumf;
}

/* ---- Q8_0 x Q8_K ------------------------------------------------------- *
 * Q8_0 weights are 32-element blocks (f16 d + int8 qs[32]); a Q8_K activation
 * block spans 8 of them. Both sides are int8, so this is a plain integer dot
 * scaled by the two per-block d's. */
static void vec_dot_q8_0(int n, float *s, const uint8_t *wr, const oq8k_block *y) {
    const int nb = n / QK_K;             /* Q8_K super-blocks */
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const int8_t *q8 = y[i].qs;
        const uint8_t *w = wr + (size_t)i * 8 * 34;  /* 8 Q8_0 blocks per 256 */
        float blk_sum = 0.0f;
        for (int sb = 0; sb < 8; sb++) {
            const uint8_t *wb = w + (size_t)sb * 34;
            float wd = oq_f16_to_f32(le_u16(wb));
            const int8_t *wq = (const int8_t *)(wb + 2);
            const int8_t *yq = q8 + sb * 32;
            int isum = 0;
            for (int l = 0; l < 32; l++) isum += (int)wq[l] * (int)yq[l];
            blk_sum += wd * (float)isum;
        }
        sumf += y[i].d * blk_sum;
    }
    *s = sumf;
}

/* ---- dispatch ---------------------------------------------------------- */

bool qdot_can(uint32_t wtype, int n) {
    if (n % QK_K != 0) return false;
    switch (wtype) {
    case OGGML_Q4_K:
    case OGGML_Q6_K:
    case OGGML_Q8_0:
        return true;
    default:
        return false;
    }
}

void qdot_vec_dot(uint32_t wtype, int n, float *s,
                  const void *wrow, const oq8k_block *yq) {
    const uint8_t *wr = (const uint8_t *)wrow;
    switch (wtype) {
    case OGGML_Q4_K: vec_dot_q4_K(n, s, wr, yq); break;
    case OGGML_Q6_K: vec_dot_q6_K(n, s, wr, yq); break;
    case OGGML_Q8_0: vec_dot_q8_0(n, s, wr, yq); break;
    default: *s = 0.0f; break;  /* unreachable when guarded by qdot_can */
    }
}
