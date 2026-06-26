/* ornith_quant.c — float conversions + ggml-compatible quant block codecs.
 *
 * Block layouts (little-endian, matching ggml):
 *   Q8_0: per 32 elems -> { f16 d; int8 qs[32] }            = 34 bytes
 *   Q4_0: per 32 elems -> { f16 d; uint8 qs[16] }           = 18 bytes
 * F32/F16/BF16 are stored flat (one value each, no blocks).
 */
#include "ornith_quant.h"
#include <string.h>
#include <math.h>

/* ---- half / bfloat16 --------------------------------------------------- */

uint16_t oq_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xffu;
    uint32_t mant =  x & 0x7fffffu;

    if (e == 0xff)                       /* inf / nan */
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));

    int32_t exp = (int32_t)e - 127 + 15; /* rebias */
    if (exp >= 0x1f)                     /* overflow -> inf */
        return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {                      /* subnormal / underflow */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift   = (uint32_t)(14 - exp);
        uint16_t half    = (uint16_t)(mant >> shift);
        uint32_t rem     = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) half++;
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    uint32_t rem  = mant & 0x1fffu;      /* round to nearest, ties to even */
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return half;
}

float oq_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1fu;
    uint32_t mant =  h & 0x3ffu;
    uint32_t f;
    if (e == 0) {
        if (mant == 0) {
            f = sign;
        } else {                          /* subnormal */
            int32_t exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            f = sign | ((uint32_t)exp << 23) | (mant << 13);
        }
    } else if (e == 0x1f) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((e - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

uint16_t oq_f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(x >> 16); /* inf/nan */
    uint32_t rounding = 0x7fffu + ((x >> 16) & 1u);               /* nearest-even */
    x += rounding;
    return (uint16_t)(x >> 16);
}

float oq_bf16_to_f32(uint16_t b) {
    uint32_t x = (uint32_t)b << 16;
    float out;
    memcpy(&out, &x, 4);
    return out;
}

/* ---- block geometry ---------------------------------------------------- */

#define QK 32  /* block size shared by Q8_0 and Q4_0 */

#define QK_K 256  /* super-block size shared by the k-quants */

size_t oq_block_elems(uint32_t type) {
    switch (type) {
    case OGGML_F32: case OGGML_F16: case OGGML_BF16: return 1;
    case OGGML_Q8_0: case OGGML_Q4_0:                return QK;
    case OGGML_Q2_K: case OGGML_Q4_K: case OGGML_Q5_K: case OGGML_Q6_K:
    case OGGML_IQ2_XXS:
        return QK_K;
    default: return 0;
    }
}

size_t oq_block_bytes(uint32_t type) {
    switch (type) {
    case OGGML_F32:  return 4;
    case OGGML_F16:  case OGGML_BF16: return 2;
    case OGGML_Q8_0: return 2 + QK;        /* f16 d + 32 int8 = 34 */
    case OGGML_Q4_0: return 2 + QK / 2;    /* f16 d + 16 nibbles = 18 */
    case OGGML_Q2_K: return 84;            /* scales[16] + qs[64] + d + dmin */
    case OGGML_Q4_K: return 144;           /* d + dmin + scales[12] + qs[128] */
    case OGGML_Q5_K: return 176;           /* + qh[32] vs Q4_K */
    case OGGML_Q6_K: return 210;           /* ql[128] + qh[64] + sc[16] + d */
    case OGGML_IQ2_XXS: return 66;         /* f16 d + uint16 qs[32] */
    default: return 0;
    }
}

size_t oq_row_bytes(uint32_t type, size_t n_elems) {
    size_t be = oq_block_elems(type);
    if (!be) return 0;
    if (n_elems % be) return 0;
    return (n_elems / be) * oq_block_bytes(type);
}

bool oq_is_implemented(uint32_t type) {
    /* Encode-capable types (oq_quantize). The quantizer uses this to decide
     * when to fall back to F16. Q2_K/Q5_K decode (oq_dequantize) but are not yet
     * encoded, so they are intentionally excluded here. */
    switch (type) {
    case OGGML_F32: case OGGML_F16: case OGGML_BF16:
    case OGGML_Q8_0: case OGGML_Q4_0:
    case OGGML_Q2_K: case OGGML_Q4_K: case OGGML_Q5_K: case OGGML_Q6_K:
    case OGGML_IQ2_XXS:
        return true;
    default:
        return false;
    }
}

/* True if oq_dequantize can decode `type` (a superset of oq_is_implemented:
 * also covers the read-only Q2_K / Q5_K paths used to load real GGUFs). */
bool oq_can_decode(uint32_t type) {
    switch (type) {
    case OGGML_IQ2_XXS:
        return true;
    default:
        return oq_is_implemented(type);
    }
}

/* ---- codecs ------------------------------------------------------------ */

static void le_put_u16(uint8_t *p, uint16_t v) { p[0] = v & 0xff; p[1] = v >> 8; }
static uint16_t le_get_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static void quantize_q8_0(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * (2 + QK);
        float amax = 0.0f;
        for (int j = 0; j < QK; j++) {
            float a = fabsf(x[j]);
            if (a > amax) amax = a;
        }
        float d  = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        le_put_u16(o, oq_f32_to_f16(d));
        for (int j = 0; j < QK; j++) {
            float v = x[j] * id;
            int32_t q = (int32_t)lroundf(v);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            o[2 + j] = (uint8_t)(int8_t)q;
        }
    }
}

static void dequantize_q8_0(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *o = src + b * (2 + QK);
        float d = oq_f16_to_f32(le_get_u16(o));
        for (int j = 0; j < QK; j++)
            dst[b * QK + j] = (float)(int8_t)o[2 + j] * d;
    }
}

static void quantize_q4_0(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK;
        uint8_t *o = dst + b * (2 + QK / 2);
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK; j++) {
            float a = fabsf(x[j]);
            if (a > amax) { amax = a; max = x[j]; }
        }
        float d  = max / -8.0f;          /* ggml convention: signed scale */
        float id = d ? 1.0f / d : 0.0f;
        le_put_u16(o, oq_f32_to_f16(d));
        for (int j = 0; j < QK / 2; j++) {
            float v0 = x[j]          * id + 8.5f;
            float v1 = x[j + QK / 2] * id + 8.5f;
            int q0 = (int)v0; if (q0 > 15) q0 = 15; if (q0 < 0) q0 = 0;
            int q1 = (int)v1; if (q1 > 15) q1 = 15; if (q1 < 0) q1 = 0;
            o[2 + j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

static void dequantize_q4_0(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *o = src + b * (2 + QK / 2);
        float d = oq_f16_to_f32(le_get_u16(o));
        for (int j = 0; j < QK / 2; j++) {
            int q0 = (o[2 + j] & 0x0f) - 8;
            int q1 = (o[2 + j] >>   4) - 8;
            dst[b * QK + j]          = (float)q0 * d;
            dst[b * QK + j + QK / 2] = (float)q1 * d;
        }
    }
}

/* ---- k-quant super-blocks (QK_K = 256), ggml-compatible byte layouts ----
 * The decoders mirror ggml's dequantize_row_* exactly so real GGUFs load
 * correctly. The Q2_K/Q4_K/Q5_K/Q6_K encoders are round-to-nearest and the
 * IQ2_XXS encoder is a greedy nearest-grid search: all emit valid, interoperable
 * blocks (quality a touch below ggml's iterative search / kmeans neighbour table,
 * which is an M3 refinement). */

/* Unpack the 6-bit scale `d` and 6-bit min `m` for sub-block j (0..7) from a
 * Q4_K/Q5_K scales[12] field. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);
    }
}

/* Inverse of get_scale_min_k4: pack 8 sub-block 6-bit scales/mins -> scales[12]. */
static void put_scale_min_k4(uint8_t *q, const uint8_t *sc, const uint8_t *mn) {
    for (int i = 0; i < 12; i++) q[i] = 0;
    for (int j = 0; j < 4; j++) { q[j] = sc[j] & 63; q[j + 4] = mn[j] & 63; }
    for (int j = 4; j < 8; j++) {
        q[j + 4] = (uint8_t)((sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4));
        q[j - 4] |= (uint8_t)(((sc[j] >> 4) & 3) << 6);
        q[j    ] |= (uint8_t)(((mn[j] >> 4) & 3) << 6);
    }
}

static void dequantize_q2_K(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *p = src + b * 84;
        const uint8_t *scales = p;        /* 16 */
        const uint8_t *q = p + 16;        /* 64 */
        float d    = oq_f16_to_f32(le_get_u16(p + 80));
        float dmin = oq_f16_to_f32(le_get_u16(p + 82));
        float *y = dst + b * QK_K;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l+16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dequantize_q4_K(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *p = src + b * 144;
        float d    = oq_f16_to_f32(le_get_u16(p));
        float dmin = oq_f16_to_f32(le_get_u16(p + 2));
        const uint8_t *scales = p + 4;    /* 12 */
        const uint8_t *q = p + 16;        /* 128 */
        float *y = dst + b * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m); float d1 = d*sc, m1 = dmin*m;
            get_scale_min_k4(is + 1, scales, &sc, &m); float d2 = d*sc, m2 = dmin*m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >>  4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dequantize_q5_K(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *p = src + b * 176;
        float d    = oq_f16_to_f32(le_get_u16(p));
        float dmin = oq_f16_to_f32(le_get_u16(p + 2));
        const uint8_t *scales = p + 4;    /* 12 */
        const uint8_t *qh = p + 16;       /* 32 */
        const uint8_t *ql = p + 48;       /* 128 */
        float *y = dst + b * QK_K;
        int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m); float d1 = d*sc, m1 = dmin*m;
            get_scale_min_k4(is + 1, scales, &sc, &m); float d2 = d*sc, m2 = dmin*m;
            for (int l = 0; l < 32; l++)
                *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *y++ = d2 * ((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequantize_q6_K(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *p = src + b * 210;
        const uint8_t *ql = p;            /* 128 */
        const uint8_t *qh = p + 128;      /* 64 */
        const int8_t  *sc = (const int8_t *)(p + 192); /* 16 */
        float d = oq_f16_to_f32(le_get_u16(p + 208));
        float *y = dst + b * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l+ 0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l+32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l+ 0] = d * sc[is + 0] * q1;
                y[l+32] = d * sc[is + 2] * q2;
                y[l+64] = d * sc[is + 4] * q3;
                y[l+96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

/* Q6_K encode (RTN): 16 groups of 16 share an int8 scale; superblock d scales
 * those. value = d * sc[group] * (q - 32), q in [0,63]. */
static void quantize_q6_K(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK_K;
        uint8_t *p = dst + b * 210;
        uint8_t *ql = p, *qh = p + 128;
        int8_t  *sc = (int8_t *)(p + 192);
        float gscale[16], maxs = 0.0f;
        for (int g = 0; g < 16; g++) {
            float amax = 0.0f;
            for (int l = 0; l < 16; l++) { float a = fabsf(x[g*16+l]); if (a > amax) amax = a; }
            gscale[g] = amax / 32.0f;
            if (gscale[g] > maxs) maxs = gscale[g];
        }
        float d = maxs / 127.0f, id = d ? 1.0f / d : 0.0f;
        le_put_u16(p + 208, oq_f32_to_f16(d));
        for (int g = 0; g < 16; g++) {
            int s = (int)lroundf(gscale[g] * id);
            s = s < 0 ? 0 : (s > 127 ? 127 : s);
            sc[g] = (int8_t)s;
        }
        uint8_t Q[256];
        for (int g = 0; g < 16; g++) {
            float gs = d * sc[g], igs = gs ? 1.0f / gs : 0.0f;
            for (int l = 0; l < 16; l++) {
                int q = (int)lroundf(x[g*16+l] * igs) + 32;
                q = q < 0 ? 0 : (q > 63 ? 63 : q);
                Q[g*16+l] = (uint8_t)q;
            }
        }
        memset(ql, 0, 128); memset(qh, 0, 64);
        for (int n = 0; n < QK_K; n += 128) {
            uint8_t *qlp = ql + (n/128)*64, *qhp = qh + (n/128)*32;
            const uint8_t *Qn = Q + n;
            for (int l = 0; l < 32; l++) {
                uint8_t q1 = Qn[l+0], q2 = Qn[l+32], q3 = Qn[l+64], q4 = Qn[l+96];
                qlp[l+ 0] = (uint8_t)((q1 & 0xF) | ((q3 & 0xF) << 4));
                qlp[l+32] = (uint8_t)((q2 & 0xF) | ((q4 & 0xF) << 4));
                qhp[l] = (uint8_t)(((q1>>4)&3) | (((q2>>4)&3)<<2) |
                                   (((q3>>4)&3)<<4) | (((q4>>4)&3)<<6));
            }
        }
    }
}

/* Q4_K encode (RTN): 8 sub-blocks of 32, each an affine (scale, min);
 * value = d*scale*q - dmin*min, q in [0,15]. */
static void quantize_q4_K(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK_K;
        uint8_t *p = dst + b * 144;
        float subscale[8], submin[8];
        for (int j = 0; j < 8; j++) {
            const float *xs = x + j*32;
            float lo = xs[0], hi = xs[0];
            for (int l = 1; l < 32; l++) { if (xs[l] < lo) lo = xs[l]; if (xs[l] > hi) hi = xs[l]; }
            if (lo > 0.0f) lo = 0.0f;          /* min term is subtracted, m>=0 */
            subscale[j] = (hi - lo) / 15.0f;
            submin[j]   = -lo;
        }
        float maxsc = 0.0f, maxmn = 0.0f;
        for (int j = 0; j < 8; j++) {
            if (subscale[j] > maxsc) maxsc = subscale[j];
            if (submin[j]   > maxmn) maxmn = submin[j];
        }
        float d = maxsc / 63.0f, dmin = maxmn / 63.0f;
        float id = d ? 1.0f/d : 0.0f, idm = dmin ? 1.0f/dmin : 0.0f;
        uint8_t sc6[8], mn6[8];
        for (int j = 0; j < 8; j++) {
            int s = (int)lroundf(subscale[j]*id);
            int m = (int)lroundf(submin[j]*idm);
            s = s < 0 ? 0 : (s > 63 ? 63 : s);
            m = m < 0 ? 0 : (m > 63 ? 63 : m);
            sc6[j] = (uint8_t)s; mn6[j] = (uint8_t)m;
        }
        le_put_u16(p,     oq_f32_to_f16(d));
        le_put_u16(p + 2, oq_f32_to_f16(dmin));
        put_scale_min_k4(p + 4, sc6, mn6);
        uint8_t *q = p + 16;
        memset(q, 0, 128);
        for (int j = 0; j < 8; j++) {
            float rs = d * sc6[j], rm = dmin * mn6[j], irs = rs ? 1.0f/rs : 0.0f;
            const float *xs = x + j*32;
            int g = j / 2; bool low = (j % 2) == 0;
            for (int l = 0; l < 32; l++) {
                int v = (int)lroundf((xs[l] + rm) * irs);
                v = v < 0 ? 0 : (v > 15 ? 15 : v);
                if (low) q[g*32 + l] |= (uint8_t)v;
                else     q[g*32 + l] |= (uint8_t)(v << 4);
            }
        }
    }
}

/* ---- IQ2_XXS (sub-2-bit i-quant), ggml-compatible -----------------------
 * Block layout (66 bytes, QK_K = 256 elems): { f16 d; uint16 qs[32] }.
 * Each 32-elem group packs into 4 uint16 = 2 uint32 (aux32[0], aux32[1]):
 *   aux32[0]: four 8-bit grid indices (one per 8-elem sub-group).
 *   aux32[1]: four 7-bit sign indices (bits 0-27) + a 4-bit scale (bits 28-31).
 * Reconstruction: y = d*(0.5 + scale)*0.25 * grid_magnitude * sign.
 * The grid codebook, ksigns and kmask tables are copied verbatim from
 * ggml/llama.cpp (MIT; ggml authors credited in LICENSE). */

static const uint8_t kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/* magnitude byte j (0..7) of grid point g, little-endian (ggml convention) */
static inline uint8_t iq2xxs_grid_byte(int g, int j) {
    return (uint8_t)((iq2xxs_grid[g] >> (8 * j)) & 0xff);
}

static void dequantize_iq2_xxs(const uint8_t *src, float *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *p = src + b * 66;
        float d = oq_f16_to_f32(le_get_u16(p));
        const uint8_t *qs = p + 2;            /* 32 uint16 */
        float *y = dst + b * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            const uint8_t *g = qs + ib32 * 8; /* 4 uint16 = 8 bytes */
            uint32_t a0 = (uint32_t)le_get_u16(g)     | ((uint32_t)le_get_u16(g + 2) << 16);
            uint32_t a1 = (uint32_t)le_get_u16(g + 4) | ((uint32_t)le_get_u16(g + 6) << 16);
            float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                int gidx = (a0 >> (8 * l)) & 0xff;
                uint8_t signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++)
                    *y++ = db * iq2xxs_grid_byte(gidx, j) *
                           ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

/* For an 8-element magnitude target `mag` (>=0) and trial scale `s`, pick the
 * grid point minimizing sum (mag[j] - s*grid[j])^2. Returns the grid index and
 * writes the squared error to *err. */
static int iq2xxs_best_grid(const float *mag, float s, float *err) {
    int best = 0; float best_e = 1e30f;
    for (int g = 0; g < 256; g++) {
        float e = 0.0f;
        for (int j = 0; j < 8; j++) {
            float d = mag[j] - s * (float)iq2xxs_grid_byte(g, j);
            e += d * d;
        }
        if (e < best_e) { best_e = e; best = g; }
    }
    if (err) *err = best_e;
    return best;
}

/* Greedy IQ2_XXS encoder. Per 8-elem sub-group it extracts magnitudes + a
 * parity-even 7-bit sign index, picks the nearest grid point at the group's
 * scale, and quantizes the per-group scale into the block f16 d + a 4-bit code.
 * Output is valid ggml; quality is below ggml's iterative neighbour search. */
static void quantize_iq2_xxs(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK_K;
        uint8_t *p = dst + b * 66;
        uint8_t *qs = p + 2;
        memset(p, 0, 66);

        float    mag[8][4][8];   /* abs magnitudes per group/sub-group        */
        uint8_t  sgn[8][4];      /* 7-bit parity-even sign index              */
        float    gscale[8];      /* continuous per-group scale                */

        for (int ib = 0; ib < 8; ib++) {
            const float *xb = x + ib * 32;
            float gamax = 0.0f;
            for (int k = 0; k < 4; k++) {
                const float *xk = xb + k * 8;
                uint8_t s = 0; int nneg = 0;
                for (int j = 0; j < 8; j++) {
                    float v = xk[j];
                    if (v < 0.0f) { mag[ib][k][j] = -v; s |= (uint8_t)(1u << j); nneg++; }
                    else            mag[ib][k][j] =  v;
                    if (mag[ib][k][j] > gamax) gamax = mag[ib][k][j];
                }
                if (nneg & 1) {                /* enforce even parity: flip min |x| */
                    int imin = 0; float vmin = mag[ib][k][0];
                    for (int j = 1; j < 8; j++)
                        if (mag[ib][k][j] < vmin) { vmin = mag[ib][k][j]; imin = j; }
                    s ^= (uint8_t)(1u << imin);
                }
                sgn[ib][k] = s & 127;          /* bit7 is parity-implied */
            }
            /* Search the per-group scale (top grid magnitude is 43). Sweep a
             * range around amax/43 and keep the scale with least total error. */
            if (gamax < 1e-12f) { gscale[ib] = 0.0f; continue; }
            float base = gamax / 43.0f, best_s = base, best_e = 1e30f;
            for (int t = 4; t <= 64; t++) {
                float s = base * 43.0f / (float)t; /* maps amax to magnitude t */
                float tot = 0.0f, e;
                for (int k = 0; k < 4; k++) { iq2xxs_best_grid(mag[ib][k], s, &e); tot += e; }
                if (tot < best_e) { best_e = tot; best_s = s; }
            }
            gscale[ib] = best_s;
        }

        float max_scale = 0.0f;
        for (int ib = 0; ib < 8; ib++) if (gscale[ib] > max_scale) max_scale = gscale[ib];
        if (max_scale <= 0.0f) { /* all-zero block */
            le_put_u16(p, oq_f32_to_f16(0.0f));
            continue;
        }
        /* group_scale = d * (0.5 + l) * 0.25, l in 0..15. Pick d so the largest
         * group maps near l=15. */
        float d = max_scale / (0.25f * 15.5f);
        le_put_u16(p, oq_f32_to_f16(d));
        d = oq_f16_to_f32(le_get_u16(p));
        float qd = 0.25f * d;

        for (int ib = 0; ib < 8; ib++) {
            int l = 0;
            if (qd > 0.0f && gscale[ib] > 0.0f)
                l = (int)lroundf(gscale[ib] / qd - 0.5f);
            if (l < 0) l = 0;
            if (l > 15) l = 15;
            float db = qd * (0.5f + (float)l);   /* effective group scale */
            uint32_t a0 = 0, a1 = 0;
            for (int k = 0; k < 4; k++) {
                int g = (db > 0.0f) ? iq2xxs_best_grid(mag[ib][k], db, NULL) : 0;
                a0 |= (uint32_t)(g & 0xff) << (8 * k);
                a1 |= (uint32_t)(sgn[ib][k] & 127) << (7 * k);
            }
            a1 |= (uint32_t)l << 28;
            uint8_t *g = qs + ib * 8;
            le_put_u16(g,     (uint16_t)(a0 & 0xffff));
            le_put_u16(g + 2, (uint16_t)(a0 >> 16));
            le_put_u16(g + 4, (uint16_t)(a1 & 0xffff));
            le_put_u16(g + 6, (uint16_t)(a1 >> 16));
        }
    }
}

/* Q2_K encode (RTN): 16 sub-blocks of 16 elems, each an affine (scale4, min4);
 * value = d*scale4*q - dmin*min4, q in [0,3]. Super d/dmin are f16. */
static void quantize_q2_K(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK_K;
        uint8_t *p = dst + b * 84;
        uint8_t *scales = p;          /* 16 */
        uint8_t *q = p + 16;          /* 64 */
        memset(p, 0, 84);

        float subscale[16], submin[16];
        for (int is = 0; is < 16; is++) {
            const float *xs = x + is * 16;
            float lo = xs[0], hi = xs[0];
            for (int l = 1; l < 16; l++) { if (xs[l] < lo) lo = xs[l]; if (xs[l] > hi) hi = xs[l]; }
            if (lo > 0.0f) lo = 0.0f;          /* min term subtracted, >= 0 */
            subscale[is] = (hi - lo) / 3.0f;
            submin[is]   = -lo;
        }
        float maxsc = 0.0f, maxmn = 0.0f;
        for (int is = 0; is < 16; is++) {
            if (subscale[is] > maxsc) maxsc = subscale[is];
            if (submin[is]   > maxmn) maxmn = submin[is];
        }
        float d = maxsc / 15.0f, dmin = maxmn / 15.0f;
        float id = d ? 1.0f / d : 0.0f, idm = dmin ? 1.0f / dmin : 0.0f;
        le_put_u16(p + 80, oq_f32_to_f16(d));
        le_put_u16(p + 82, oq_f32_to_f16(dmin));
        d = oq_f16_to_f32(le_get_u16(p + 80));
        dmin = oq_f16_to_f32(le_get_u16(p + 82));

        for (int is = 0; is < 16; is++) {
            int s = (int)lroundf(subscale[is] * id);
            int m = (int)lroundf(submin[is]   * idm);
            s = s < 0 ? 0 : (s > 15 ? 15 : s);
            m = m < 0 ? 0 : (m > 15 ? 15 : m);
            scales[is] = (uint8_t)(s | (m << 4));
        }
        /* qs byte index for sub-block is, position l: group = is/8 picks the
         * 32-byte half; (is%8)/2 is the 2-bit shift; (is%8)%2 picks low/high 16. */
        for (int is = 0; is < 16; is++) {
            const float *xs = x + is * 16;
            float dl = d * (scales[is] & 0xF), ml = dmin * (scales[is] >> 4);
            float idl = dl ? 1.0f / dl : 0.0f;
            int group = is / 8, idx = is % 8;
            int shift = (idx / 2) * 2, half = (idx % 2) * 16;
            uint8_t *qp = q + group * 32 + half;
            for (int l = 0; l < 16; l++) {
                int v = (int)lroundf((xs[l] + ml) * idl);
                v = v < 0 ? 0 : (v > 3 ? 3 : v);
                qp[l] |= (uint8_t)(v << shift);
            }
        }
    }
}

/* Q5_K encode (RTN): 8 sub-blocks of 32, each an affine (scale, min);
 * value = d*scale*q - dmin*min, q in [0,31] (low nibble in ql, bit 5 in qh). */
static void quantize_q5_K(const float *src, uint8_t *dst, size_t nblk) {
    for (size_t b = 0; b < nblk; b++) {
        const float *x = src + b * QK_K;
        uint8_t *p = dst + b * 176;
        uint8_t *qh = p + 16;         /* 32 */
        uint8_t *ql = p + 48;         /* 128 */
        memset(p, 0, 176);

        float subscale[8], submin[8];
        for (int j = 0; j < 8; j++) {
            const float *xs = x + j * 32;
            float lo = xs[0], hi = xs[0];
            for (int l = 1; l < 32; l++) { if (xs[l] < lo) lo = xs[l]; if (xs[l] > hi) hi = xs[l]; }
            if (lo > 0.0f) lo = 0.0f;
            subscale[j] = (hi - lo) / 31.0f;
            submin[j]   = -lo;
        }
        float maxsc = 0.0f, maxmn = 0.0f;
        for (int j = 0; j < 8; j++) {
            if (subscale[j] > maxsc) maxsc = subscale[j];
            if (submin[j]   > maxmn) maxmn = submin[j];
        }
        float d = maxsc / 63.0f, dmin = maxmn / 63.0f;
        float id = d ? 1.0f / d : 0.0f, idm = dmin ? 1.0f / dmin : 0.0f;
        uint8_t sc6[8], mn6[8];
        for (int j = 0; j < 8; j++) {
            int s = (int)lroundf(subscale[j] * id);
            int m = (int)lroundf(submin[j]   * idm);
            s = s < 0 ? 0 : (s > 63 ? 63 : s);
            m = m < 0 ? 0 : (m > 63 ? 63 : m);
            sc6[j] = (uint8_t)s; mn6[j] = (uint8_t)m;
        }
        le_put_u16(p,     oq_f32_to_f16(d));
        le_put_u16(p + 2, oq_f32_to_f16(dmin));
        put_scale_min_k4(p + 4, sc6, mn6);
        d = oq_f16_to_f32(le_get_u16(p));
        dmin = oq_f16_to_f32(le_get_u16(p + 2));

        for (int j = 0; j < 8; j++) {
            float rs = d * sc6[j], rm = dmin * mn6[j], irs = rs ? 1.0f / rs : 0.0f;
            const float *xs = x + j * 32;
            uint8_t *qlp = ql + (j / 2) * 32;   /* 32 bytes shared by the pair */
            bool low = (j % 2) == 0;
            for (int l = 0; l < 32; l++) {
                int v = (int)lroundf((xs[l] + rm) * irs);
                v = v < 0 ? 0 : (v > 31 ? 31 : v);
                if (low) qlp[l] |= (uint8_t)(v & 0xF);
                else     qlp[l] |= (uint8_t)((v & 0xF) << 4);
                if (v & 0x10) qh[l] |= (uint8_t)(1u << j);
            }
        }
    }
}

ornith_status oq_quantize(uint32_t type, const float *src, void *dst,
                          size_t n_elems) {
    switch (type) {
    case OGGML_F32:
        memcpy(dst, src, n_elems * 4);
        return ORNITH_OK;
    case OGGML_F16: {
        uint8_t *o = dst;
        for (size_t i = 0; i < n_elems; i++) le_put_u16(o + i * 2,
                                                        oq_f32_to_f16(src[i]));
        return ORNITH_OK;
    }
    case OGGML_BF16: {
        uint8_t *o = dst;
        for (size_t i = 0; i < n_elems; i++) le_put_u16(o + i * 2,
                                                        oq_f32_to_bf16(src[i]));
        return ORNITH_OK;
    }
    case OGGML_Q8_0:
        if (n_elems % QK) { ornith_set_error("Q8_0 needs multiple of %d", QK);
                            return ORNITH_ERR_FORMAT; }
        quantize_q8_0(src, dst, n_elems / QK);
        return ORNITH_OK;
    case OGGML_Q4_0:
        if (n_elems % QK) { ornith_set_error("Q4_0 needs multiple of %d", QK);
                            return ORNITH_ERR_FORMAT; }
        quantize_q4_0(src, dst, n_elems / QK);
        return ORNITH_OK;
    case OGGML_Q4_K:
        if (n_elems % QK_K) { ornith_set_error("Q4_K needs multiple of %d", QK_K);
                              return ORNITH_ERR_FORMAT; }
        quantize_q4_K(src, dst, n_elems / QK_K);
        return ORNITH_OK;
    case OGGML_Q6_K:
        if (n_elems % QK_K) { ornith_set_error("Q6_K needs multiple of %d", QK_K);
                              return ORNITH_ERR_FORMAT; }
        quantize_q6_K(src, dst, n_elems / QK_K);
        return ORNITH_OK;
    case OGGML_Q2_K:
        if (n_elems % QK_K) { ornith_set_error("Q2_K needs multiple of %d", QK_K);
                              return ORNITH_ERR_FORMAT; }
        quantize_q2_K(src, dst, n_elems / QK_K);
        return ORNITH_OK;
    case OGGML_Q5_K:
        if (n_elems % QK_K) { ornith_set_error("Q5_K needs multiple of %d", QK_K);
                              return ORNITH_ERR_FORMAT; }
        quantize_q5_K(src, dst, n_elems / QK_K);
        return ORNITH_OK;
    case OGGML_IQ2_XXS:
        if (n_elems % QK_K) { ornith_set_error("IQ2_XXS needs multiple of %d", QK_K);
                              return ORNITH_ERR_FORMAT; }
        quantize_iq2_xxs(src, dst, n_elems / QK_K);
        return ORNITH_OK;
    default:
        ornith_set_error("oq_quantize: type %s not implemented",
                         oggml_type_name(type));
        return ORNITH_ERR_UNSUPPORTED;
    }
}

ornith_status oq_dequantize(uint32_t type, const void *src, float *dst,
                            size_t n_elems) {
    const uint8_t *s = src;
    switch (type) {
    case OGGML_F32:
        memcpy(dst, src, n_elems * 4);
        return ORNITH_OK;
    case OGGML_F16:
        for (size_t i = 0; i < n_elems; i++) dst[i] =
            oq_f16_to_f32(le_get_u16(s + i * 2));
        return ORNITH_OK;
    case OGGML_BF16:
        for (size_t i = 0; i < n_elems; i++) dst[i] =
            oq_bf16_to_f32(le_get_u16(s + i * 2));
        return ORNITH_OK;
    case OGGML_Q8_0:
        if (n_elems % QK) { ornith_set_error("Q8_0 needs multiple of %d", QK);
                            return ORNITH_ERR_FORMAT; }
        dequantize_q8_0(s, dst, n_elems / QK);
        return ORNITH_OK;
    case OGGML_Q4_0:
        if (n_elems % QK) { ornith_set_error("Q4_0 needs multiple of %d", QK);
                            return ORNITH_ERR_FORMAT; }
        dequantize_q4_0(s, dst, n_elems / QK);
        return ORNITH_OK;
    case OGGML_Q2_K: case OGGML_Q4_K: case OGGML_Q5_K: case OGGML_Q6_K:
        if (n_elems % QK_K) { ornith_set_error("%s needs multiple of %d",
                              oggml_type_name(type), QK_K);
                              return ORNITH_ERR_FORMAT; }
        if      (type == OGGML_Q2_K) dequantize_q2_K(s, dst, n_elems / QK_K);
        else if (type == OGGML_Q4_K) dequantize_q4_K(s, dst, n_elems / QK_K);
        else if (type == OGGML_Q5_K) dequantize_q5_K(s, dst, n_elems / QK_K);
        else                         dequantize_q6_K(s, dst, n_elems / QK_K);
        return ORNITH_OK;
    case OGGML_IQ2_XXS:
        if (n_elems % QK_K) { ornith_set_error("IQ2_XXS needs multiple of %d", QK_K);
                              return ORNITH_ERR_FORMAT; }
        dequantize_iq2_xxs(s, dst, n_elems / QK_K);
        return ORNITH_OK;
    default:
        ornith_set_error("oq_dequantize: type %s not implemented",
                         oggml_type_name(type));
        return ORNITH_ERR_UNSUPPORTED;
    }
}

/* ---- POLICY.md tensor -> quant-type mapping --------------------------- */

uint32_t oq_policy_target(const char *name, uint32_t base) {
    if (!name) return base;
    #define HAS(s) (strstr(name, (s)) != NULL)

    /* Real Ornith GGUF tensor names (verified against the official 9B GGUF):
     * SSM/linear-attn blocks use ssm_* tensors; full-attn blocks use attn_q/k/v
     * plus QK-norms; norms are *_norm / output_norm / post_attention_norm. */

    /* SSM recurrent *dynamics* — the decay/gating/conv that iterate the state.
     * The official quant kept these in F32 and so do we (matches our
     * "recurrent error compounds over 262K steps" rule). Check before norms so
     * intent is explicit; ssm_norm is caught by the norm rule below anyway. */
    if (HAS("ssm_a") || HAS("ssm_dt") || HAS("ssm_conv1d")) return OGGML_F32;

    /* All norms and scales (attn_norm, attn_q_norm, attn_k_norm, ssm_norm,
     * post_attention_norm, output_norm, ...). Negligible size; full precision. */
    if (HAS("_norm") || HAS("norm.") || HAS("scale")) return OGGML_F32;

    /* Router gate: a wrong route is unrecoverable -> F16. Precedes the generic
     * ffn_gate_* expert rules ("ffn_gate_inp" contains no "exps"). */
    if (HAS("ffn_gate_inp")) return OGGML_F16;

    /* Routed experts: the cold, dominant mass. Up/gate hardest, down sensitive. */
    if (HAS("ffn_gate_exps") || HAS("ffn_up_exps")) return OGGML_IQ2_XXS;
    if (HAS("ffn_down_exps"))                       return OGGML_Q2_K;

    /* Shared expert runs every token. */
    if (HAS("shexp")) return OGGML_Q5_K;

    /* SSM/linear-attn *projections* (alpha/beta gates, output, fused qkv input,
     * output gate). Recurrent path -> keep high precision Q8_0. */
    if (HAS("ssm_alpha") || HAS("ssm_beta") || HAS("ssm_out") ||
        HAS("attn_qkv")  || HAS("attn_gate")) return OGGML_Q8_0;

    /* Full-attention projections, every token. */
    if (HAS("attn_q") || HAS("attn_k") || HAS("attn_v") || HAS("attn_output"))
        return OGGML_Q6_K;

    /* Embeddings (input side) and the untied lm_head. */
    if (HAS("token_embd")) return OGGML_Q5_K;
    if (HAS("output"))     return OGGML_Q6_K;  /* output.weight (lm_head) */

    /* Dense (non-expert) FFN down-projection writes into the residual. */
    if (HAS("ffn_down")) return OGGML_Q6_K;

    #undef HAS
    return base;
}
