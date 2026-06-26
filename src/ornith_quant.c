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
    case OGGML_Q4_K: case OGGML_Q6_K:
        return true;
    default:
        return false;
    }
}

/* True if oq_dequantize can decode `type` (a superset of oq_is_implemented:
 * also covers the read-only Q2_K / Q5_K paths used to load real GGUFs). */
bool oq_can_decode(uint32_t type) {
    switch (type) {
    case OGGML_Q2_K: case OGGML_Q5_K:
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
 * correctly. The Q4_K/Q6_K encoders are round-to-nearest: they emit valid,
 * interoperable blocks (quality a touch below ggml's iterative search, which is
 * an M3 refinement). Q2_K/Q5_K are decode-only for now (enough to run the
 * published quants). */

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
