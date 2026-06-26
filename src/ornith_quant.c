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

size_t oq_block_elems(uint32_t type) {
    switch (type) {
    case OGGML_F32: case OGGML_F16: case OGGML_BF16: return 1;
    case OGGML_Q8_0: case OGGML_Q4_0:                return QK;
    default: return 0;
    }
}

size_t oq_block_bytes(uint32_t type) {
    switch (type) {
    case OGGML_F32:  return 4;
    case OGGML_F16:  case OGGML_BF16: return 2;
    case OGGML_Q8_0: return 2 + QK;        /* f16 d + 32 int8 = 34 */
    case OGGML_Q4_0: return 2 + QK / 2;    /* f16 d + 16 nibbles = 18 */
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
    return oq_block_elems(type) != 0;
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
