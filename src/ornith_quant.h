/* ornith_quant.h — float conversions and ggml-compatible quant block codecs.
 *
 * The on-disk block layouts here match ggml/llama.cpp byte-for-byte so the
 * GGUFs we emit are interoperable with the wider ecosystem. We implement the
 * conversions ornithology's own quantizer (ROADMAP M1) needs end to end —
 * F32, F16, BF16, Q8_0 and Q4_0 — and expose a policy lookup that encodes the
 * tensor-name -> quant-type table from tools/quantize/POLICY.md.
 *
 * Types we do NOT yet encode (the k-/i-quants: Q2_K, Q5_K, Q6_K, IQ2_XXS,
 * IQ3_S, ...) are reported as unimplemented so the quantizer can fall back to
 * F16 and log it honestly rather than emit a corrupt block.
 */
#ifndef ORNITH_QUANT_H
#define ORNITH_QUANT_H

#include "ornith.h"
#include "ornith_gguf.h"   /* oggml_type */

/* ---- half / bfloat16 scalar conversions ------------------------------- */

uint16_t oq_f32_to_f16(float f);
float    oq_f16_to_f32(uint16_t h);
uint16_t oq_f32_to_bf16(float f);
float    oq_bf16_to_f32(uint16_t b);

/* ---- block geometry ---------------------------------------------------- */

/* Elements per quant block (1 for the non-blocked F32/F16/BF16). 0 if the
 * type is not one we encode. */
size_t oq_block_elems(uint32_t type);
/* Bytes occupied by one block of `type`. 0 if not encoded. */
size_t oq_block_bytes(uint32_t type);
/* Bytes needed to store `n_elems` of `type`. 0 if not encoded or if `n_elems`
 * is not a whole number of blocks for a blocked type. */
size_t oq_row_bytes(uint32_t type, size_t n_elems);

/* True if oq_quantize/oq_dequantize can handle `type`. */
bool oq_is_implemented(uint32_t type);

/* ---- codecs ------------------------------------------------------------ */

/* Quantize `n_elems` f32 values from `src` into `dst`. `dst` must be at least
 * oq_row_bytes(type, n_elems) bytes. Returns ORNITH_ERR_UNSUPPORTED for a type
 * we don't encode, ORNITH_ERR_FORMAT if `n_elems` is not block-aligned. */
ornith_status oq_quantize(uint32_t type, const float *src, void *dst,
                          size_t n_elems);
/* Inverse of oq_quantize: decode `n_elems` of `type` from `src` to f32. */
ornith_status oq_dequantize(uint32_t type, const void *src, float *dst,
                            size_t n_elems);

/* ---- POLICY.md tensor -> quant-type mapping ---------------------------- */

/* Ideal target quant type for a tensor, by GGUF name (substring match, first
 * rule wins; see tools/quantize/POLICY.md). Returns `base` for names with no
 * specific rule. The returned type may be one oq_quantize cannot yet encode
 * (a k-/i-quant); the caller is expected to fall back via oq_is_implemented. */
uint32_t oq_policy_target(const char *tensor_name, uint32_t base);

#endif /* ORNITH_QUANT_H */
