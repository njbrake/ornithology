/* ornith_qdot.h — quant-aware integer dot products (ds4 / llama.cpp style).
 *
 * The CPU matvec hot path used to dequantize each weight row to f32 and then
 * do a float dot against the f32 activation. That materializes a full f32 row
 * per output and wastes the structure of the quantized weight. Instead we do
 * what ds4 / ggml do: quantize the ACTIVATION row once to Q8_K, then for each
 * weight row run a type-specific INTEGER dot directly against the quantized
 * weight bytes (no f32 weight materialization). Weights stay resident in their
 * quantized form, so this remains compatible with the very large models.
 *
 * Block layouts are ggml-compatible and match the decoders in ornith_quant.c
 * byte-for-byte (so a row of weight bytes can be fed straight in). The Q8_K
 * activation block mirrors ds4's block_q8_K (see ds4-ref/ds4.c ~L364).
 *
 * Attribution: derived from ggml (MIT) via the ds4 reference clone; see LICENSE.
 */
#ifndef ORNITH_QDOT_H
#define ORNITH_QDOT_H

#include "ornith.h"
#include <stdint.h>

#define OQDOT_QK_K 256  /* super-block size shared by the k-quants / Q8_K */

/* Q8_K quantized activation super-block (292 bytes, ggml/ds4 layout):
 *   d      — f32 scale (1/iscale)
 *   qs[256]— int8 quantized values
 *   bsums  — int16 group sums (sum of qs over each run of 16), used to fold
 *            the k-quant per-sub-block MINS into the dot cheaply. */
typedef struct {
    float   d;
    int8_t  qs[OQDOT_QK_K];
    int16_t bsums[OQDOT_QK_K / 16];
} oq8k_block;

/* Quantize a length-`k` (must be a multiple of 256) f32 activation row into
 * `k/256` Q8_K blocks. */
void qdot_quantize_row_q8_K(const float *x, oq8k_block *y, int64_t k);

/* True if there is an integer vec_dot for weight type `wtype` over a row of
 * `n` elements (i.e. `n` is 256-block aligned and the type is supported:
 * Q4_K, Q6_K, Q8_0). When false the caller must fall back to dequant + f32. */
bool qdot_can(uint32_t wtype, int n);

/* Integer dot of one quantized weight row (`wrow`, raw block bytes of `wtype`)
 * with a Q8_K-quantized activation row (`yq`, n/256 blocks). Result in `*s`.
 * Precondition: qdot_can(wtype, n) == true. */
void qdot_vec_dot(uint32_t wtype, int n, float *s,
                  const void *wrow, const oq8k_block *yq);

#endif /* ORNITH_QDOT_H */
