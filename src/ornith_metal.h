/* ornith_metal.h — Metal (Apple GPU) backend interface.
 *
 * FIRST CUT. This declares a small, standalone backend seam that mirrors the
 * CPU quant-aware matvec oracle in ornith_qdot.c: quantize the activation row
 * once to Q8_K, then run a type-specific INTEGER dot directly against the
 * quantized weight bytes (Q4_K / Q6_K / Q8_0), with weights resident on-device
 * in their packed GGUF form. The numerics are transcribed block-for-block from
 * ornith_qdot.c so GPU output should match the CPU reference (modulo float
 * rounding in the activation-quant and the float accumulation order).
 *
 * The whole backend is gated behind ORNITH_BACKEND_METAL. On non-Metal builds
 * (the default CPU build, the Linux CI/tests) ornith_metal.m compiles to
 * nothing and these symbols are simply absent — nothing here is referenced by
 * the CPU engine yet (CPU-engine integration is a later pass; see METAL.md).
 *
 * This header is plain C so the CPU engine (also C) can include it later and
 * call these entry points without an Objective-C dependency leaking out.
 *
 * Attribution: the quant block layouts and the integer-dot math are derived
 * from ggml (MIT) via the ds4 reference; the Objective-C / Metal scaffolding
 * (device, queue, newLibraryWithSource, buffers, dispatch) follows the same
 * structure as ds4-ref/ds4_metal.m. See LICENSE.
 */
#ifndef ORNITH_METAL_H
#define ORNITH_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Metal context: device, command queue, compiled library, pipelines. */
typedef struct ornith_metal_ctx ornith_metal_ctx;

/* Opaque handle to a quantized weight tensor resident in a MTLBuffer, together
 * with its logical shape (n_cols = elements per row, n_rows = output count) and
 * its oggml_type. Produced by ornith_metal_upload_weight. */
typedef struct ornith_metal_tensor ornith_metal_tensor;

/* 1 if a usable Metal device is present, 0 otherwise. Safe to call without a
 * context (creates and releases a transient device query). */
int ornith_backend_metal_available(void);

/* Create the backend context: pick the system default device, build a command
 * queue, compile the embedded MSL library, and prebuild the matvec + Q8_K
 * activation-quant pipelines. Returns NULL on any failure (diagnostics are
 * printed to stderr, ds4-style). Free with ornith_metal_free. */
ornith_metal_ctx *ornith_metal_init(void);

/* Tear down a context created by ornith_metal_init. NULL-safe. */
void ornith_metal_free(ornith_metal_ctx *ctx);

/* Upload one quantized weight tensor to the GPU.
 *
 *   wtype   — oggml_type of the blocks (OGGML_Q4_K / OGGML_Q6_K / OGGML_Q8_0).
 *   data    — raw GGUF block bytes, row-major, n_rows rows of row_bytes each.
 *   nbytes  — total byte length of `data` (sanity-checked against the layout).
 *   n_cols  — elements per row (must be a multiple of 256 for the k-quants).
 *   n_rows  — number of weight rows (== number of matvec outputs).
 *
 * The bytes are copied into a shared MTLBuffer (unified memory). Returns NULL
 * on bad arguments or allocation failure. Free with ornith_metal_free_tensor. */
ornith_metal_tensor *ornith_metal_upload_weight(ornith_metal_ctx *ctx,
                                                uint32_t wtype,
                                                const void *data, size_t nbytes,
                                                int64_t n_cols, int64_t n_rows);

/* Release a tensor handle and its device buffer. NULL-safe. */
void ornith_metal_free_tensor(ornith_metal_tensor *t);

/* Quant-aware matvec: out[r] = sum_c W[r,c] * x[c], r in [0, n_rows).
 *
 *   x   — f32 activation row of length w->n_cols (host pointer).
 *   out — f32 result of length w->n_rows (host pointer).
 *
 * Internally: x is uploaded and quantized to Q8_K on the GPU (mirrors
 * qdot_quantize_row_q8_K), then the type-specific integer-dot matvec kernel is
 * dispatched one threadgroup per output row. Blocks until the result is ready
 * (synchronous; batching/streaming is future work). Returns 0 on success,
 * non-zero on error.
 *
 * The _q4k / _q6k variants assert the tensor type; ornith_metal_matvec
 * dispatches on w->wtype and also handles Q8_0. */
int ornith_metal_matvec_q4k(ornith_metal_ctx *ctx, const ornith_metal_tensor *w,
                            const float *x, float *out);
int ornith_metal_matvec_q6k(ornith_metal_ctx *ctx, const ornith_metal_tensor *w,
                            const float *x, float *out);
int ornith_metal_matvec(ornith_metal_ctx *ctx, const ornith_metal_tensor *w,
                        const float *x, float *out);

#ifdef __cplusplus
}
#endif

#endif /* ORNITH_METAL_H */
