/* ornith_imatrix.h — importance-matrix (imatrix) collection + on-disk format.
 *
 * An imatrix records, per weight tensor, how "important" each input channel is
 * to the model's output on a calibration corpus. We use the standard llama.cpp
 * proxy: the sum of squared activations seen at each input channel (plus a token
 * count). The sub-2-bit quantizer then weights its rounding error by this
 * importance so the channels that actually drive the model keep their precision
 * — the difference between a 2-bit model that works and one that babbles.
 *
 * Collection is a cheap, off-by-default hook in the matvec hot path
 * (ornith_rforward.c). The CLI's `imatrix` command flips it on, runs prefill
 * over a corpus, and saves the result.
 *
 * File format (imatrix.dat), all little-endian (host) scalars:
 *   char  magic[4]  = "OIMX"
 *   u32   version   = 1
 *   u64   n_tensors
 *   per tensor:
 *     u32   name_len
 *     char  name[name_len]      (no NUL terminator)
 *     u64   n                   (per-input-channel vector length)
 *     u64   count               (tokens accumulated into this tensor)
 *     f32   sum[n]              (sum of squared activations per channel)
 */
#ifndef ORNITH_IMATRIX_H
#define ORNITH_IMATRIX_H

#include "ornith.h"

typedef struct oimatrix oimatrix;

oimatrix *oimatrix_new(void);
void      oimatrix_free(oimatrix *im);

/* Accumulate per-input-channel sum of squared activations for tensor `name`.
 * `X` is the activation block fed to a matvec: `Tn` columns, each `in` floats,
 * laid out column-major (column t starts at X + (size_t)t*in). Adds `Tn` to the
 * tensor's token count. Cheap (O(Tn*in)); single-threaded by design. */
void oimatrix_accumulate(oimatrix *im, const char *name,
                         const float *X, int in, int Tn);

/* Fetch the raw per-channel sum-of-squares vector for `name` (length *n_out,
 * tokens *count_out — either out-pointer may be NULL). Returns NULL if absent. */
const float *oimatrix_get(const oimatrix *im, const char *name,
                          size_t *n_out, uint64_t *count_out);

size_t oimatrix_count(const oimatrix *im);     /* number of tensors recorded   */
const char *oimatrix_name_at(const oimatrix *im, size_t i); /* tensor i's name  */

ornith_status oimatrix_save(const oimatrix *im, const char *path);
ornith_status oimatrix_load(const char *path, oimatrix **out);

/* ---- global collection hook (driven by the matvec hot path) ------------ */

/* While a collector is set, every oimatrix_on_matmul call accumulates into it.
 * begin/end are not reentrant; collection runs single-threaded (the matvec
 * accumulates before it spawns its worker threads). */
void oimatrix_collect_begin(oimatrix *im);
void oimatrix_collect_end(void);

/* Hot-path hook: a no-op (one pointer compare) unless collection is active. */
void oimatrix_on_matmul(const char *name, const float *X, int in, int Tn);

#endif /* ORNITH_IMATRIX_H */
