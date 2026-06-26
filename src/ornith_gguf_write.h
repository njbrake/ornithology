/* ornith_gguf_write.h — a GGUF v3 writer (the inverse of ornith_gguf.c).
 *
 * The reader in ornith_gguf.c tells you what is in a file; this writer produces
 * files that reader (and the wider ggml ecosystem) can consume byte-for-byte:
 * magic/version/counts, the metadata key/value table, the tensor index, and a
 * 32-byte-aligned tensor data section.
 *
 * Usage:
 *   ogguf_writer *w = ogguf_writer_new();
 *   ogguf_w_string(w, "general.architecture", "qwen3_5_moe");
 *   ogguf_w_tensor(w, "token_embd.weight", 2, dims, OGGML_F16, data, nbytes);
 *   ogguf_writer_write(w, "out.gguf");
 *   ogguf_writer_free(w);
 *
 * Tensor data is BORROWED: the pointers passed to ogguf_w_tensor must stay
 * valid until ogguf_writer_write returns.
 *
 * Companion to the writer: ogguf_load reads an entire GGUF (metadata payloads
 * AND tensor data) into memory so it can be transcoded and re-emitted, and
 * ornith_quantize_file drives the writer + ornith_quant codecs to apply the
 * POLICY.md mapping.
 */
#ifndef ORNITH_GGUF_WRITE_H
#define ORNITH_GGUF_WRITE_H

#include "ornith.h"
#include "ornith_gguf.h"

typedef struct ogguf_writer ogguf_writer;

ogguf_writer *ogguf_writer_new(void);
void          ogguf_writer_free(ogguf_writer *w);

/* ---- metadata key/value entries (GGUF value-type tags handled inside) --- */

ornith_status ogguf_w_string(ogguf_writer *w, const char *key, const char *val);
ornith_status ogguf_w_u32   (ogguf_writer *w, const char *key, uint32_t val);
ornith_status ogguf_w_u64   (ogguf_writer *w, const char *key, uint64_t val);
ornith_status ogguf_w_i32   (ogguf_writer *w, const char *key, int32_t  val);
ornith_status ogguf_w_f32   (ogguf_writer *w, const char *key, float    val);
ornith_status ogguf_w_bool  (ogguf_writer *w, const char *key, bool     val);
ornith_status ogguf_w_arr_str(ogguf_writer *w, const char *key,
                              const char *const *vals, size_t n);
ornith_status ogguf_w_arr_i32(ogguf_writer *w, const char *key,
                              const int32_t *vals, size_t n);
ornith_status ogguf_w_arr_f32(ogguf_writer *w, const char *key,
                              const float *vals, size_t n);
/* Copy an already-serialized value payload (the bytes that follow the value
 * type tag) verbatim — used to pass metadata through during transcode. */
ornith_status ogguf_w_raw_kv(ogguf_writer *w, const char *key, uint32_t vtype,
                             const void *payload, size_t len);

/* ---- tensor index entries --------------------------------------------- */

/* Add a tensor. `data`/`nbytes` are borrowed until ogguf_writer_write. */
ornith_status ogguf_w_tensor(ogguf_writer *w, const char *name,
                             uint32_t n_dims, const uint64_t *dims,
                             uint32_t type, const void *data, uint64_t nbytes);

/* Serialize everything to `path`. */
ornith_status ogguf_writer_write(ogguf_writer *w, const char *path);

/* ---- whole-file load (for transcode) ---------------------------------- */

typedef struct {
    char     name[256];
    uint32_t n_dims;
    uint64_t dims[OGGUF_MAX_DIMS];
    uint64_t n_elements;
    uint32_t type;
    void    *data;      /* owned: oq-decodable bytes for this tensor          */
    uint64_t nbytes;
} ogguf_ltensor;

typedef struct {
    char    *key;
    uint32_t vtype;
    uint8_t *payload;   /* owned: serialized value bytes after the type tag   */
    size_t   payload_len;
} ogguf_lkv;

typedef struct {
    uint32_t       version;
    uint32_t       alignment;
    ogguf_lkv     *kv;
    uint64_t       n_kv;
    ogguf_ltensor *tensors;
    uint64_t       n_tensors;
} ogguf_loaded;

ornith_status ogguf_load(const char *path, ogguf_loaded *out);
void          ogguf_loaded_free(ogguf_loaded *l);

/* ---- quantizer driver -------------------------------------------------- */

/* Read `in_path`, requantize each tensor to the type chosen by
 * oq_policy_target() (base type `base` for unmatched tensors; pass
 * OGGML_F16 for a sensible default), and write `out_path`. Metadata is copied
 * through unchanged. Policy targets we cannot yet encode fall back to F16 and
 * are logged to `log` (may be NULL). */
ornith_status ornith_quantize_file(const char *in_path, const char *out_path,
                                   uint32_t base, FILE *log);

#endif /* ORNITH_GGUF_WRITE_H */
