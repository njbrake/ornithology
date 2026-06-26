/* ornith_gguf_write.c — GGUF v3 writer, whole-file loader, and the quantizer
 * driver. The on-disk layout mirrors ornith_gguf.c's reader exactly:
 *
 *   u32 "GGUF" | u32 version=3 | u64 n_tensors | u64 n_kv
 *   n_kv  x [ str key | u32 vtype | value ]
 *   n_tns x [ str name | u32 n_dims | u64 dims[n_dims] | u32 type | u64 offset ]
 *   <pad to alignment> | tensor data (each tensor padded to alignment)
 *
 * str = u64 len | bytes (no NUL). All scalars little-endian. Offsets are
 * relative to the start of the (aligned) data section.
 */
#define _POSIX_C_SOURCE 200809L
#include "ornith_gguf_write.h"
#include "ornith_quant.h"
#include "ornith_imatrix.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* GGUF metadata value type tags (must match the reader). */
enum {
    GGUF_T_UINT8=0, GGUF_T_INT8=1, GGUF_T_UINT16=2, GGUF_T_INT16=3,
    GGUF_T_UINT32=4, GGUF_T_INT32=5, GGUF_T_FLOAT32=6, GGUF_T_BOOL=7,
    GGUF_T_STRING=8, GGUF_T_ARRAY=9, GGUF_T_UINT64=10, GGUF_T_INT64=11,
    GGUF_T_FLOAT64=12,
};

#define GGUF_DEFAULT_ALIGNMENT 32

static uint64_t align_up(uint64_t x, uint64_t a) {
    return a ? ((x + a - 1) / a) * a : x;
}

/* ---- growable byte buffer --------------------------------------------- */

typedef struct { uint8_t *p; size_t len, cap; bool oom; } buf;

static void buf_reserve(buf *b, size_t extra) {
    if (b->oom) return;
    if (b->len + extra <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 64;
    while (cap < b->len + extra) cap *= 2;
    uint8_t *np = realloc(b->p, cap);
    if (!np) { b->oom = true; return; }
    b->p = np; b->cap = cap;
}
static void buf_put(buf *b, const void *src, size_t n) {
    buf_reserve(b, n);
    if (b->oom) return;
    memcpy(b->p + b->len, src, n);
    b->len += n;
}
static void buf_u32(buf *b, uint32_t v) { buf_put(b, &v, 4); }
static void buf_u64(buf *b, uint64_t v) { buf_put(b, &v, 8); }
static void buf_str(buf *b, const char *s) {
    uint64_t n = strlen(s);
    buf_u64(b, n);
    buf_put(b, s, n);
}

/* ---- writer ------------------------------------------------------------ */

typedef struct {
    char    *key;
    uint32_t vtype;
    uint8_t *payload;   /* serialized value bytes (after the type tag) */
    size_t   len;
} wkv;

typedef struct {
    char        name[256];
    uint32_t    n_dims;
    uint64_t    dims[OGGUF_MAX_DIMS];
    uint32_t    type;
    const void *data;   /* borrowed */
    uint64_t    nbytes;
} wtensor;

struct ogguf_writer {
    wkv     *kv;  size_t n_kv,  cap_kv;
    wtensor *t;   size_t n_t,   cap_t;
    bool     oom;
};

ogguf_writer *ogguf_writer_new(void) {
    return calloc(1, sizeof(ogguf_writer));
}

void ogguf_writer_free(ogguf_writer *w) {
    if (!w) return;
    for (size_t i = 0; i < w->n_kv; i++) { free(w->kv[i].key); free(w->kv[i].payload); }
    free(w->kv);
    free(w->t);
    free(w);
}

/* Append a KV, taking ownership of `payload`. */
static ornith_status push_kv(ogguf_writer *w, const char *key, uint32_t vtype,
                             uint8_t *payload, size_t len) {
    if (w->n_kv == w->cap_kv) {
        size_t cap = w->cap_kv ? w->cap_kv * 2 : 16;
        wkv *nk = realloc(w->kv, cap * sizeof(wkv));
        if (!nk) { free(payload); ornith_set_error("oom (kv)"); return ORNITH_ERR_OOM; }
        w->kv = nk; w->cap_kv = cap;
    }
    size_t klen = strlen(key) + 1;
    char *k = malloc(klen);
    if (!k) { free(payload); ornith_set_error("oom (key)"); return ORNITH_ERR_OOM; }
    memcpy(k, key, klen);
    w->kv[w->n_kv++] = (wkv){ .key = k, .vtype = vtype,
                              .payload = payload, .len = len };
    return ORNITH_OK;
}

/* Build a KV from a buf, transferring its storage. */
static ornith_status push_buf_kv(ogguf_writer *w, const char *key,
                                 uint32_t vtype, buf *b) {
    if (b->oom) { free(b->p); ornith_set_error("oom (value)"); return ORNITH_ERR_OOM; }
    return push_kv(w, key, vtype, b->p, b->len);
}

ornith_status ogguf_w_string(ogguf_writer *w, const char *key, const char *val) {
    buf b = {0};
    buf_str(&b, val);
    return push_buf_kv(w, key, GGUF_T_STRING, &b);
}
ornith_status ogguf_w_u32(ogguf_writer *w, const char *key, uint32_t v) {
    buf b = {0}; buf_put(&b, &v, 4); return push_buf_kv(w, key, GGUF_T_UINT32, &b);
}
ornith_status ogguf_w_u64(ogguf_writer *w, const char *key, uint64_t v) {
    buf b = {0}; buf_put(&b, &v, 8); return push_buf_kv(w, key, GGUF_T_UINT64, &b);
}
ornith_status ogguf_w_i32(ogguf_writer *w, const char *key, int32_t v) {
    buf b = {0}; buf_put(&b, &v, 4); return push_buf_kv(w, key, GGUF_T_INT32, &b);
}
ornith_status ogguf_w_f32(ogguf_writer *w, const char *key, float v) {
    buf b = {0}; buf_put(&b, &v, 4); return push_buf_kv(w, key, GGUF_T_FLOAT32, &b);
}
ornith_status ogguf_w_bool(ogguf_writer *w, const char *key, bool v) {
    buf b = {0}; uint8_t x = v ? 1 : 0; buf_put(&b, &x, 1);
    return push_buf_kv(w, key, GGUF_T_BOOL, &b);
}
ornith_status ogguf_w_arr_str(ogguf_writer *w, const char *key,
                              const char *const *vals, size_t n) {
    buf b = {0};
    buf_u32(&b, GGUF_T_STRING);
    buf_u64(&b, n);
    for (size_t i = 0; i < n; i++) buf_str(&b, vals[i]);
    return push_buf_kv(w, key, GGUF_T_ARRAY, &b);
}
ornith_status ogguf_w_arr_i32(ogguf_writer *w, const char *key,
                              const int32_t *vals, size_t n) {
    buf b = {0};
    buf_u32(&b, GGUF_T_INT32);
    buf_u64(&b, n);
    buf_put(&b, vals, n * 4);
    return push_buf_kv(w, key, GGUF_T_ARRAY, &b);
}
ornith_status ogguf_w_arr_f32(ogguf_writer *w, const char *key,
                              const float *vals, size_t n) {
    buf b = {0};
    buf_u32(&b, GGUF_T_FLOAT32);
    buf_u64(&b, n);
    buf_put(&b, vals, n * 4);
    return push_buf_kv(w, key, GGUF_T_ARRAY, &b);
}
ornith_status ogguf_w_raw_kv(ogguf_writer *w, const char *key, uint32_t vtype,
                             const void *payload, size_t len) {
    uint8_t *cp = malloc(len ? len : 1);
    if (!cp) { ornith_set_error("oom (raw kv)"); return ORNITH_ERR_OOM; }
    memcpy(cp, payload, len);
    return push_kv(w, key, vtype, cp, len);
}

ornith_status ogguf_w_tensor(ogguf_writer *w, const char *name,
                             uint32_t n_dims, const uint64_t *dims,
                             uint32_t type, const void *data, uint64_t nbytes) {
    if (n_dims > OGGUF_MAX_DIMS) {
        ornith_set_error("tensor '%s' has %u dims (max %d)", name, n_dims,
                         OGGUF_MAX_DIMS);
        return ORNITH_ERR_FORMAT;
    }
    if (w->n_t == w->cap_t) {
        size_t cap = w->cap_t ? w->cap_t * 2 : 16;
        wtensor *nt = realloc(w->t, cap * sizeof(wtensor));
        if (!nt) { ornith_set_error("oom (tensor)"); return ORNITH_ERR_OOM; }
        w->t = nt; w->cap_t = cap;
    }
    wtensor *t = &w->t[w->n_t++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = n_dims;
    for (uint32_t d = 0; d < n_dims; d++) t->dims[d] = dims[d];
    t->type = type;
    t->data = data;
    t->nbytes = nbytes;
    return ORNITH_OK;
}

/* zero-pad the file to a multiple of `align` from the current position */
static bool pad_to(FILE *f, uint64_t align) {
    long pos = ftell(f);
    if (pos < 0) return false;
    uint64_t want = align_up((uint64_t)pos, align);
    static const uint8_t zeros[64] = {0};
    while ((uint64_t)pos < want) {
        size_t n = (size_t)(want - pos);
        if (n > sizeof(zeros)) n = sizeof(zeros);
        if (fwrite(zeros, 1, n, f) != n) return false;
        pos += (long)n;
    }
    return true;
}

ornith_status ogguf_writer_write(ogguf_writer *w, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { ornith_set_error("cannot open %s for writing", path); return ORNITH_ERR_IO; }

    buf hdr = {0};
    buf_put(&hdr, "GGUF", 4);
    buf_u32(&hdr, 3);                 /* version */
    buf_u64(&hdr, w->n_t);            /* n_tensors */
    buf_u64(&hdr, w->n_kv);           /* n_kv */

    /* metadata KV table */
    for (size_t i = 0; i < w->n_kv; i++) {
        buf_str(&hdr, w->kv[i].key);
        buf_u32(&hdr, w->kv[i].vtype);
        buf_put(&hdr, w->kv[i].payload, w->kv[i].len);
    }

    /* tensor index — assign aligned, cumulative offsets */
    uint64_t off = 0;
    for (size_t i = 0; i < w->n_t; i++) {
        wtensor *t = &w->t[i];
        buf_str(&hdr, t->name);
        buf_u32(&hdr, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) buf_u64(&hdr, t->dims[d]);
        buf_u32(&hdr, t->type);
        buf_u64(&hdr, off);
        off = align_up(off + t->nbytes, GGUF_DEFAULT_ALIGNMENT);
    }

    if (hdr.oom) { free(hdr.p); fclose(f); ornith_set_error("oom (header)");
                   return ORNITH_ERR_OOM; }
    if (fwrite(hdr.p, 1, hdr.len, f) != hdr.len) {
        free(hdr.p); fclose(f); ornith_set_error("short write (header)");
        return ORNITH_ERR_IO;
    }
    free(hdr.p);

    /* data section: align, then write each tensor padded to alignment */
    if (!pad_to(f, GGUF_DEFAULT_ALIGNMENT)) goto io_err;
    for (size_t i = 0; i < w->n_t; i++) {
        wtensor *t = &w->t[i];
        if (t->nbytes && fwrite(t->data, 1, t->nbytes, f) != t->nbytes) goto io_err;
        if (!pad_to(f, GGUF_DEFAULT_ALIGNMENT)) goto io_err;
    }

    if (fclose(f) != 0) { ornith_set_error("close failed for %s", path);
                          return ORNITH_ERR_IO; }
    return ORNITH_OK;

io_err:
    fclose(f);
    ornith_set_error("short write to %s", path);
    return ORNITH_ERR_IO;
}

/* ---- whole-file loader ------------------------------------------------- */

static bool rd(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n; }
static bool rd_u32(FILE *f, uint32_t *v) { return rd(f, v, 4); }
static bool rd_u64(FILE *f, uint64_t *v) { return rd(f, v, 8); }

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL: return 1;
    case GGUF_T_UINT16: case GGUF_T_INT16: return 2;
    case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
    case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
    default: return 0;
    }
}

/* Append the full serialized form of one value (no leading type tag) of
 * `vtype` from `f` into `out`. Also used for array elements. */
static bool load_value(FILE *f, uint32_t vtype, buf *out) {
    if (vtype == GGUF_T_STRING) {
        uint64_t len;
        if (!rd_u64(f, &len)) return false;
        buf_u64(out, len);
        if (len) {
            buf_reserve(out, len);
            if (out->oom) return false;
            if (fread(out->p + out->len, 1, len, f) != len) return false;
            out->len += len;
        }
        return true;
    }
    if (vtype == GGUF_T_ARRAY) {
        uint32_t etype; uint64_t n;
        if (!rd_u32(f, &etype) || !rd_u64(f, &n)) return false;
        buf_u32(out, etype);
        buf_u64(out, n);
        for (uint64_t i = 0; i < n; i++)
            if (!load_value(f, etype, out)) return false;
        return true;
    }
    size_t sz = scalar_size(vtype);
    if (!sz) return false;
    buf_reserve(out, sz);
    if (out->oom) return false;
    if (fread(out->p + out->len, 1, sz, f) != sz) return false;
    out->len += sz;
    return true;
}

static bool load_string(FILE *f, char **out) {
    uint64_t len;
    if (!rd_u64(f, &len)) return false;
    char *s = malloc(len + 1);
    if (!s) return false;
    if (len && fread(s, 1, len, f) != len) { free(s); return false; }
    s[len] = '\0';
    *out = s;
    return true;
}

ornith_status ogguf_load(const char *path, ogguf_loaded *out) {
    memset(out, 0, sizeof(*out));
    out->alignment = GGUF_DEFAULT_ALIGNMENT;

    FILE *f = fopen(path, "rb");
    if (!f) { ornith_set_error("cannot open %s", path); return ORNITH_ERR_IO; }

    char magic[4];
    if (!rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0) {
        fclose(f); ornith_set_error("not a GGUF file (bad magic)");
        return ORNITH_ERR_FORMAT;
    }
    if (!rd_u32(f, &out->version) ||
        !rd_u64(f, &out->n_tensors) || !rd_u64(f, &out->n_kv)) {
        fclose(f); ornith_set_error("truncated GGUF header");
        return ORNITH_ERR_FORMAT;
    }
    if (out->version < 2 || out->version > 3) {
        fclose(f); ornith_set_error("unsupported GGUF version %u", out->version);
        return ORNITH_ERR_UNSUPPORTED;
    }
    if (out->n_tensors > (1u << 24) || out->n_kv > (1u << 24)) {
        fclose(f); ornith_set_error("implausible counts");
        return ORNITH_ERR_FORMAT;
    }

    out->kv = calloc(out->n_kv ? out->n_kv : 1, sizeof(ogguf_lkv));
    out->tensors = calloc(out->n_tensors ? out->n_tensors : 1, sizeof(ogguf_ltensor));
    if (!out->kv || !out->tensors) {
        fclose(f); ogguf_loaded_free(out); ornith_set_error("oom (load)");
        return ORNITH_ERR_OOM;
    }

    for (uint64_t i = 0; i < out->n_kv; i++) {
        ogguf_lkv *e = &out->kv[i];
        uint32_t vtype;
        if (!load_string(f, &e->key) || !rd_u32(f, &vtype)) {
            fclose(f); ogguf_loaded_free(out);
            ornith_set_error("bad kv #%llu", (unsigned long long)i);
            return ORNITH_ERR_FORMAT;
        }
        e->vtype = vtype;
        buf b = {0};
        if (!load_value(f, vtype, &b)) {
            free(b.p); fclose(f); ogguf_loaded_free(out);
            ornith_set_error("bad value for key '%s'", e->key);
            return ORNITH_ERR_FORMAT;
        }
        e->payload = b.p; e->payload_len = b.len;
        if (strcmp(e->key, "general.alignment") == 0 &&
            vtype == GGUF_T_UINT32 && b.len >= 4) {
            uint32_t a; memcpy(&a, b.p, 4);
            if (a) out->alignment = a;
        }
    }

    for (uint64_t i = 0; i < out->n_tensors; i++) {
        ogguf_ltensor *t = &out->tensors[i];
        char *nm = NULL;
        if (!load_string(f, &nm) || !rd_u32(f, &t->n_dims)) {
            free(nm); fclose(f); ogguf_loaded_free(out);
            ornith_set_error("bad tensor record #%llu", (unsigned long long)i);
            return ORNITH_ERR_FORMAT;
        }
        snprintf(t->name, sizeof(t->name), "%s", nm);
        free(nm);
        if (t->n_dims > OGGUF_MAX_DIMS) {
            fclose(f); ogguf_loaded_free(out);
            ornith_set_error("tensor '%s' has %u dims", t->name, t->n_dims);
            return ORNITH_ERR_FORMAT;
        }
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (!rd_u64(f, &t->dims[d])) {
                fclose(f); ogguf_loaded_free(out);
                ornith_set_error("bad dims for '%s'", t->name);
                return ORNITH_ERR_FORMAT;
            }
            t->n_elements *= t->dims[d];
        }
        uint64_t offset;
        if (!rd_u32(f, &t->type) || !rd_u64(f, &offset)) {
            fclose(f); ogguf_loaded_free(out);
            ornith_set_error("bad type/offset for '%s'", t->name);
            return ORNITH_ERR_FORMAT;
        }
        t->data = (void *)(uintptr_t)offset; /* stash offset; resolve below */
    }

    /* data section begins at the next alignment boundary after the index */
    long pos = ftell(f);
    if (pos < 0) { fclose(f); ogguf_loaded_free(out);
                   ornith_set_error("ftell failed"); return ORNITH_ERR_IO; }
    uint64_t data_start = align_up((uint64_t)pos, out->alignment);

    /* mmap the whole file read-only, so weights page in on demand instead of
     * being malloc'd + read up front. The mapping outlives the fd. */
    int fd = fileno(f);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        fclose(f); ogguf_loaded_free(out);
        ornith_set_error("fstat failed for %s", path); return ORNITH_ERR_IO;
    }
    size_t fsize = (size_t)st.st_size;
    if (data_start > fsize) {
        fclose(f); ogguf_loaded_free(out);
        ornith_set_error("data section starts past EOF in %s", path);
        return ORNITH_ERR_FORMAT;
    }
    void *map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        fclose(f); ogguf_loaded_free(out);
        ornith_set_error("mmap failed for %s", path); return ORNITH_ERR_IO;
    }
    /* fd no longer needed; the mapping persists after close. */
    fclose(f);
    out->map = map;
    out->map_size = fsize;

    for (uint64_t i = 0; i < out->n_tensors; i++) {
        ogguf_ltensor *t = &out->tensors[i];
        uint64_t offset = (uint64_t)(uintptr_t)t->data;
        t->data = NULL;
        t->nbytes = oq_row_bytes(t->type, t->n_elements);
        if (!t->nbytes && t->n_elements) {
            ogguf_loaded_free(out);
            ornith_set_error("tensor '%s': cannot size type %s for load",
                             t->name, oggml_type_name(t->type));
            return ORNITH_ERR_UNSUPPORTED;
        }
        /* bounds-check before handing out a pointer into the mapping */
        if (offset > fsize - data_start ||
            t->nbytes > fsize - data_start - offset) {
            ogguf_loaded_free(out);
            ornith_set_error("tensor '%s' data extends past EOF", t->name);
            return ORNITH_ERR_FORMAT;
        }
        t->data = (uint8_t *)map + data_start + offset;
    }

    return ORNITH_OK;
}

void ogguf_loaded_free(ogguf_loaded *l) {
    if (!l) return;
    if (l->kv) {
        for (uint64_t i = 0; i < l->n_kv; i++) {
            free(l->kv[i].key);
            free(l->kv[i].payload);
        }
        free(l->kv);
    }
    /* Tensor `data` pointers reference the mmap, not malloc'd buffers, so we
     * must NOT free them individually — just release the array and the map. */
    free(l->tensors);
    if (l->map && l->map != MAP_FAILED) munmap(l->map, l->map_size);
    memset(l, 0, sizeof(*l));
}

/* ---- quantizer driver -------------------------------------------------- */

ornith_status ornith_quantize_file(const char *in_path, const char *out_path,
                                   uint32_t base, FILE *log) {
    return ornith_quantize_file_imatrix(in_path, out_path, base, NULL, log);
}

ornith_status ornith_quantize_file_imatrix(const char *in_path,
                                           const char *out_path, uint32_t base,
                                           const void *imatrix, FILE *log) {
    const oimatrix *im = (const oimatrix *)imatrix;
    ogguf_loaded in;
    ornith_status s = ogguf_load(in_path, &in);
    if (s != ORNITH_OK) return s;

    ogguf_writer *w = ogguf_writer_new();
    if (!w) { ogguf_loaded_free(&in); ornith_set_error("oom (writer)");
              return ORNITH_ERR_OOM; }

    /* one output buffer kept alive per tensor until the file is written */
    void **keep = calloc(in.n_tensors ? in.n_tensors : 1, sizeof(void *));
    if (!keep) { s = ORNITH_ERR_OOM; ornith_set_error("oom (keep)"); goto done; }

    /* metadata passes through unchanged */
    for (uint64_t i = 0; i < in.n_kv; i++) {
        s = ogguf_w_raw_kv(w, in.kv[i].key, in.kv[i].vtype,
                           in.kv[i].payload, in.kv[i].payload_len);
        if (s != ORNITH_OK) goto done;
    }

    if (log) fprintf(log, "quantize: %llu tensors (%s -> policy)\n",
                     (unsigned long long)in.n_tensors, oggml_type_name(base));

    for (uint64_t i = 0; i < in.n_tensors; i++) {
        ogguf_ltensor *t = &in.tensors[i];

        /* decode source to f32 */
        float *f32 = malloc((t->n_elements ? t->n_elements : 1) * sizeof(float));
        if (!f32) { s = ORNITH_ERR_OOM; ornith_set_error("oom (f32)"); goto done; }
        s = oq_dequantize(t->type, t->data, f32, t->n_elements);
        if (s != ORNITH_OK) { free(f32); goto done; }

        /* choose target via policy, falling back honestly when unimplemented */
        uint32_t want = oq_policy_target(t->name, base);
        uint32_t tgt  = want;
        const char *why = NULL;
        if (!oq_is_implemented(tgt)) { why = "encoder not implemented"; tgt = OGGML_F16; }
        else if (oq_row_bytes(tgt, t->n_elements) == 0) {
            why = "not block-aligned"; tgt = OGGML_F16;
        }

        size_t obytes = oq_row_bytes(tgt, t->n_elements);
        uint8_t *odata = malloc(obytes ? obytes : 1);
        if (!odata) { free(f32); s = ORNITH_ERR_OOM; ornith_set_error("oom (odata)");
                      goto done; }

        /* Importance-weighted path: a recorded per-input-channel vector whose
         * length matches the tensor's input width (dims[0]) is the SAME for
         * every output row, so we quantize row by row, broadcasting it. */
        const float *chan_imp = NULL; size_t cn = 0; uint64_t ccount = 0;
        if (im) chan_imp = oimatrix_get(im, t->name, &cn, &ccount);
        size_t in_w  = t->n_dims > 0 ? (size_t)t->dims[0] : t->n_elements;
        size_t row_b = oq_row_bytes(tgt, in_w);
        int    weighted = chan_imp && in_w && cn == in_w && row_b &&
                          (t->n_elements % in_w) == 0;
        if (weighted) {
            size_t nrows = t->n_elements / in_w;
            for (size_t r = 0; r < nrows && s == ORNITH_OK; r++)
                s = oq_quantize_imatrix(tgt, f32 + r * in_w, odata + r * row_b,
                                        in_w, chan_imp);
        } else {
            s = oq_quantize(tgt, f32, odata, t->n_elements);
        }
        free(f32);
        if (s != ORNITH_OK) { free(odata); goto done; }

        keep[i] = odata;
        s = ogguf_w_tensor(w, t->name, t->n_dims, t->dims, tgt, odata, obytes);
        if (s != ORNITH_OK) goto done;

        if (log) {
            const char *imtag = weighted ? " [imatrix]" : "";
            if (why)
                fprintf(log, "  %-40s %-8s -> %-6s (policy %s; fell back: %s)%s\n",
                        t->name, oggml_type_name(t->type), oggml_type_name(tgt),
                        oggml_type_name(want), why, imtag);
            else
                fprintf(log, "  %-40s %-8s -> %-6s%s\n", t->name,
                        oggml_type_name(t->type), oggml_type_name(tgt), imtag);
        }
    }

    s = ogguf_writer_write(w, out_path);

done:
    /* keep[] held the per-tensor output buffers alive (borrowed by the writer)
     * until the file was written; release them now. */
    if (keep) {
        for (uint64_t i = 0; i < in.n_tensors; i++) free(keep[i]);
        free(keep);
    }
    ogguf_writer_free(w);
    ogguf_loaded_free(&in);
    return s;
}
