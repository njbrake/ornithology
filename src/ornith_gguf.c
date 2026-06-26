/* ornith_gguf.c — GGUF v2/v3 header, metadata and tensor-index reader.
 *
 * Format reference (little-endian):
 *   u32 magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
 *   n_kv  x  [ gguf_string key | u32 value_type | value ]
 *   n_tns x  [ gguf_string name | u32 n_dims | u64 dims[n_dims]
 *              | u32 ggml_type | u64 offset ]
 * gguf_string = u64 len | bytes (no NUL).
 */
#include "ornith_gguf.h"
#include <stdlib.h>
#include <string.h>

/* GGUF metadata value type tags. */
enum {
    GGUF_T_UINT8=0, GGUF_T_INT8=1, GGUF_T_UINT16=2, GGUF_T_INT16=3,
    GGUF_T_UINT32=4, GGUF_T_INT32=5, GGUF_T_FLOAT32=6, GGUF_T_BOOL=7,
    GGUF_T_STRING=8, GGUF_T_ARRAY=9, GGUF_T_UINT64=10, GGUF_T_INT64=11,
    GGUF_T_FLOAT64=12,
};

const char *oggml_type_name(uint32_t t) {
    switch (t) {
    case OGGML_F32: return "F32";   case OGGML_F16: return "F16";
    case OGGML_Q4_0:return "Q4_0";  case OGGML_Q4_1:return "Q4_1";
    case OGGML_Q5_0:return "Q5_0";  case OGGML_Q5_1:return "Q5_1";
    case OGGML_Q8_0:return "Q8_0";  case OGGML_Q8_1:return "Q8_1";
    case OGGML_Q2_K:return "Q2_K";  case OGGML_Q3_K:return "Q3_K";
    case OGGML_Q4_K:return "Q4_K";  case OGGML_Q5_K:return "Q5_K";
    case OGGML_Q6_K:return "Q6_K";  case OGGML_Q8_K:return "Q8_K";
    case OGGML_IQ2_XXS:return "IQ2_XXS"; case OGGML_IQ2_XS:return "IQ2_XS";
    case OGGML_IQ3_XXS:return "IQ3_XXS"; case OGGML_IQ2_S:return "IQ2_S";
    case OGGML_IQ3_S:return "IQ3_S";     case OGGML_BF16:return "BF16";
    default: return "?";
    }
}

double oggml_type_bpw(uint32_t t) {
    switch (t) {
    case OGGML_F32: return 32.0;  case OGGML_BF16: case OGGML_F16: return 16.0;
    case OGGML_Q8_0: return 8.5;  case OGGML_Q8_K: return 8.0;
    case OGGML_Q6_K: return 6.56; case OGGML_Q5_K: return 5.5;
    case OGGML_Q5_0: case OGGML_Q5_1: return 5.5;
    case OGGML_Q4_K: return 4.5;  case OGGML_Q4_0: case OGGML_Q4_1: return 4.5;
    case OGGML_Q3_K: return 3.44; case OGGML_IQ3_S: case OGGML_IQ3_XXS: return 3.1;
    case OGGML_Q2_K: return 2.63; case OGGML_IQ2_S: return 2.5;
    case OGGML_IQ2_XS: return 2.31; case OGGML_IQ2_XXS: return 2.06;
    default: return 0.0;
    }
}

/* ---- little-endian readers -------------------------------------------- */

static bool rd(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}
static bool rd_u32(FILE *f, uint32_t *v) { return rd(f, v, 4); }
static bool rd_u64(FILE *f, uint64_t *v) { return rd(f, v, 8); }

/* Read a gguf_string into `buf` (truncated to cap-1); always consumes it. */
static bool rd_str(FILE *f, char *buf, size_t cap) {
    uint64_t len;
    if (!rd_u64(f, &len)) return false;
    size_t take = (buf && cap) ? (len < cap - 1 ? (size_t)len : cap - 1) : 0;
    if (take && fread(buf, 1, take, f) != take) return false;
    if (buf && cap) buf[take] = '\0';
    /* skip any remainder */
    uint64_t rest = len - take;
    if (rest && fseek(f, (long)rest, SEEK_CUR) != 0) return false;
    return true;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL: return 1;
    case GGUF_T_UINT16: case GGUF_T_INT16: return 2;
    case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
    case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
    default: return 0;
    }
}

/* Consume a metadata value of `type`, capturing it as a string if `capture`. */
static bool skip_value(FILE *f, uint32_t type, char *capture, size_t cap) {
    if (type == GGUF_T_STRING) return rd_str(f, capture, cap);
    if (type == GGUF_T_ARRAY) {
        uint32_t etype; uint64_t n;
        if (!rd_u32(f, &etype) || !rd_u64(f, &n)) return false;
        if (etype == GGUF_T_STRING) {
            for (uint64_t i = 0; i < n; i++)
                if (!rd_str(f, NULL, 0)) return false;
        } else {
            size_t sz = scalar_size(etype);
            if (!sz) return false;
            if (fseek(f, (long)(sz * n), SEEK_CUR) != 0) return false;
        }
        return true;
    }
    size_t sz = scalar_size(type);
    if (!sz) return false;
    if (capture && cap) capture[0] = '\0';   /* numbers not captured here */
    return fseek(f, (long)sz, SEEK_CUR) == 0;
}

ornith_status ogguf_open(const char *path, ogguf_file *out) {
    memset(out, 0, sizeof(*out));
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
    if (out->n_tensors > (1u << 24)) {
        fclose(f); ornith_set_error("implausible tensor count");
        return ORNITH_ERR_FORMAT;
    }

    /* metadata KVs — capture a couple we care about, skip the rest */
    for (uint64_t i = 0; i < out->n_kv; i++) {
        char key[256];
        uint32_t vtype;
        if (!rd_str(f, key, sizeof(key)) || !rd_u32(f, &vtype)) {
            fclose(f); ornith_set_error("bad metadata kv #%llu",
                (unsigned long long)i);
            return ORNITH_ERR_FORMAT;
        }
        bool want_arch = strcmp(key, "general.architecture") == 0;
        bool want_name = strcmp(key, "general.name") == 0;
        char tmp[128] = {0};
        if (!skip_value(f, vtype, (want_arch || want_name) ? tmp : NULL,
                        sizeof(tmp))) {
            fclose(f); ornith_set_error("bad value for key '%s'", key);
            return ORNITH_ERR_FORMAT;
        }
        if (want_arch) snprintf(out->model_arch, sizeof(out->model_arch),
                                "%s", tmp);
        if (want_name) snprintf(out->model_name, sizeof(out->model_name),
                                "%s", tmp);
    }

    /* tensor index */
    out->tensors = calloc(out->n_tensors, sizeof(ogguf_tensor));
    if (!out->tensors && out->n_tensors) {
        fclose(f); ornith_set_error("oom for %llu tensors",
            (unsigned long long)out->n_tensors);
        return ORNITH_ERR_OOM;
    }
    for (uint64_t i = 0; i < out->n_tensors; i++) {
        ogguf_tensor *t = &out->tensors[i];
        if (!rd_str(f, t->name, sizeof(t->name)) || !rd_u32(f, &t->n_dims)) {
            fclose(f); free(out->tensors); out->tensors = NULL;
            ornith_set_error("bad tensor record #%llu",
                (unsigned long long)i);
            return ORNITH_ERR_FORMAT;
        }
        if (t->n_dims > OGGUF_MAX_DIMS) {
            fclose(f); free(out->tensors); out->tensors = NULL;
            ornith_set_error("tensor '%s' has %u dims", t->name, t->n_dims);
            return ORNITH_ERR_FORMAT;
        }
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (!rd_u64(f, &t->dims[d])) {
                fclose(f); free(out->tensors); out->tensors = NULL;
                ornith_set_error("bad dims for '%s'", t->name);
                return ORNITH_ERR_FORMAT;
            }
            t->n_elements *= t->dims[d];
        }
        if (!rd_u32(f, &t->type) || !rd_u64(f, &t->offset)) {
            fclose(f); free(out->tensors); out->tensors = NULL;
            ornith_set_error("bad type/offset for '%s'", t->name);
            return ORNITH_ERR_FORMAT;
        }
        out->total_elements += t->n_elements;
        if (t->type < OGGML_TYPE_COUNT)
            out->type_elem_count[t->type] += t->n_elements;
        if (strstr(t->name, "exps") || strstr(t->name, "expert"))
            out->n_expert_tensors++;
    }

    fclose(f);
    return ORNITH_OK;
}

void ogguf_close(ogguf_file *f) {
    if (!f) return;
    free(f->tensors);
    f->tensors = NULL;
}

void ogguf_print(const ogguf_file *f, FILE *out, bool list_tensors) {
    fprintf(out, "GGUF file\n");
    fprintf(out, "  version           : %u\n", f->version);
    fprintf(out, "  general.arch      : %s\n",
            f->model_arch[0] ? f->model_arch : "(unset)");
    fprintf(out, "  general.name      : %s\n",
            f->model_name[0] ? f->model_name : "(unset)");
    fprintf(out, "  tensors           : %llu\n",
            (unsigned long long)f->n_tensors);
    fprintf(out, "  metadata kv       : %llu\n", (unsigned long long)f->n_kv);
    fprintf(out, "  expert tensors    : %d\n", f->n_expert_tensors);
    fprintf(out, "  total elements    : %.2fB\n", f->total_elements / 1e9);

    fprintf(out, "  quant histogram (by element share):\n");
    double est_bits = 0.0;
    for (uint32_t t = 0; t < OGGML_TYPE_COUNT; t++) {
        uint64_t c = f->type_elem_count[t];
        if (!c) continue;
        double pct = 100.0 * (double)c / (double)(f->total_elements ?
                     f->total_elements : 1);
        double bpw = oggml_type_bpw(t);
        est_bits += (double)c * bpw;
        fprintf(out, "    %-8s %6.2f%%  (%.2fB elems, %.2f bpw)\n",
                oggml_type_name(t), pct, c / 1e9, bpw);
    }
    fprintf(out, "  est. weight size  : ~%.1f GB\n", est_bits / 8.0 / 1e9);

    if (list_tensors) {
        fprintf(out, "  tensors:\n");
        for (uint64_t i = 0; i < f->n_tensors; i++) {
            const ogguf_tensor *t = &f->tensors[i];
            fprintf(out, "    %-44s %-8s [", t->name, oggml_type_name(t->type));
            for (uint32_t d = 0; d < t->n_dims; d++)
                fprintf(out, "%llu%s", (unsigned long long)t->dims[d],
                        d + 1 < t->n_dims ? "," : "");
            fprintf(out, "]\n");
        }
    }
}
