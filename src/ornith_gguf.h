/* ornith_gguf.h — a minimal reader for the GGUF container format (v2/v3).
 *
 * We reuse the GGUF on-disk format (and, later, llama.cpp's quant block
 * layouts) but link no external library: this reader parses the header, the
 * metadata key/value table, and the tensor index. It does not mmap weight data
 * — that is the backend loader's job. Its purpose here is `ornith inspect`:
 * tell you what is in a file and whether it is a plausible Ornith model.
 */
#ifndef ORNITH_GGUF_H
#define ORNITH_GGUF_H

#include "ornith.h"
#include "ornith_model.h"

/* Subset of GGML tensor types we care about naming. Numeric values match
 * ggml's enum so they can be compared against GGUF tensor type fields. */
typedef enum {
    OGGML_F32   = 0,
    OGGML_F16   = 1,
    OGGML_Q4_0  = 2,
    OGGML_Q4_1  = 3,
    OGGML_Q5_0  = 6,
    OGGML_Q5_1  = 7,
    OGGML_Q8_0  = 8,
    OGGML_Q8_1  = 9,
    OGGML_Q2_K  = 10,
    OGGML_Q3_K  = 11,
    OGGML_Q4_K  = 12,
    OGGML_Q5_K  = 13,
    OGGML_Q6_K  = 14,
    OGGML_Q8_K  = 15,
    OGGML_IQ2_XXS = 16,
    OGGML_IQ2_XS  = 17,
    OGGML_IQ3_XXS = 18,
    OGGML_IQ2_S   = 21,
    OGGML_IQ3_S   = 22,
    OGGML_BF16    = 30,
    OGGML_TYPE_COUNT = 64,
} oggml_type;

const char *oggml_type_name(uint32_t t);
/* Bits-per-weight for a type (approx for k-/i-quants). 0 if unknown. */
double oggml_type_bpw(uint32_t t);

#define OGGUF_MAX_DIMS 4

typedef struct {
    char     name[256];
    uint32_t n_dims;
    uint64_t dims[OGGUF_MAX_DIMS];
    uint32_t type;          /* oggml_type                                    */
    uint64_t offset;        /* byte offset into the data section             */
    uint64_t n_elements;    /* product of dims                               */
} ogguf_tensor;

typedef struct {
    uint32_t      version;
    uint64_t      n_tensors;
    uint64_t      n_kv;
    char          model_arch[64];  /* "general.architecture" if present       */
    char          model_name[128]; /* "general.name" if present               */
    ogguf_tensor *tensors;         /* n_tensors entries                       */

    /* Aggregate stats computed while scanning the tensor index. */
    uint64_t      total_elements;
    uint64_t      type_elem_count[OGGML_TYPE_COUNT]; /* elems per type        */
    int           n_expert_tensors;
} ogguf_file;

/* Read header + metadata + tensor index from `path`. Caller frees with
 * ogguf_close(). Weight data is NOT read. */
ornith_status ogguf_open(const char *path, ogguf_file *out);
void          ogguf_close(ogguf_file *f);

/* Pretty-print a summary (counts, type histogram, estimated size). */
void ogguf_print(const ogguf_file *f, FILE *out, bool list_tensors);

#endif /* ORNITH_GGUF_H */
