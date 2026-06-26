/* ornith_imatrix.c — importance-matrix collection, lookup, and file I/O.
 *
 * Storage is a flat, append-on-first-sight array of per-tensor entries keyed by
 * name (linear search: a model has a few hundred matvec tensors, so this is not
 * a hot path — the accumulation inside each entry is). Sums are kept in double
 * for accumulation robustness over long corpora and narrowed to f32 on save, to
 * match the documented file format.
 */
#include "ornith_imatrix.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    char    *name;
    size_t   n;       /* per-input-channel vector length                */
    uint64_t count;   /* tokens accumulated                             */
    double  *sum;     /* sum of squared activations per channel (n)     */
    float   *view;    /* lazily-built f32 mirror returned by _get (n)   */
} imat_entry;

struct oimatrix {
    imat_entry *e;
    size_t      n, cap;
};

oimatrix *oimatrix_new(void) {
    return calloc(1, sizeof(oimatrix));
}

void oimatrix_free(oimatrix *im) {
    if (!im) return;
    for (size_t i = 0; i < im->n; i++) {
        free(im->e[i].name); free(im->e[i].sum); free(im->e[i].view);
    }
    free(im->e);
    free(im);
}

static imat_entry *find_entry(oimatrix *im, const char *name) {
    for (size_t i = 0; i < im->n; i++)
        if (strcmp(im->e[i].name, name) == 0) return &im->e[i];
    return NULL;
}

/* Find or create the entry for `name` with channel-vector length `n`. Returns
 * NULL on OOM or if an existing entry's length disagrees (defensive: a tensor
 * seen with two different input widths is skipped rather than corrupted). */
static imat_entry *get_or_add(oimatrix *im, const char *name, size_t n) {
    imat_entry *e = find_entry(im, name);
    if (e) return (e->n == n) ? e : NULL;
    if (im->n == im->cap) {
        size_t cap = im->cap ? im->cap * 2 : 64;
        imat_entry *ne = realloc(im->e, cap * sizeof(imat_entry));
        if (!ne) return NULL;
        im->e = ne; im->cap = cap;
    }
    e = &im->e[im->n];
    memset(e, 0, sizeof(*e));
    e->name = malloc(strlen(name) + 1);
    e->sum  = calloc(n ? n : 1, sizeof(double));
    if (!e->name || !e->sum) { free(e->name); free(e->sum); return NULL; }
    strcpy(e->name, name);
    e->n = n;
    im->n++;
    return e;
}

void oimatrix_accumulate(oimatrix *im, const char *name,
                         const float *X, int in, int Tn) {
    if (!im || in <= 0 || Tn <= 0) return;
    imat_entry *e = get_or_add(im, name, (size_t)in);
    if (!e) return;
    for (int t = 0; t < Tn; t++) {
        const float *col = X + (size_t)t * in;
        for (int i = 0; i < in; i++) e->sum[i] += (double)col[i] * col[i];
    }
    e->count += (uint64_t)Tn;
}

const float *oimatrix_get(const oimatrix *im, const char *name,
                          size_t *n_out, uint64_t *count_out) {
    /* The public vector is f32 (the documented importance type); materialize it
     * lazily into a per-entry f32 mirror so callers get a stable pointer. */
    for (size_t i = 0; i < im->n; i++) {
        if (strcmp(im->e[i].name, name) != 0) continue;
        imat_entry *e = &im->e[i];
        if (!e->view) {
            e->view = malloc((e->n ? e->n : 1) * sizeof(float));
            if (!e->view) return NULL;
        }
        for (size_t k = 0; k < e->n; k++) e->view[k] = (float)e->sum[k];
        if (n_out) *n_out = e->n;
        if (count_out) *count_out = e->count;
        return e->view;
    }
    return NULL;
}

size_t oimatrix_count(const oimatrix *im) { return im ? im->n : 0; }

const char *oimatrix_name_at(const oimatrix *im, size_t i) {
    return (im && i < im->n) ? im->e[i].name : NULL;
}

/* ---- file I/O ---------------------------------------------------------- */

ornith_status oimatrix_save(const oimatrix *im, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { ornith_set_error("cannot open %s for writing", path); return ORNITH_ERR_IO; }
    uint32_t version = 1;
    uint64_t nt = im->n;
    int ok = 1;
    ok &= fwrite("OIMX", 1, 4, f) == 4;
    ok &= fwrite(&version, sizeof(version), 1, f) == 1;
    ok &= fwrite(&nt, sizeof(nt), 1, f) == 1;
    for (size_t i = 0; i < im->n && ok; i++) {
        imat_entry *e = &im->e[i];
        uint32_t name_len = (uint32_t)strlen(e->name);
        uint64_t n = e->n, count = e->count;
        ok &= fwrite(&name_len, sizeof(name_len), 1, f) == 1;
        ok &= fwrite(e->name, 1, name_len, f) == name_len;
        ok &= fwrite(&n, sizeof(n), 1, f) == 1;
        ok &= fwrite(&count, sizeof(count), 1, f) == 1;
        for (size_t k = 0; k < e->n && ok; k++) {
            float v = (float)e->sum[k];
            ok &= fwrite(&v, sizeof(v), 1, f) == 1;
        }
    }
    if (!ok) { fclose(f); ornith_set_error("short write to %s", path); return ORNITH_ERR_IO; }
    if (fclose(f) != 0) { ornith_set_error("close failed for %s", path); return ORNITH_ERR_IO; }
    return ORNITH_OK;
}

ornith_status oimatrix_load(const char *path, oimatrix **out) {
    FILE *f = fopen(path, "rb");
    if (!f) { ornith_set_error("cannot open %s", path); return ORNITH_ERR_IO; }
    char magic[4];
    uint32_t version = 0;
    uint64_t nt = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OIMX", 4) != 0) {
        fclose(f); ornith_set_error("not an imatrix file (bad magic)");
        return ORNITH_ERR_FORMAT;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 ||
        fread(&nt, sizeof(nt), 1, f) != 1) {
        fclose(f); ornith_set_error("truncated imatrix header");
        return ORNITH_ERR_FORMAT;
    }
    if (version != 1) {
        fclose(f); ornith_set_error("unsupported imatrix version %u", version);
        return ORNITH_ERR_UNSUPPORTED;
    }
    if (nt > (1u << 24)) {
        fclose(f); ornith_set_error("implausible imatrix tensor count");
        return ORNITH_ERR_FORMAT;
    }
    oimatrix *im = oimatrix_new();
    if (!im) { fclose(f); ornith_set_error("oom (imatrix)"); return ORNITH_ERR_OOM; }

    for (uint64_t i = 0; i < nt; i++) {
        uint32_t name_len = 0;
        uint64_t n = 0, count = 0;
        if (fread(&name_len, sizeof(name_len), 1, f) != 1 || name_len > (1u << 16)) goto bad;
        char *name = malloc((size_t)name_len + 1);
        if (!name) { oimatrix_free(im); fclose(f); ornith_set_error("oom (name)"); return ORNITH_ERR_OOM; }
        if (fread(name, 1, name_len, f) != name_len) { free(name); goto bad; }
        name[name_len] = '\0';
        if (fread(&n, sizeof(n), 1, f) != 1 || fread(&count, sizeof(count), 1, f) != 1 ||
            n > (1u << 28)) { free(name); goto bad; }
        imat_entry *e = get_or_add(im, name, (size_t)n);
        free(name);
        if (!e) { oimatrix_free(im); fclose(f); ornith_set_error("oom (entry)"); return ORNITH_ERR_OOM; }
        e->count = count;
        for (size_t k = 0; k < (size_t)n; k++) {
            float v;
            if (fread(&v, sizeof(v), 1, f) != 1) goto bad;
            e->sum[k] = (double)v;
        }
    }
    fclose(f);
    *out = im;
    return ORNITH_OK;

bad:
    oimatrix_free(im);
    fclose(f);
    ornith_set_error("truncated/corrupt imatrix in %s", path);
    return ORNITH_ERR_FORMAT;
}

/* ---- global collection hook ------------------------------------------- */

static oimatrix *g_collector = NULL;

void oimatrix_collect_begin(oimatrix *im) { g_collector = im; }
void oimatrix_collect_end(void)           { g_collector = NULL; }

void oimatrix_on_matmul(const char *name, const float *X, int in, int Tn) {
    oimatrix *im = g_collector;
    if (!im) return;
    oimatrix_accumulate(im, name, X, in, Tn);
}
