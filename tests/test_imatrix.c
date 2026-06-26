/* test_imatrix.c — coverage for the importance matrix:
 *
 *   1. accumulate + get: per-input-channel sum of squared activations over
 *      several columns, plus token count.
 *   2. save/load round-trip: write imatrix.dat, read it back, assert the
 *      per-tensor vectors, lengths, names, and counts survive.
 *   3. importance-weighted quantization quality: build a synthetic tensor where
 *      a few channels matter, mark them important, and assert that
 *      oq_quantize_imatrix gives LOWER reconstruction error on those channels
 *      than plain RTN (oq_quantize) does.
 */
#include "ornith_imatrix.h"
#include "ornith_quant.h"
#include "ornith_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

/* deterministic LCG so the synthetic tensor is reproducible */
static uint32_t rng_state = 12345u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; /* [-1,1) */
}

/* sum of squared (recon - ref) over a set of channel indices */
static double err_over(const float *ref, const float *recon,
                       const int *chans, int nch) {
    double e = 0;
    for (int i = 0; i < nch; i++) {
        float d = recon[chans[i]] - ref[chans[i]];
        e += (double)d * d;
    }
    return e;
}

/* Round-trip a quant type with and without importance; return the two
 * important-channel errors via out params. */
static void quant_quality(uint32_t type, const float *src, int n,
                          const float *imp, const int *chans, int nch,
                          double *rtn_err, double *imat_err) {
    size_t rb = oq_row_bytes(type, (size_t)n);
    uint8_t *q = malloc(rb);
    float *dec = malloc((size_t)n * sizeof(float));

    /* unweighted (RTN) */
    oq_quantize(type, src, q, (size_t)n);
    oq_dequantize(type, q, dec, (size_t)n);
    *rtn_err = err_over(src, dec, chans, nch);

    /* importance-weighted */
    oq_quantize_imatrix(type, src, q, (size_t)n, imp);
    oq_dequantize(type, q, dec, (size_t)n);
    *imat_err = err_over(src, dec, chans, nch);

    free(q); free(dec);
}

int main(void) {
    printf("== accumulate + get ==\n");
    {
        oimatrix *im = oimatrix_new();
        CHECK(im != NULL, "imatrix alloc");
        /* 2 columns of 3 channels, column-major: col0={1,2,3}, col1={4,0,-1} */
        float X[6] = { 1, 2, 3,  4, 0, -1 };
        oimatrix_accumulate(im, "blk.0.attn_q.weight", X, 3, 2);
        size_t n = 0; uint64_t cnt = 0;
        const float *v = oimatrix_get(im, "blk.0.attn_q.weight", &n, &cnt);
        CHECK(v != NULL, "entry present after accumulate");
        CHECK(n == 3, "vector length == in (3)");
        CHECK(cnt == 2, "token count == Tn (2)");
        /* sums of squares per channel: 1+16=17, 4+0=4, 9+1=10 */
        CHECK(fabsf(v[0] - 17.f) < 1e-4f, "channel 0 sum = 17");
        CHECK(fabsf(v[1] -  4.f) < 1e-4f, "channel 1 sum = 4");
        CHECK(fabsf(v[2] - 10.f) < 1e-4f, "channel 2 sum = 10");
        /* a second accumulation adds on */
        oimatrix_accumulate(im, "blk.0.attn_q.weight", X, 3, 2);
        v = oimatrix_get(im, "blk.0.attn_q.weight", &n, &cnt);
        CHECK(cnt == 4, "count doubles after second accumulate");
        CHECK(fabsf(v[0] - 34.f) < 1e-4f, "channel 0 sum doubles to 34");
        CHECK(oimatrix_get(im, "nonexistent", &n, &cnt) == NULL, "missing tensor -> NULL");
        oimatrix_free(im);
    }

    printf("== save / load round-trip ==\n");
    {
        oimatrix *im = oimatrix_new();
        float A[8]; for (int i = 0; i < 8; i++) A[i] = (float)(i + 1);
        float B[12]; for (int i = 0; i < 12; i++) B[i] = sinf((float)i);
        oimatrix_accumulate(im, "blk.0.ffn_down.weight", A, 4, 2); /* in=4, 2 cols */
        oimatrix_accumulate(im, "output.weight",        B, 3, 4); /* in=3, 4 cols */

        const char *path = "/tmp/ornith_test_imatrix.dat";
        CHECK(oimatrix_save(im, path) == ORNITH_OK, "save ok");

        oimatrix *im2 = NULL;
        CHECK(oimatrix_load(path, &im2) == ORNITH_OK, "load ok");
        CHECK(oimatrix_count(im2) == 2, "tensor count preserved");

        for (size_t t = 0; t < oimatrix_count(im); t++) {
            const char *nm = oimatrix_name_at(im, t);
            size_t n1 = 0, n2 = 0; uint64_t c1 = 0, c2 = 0;
            const float *v1 = oimatrix_get(im,  nm, &n1, &c1);
            const float *v2 = oimatrix_get(im2, nm, &n2, &c2);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s present after load", nm);
            CHECK(v2 != NULL && n1 == n2 && c1 == c2, msg);
            int eq = (v1 && v2 && n1 == n2);
            for (size_t k = 0; eq && k < n1; k++)
                if (fabsf(v1[k] - v2[k]) > 1e-4f) eq = 0;
            snprintf(msg, sizeof(msg), "%s values round-trip", nm);
            CHECK(eq, msg);
        }
        oimatrix_free(im);
        oimatrix_free(im2);
        remove(path);
    }

    printf("== bad file rejected ==\n");
    {
        const char *path = "/tmp/ornith_test_imatrix_bad.dat";
        FILE *f = fopen(path, "wb");
        if (f) { fputs("NOPE....garbage", f); fclose(f); }
        oimatrix *im = NULL;
        CHECK(oimatrix_load(path, &im) != ORNITH_OK, "garbage file rejected");
        remove(path);
    }

    printf("== importance-weighted quant beats RTN on important channels ==\n");
    {
        /* 256-element row = one k-quant super-block. Put one "important" channel
         * in each 32-wide sub-block (8 total), each holding a mid-range value the
         * coarse RTN scale represents poorly; the rest is wide-range noise that
         * pins the unweighted scale. High importance on the 8 channels should
         * pull the weighted fit toward them. */
        enum { N = 256, NCH = 8 };
        float src[N], imp[N];
        int chans[NCH];
        for (int i = 0; i < N; i++) { src[i] = frand() * 2.0f; imp[i] = 1.0f; }
        for (int s = 0; s < NCH; s++) {
            int c = s * 32 + 5;       /* one important channel per sub-block */
            chans[s] = c;
            src[c] = 0.137f + 0.05f * (float)s;   /* deliberate mid-range value */
            imp[c] = 1000.0f;                       /* high importance           */
        }

        struct { uint32_t type; const char *name; } types[] = {
            { OGGML_Q4_K, "Q4_K" },
            { OGGML_Q2_K, "Q2_K" },
            { OGGML_Q5_K, "Q5_K" },
            { OGGML_IQ2_XXS, "IQ2_XXS" },
        };
        for (size_t ti = 0; ti < sizeof(types)/sizeof(types[0]); ti++) {
            double rtn = 0, imat = 0;
            quant_quality(types[ti].type, src, N, imp, chans, NCH, &rtn, &imat);
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "%-8s important-channel err: RTN=%.5g  imatrix=%.5g  (imatrix lower)",
                     types[ti].name, rtn, imat);
            CHECK(imat < rtn, msg);
        }

        /* sanity: with NULL importance, the weighted entry point is identical
         * to plain RTN (byte-for-byte). */
        size_t rb = oq_row_bytes(OGGML_Q4_K, N);
        uint8_t *a = malloc(rb), *b = malloc(rb);
        oq_quantize(OGGML_Q4_K, src, a, N);
        oq_quantize_imatrix(OGGML_Q4_K, src, b, N, NULL);
        CHECK(memcmp(a, b, rb) == 0, "oq_quantize_imatrix(NULL) == oq_quantize");
        free(a); free(b);
    }

    if (failures) { printf("\n%d check(s) FAILED\n", failures); return 1; }
    printf("\nall imatrix checks passed\n");
    return 0;
}
