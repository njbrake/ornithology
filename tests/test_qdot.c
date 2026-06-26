/* test_qdot.c — quant-aware integer dot products (ornith_qdot).
 *
 * For each supported weight type (Q4_K, Q6_K, Q8_0):
 *   - build a weight row and an activation row,
 *   - quantize the weight (oq_quantize) and the activation (qdot_quantize_row_q8_K),
 *   - compute the integer dot (qdot_vec_dot),
 *   - compare against the reference: dequantize the SAME weight bytes to f32 and
 *     float-dot with the ORIGINAL f32 activation.
 * The only difference between the two is the Q8_K rounding of the activation, so
 * the relative error must be small (~1e-2). Also checks block size and the
 * qdot_can() gate. All synthetic; no weights, no network.
 */
#include "ornith_qdot.h"
#include "ornith_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

static double ref_dot(const float *w, const float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)w[i] * (double)x[i];
    return s;
}

/* run one type over an `n`-element row; returns relative error vs f32 ref. */
static double run_type(uint32_t wtype, int n, const char *label) {
    float *w  = malloc((size_t)n * sizeof(float));
    float *x  = malloc((size_t)n * sizeof(float));
    float *wd = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        w[i] = sinf((float)i * 0.11f) * 0.6f + cosf((float)i * 0.37f) * 0.2f;
        x[i] = cosf((float)i * 0.07f) * 1.3f - sinf((float)i * 0.19f) * 0.5f;
    }
    size_t rb = oq_row_bytes(wtype, (size_t)n);
    uint8_t *wq = malloc(rb);
    if (oq_quantize(wtype, w, wq, (size_t)n) != ORNITH_OK) { failures++; printf("  FAIL: quantize %s\n", label); }
    oq_dequantize(wtype, wq, wd, (size_t)n);   /* same bytes the qdot reads */

    oq8k_block *xq = malloc((size_t)(n / OQDOT_QK_K) * sizeof(oq8k_block));
    qdot_quantize_row_q8_K(x, xq, n);

    float got = 0.0f;
    CHECK(qdot_can(wtype, n), label);
    qdot_vec_dot(wtype, n, &got, wq, xq);

    double ref = ref_dot(wd, x, n);   /* dequant-weight . f32-activation */
    double rel = fabs((double)got - ref) / (fabs(ref) + 1e-6);
    printf("    %s: qdot=%.5f ref=%.5f rel=%.2e\n", label, (double)got, ref, rel);

    free(w); free(x); free(wd); free(wq); free(xq);
    return rel;
}

int main(void) {
    printf("== Q8_K block layout ==\n");
    CHECK(sizeof(oq8k_block) == 292, "oq8k_block == 292 bytes");
    CHECK(!qdot_can(OGGML_Q4_K, 200), "qdot_can rejects non-256-aligned n");
    CHECK(!qdot_can(OGGML_Q5_K, 256), "qdot_can rejects unsupported type (Q5_K)");
    CHECK(!qdot_can(OGGML_F32, 256), "qdot_can rejects F32 (uses f32 path)");

    printf("== Q8_K quantize: zero row stays zero ==\n");
    {
        float z[256]; memset(z, 0, sizeof(z));
        oq8k_block b; qdot_quantize_row_q8_K(z, &b, 256);
        int allzero = (b.d == 0.0f);
        for (int i = 0; i < 256; i++) if (b.qs[i]) allzero = 0;
        CHECK(allzero, "zero activation -> zero block");
    }

    printf("== quant-aware dot vs f32 dequant-dot (n=512, 2 super-blocks) ==\n");
    {
        double r4 = run_type(OGGML_Q4_K, 512, "Q4_K");
        double r6 = run_type(OGGML_Q6_K, 512, "Q6_K");
        double r8 = run_type(OGGML_Q8_0, 512, "Q8_0");
        /* Difference is purely the Q8 activation rounding: ~1e-2 relative. */
        CHECK(r4 < 3e-2, "Q4_K dot matches f32 within Q8 activation tol");
        CHECK(r6 < 3e-2, "Q6_K dot matches f32 within Q8 activation tol");
        CHECK(r8 < 3e-2, "Q8_0 dot matches f32 within Q8 activation tol");
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "TESTS FAILED" : "ALL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
