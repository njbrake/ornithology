/* test_attn.c — attention correctness.
 *
 * The headline test of milestone M2: the gated delta-net chunked prefill must
 * reproduce the step-by-step decode recurrence exactly (within fp tolerance),
 * for any chunk size, with and without a nonzero incoming state. Also: full GQA
 * step path == naive O(T^2) reference, and conv prefill == conv steps.
 */
#include "ornith_attn.h"
#include "ornith_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

static float maxdiff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d>m) m=d; }
    return m;
}

/* ---- gated delta-net: prefill (chunked scan) == decode (step) ---------- */

static void test_delta_parity(void) {
    printf("== delta-net: prefill (chunked scan) == decode (step) ==\n");
    int dk = 6, dv = 5, T = 17;
    ot_rng r = ot_rng_seed(0xABCDEF);

    float *Q = malloc(T*dk*sizeof(float));
    float *K = malloc(T*dk*sizeof(float));
    float *V = malloc(T*dv*sizeof(float));
    float *al = malloc(T*sizeof(float));
    float *be = malloc(T*sizeof(float));
    ot_rng_fill(&r, Q, T*dk, -1, 1);
    ot_rng_fill(&r, K, T*dk, -1, 1);
    ot_rng_fill(&r, V, T*dv, -1, 1);
    /* decay alpha in (0.85,0.999), beta in (0.1,0.9): realistic gated ranges. */
    for (int t=0;t<T;t++){ al[t]=ot_rng_uniform(&r,0.85f,0.999f);
                           be[t]=ot_rng_uniform(&r,0.1f,0.9f); }

    /* reference: step decode */
    float *O_dec = malloc(T*dv*sizeof(float));
    ornith_delta_state sd; ornith_delta_init(&sd, dk, dv);
    for (int t=0;t<T;t++)
        ornith_delta_step(&sd, Q+t*dk, K+t*dk, V+t*dv, al[t], be[t], O_dec+t*dv);

    /* chunked prefill for several chunk sizes, all must match decode */
    int chunks[] = {1, 2, 3, 4, 5, 8, 16, T};
    for (unsigned ci=0; ci<sizeof(chunks)/sizeof(chunks[0]); ci++) {
        int C = chunks[ci];
        float *O_pre = malloc(T*dv*sizeof(float));
        ornith_delta_state sp; ornith_delta_init(&sp, dk, dv);
        ornith_delta_prefill(&sp, Q, K, V, al, be, T, C, O_pre);
        float od = maxdiff(O_dec, O_pre, T*dv);
        float sdf = maxdiff(sd.S, sp.S, dk*dv);
        char msg[96];
        snprintf(msg,sizeof(msg),"chunk=%-2d  out diff=%.2e  state diff=%.2e",
                 C, od, sdf);
        CHECK(od < 2e-4f && sdf < 2e-4f, msg);
        ornith_delta_free(&sp); free(O_pre);
    }
    ornith_delta_free(&sd);

    /* parity must also hold when the chunk boundary lands mid-stream with a
     * NONZERO incoming state: prefill[0:s] then prefill[s:T] == decode. */
    {
        int split = 7;
        float *O_a = malloc(T*dv*sizeof(float));
        ornith_delta_state sp; ornith_delta_init(&sp, dk, dv);
        ornith_delta_prefill(&sp, Q, K, V, al, be, split, 3, O_a);
        ornith_delta_prefill(&sp, Q+split*dk, K+split*dk, V+split*dv,
                             al+split, be+split, T-split, 4, O_a+split*dv);
        float od = maxdiff(O_dec, O_a, T*dv);
        CHECK(od < 2e-4f, "split prefill (nonzero carried state) == decode");
        ornith_delta_free(&sp); free(O_a);
    }

    free(Q); free(K); free(V); free(al); free(be); free(O_dec);
}

/* ---- full GQA: step path == naive O(T^2) reference -------------------- */

static void test_gqa(void) {
    printf("== full GQA: step (KV cache) == naive reference ==\n");
    int nh = 4, nkv = 2, hd = 8, T = 11;
    int qd = nh*hd, kvd = nkv*hd;
    ot_rng r = ot_rng_seed(42);
    float *Q = malloc(T*qd*sizeof(float));
    float *K = malloc(T*kvd*sizeof(float));
    float *V = malloc(T*kvd*sizeof(float));
    ot_rng_fill(&r, Q, T*qd, -1, 1);
    ot_rng_fill(&r, K, T*kvd, -1, 1);
    ot_rng_fill(&r, V, T*kvd, -1, 1);

    float *O_ref = malloc(T*qd*sizeof(float));
    ornith_gqa_reference(Q, K, V, T, nh, nkv, hd, O_ref, 0.0f);

    ornith_kv_cache c; ornith_kv_init(&c, T, nkv, hd);
    float *O_step = malloc(T*qd*sizeof(float));
    for (int t=0;t<T;t++)
        ornith_gqa_step(&c, Q+t*qd, K+t*kvd, V+t*kvd, nh, O_step+t*qd, 0.0f);
    CHECK(maxdiff(O_ref, O_step, T*qd) < 1e-5f, "GQA step == reference");

    /* sanity: with T=1, output is exactly V (softmax over one element). */
    {
        ornith_kv_cache c1; ornith_kv_init(&c1, 1, nkv, hd);
        float o1[32];
        ornith_gqa_step(&c1, Q, K, V, nh, o1, 0.0f);
        /* head 0 uses kv head 0 -> equals V[kvhead0] */
        CHECK(maxdiff(o1, V, hd) < 1e-5f, "GQA T=1 returns V");
        ornith_kv_free(&c1);
    }
    ornith_kv_free(&c);
    free(Q); free(K); free(V); free(O_ref); free(O_step);
}

/* ---- causal conv: prefill == steps, causality holds ------------------- */

static void test_conv(void) {
    printf("== causal depthwise conv: prefill == step, causal ==\n");
    int K = 4, C = 3, T = 9;
    ot_rng r = ot_rng_seed(7);
    float *X = malloc(T*C*sizeof(float));
    float *w = malloc(C*K*sizeof(float));
    float *b = malloc(C*sizeof(float));
    ot_rng_fill(&r, X, T*C, -1, 1);
    ot_rng_fill(&r, w, C*K, -1, 1);
    ot_rng_fill(&r, b, C, -0.2f, 0.2f);

    float *O_pre = malloc(T*C*sizeof(float));
    ornith_conv_state cs; ornith_conv_init(&cs, K, C);
    ornith_conv_prefill(&cs, X, w, b, T, O_pre);
    ornith_conv_free(&cs);

    float *O_step = malloc(T*C*sizeof(float));
    ornith_conv_state cs2; ornith_conv_init(&cs2, K, C);
    for (int t=0;t<T;t++)
        ornith_conv_step(&cs2, X+t*C, w, b, O_step+t*C);
    CHECK(maxdiff(O_pre,O_step,T*C) < 1e-6f, "conv prefill == step");

    /* causality: first output uses only x_0 (history zero) => out0 = b + w[K-1]*x0 */
    for (int c=0;c<C;c++) {
        float expect = b[c] + w[c*K + (K-1)]*X[c];
        if (fabsf(O_pre[c]-expect) > 1e-5f) { CHECK(0,"conv causal t=0"); break; }
        if (c==C-1) CHECK(1,"conv causal t=0");
    }
    ornith_conv_free(&cs2);
    free(X); free(w); free(b); free(O_pre); free(O_step);
}

int main(void) {
    test_delta_parity();
    test_gqa();
    test_conv();
    printf("\n%s (%d failure%s)\n",
           failures ? "ATTN TESTS FAILED" : "ALL ATTN TESTS PASSED",
           failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
