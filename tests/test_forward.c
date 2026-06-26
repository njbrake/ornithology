/* test_forward.c — the full qwen3_5_moe decoder on a tiny seeded model.
 *
 * The system-level expression of the M2 correctness property: running a prompt
 * through prefill (chunked linear-attn scan + causal full-attn pass) must give
 * the same per-position logits as feeding the same tokens one at a time through
 * decode (step recurrence + KV append). Also checks determinism, finiteness,
 * and a checked-in golden top token (regression guard).
 */
#include "ornith_forward.h"
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
    float m=0; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;} return m;
}
static int all_finite(const float *x, int n) {
    for (int i=0;i<n;i++) if (!isfinite(x[i])) return 0;
    return 1;
}

int main(void) {
    ornith_arch a; ornith_arch_tiny(&a);
    of_model *m = of_model_build_synthetic(&a, 0x1234567);
    CHECK(m != NULL, "build synthetic tiny model");
    int V = a.vocab_size;

    int32_t tokens[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    int T = (int)(sizeof(tokens)/sizeof(tokens[0]));

    printf("== forward: prefill logits all-finite ==\n");
    float *pre = malloc((size_t)T*V*sizeof(float));
    of_state *sp = of_state_new(m, 64);
    for (int chunk = 1; chunk <= T; chunk++) {
        /* (use one chunk size for the finiteness check; parity loop below) */
        if (chunk != 4) continue;
        of_state_reset(sp);
        ornith_status st = of_forward_prefill(m, sp, tokens, T, chunk, pre);
        CHECK(st == ORNITH_OK, "prefill ok");
    }
    CHECK(all_finite(pre, T*V), "all prefill logits finite");

    printf("== forward: prefill == decode (system-level parity) ==\n");
    /* decode one token at a time on a fresh state */
    float *dec = malloc((size_t)T*V*sizeof(float));
    of_state *sd = of_state_new(m, 64);
    for (int t=0;t<T;t++) {
        ornith_status st = of_forward_decode(m, sd, tokens[t], dec + (size_t)t*V);
        CHECK(st == ORNITH_OK, t==0 ? "decode step ok" : "decode step ok (cont)");
        if (st != ORNITH_OK) break;
    }
    /* compare against prefill for every chunk size: all must match decode */
    int parity_ok = 1;
    float worst = 0;
    for (int chunk=1; chunk<=T+2; chunk++) {
        of_state_reset(sp);
        of_forward_prefill(m, sp, tokens, T, chunk, pre);
        float md = maxdiff(pre, dec, T*V);
        if (md > worst) worst = md;
        if (md > 1e-3f) { parity_ok = 0;
            printf("    chunk=%d maxdiff=%.3e\n", chunk, md); }
    }
    char msg[96];
    snprintf(msg,sizeof(msg),"prefill==decode for all chunk sizes (worst=%.2e)",
             worst);
    CHECK(parity_ok, msg);

    printf("== forward: determinism ==\n");
    {
        of_model *m2 = of_model_build_synthetic(&a, 0x1234567);
        float *pre2 = malloc((size_t)T*V*sizeof(float));
        of_state *s2 = of_state_new(m2, 64);
        of_forward_prefill(m2, s2, tokens, T, 4, pre2);
        of_state_reset(sp); of_forward_prefill(m, sp, tokens, T, 4, pre);
        CHECK(maxdiff(pre, pre2, T*V) == 0.0f, "same seed -> identical logits");
        of_state_free(s2); of_model_free(m2); free(pre2);
    }

    printf("== forward: golden top token (regression) ==\n");
    {
        of_state_reset(sp);
        of_forward_prefill(m, sp, tokens, T, 4, pre);
        int top = of_argmax(pre + (size_t)(T-1)*V, V);
        printf("    last-position top token = %d\n", top);
        /* GOLDEN: pinned after first green run; guards against silent drift. */
        CHECK(top == GOLDEN_TOP_TOKEN, "golden top token matches");
    }

    free(pre); free(dec);
    of_state_free(sp); of_state_free(sd); of_model_free(m);
    printf("\n%s (%d failure%s)\n",
           failures ? "FORWARD TESTS FAILED" : "ALL FORWARD TESTS PASSED",
           failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
