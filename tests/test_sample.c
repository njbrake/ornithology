/* test_sample.c — unit tests for the logits sampler (ornith_sample).
 * Run via `make test`. Covers: greedy == argmax, greedy determinism,
 * top_k=1 == argmax, fixed-seed reproducibility, top-p nucleus restriction,
 * min-p restriction, and repeat-penalty demotion. */
#include "ornith_sample.h"
#include "ornith_tensor.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

static int32_t ref_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int main(void) {
    printf("== sample: greedy == argmax ==\n");
    {
        float logits[6] = { 0.1f, -2.0f, 3.5f, 1.2f, 3.4f, -0.5f };
        osample_params p = osample_params_default();  /* temperature 0 */
        ot_rng rng = ot_rng_seed(123);
        int32_t got = osample(logits, 6, &p, &rng, NULL, 0);
        CHECK(got == ref_argmax(logits, 6) && got == 2, "greedy returns argmax");
    }

    printf("== sample: greedy deterministic ==\n");
    {
        float base[8] = { 1, 2, 9, 4, 8, 3, 0, 5 };
        osample_params p = osample_params_default();
        int32_t first = -1; int stable = 1;
        for (int t = 0; t < 20; t++) {
            float logits[8]; for (int i = 0; i < 8; i++) logits[i] = base[i];
            ot_rng rng = ot_rng_seed((uint64_t)t * 7919u);
            int32_t got = osample(logits, 8, &p, &rng, NULL, 0);
            if (first < 0) first = got;
            if (got != first || got != 2) stable = 0;
        }
        CHECK(stable, "temperature<=0 deterministic across seeds");
    }

    printf("== sample: top_k=1 == argmax ==\n");
    {
        float base[6] = { 0.5f, 4.0f, 1.0f, 3.9f, -1.0f, 2.0f };
        osample_params p = osample_params_default();
        p.temperature = 1.0f;   /* sampling enabled */
        p.top_k = 1;
        int ok = 1;
        for (int t = 0; t < 50; t++) {
            float logits[6]; for (int i = 0; i < 6; i++) logits[i] = base[i];
            ot_rng rng = ot_rng_seed((uint64_t)t + 1);
            if (osample(logits, 6, &p, &rng, NULL, 0) != 1) ok = 0;
        }
        CHECK(ok, "top_k=1 always selects argmax");
    }

    printf("== sample: fixed seed reproducible ==\n");
    {
        float base[10] = { 0.2f, 1.1f, 0.9f, 2.0f, 1.5f, 0.3f, 1.8f, 0.7f, 2.1f, 1.0f };
        osample_params p = osample_params_default();
        p.temperature = 1.0f; p.seed = 42;
        float l1[10], l2[10];
        for (int i = 0; i < 10; i++) { l1[i] = base[i]; l2[i] = base[i]; }
        ot_rng r1 = ot_rng_seed(p.seed);
        ot_rng r2 = ot_rng_seed(p.seed);
        int32_t a = osample(l1, 10, &p, &r1, NULL, 0);
        int32_t b = osample(l2, 10, &p, &r2, NULL, 0);
        CHECK(a == b, "same seed => same token");

        /* A full sequence is reproducible too. */
        int seqA[16], seqB[16]; int matched = 1;
        ot_rng ra = ot_rng_seed(p.seed), rb = ot_rng_seed(p.seed);
        for (int s = 0; s < 16; s++) {
            float la[10], lb[10];
            for (int i = 0; i < 10; i++) { la[i] = base[i]; lb[i] = base[i]; }
            seqA[s] = osample(la, 10, &p, &ra, NULL, 0);
            seqB[s] = osample(lb, 10, &p, &rb, NULL, 0);
            if (seqA[s] != seqB[s]) matched = 0;
        }
        CHECK(matched, "same seed => same 16-token sequence");
    }

    printf("== sample: top-p keeps the nucleus ==\n");
    {
        /* Two dominant logits; everything else far lower. With a tight top_p,
         * only the top two tokens should ever be drawn. */
        float base[6] = { 10.0f, 9.0f, -5.0f, -6.0f, -7.0f, -8.0f };
        osample_params p = osample_params_default();
        p.temperature = 1.0f; p.top_p = 0.95f;
        int in_nucleus = 1, saw0 = 0, saw1 = 0;
        for (int t = 0; t < 400; t++) {
            float logits[6]; for (int i = 0; i < 6; i++) logits[i] = base[i];
            ot_rng rng = ot_rng_seed((uint64_t)t * 2654435761u + 1u);
            int32_t got = osample(logits, 6, &p, &rng, NULL, 0);
            if (got != 0 && got != 1) in_nucleus = 0;
            if (got == 0) saw0 = 1;
            if (got == 1) saw1 = 1;
        }
        CHECK(in_nucleus, "top-p draws stay inside the nucleus {0,1}");
        CHECK(saw0 && saw1, "top-p still samples both nucleus tokens");
    }

    printf("== sample: min-p restriction ==\n");
    {
        /* One token dominates; a high min-p prunes everything else. */
        float base[5] = { 8.0f, 2.0f, 1.0f, 0.0f, -1.0f };
        osample_params p = osample_params_default();
        p.temperature = 1.0f; p.min_p = 0.5f;
        int only_top = 1;
        for (int t = 0; t < 200; t++) {
            float logits[5]; for (int i = 0; i < 5; i++) logits[i] = base[i];
            ot_rng rng = ot_rng_seed((uint64_t)t * 40503u + 7u);
            if (osample(logits, 5, &p, &rng, NULL, 0) != 0) only_top = 0;
        }
        CHECK(only_top, "high min-p keeps only the dominant token");
    }

    printf("== sample: repeat penalty demotes a repeated token ==\n");
    {
        /* token 2 is the argmax; token 4 is a close runner-up. A repeat penalty
         * over a window that contains token 2 should pull it below token 4. */
        float base[6] = { 0.0f, 1.0f, 5.0f, 1.0f, 4.0f, 0.0f };
        int32_t recent[3] = { 2, 2, 2 };

        /* Without penalty (greedy): argmax is token 2. */
        {
            float logits[6]; for (int i = 0; i < 6; i++) logits[i] = base[i];
            osample_params p = osample_params_default();  /* greedy, no penalty */
            ot_rng rng = ot_rng_seed(1);
            CHECK(osample(logits, 6, &p, &rng, recent, 3) == 2,
                  "no penalty: token 2 wins");
        }
        /* With penalty (greedy): token 2 is demoted, token 4 wins. */
        {
            float logits[6]; for (int i = 0; i < 6; i++) logits[i] = base[i];
            osample_params p = osample_params_default();
            p.repeat_penalty = 2.0f; p.repeat_last_n = 64;
            ot_rng rng = ot_rng_seed(1);
            CHECK(osample(logits, 6, &p, &rng, recent, 3) == 4,
                  "penalty demotes repeated token 2 below token 4");
        }
        /* The penalty respects the window: an empty window leaves token 2 on top. */
        {
            float logits[6]; for (int i = 0; i < 6; i++) logits[i] = base[i];
            osample_params p = osample_params_default();
            p.repeat_penalty = 2.0f;
            ot_rng rng = ot_rng_seed(1);
            CHECK(osample(logits, 6, &p, &rng, NULL, 0) == 2,
                  "penalty with empty history is a no-op");
        }
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
