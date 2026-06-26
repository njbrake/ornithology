/* ornith_bench.c — see ornith_bench.h.
 *
 * Everything runs over the public rforward session API so the harness exercises
 * exactly the prefill/decode paths real generation uses. Prefill throughput is
 * a single rmodel_session_eval of the whole synthetic prompt; decode throughput
 * is N single-token evals, each fed the previous step's argmax, so it is a clean
 * per-token decode cost (independent of EOS / sampling). Peak RSS is read from
 * getrusage after all work, capturing the loaded model + caches.
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, getrusage */
#include "ornith_bench.h"
#include "ornith_rforward.h"
#include "ornith_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Peak resident set in kilobytes. ru_maxrss is KB on Linux, bytes on macOS. */
static long peak_rss_kb(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}

static int argmax_f(const float *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

int ornith_bench_main(int argc, char **argv) {
    const char *path = NULL;
    int P = 64, N = 32, R = 3;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt-len") && i + 1 < argc) P = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc)   N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)  R = atoi(argv[++i]);
        else path = argv[i];
    }
    if (!path) {
        fprintf(stderr,
            "usage: ornith bench <model.gguf> [--prompt-len P] [--gen N] [--reps R]\n");
        return 1;
    }
    if (P < 1) P = 1;
    if (N < 1) N = 1;
    if (R < 1) R = 1;

    fprintf(stderr, "bench: loading %s ...\n", path);
    double t0 = now_sec();
    rmodel *m = NULL;
    ornith_status s = rmodel_load(path, &m);
    double load_s = now_sec() - t0;
    if (s != ORNITH_OK) {
        fprintf(stderr, "bench: load failed: %s\n", ornith_last_error());
        return 1;
    }

    const ornith_arch *a = rmodel_arch(m);
    int V = a->vocab_size;
    if (V < 2) { fprintf(stderr, "bench: degenerate vocab\n"); rmodel_free(m); return 1; }

    /* Deterministic, in-vocab synthetic prompt (avoids depending on any corpus
     * and avoids id 0 / the eos id at the very end). */
    int32_t *prompt = malloc((size_t)P * sizeof(int32_t));
    if (!prompt) { fprintf(stderr, "bench: oom\n"); rmodel_free(m); return 1; }
    for (int i = 0; i < P; i++) prompt[i] = (int32_t)(((long)i * 7 + 1) % (V - 1));

    fprintf(stderr,
            "bench: hidden %d, %d layers (%d full-attn), vocab %d | "
            "prompt-len %d, gen %d, reps %d\n",
            a->hidden_size, a->num_layers, ornith_count_full_attn_layers(a),
            V, P, N, R);

    double best_pre = 0, sum_pre = 0, best_dec = 0, sum_dec = 0;
    int ok_reps = 0;
    for (int r = 0; r < R; r++) {
        rsession *sess = rmodel_session_new(m, P + N + 8);
        if (!sess) { fprintf(stderr, "bench: session alloc failed\n"); break; }

        /* prefill: one eval of the whole prompt */
        double tp = now_sec();
        s = rmodel_session_eval(sess, prompt, P);
        double pre_s = now_sec() - tp;
        if (s != ORNITH_OK) {
            fprintf(stderr, "bench: prefill failed: %s\n", ornith_last_error());
            rmodel_session_free(sess);
            break;
        }
        double pre_tps = pre_s > 0 ? (double)P / pre_s : 0.0;

        /* decode: N single-token evals, each fed the previous step's argmax */
        double td = now_sec();
        int produced = 0;
        for (int k = 0; k < N; k++) {
            const float *lg = rmodel_session_logits(sess);
            if (!lg) break;
            int32_t nx = (int32_t)argmax_f(lg, V);
            if (rmodel_session_eval(sess, &nx, 1) != ORNITH_OK) break;
            produced++;
        }
        double dec_s = now_sec() - td;
        double dec_tps = dec_s > 0 ? (double)produced / dec_s : 0.0;
        rmodel_session_free(sess);

        sum_pre += pre_tps; if (pre_tps > best_pre) best_pre = pre_tps;
        sum_dec += dec_tps; if (dec_tps > best_dec) best_dec = dec_tps;
        ok_reps++;
        fprintf(stderr,
                "  rep %d/%d: prefill %d tok in %.3fs (%.1f tok/s) | "
                "decode %d tok in %.3fs (%.1f tok/s)\n",
                r + 1, R, P, pre_s, pre_tps, produced, dec_s, dec_tps);
    }

    long rss_kb = peak_rss_kb();
    free(prompt);

    if (ok_reps == 0) { rmodel_free(m); return 1; }

    double mean_pre = sum_pre / ok_reps;
    double mean_dec = sum_dec / ok_reps;
    const char *name = rmodel_name(m);

    printf("\n");
    printf("ornithology bench — %s\n", name ? name : "(unnamed model)");
    printf("  model : %s\n", path);
    printf("  arch  : hidden %d, %d layers (%d full-attn), vocab %d\n",
           a->hidden_size, a->num_layers, ornith_count_full_attn_layers(a), V);
    printf("  config: prompt-len %d, gen %d, reps %d\n\n", P, N, R);

    printf("  %-22s %14s %14s\n", "metric", "mean", "best");
    printf("  %-22s %14s %14s\n", "----------------------",
           "--------------", "--------------");
    printf("  %-22s %14.3f %14s\n", "model load (s)", load_s, "-");
    printf("  %-22s %14.1f %14.1f\n", "prefill (tok/s)", mean_pre, best_pre);
    printf("  %-22s %14.1f %14.1f\n", "decode (tok/s)", mean_dec, best_dec);
    if (mean_dec > 0)
        printf("  %-22s %14.2f %14.2f\n", "decode latency (ms/tok)",
               1000.0 / mean_dec, 1000.0 / best_dec);
    printf("  %-22s %14.1f %14s\n", "peak RSS (MB)",
           (double)rss_kb / 1024.0, "-");
    printf("\n");

    rmodel_free(m);
    return 0;
}
