/* test_eval.c — perplexity / NLL harness (ornith_eval).
 *
 * Two layers of coverage:
 *
 *   1. Deterministic math: ornith_eval_token_nll on hand-computed logits must
 *      equal -log(softmax)[next] (log-sum-exp), be >= 0, give log(n)/ppl==n on
 *      uniform logits, and be invariant to a constant logit shift.
 *
 *   2. Real session pipeline on a tiny but structurally complete qwen35 GGUF
 *      (same fixture shape as test_session.c): ornith_eval_ids must produce a
 *      finite mean NLL with perplexity >= 1, and a sequence built FROM the
 *      model's own predictions (each next token = the model's argmax) must score
 *      a strictly lower perplexity than a fixed "random" sequence — i.e. the
 *      metric actually rewards sequences the model predicts well.
 */
#define _POSIX_C_SOURCE 200112L
#include "ornith_eval.h"
#include "ornith_rforward.h"
#include "ornith_gguf_write.h"
#include "ornith_gguf.h"
#include "ornith_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

/* ---- 1. deterministic NLL math ----------------------------------------- */

static void test_math(void) {
    printf("== nll math ==\n");

    /* logits {1,2,3}: max=3, sum exp(.-3)=e^-2+e^-1+1=1.5032147,
     * logsumexp=3+log(1.5032147)=3.407606. nll[i]=logsumexp-logit[i]. */
    float L[3] = {1.0f, 2.0f, 3.0f};
    double lse = 3.0 + log(exp(-2.0) + exp(-1.0) + 1.0);
    double n2 = ornith_eval_token_nll(L, 3, 2);
    double n0 = ornith_eval_token_nll(L, 3, 0);
    char msg[160];
    snprintf(msg, sizeof msg, "nll(next=2)=%.6f == lse-3=%.6f", n2, lse - 3.0);
    CHECK(fabs(n2 - (lse - 3.0)) < 1e-9, msg);
    snprintf(msg, sizeof msg, "nll(next=0)=%.6f == lse-1=%.6f", n0, lse - 1.0);
    CHECK(fabs(n0 - (lse - 1.0)) < 1e-9, msg);
    CHECK(n2 >= 0.0 && n0 >= 0.0, "nll always >= 0");
    CHECK(n2 < n0, "lower nll for the higher-logit token");

    /* uniform logits over V: every token has p=1/V, so nll=log(V), ppl=V. */
    enum { U = 8 };
    float Lu[U]; for (int i = 0; i < U; i++) Lu[i] = 0.5f;  /* any constant */
    double nu = ornith_eval_token_nll(Lu, U, 3);
    snprintf(msg, sizeof msg, "uniform nll=%.6f == log(%d)=%.6f", nu, U, log((double)U));
    CHECK(fabs(nu - log((double)U)) < 1e-9, msg);
    CHECK(fabs(exp(nu) - (double)U) < 1e-6, "uniform perplexity == vocab size");

    /* constant-shift invariance (softmax is shift-invariant). */
    float Ls[3] = {1001.0f, 1002.0f, 1003.0f};
    double ns = ornith_eval_token_nll(Ls, 3, 2);
    snprintf(msg, sizeof msg, "shift-invariant: %.6f == %.6f", ns, n2);
    CHECK(fabs(ns - n2) < 1e-6, msg);

    /* out-of-range next -> NaN (defensive). */
    CHECK(isnan(ornith_eval_token_nll(L, 3, 5)), "out-of-range next -> NaN");
}

/* ---- tiny GGUF fixture (mirrors tests/test_session.c) ------------------ */

enum { H = 32, V = 16, NL = 4, HD = 8, NH = 2, NKV = 1,
       NV = 2, DV = 8, CC = 32, CONVK = 4, FFI = 64, VD = NV * DV };

#define MAX_BUFS 512
static void *g_bufs[MAX_BUFS];
static int   g_nbuf = 0;
static unsigned g_seed = 1;

static float *mkbuf(int n, float lo, float hi) {
    float *b = malloc((size_t)n * sizeof(float));
    ot_rng r = ot_rng_seed(0x9E37 * (g_seed++) + 7);
    ot_rng_fill(&r, b, n, lo, hi);
    g_bufs[g_nbuf++] = b;
    return b;
}
static void add2d(ogguf_writer *w, const char *name, int in, int out,
                  float lo, float hi) {
    uint64_t dims[2] = { (uint64_t)in, (uint64_t)out };
    float *b = mkbuf(in * out, lo, hi);
    ogguf_w_tensor(w, name, 2, dims, OGGML_F32, b,
                   (uint64_t)in * out * sizeof(float));
}
static void add1d(ogguf_writer *w, const char *name, int n, float lo, float hi) {
    uint64_t dims[1] = { (uint64_t)n };
    float *b = mkbuf(n, lo, hi);
    ogguf_w_tensor(w, name, 1, dims, OGGML_F32, b, (uint64_t)n * sizeof(float));
}
static void addconv(ogguf_writer *w, const char *name) {
    uint64_t dims[2] = { (uint64_t)CONVK, (uint64_t)CC };
    float *b = mkbuf(CONVK * CC, -0.2f, 0.2f);
    ogguf_w_tensor(w, name, 2, dims, OGGML_F32, b,
                   (uint64_t)CONVK * CC * sizeof(float));
}

static const char *write_fixture(void) {
    static const char *path = "build/_eval_fixture.gguf";
    ogguf_writer *w = ogguf_writer_new();
    ogguf_w_string(w, "general.architecture", "qwen35");

    char nm[64];
    char tokstrs[V][4]; const char *tokptr[V];
    for (int i = 0; i < V; i++) { tokstrs[i][0] = (char)('a' + i); tokstrs[i][1] = 0;
                                  tokptr[i] = tokstrs[i]; }
    ogguf_w_arr_str(w, "tokenizer.ggml.tokens", tokptr, V);
    ogguf_w_u32(w, "tokenizer.ggml.eos_token_id", V - 1);

    add2d(w, "token_embd.weight", H, V, -0.25f, 0.25f);
    add1d(w, "output_norm.weight", H, 0.9f, 1.1f);
    add2d(w, "output.weight", H, V, -0.25f, 0.25f);

    for (int L = 0; L < NL; L++) {
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", L);
        add1d(w, nm, H, 0.9f, 1.1f);
        snprintf(nm, sizeof nm, "blk.%d.post_attention_norm.weight", L);
        add1d(w, nm, H, 0.9f, 1.1f);

        if (L == NL - 1) {
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", L);
            add2d(w, nm, H, 2 * NH * HD, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", L);
            add2d(w, nm, H, NKV * HD, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", L);
            add2d(w, nm, H, NKV * HD, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", L);
            add1d(w, nm, HD, 0.9f, 1.1f);
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", L);
            add1d(w, nm, HD, 0.9f, 1.1f);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", L);
            add2d(w, nm, NH * HD, H, -0.25f, 0.25f);
        } else {
            snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", L);
            add2d(w, nm, H, CC, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.ssm_conv1d.weight", L);
            addconv(w, nm);
            snprintf(nm, sizeof nm, "blk.%d.ssm_a", L);
            add1d(w, nm, NV, -0.8f, -0.2f);
            snprintf(nm, sizeof nm, "blk.%d.ssm_dt.bias", L);
            add1d(w, nm, NV, -0.05f, 0.05f);
            snprintf(nm, sizeof nm, "blk.%d.ssm_norm.weight", L);
            add1d(w, nm, DV, 0.9f, 1.1f);
            snprintf(nm, sizeof nm, "blk.%d.ssm_alpha.weight", L);
            add2d(w, nm, H, NV, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.ssm_beta.weight", L);
            add2d(w, nm, H, NV, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", L);
            add2d(w, nm, H, VD, -0.25f, 0.25f);
            snprintf(nm, sizeof nm, "blk.%d.ssm_out.weight", L);
            add2d(w, nm, VD, H, -0.25f, 0.25f);
        }
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", L);
        add2d(w, nm, H, FFI, -0.25f, 0.25f);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", L);
        add2d(w, nm, H, FFI, -0.25f, 0.25f);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", L);
        add2d(w, nm, FFI, H, -0.25f, 0.25f);
    }

    ornith_status s = ogguf_writer_write(w, path);
    ogguf_writer_free(w);
    for (int i = 0; i < g_nbuf; i++) free(g_bufs[i]);
    g_nbuf = 0;
    if (s != ORNITH_OK) { printf("  FAIL: write fixture (%s)\n", ornith_last_error());
                          exit(2); }
    return path;
}

/* Build a greedy sequence: start with `start`, then each next token is the
 * model's argmax given everything so far. The model predicts these tokens by
 * construction, so the sequence should score a low NLL. */
static void greedy_seq(rmodel *m, int32_t start, int32_t *out, int n) {
    int Vv = rmodel_arch(m)->vocab_size;
    rsession *s = rmodel_session_new(m, n + 2);
    out[0] = start;
    rmodel_session_eval(s, &out[0], 1);
    for (int i = 1; i < n; i++) {
        const float *lg = rmodel_session_logits(s);
        int b = 0; for (int k = 1; k < Vv; k++) if (lg[k] > lg[b]) b = k;
        out[i] = (int32_t)b;
        rmodel_session_eval(s, &out[i], 1);
    }
    rmodel_session_free(s);
}

/* ---- 2. real session pipeline ------------------------------------------ */

static void test_pipeline(rmodel *m) {
    printf("== eval pipeline (synthetic model) ==\n");
    enum { SEQ = 12 };

    /* a fixed arbitrary ("random") sequence */
    int32_t rnd[SEQ] = {3, 7, 1, 12, 5, 2, 9, 4, 11, 6, 8, 0};
    double mean_r = 0, ppl_r = 0; long cnt_r = 0;
    ornith_status s = ornith_eval_ids(m, rnd, SEQ, &mean_r, &ppl_r, &cnt_r);
    CHECK(s == ORNITH_OK, "eval_ids on random sequence ok");
    CHECK(cnt_r == SEQ - 1, "scored count == n-1");
    char msg[160];
    snprintf(msg, sizeof msg, "random ppl finite (mean=%.4f ppl=%.4f)", mean_r, ppl_r);
    CHECK(isfinite(mean_r) && isfinite(ppl_r), msg);
    CHECK(ppl_r >= 1.0, "perplexity >= 1");
    CHECK(mean_r >= 0.0, "mean NLL >= 0");

    /* a sequence the model itself predicts (greedy) should score lower ppl */
    int32_t grd[SEQ];
    greedy_seq(m, rnd[0], grd, SEQ);
    double mean_g = 0, ppl_g = 0; long cnt_g = 0;
    s = ornith_eval_ids(m, grd, SEQ, &mean_g, &ppl_g, &cnt_g);
    CHECK(s == ORNITH_OK, "eval_ids on greedy sequence ok");
    CHECK(isfinite(ppl_g) && ppl_g >= 1.0, "greedy ppl finite and >= 1");
    snprintf(msg, sizeof msg,
             "greedy ppl %.4f < random ppl %.4f (metric rewards predictable seq)",
             ppl_g, ppl_r);
    CHECK(ppl_g < ppl_r, msg);

    /* too-short corpus rejected */
    int32_t one[1] = {5};
    CHECK(ornith_eval_ids(m, one, 1, NULL, NULL, NULL) != ORNITH_OK,
          "single-token sequence rejected");
}

int main(void) {
    test_math();

    const char *path = write_fixture();
    rmodel *m = NULL;
    ornith_status s = rmodel_load(path, &m);
    if (s != ORNITH_OK) {
        printf("  FAIL: rmodel_load fixture: %s\n", ornith_last_error());
        return 1;
    }
    printf("loaded fixture: hidden %d, %d layers, vocab %d\n",
           rmodel_arch(m)->hidden_size, rmodel_arch(m)->num_layers,
           rmodel_arch(m)->vocab_size);

    test_pipeline(m);
    rmodel_free(m);

    printf("\n%s (%d failure%s)\n",
           failures ? "EVAL TESTS FAILED" : "ALL EVAL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
