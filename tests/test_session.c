/* test_session.c — persistent / resumable KV-cache sessions.
 *
 * Builds a tiny but structurally complete qwen35 GGUF (F32 weights, real
 * hybrid layout: 3 linear + 1 full-attention layer, dense FFN) on disk, loads
 * it through the real rmodel forward, and exercises the session save/restore
 * round-trip:
 *
 *   1. feed a prompt to session A, snapshot it, restore into a fresh session B,
 *      and assert B reproduces A's next-token logits/argmax BIT-FOR-BIT;
 *   2. continue both A and B with the SAME tokens and assert the resulting
 *      logits are still bit-identical — i.e. the restored recurrent/KV state is
 *      exactly the original, so a chat resumes without re-prefilling;
 *   3. the same fidelity holds with the int8 (Q8) KV cache (ORNITH_KV_Q8);
 *   4. negative cases: missing file, bad magic, and an arch-fingerprint
 *      mismatch are all rejected.
 */
#define _POSIX_C_SOURCE 200112L   /* setenv */
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

static float maxdiff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; }
    return m;
}
static int argmax(const float *v, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b;
}

/* ---- tiny GGUF fixture ------------------------------------------------- */
/* Geometry: hidden 32, vocab 16, 4 layers (interval 4 -> L0..L2 linear, L3
 * full attn), head_dim 8, 2 attn heads, 1 kv head, 2 value heads, dv=dk=8,
 * conv kernel 4, dense FFN inter 64. */
enum { H = 32, V = 16, NL = 4, HD = 8, NH = 2, NKV = 1,
       NV = 2, DV = 8, DK = 8, CC = 32, CONVK = 4, FFI = 64, VD = NV * DV };

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
/* 2D weight [in, out] (GGUF dims {in, out}) — matches rforward's W[out,in]. */
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
    uint64_t dims[2] = { (uint64_t)CONVK, (uint64_t)CC };  /* dims[0]=kernel */
    float *b = mkbuf(CONVK * CC, -0.2f, 0.2f);
    ogguf_w_tensor(w, name, 2, dims, OGGML_F32, b,
                   (uint64_t)CONVK * CC * sizeof(float));
}

static const char *write_fixture(void) {
    static const char *path = "build/_session_fixture.gguf";
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

        if (L == NL - 1) {            /* full attention */
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
        } else {                      /* linear (gated delta-net) */
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

/* ---- round-trip fidelity ----------------------------------------------- */

static void roundtrip(rmodel *m, const char *label) {
    printf("== session round-trip (%s) ==\n", label);
    int Vv = rmodel_arch(m)->vocab_size;
    const char *sf = "build/_session.bin";

    int32_t P[] = { 1, 2, 3, 0, 5, 4 };  int np = 6;   /* the "original prompt" */
    int32_t C[] = { 7, 2, 9 };           int nc = 3;   /* the continuation     */

    /* session A: feed the prompt, snapshot, then continue with C */
    rsession *A = rmodel_session_new(m, 64);
    CHECK(A != NULL, "session_new");
    CHECK(rmodel_session_eval(A, P, np) == ORNITH_OK, "eval prompt");
    int posP = rmodel_session_pos(A);
    CHECK(posP == np, "pos == prompt length after eval");

    float *logP = malloc((size_t)Vv * sizeof(float));
    memcpy(logP, rmodel_session_logits(A), (size_t)Vv * sizeof(float));
    int amP = argmax(logP, Vv);

    CHECK(rmodel_session_save(A, sf) == ORNITH_OK, "session_save");

    CHECK(rmodel_session_eval(A, C, nc) == ORNITH_OK, "continue A with C");
    float *logAC = malloc((size_t)Vv * sizeof(float));
    memcpy(logAC, rmodel_session_logits(A), (size_t)Vv * sizeof(float));

    /* session B: restore from disk — must match A at the snapshot point */
    rsession *B = NULL;
    CHECK(rmodel_session_load(m, sf, 64, &B) == ORNITH_OK, "session_load");
    CHECK(rmodel_session_pos(B) == posP, "restored pos matches");

    float dPB = maxdiff(logP, rmodel_session_logits(B), Vv);
    char msg[128];
    snprintf(msg, sizeof msg, "restored next-token logits == original (maxdiff=%.2e)", dPB);
    CHECK(dPB == 0.0f, msg);
    CHECK(argmax(rmodel_session_logits(B), Vv) == amP, "restored argmax == original");

    /* continue B with the SAME C — restored state is exact, so logits match */
    CHECK(rmodel_session_eval(B, C, nc) == ORNITH_OK, "continue B with C");
    float dBC = maxdiff(logAC, rmodel_session_logits(B), Vv);
    snprintf(msg, sizeof msg, "resumed continuation == original (maxdiff=%.2e)", dBC);
    CHECK(dBC == 0.0f, msg);

    rmodel_session_free(A);
    rmodel_session_free(B);
    free(logP); free(logAC);
}

/* ---- negative cases ---------------------------------------------------- */

static void negatives(rmodel *m) {
    printf("== session load rejections ==\n");
    rsession *s = NULL;

    CHECK(rmodel_session_load(m, "build/_does_not_exist.bin", 8, &s) != ORNITH_OK,
          "missing file rejected");

    FILE *f = fopen("build/_bad_magic.bin", "wb");
    fwrite("NOTASESS", 1, 8, f); fwrite("garbagegarbage", 1, 14, f); fclose(f);
    CHECK(rmodel_session_load(m, "build/_bad_magic.bin", 8, &s) != ORNITH_OK,
          "bad magic rejected");

    /* corrupt the arch fingerprint of a valid snapshot: flip the first
     * fingerprint int (hidden_size) at offset 8 (magic) + 4 (version). */
    rsession *A = rmodel_session_new(m, 16);
    int32_t P[] = { 1, 2, 3 };
    rmodel_session_eval(A, P, 3);
    rmodel_session_save(A, "build/_fp.bin");
    rmodel_session_free(A);
    f = fopen("build/_fp.bin", "r+b");
    int32_t hidden = 0;
    fseek(f, 12, SEEK_SET);
    if (fread(&hidden, 4, 1, f) != 1) hidden = 0;
    hidden += 999;
    fseek(f, 12, SEEK_SET); fwrite(&hidden, 4, 1, f);
    fclose(f);
    CHECK(rmodel_session_load(m, "build/_fp.bin", 8, &s) != ORNITH_OK,
          "arch fingerprint mismatch rejected");
}

int main(void) {
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

    roundtrip(m, "fp32 KV");

    /* same fidelity with the int8 KV cache */
    setenv("ORNITH_KV_Q8", "1", 1);
    roundtrip(m, "Q8 KV");
    setenv("ORNITH_KV_Q8", "0", 1);

    negatives(m);

    rmodel_free(m);
    printf("\n%s (%d failure%s)\n",
           failures ? "SESSION TESTS FAILED" : "ALL SESSION TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
