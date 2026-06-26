/* test_quant.c — M1 coverage: the GGUF writer, the quant block codecs, and the
 * `quantize` transcode path. All synthetic; no weights, no network.
 *
 *   1. writer round-trip: write metadata + several tensors with ogguf_writer,
 *      read back with the existing ogguf_open reader, assert everything matches
 *      (names, dims, types, offsets, arch/name, quant histogram).
 *   2. codecs: f16/bf16 conversion and Q8_0/Q4_0 quantize/dequantize round-trip
 *      with bounded error on synthetic data; assert ggml-exact block sizes.
 *   3. policy + quantize: build a full-precision GGUF, run ornith_quantize_file,
 *      reopen and assert each tensor took the POLICY.md type (or honest F16
 *      fallback for not-yet-implemented k-/i-quants).
 */
#include "ornith_gguf.h"
#include "ornith_gguf_write.h"
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

/* find a tensor by name in an opened file */
static const ogguf_tensor *find_tensor(const ogguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

int main(void) {
    printf("== writer round-trip ==\n");
    {
        /* three tensors of different types/shapes + metadata */
        float embd[64], expert[96];
        for (int i = 0; i < 64; i++) embd[i]   = (float)(i - 32) * 0.03125f;
        for (int i = 0; i < 96; i++) expert[i] = sinf((float)i * 0.1f);

        uint8_t embd_q[1024], expert_q[1024], gate_f32[256];
        memcpy(gate_f32, embd, sizeof(embd)); /* reuse as raw F32 payload */

        size_t embd_bytes   = oq_row_bytes(OGGML_F16, 64);
        size_t expert_bytes = oq_row_bytes(OGGML_Q8_0, 96);
        oq_quantize(OGGML_F16,  embd,   embd_q,   64);
        oq_quantize(OGGML_Q8_0, expert, expert_q, 96);

        ogguf_writer *w = ogguf_writer_new();
        CHECK(w != NULL, "writer alloc");
        CHECK(ogguf_w_string(w, "general.architecture", "qwen3_5_moe") == ORNITH_OK,
              "add arch");
        CHECK(ogguf_w_string(w, "general.name", "Ornith-1.0-35B") == ORNITH_OK,
              "add name");
        CHECK(ogguf_w_u32(w, "qwen3_5_moe.block_count", 40) == ORNITH_OK, "add u32");
        CHECK(ogguf_w_f32(w, "qwen3_5_moe.rope.freq_base", 1e7f) == ORNITH_OK,
              "add f32");
        const char *toks[] = { "<bos>", "<eos>", "hello" };
        CHECK(ogguf_w_arr_str(w, "tokenizer.test", toks, 3) == ORNITH_OK,
              "add str array");

        uint64_t d_embd[2]   = { 32, 2 };        /* 64 elems   */
        uint64_t d_expert[2] = { 32, 3 };        /* 96 elems   */
        uint64_t d_gate[1]   = { 64 };           /* 64 F32     */
        CHECK(ogguf_w_tensor(w, "token_embd.weight", 2, d_embd, OGGML_F16,
                             embd_q, embd_bytes) == ORNITH_OK, "add F16 tensor");
        CHECK(ogguf_w_tensor(w, "blk.0.ffn_gate_exps.weight", 2, d_expert,
                             OGGML_Q8_0, expert_q, expert_bytes) == ORNITH_OK,
              "add Q8_0 tensor");
        CHECK(ogguf_w_tensor(w, "blk.0.attn_norm.weight", 1, d_gate, OGGML_F32,
                             gate_f32, sizeof(embd)) == ORNITH_OK, "add F32 tensor");

        const char *path = "build/_w_roundtrip.gguf";
        CHECK(ogguf_writer_write(w, path) == ORNITH_OK, "write file");
        ogguf_writer_free(w);

        ogguf_file g;
        CHECK(ogguf_open(path, &g) == ORNITH_OK, "reader opens our file");
        CHECK(g.version == 3, "version 3");
        CHECK(g.n_tensors == 3, "3 tensors");
        CHECK(g.n_kv == 5, "5 kv");
        CHECK(strcmp(g.model_arch, "qwen3_5_moe") == 0, "arch round-trips");
        CHECK(strcmp(g.model_name, "Ornith-1.0-35B") == 0, "name round-trips");
        CHECK(g.n_expert_tensors == 1, "1 expert tensor counted");

        const ogguf_tensor *te = find_tensor(&g, "token_embd.weight");
        CHECK(te && te->type == OGGML_F16, "F16 type round-trips");
        CHECK(te && te->n_dims == 2 && te->dims[0] == 32 && te->dims[1] == 2,
              "F16 dims round-trip");
        CHECK(te && te->offset == 0, "first tensor offset 0");

        const ogguf_tensor *ex = find_tensor(&g, "blk.0.ffn_gate_exps.weight");
        CHECK(ex && ex->type == OGGML_Q8_0, "Q8_0 type round-trips");
        CHECK(ex && ex->n_elements == 96, "Q8_0 n_elements");
        /* offset of 2nd tensor = align_up(embd_bytes, 32) = align_up(128,32)=128 */
        CHECK(ex && ex->offset == 128, "second tensor offset aligned");

        CHECK(g.type_elem_count[OGGML_F16]  == 64, "histogram F16 elems");
        CHECK(g.type_elem_count[OGGML_Q8_0] == 96, "histogram Q8_0 elems");
        CHECK(g.type_elem_count[OGGML_F32]  == 64, "histogram F32 elems");
        ogguf_close(&g);
    }

    printf("== codecs ==\n");
    {
        /* block geometry must match ggml exactly */
        CHECK(oq_block_bytes(OGGML_Q8_0) == 34, "Q8_0 block = 34 bytes");
        CHECK(oq_block_bytes(OGGML_Q4_0) == 18, "Q4_0 block = 18 bytes");
        CHECK(oq_row_bytes(OGGML_Q8_0, 64) == 68, "Q8_0 row bytes");
        CHECK(oq_row_bytes(OGGML_Q8_0, 7) == 0, "Q8_0 rejects non-block size");

        /* f16: exact representable values round-trip exactly */
        CHECK(oq_f16_to_f32(oq_f32_to_f16(1.0f)) == 1.0f, "f16 1.0 exact");
        CHECK(oq_f16_to_f32(oq_f32_to_f16(-0.5f)) == -0.5f, "f16 -0.5 exact");
        CHECK(oq_f16_to_f32(oq_f32_to_f16(0.0f)) == 0.0f, "f16 0.0 exact");
        /* bf16 keeps the top 8 mantissa bits */
        CHECK(oq_bf16_to_f32(oq_f32_to_bf16(2.0f)) == 2.0f, "bf16 2.0 exact");

        /* f16 bounded relative error over a range */
        double max_rel = 0.0;
        for (int i = 1; i <= 1000; i++) {
            float x = (float)i * 0.137f;
            float y = oq_f16_to_f32(oq_f32_to_f16(x));
            double rel = fabs((double)y - x) / x;
            if (rel > max_rel) max_rel = rel;
        }
        CHECK(max_rel < 1e-3, "f16 relative error < 1e-3");

        /* Q8_0 round-trip: bounded by quant step (amax/127) */
        float src[64], deq[64];
        float amax = 0.0f;
        for (int i = 0; i < 64; i++) { src[i] = sinf((float)i * 0.3f) * 5.0f;
                                       if (fabsf(src[i]) > amax) amax = fabsf(src[i]); }
        uint8_t q8[68];
        CHECK(oq_quantize(OGGML_Q8_0, src, q8, 64) == ORNITH_OK, "Q8_0 quantize");
        CHECK(oq_dequantize(OGGML_Q8_0, q8, deq, 64) == ORNITH_OK, "Q8_0 dequant");
        double q8_maxerr = 0.0;
        for (int i = 0; i < 64; i++) {
            double e = fabs((double)deq[i] - src[i]);
            if (e > q8_maxerr) q8_maxerr = e;
        }
        CHECK(q8_maxerr <= amax / 127.0 + 1e-4, "Q8_0 error within one step");

        /* Q4_0 round-trip: coarser, bound by step amax/8 */
        float deq4[64];
        uint8_t q4[36];
        CHECK(oq_quantize(OGGML_Q4_0, src, q4, 64) == ORNITH_OK, "Q4_0 quantize");
        CHECK(oq_dequantize(OGGML_Q4_0, q4, deq4, 64) == ORNITH_OK, "Q4_0 dequant");
        double q4_maxerr = 0.0;
        for (int i = 0; i < 64; i++) {
            double e = fabs((double)deq4[i] - src[i]);
            if (e > q4_maxerr) q4_maxerr = e;
        }
        CHECK(q4_maxerr <= amax / 8.0 + 1e-3, "Q4_0 error within one step");

        CHECK(oq_quantize(OGGML_IQ3_S, src, q8, 64) == ORNITH_ERR_UNSUPPORTED,
              "unimplemented type reported");
        CHECK(!oq_is_implemented(OGGML_IQ3_S), "IQ3_S not implemented");
        CHECK(oq_is_implemented(OGGML_Q8_0), "Q8_0 implemented");

        /* k-quants: block geometry must match ggml exactly */
        CHECK(oq_block_bytes(OGGML_Q2_K) == 84,  "Q2_K block = 84 bytes");
        CHECK(oq_block_bytes(OGGML_Q4_K) == 144, "Q4_K block = 144 bytes");
        CHECK(oq_block_bytes(OGGML_Q5_K) == 176, "Q5_K block = 176 bytes");
        CHECK(oq_block_bytes(OGGML_Q6_K) == 210, "Q6_K block = 210 bytes");
        CHECK(oq_block_elems(OGGML_Q4_K) == 256, "Q4_K block = 256 elems");
        CHECK(oq_is_implemented(OGGML_Q4_K), "Q4_K encode implemented");
        CHECK(oq_is_implemented(OGGML_Q6_K), "Q6_K encode implemented");
        CHECK(oq_can_decode(OGGML_Q2_K), "Q2_K decodable");
        CHECK(oq_can_decode(OGGML_Q5_K), "Q5_K decodable");
        CHECK(oq_is_implemented(OGGML_Q2_K), "Q2_K now encodable");
        CHECK(oq_is_implemented(OGGML_Q5_K), "Q5_K now encodable");

        /* IQ2_XXS block geometry must match ggml exactly */
        CHECK(oq_block_bytes(OGGML_IQ2_XXS) == 66, "IQ2_XXS block = 66 bytes");
        CHECK(oq_block_elems(OGGML_IQ2_XXS) == 256, "IQ2_XXS block = 256 elems");
        CHECK(oq_row_bytes(OGGML_IQ2_XXS, 512) == 132, "IQ2_XXS row bytes");
        CHECK(oq_is_implemented(OGGML_IQ2_XXS), "IQ2_XXS encode implemented");
        CHECK(oq_can_decode(OGGML_IQ2_XXS), "IQ2_XXS decodable");

        /* Q4_K / Q6_K round-trip over a 256-element super-block. Weight-like
         * data (roughly symmetric around 0). Error bounded by the quant step. */
        float ksrc[256], kdeq[256];
        float kamax = 0.0f;
        for (int i = 0; i < 256; i++) {
            ksrc[i] = sinf((float)i * 0.11f) * 0.6f + cosf((float)i * 0.37f) * 0.2f;
            if (fabsf(ksrc[i]) > kamax) kamax = fabsf(ksrc[i]);
        }
        uint8_t kbuf[210];
        CHECK(oq_quantize(OGGML_Q6_K, ksrc, kbuf, 256) == ORNITH_OK, "Q6_K quantize");
        CHECK(oq_dequantize(OGGML_Q6_K, kbuf, kdeq, 256) == ORNITH_OK, "Q6_K dequant");
        double q6_maxerr = 0.0;
        for (int i = 0; i < 256; i++) {
            double e = fabs((double)kdeq[i] - ksrc[i]);
            if (e > q6_maxerr) q6_maxerr = e;
        }
        CHECK(q6_maxerr < kamax / 30.0, "Q6_K round-trip error small");

        CHECK(oq_quantize(OGGML_Q4_K, ksrc, kbuf, 256) == ORNITH_OK, "Q4_K quantize");
        CHECK(oq_dequantize(OGGML_Q4_K, kbuf, kdeq, 256) == ORNITH_OK, "Q4_K dequant");
        double q4k_maxerr = 0.0;
        for (int i = 0; i < 256; i++) {
            double e = fabs((double)kdeq[i] - ksrc[i]);
            if (e > q4k_maxerr) q4k_maxerr = e;
        }
        CHECK(q4k_maxerr < kamax / 8.0, "Q4_K round-trip error within a 4-bit step");

        /* Q5_K round-trip: finer than Q4_K. */
        CHECK(oq_quantize(OGGML_Q5_K, ksrc, kbuf, 256) == ORNITH_OK, "Q5_K quantize");
        CHECK(oq_dequantize(OGGML_Q5_K, kbuf, kdeq, 256) == ORNITH_OK, "Q5_K dequant");
        double q5k_maxerr = 0.0;
        for (int i = 0; i < 256; i++) {
            double e = fabs((double)kdeq[i] - ksrc[i]);
            if (e > q5k_maxerr) q5k_maxerr = e;
        }
        CHECK(q5k_maxerr < kamax / 16.0, "Q5_K round-trip error within a 5-bit step");

        /* Q2_K round-trip: coarse (2-bit + per-16 affine). Bound loosely. */
        CHECK(oq_quantize(OGGML_Q2_K, ksrc, kbuf, 256) == ORNITH_OK, "Q2_K quantize");
        CHECK(oq_dequantize(OGGML_Q2_K, kbuf, kdeq, 256) == ORNITH_OK, "Q2_K dequant");
        double q2k_maxerr = 0.0, q2k_sse = 0.0, ksse = 0.0;
        for (int i = 0; i < 256; i++) {
            double e = fabs((double)kdeq[i] - ksrc[i]);
            if (e > q2k_maxerr) q2k_maxerr = e;
            q2k_sse += e * e; ksse += (double)ksrc[i] * ksrc[i];
        }
        CHECK(isfinite(q2k_maxerr) && q2k_maxerr < kamax / 2.0,
              "Q2_K round-trip error bounded");
        CHECK(q2k_sse < 0.15 * ksse, "Q2_K captures most signal energy");

        /* IQ2_XXS round-trip: sub-2-bit, loose tolerance; just much better than
         * dropping the tensor (relative SSE well under 1) and finite. */
        uint8_t iqbuf[66];
        float iqdeq[256];
        CHECK(oq_quantize(OGGML_IQ2_XXS, ksrc, iqbuf, 256) == ORNITH_OK,
              "IQ2_XXS quantize");
        CHECK(oq_dequantize(OGGML_IQ2_XXS, iqbuf, iqdeq, 256) == ORNITH_OK,
              "IQ2_XXS dequant");
        double iq_sse = 0.0; int iq_finite = 1;
        for (int i = 0; i < 256; i++) {
            if (!isfinite(iqdeq[i])) iq_finite = 0;
            double e = (double)iqdeq[i] - ksrc[i];
            iq_sse += e * e;
        }
        CHECK(iq_finite, "IQ2_XXS output all finite");
        CHECK(iq_sse < 0.5 * ksse, "IQ2_XXS much better than zeroing the tensor");

        /* IQ2_XXS preserves sign of large-magnitude weights (it is a signed
         * codebook). Check the few biggest entries keep their sign. */
        int sign_ok = 1;
        for (int i = 0; i < 256; i++)
            if (fabsf(ksrc[i]) > 0.5f * kamax &&
                (ksrc[i] > 0) != (iqdeq[i] >= 0)) sign_ok = 0;
        CHECK(sign_ok, "IQ2_XXS keeps sign of large weights");
    }

    printf("== policy ==\n");
    {
        CHECK(oq_policy_target("blk.3.ffn_gate_exps.weight", OGGML_F16)
              == OGGML_IQ2_XXS, "gate_exps -> IQ2_XXS");
        CHECK(oq_policy_target("blk.3.ffn_down_exps.weight", OGGML_F16)
              == OGGML_Q2_K, "down_exps -> Q2_K");
        CHECK(oq_policy_target("blk.3.ffn_gate_inp.weight", OGGML_F16)
              == OGGML_F16, "router -> F16");
        CHECK(oq_policy_target("blk.3.ffn_down_shexp.weight", OGGML_F16)
              == OGGML_Q5_K, "shexp -> Q5_K");
        CHECK(oq_policy_target("blk.3.attn_q.weight", OGGML_F16)
              == OGGML_Q6_K, "attn_q -> Q6_K");
        CHECK(oq_policy_target("blk.3.attn_norm.weight", OGGML_F16)
              == OGGML_F32, "norm -> F32 (before attn rule)");
        CHECK(oq_policy_target("token_embd.weight", OGGML_F16)
              == OGGML_Q5_K, "token_embd -> Q5_K");
        CHECK(oq_policy_target("output.weight", OGGML_F16)
              == OGGML_Q6_K, "lm_head -> Q6_K");
        CHECK(oq_policy_target("blk.2.ssm_conv1d.weight", OGGML_F16)
              == OGGML_F32, "ssm_conv1d -> F32 (dynamics)");
        CHECK(oq_policy_target("blk.2.ssm_a", OGGML_F16)
              == OGGML_F32, "ssm_a -> F32 (dynamics)");
        CHECK(oq_policy_target("blk.2.ssm_out.weight", OGGML_F16)
              == OGGML_Q8_0, "ssm_out projection -> Q8_0");
        CHECK(oq_policy_target("blk.2.attn_qkv.weight", OGGML_F16)
              == OGGML_Q8_0, "ssm attn_qkv -> Q8_0");
        CHECK(oq_policy_target("blk.2.something_else.weight", OGGML_BF16)
              == OGGML_BF16, "unmatched -> base");
    }

    printf("== quantize transcode ==\n");
    {
        /* build a small full-precision (F32) GGUF with policy-relevant names */
        ogguf_writer *w = ogguf_writer_new();
        ogguf_w_string(w, "general.architecture", "qwen3_5_moe");
        ogguf_w_string(w, "general.name", "synthetic");

        static float gate_exps[256], conv[64], norm[32], router[128], embd[128];
        for (int i = 0; i < 256; i++) gate_exps[i] = sinf((float)i * 0.05f);
        for (int i = 0; i < 64;  i++) conv[i]   = cosf((float)i * 0.2f);
        for (int i = 0; i < 32;  i++) norm[i]   = 1.0f + (float)i * 0.01f;
        for (int i = 0; i < 128; i++) router[i] = (float)(i % 7) * 0.1f;
        for (int i = 0; i < 128; i++) embd[i]   = (float)(i - 64) * 0.02f;

        uint64_t d256[1] = {256}, d64[1] = {64}, d32[1] = {32}, d128[1] = {128};
        ogguf_w_tensor(w, "blk.0.ffn_gate_exps.weight", 1, d256, OGGML_F32,
                       gate_exps, sizeof(gate_exps));
        ogguf_w_tensor(w, "blk.0.ssm_out.weight", 1, d64, OGGML_F32,
                       conv, sizeof(conv));
        ogguf_w_tensor(w, "blk.0.attn_norm.weight", 1, d32, OGGML_F32,
                       norm, sizeof(norm));
        ogguf_w_tensor(w, "blk.0.ffn_gate_inp.weight", 1, d128, OGGML_F32,
                       router, sizeof(router));
        ogguf_w_tensor(w, "token_embd.weight", 1, d128, OGGML_F32,
                       embd, sizeof(embd));
        const char *fp = "build/_full.gguf";
        CHECK(ogguf_writer_write(w, fp) == ORNITH_OK, "write full-precision gguf");
        ogguf_writer_free(w);

        const char *qp = "build/_quant.gguf";
        CHECK(ornith_quantize_file(fp, qp, OGGML_F16, NULL) == ORNITH_OK,
              "quantize_file succeeds");

        ogguf_file g;
        CHECK(ogguf_open(qp, &g) == ORNITH_OK, "reopen quantized");
        CHECK(g.n_tensors == 5, "all tensors preserved");
        CHECK(strcmp(g.model_arch, "qwen3_5_moe") == 0, "metadata passed through");

        const ogguf_tensor *t;
        t = find_tensor(&g, "blk.0.ssm_out.weight");
        CHECK(t && t->type == OGGML_Q8_0, "ssm_out quantized to Q8_0");
        t = find_tensor(&g, "blk.0.attn_norm.weight");
        CHECK(t && t->type == OGGML_F32, "norm kept F32");
        t = find_tensor(&g, "blk.0.ffn_gate_inp.weight");
        CHECK(t && t->type == OGGML_F16, "router kept F16");
        t = find_tensor(&g, "blk.0.ffn_gate_exps.weight");
        CHECK(t && t->type == OGGML_IQ2_XXS,
              "IQ2_XXS policy now encodes (256 elems, block-aligned)");
        t = find_tensor(&g, "token_embd.weight");
        CHECK(t && t->type == OGGML_F16,
              "Q5_K policy falls back to F16 (128 elems not 256-aligned)");
        ogguf_close(&g);

        /* and the F16-fallback data is still numerically sane */
        ogguf_loaded l;
        CHECK(ogguf_load(qp, &l) == ORNITH_OK, "load quantized for value check");
        for (uint64_t i = 0; i < l.n_tensors; i++) {
            if (strcmp(l.tensors[i].name, "blk.0.ssm_out.weight")) continue;
            float deq[64];
            oq_dequantize(l.tensors[i].type, l.tensors[i].data, deq, 64);
            double maxerr = 0.0;
            for (int j = 0; j < 64; j++) {
                double e = fabs((double)deq[j] - conv[j]);
                if (e > maxerr) maxerr = e;
            }
            CHECK(maxerr < 0.02, "Q8_0 ssm_out values survive transcode");
        }
        ogguf_loaded_free(&l);
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "TESTS FAILED" : "ALL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
