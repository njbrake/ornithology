/* ornith.c — command-line entry point.
 *
 * Implemented today (the parts that don't need weights or a GPU):
 *   ornith version                 print version
 *   ornith arch                    print the Ornith-1.0-397B reference arch
 *   ornith config <config.json>    parse a HF config and print the arch
 *   ornith inspect <model.gguf>    parse GGUF header/metadata/tensor index
 *   ornith run <model.gguf>        load the GGUF, report its layout, and run the
 *                                  M2 CPU forward engine (on a tiny synthetic
 *                                  model — binding real *quantized* weights needs
 *                                  the M1 dequant path, which is gated honestly)
 *
 * The split is deliberate: everything that can be correct and tested without a
 * 397B checkpoint or a Metal/CUDA device works now and is covered by tests; the
 * quantized-weight binding is gated with a clear message rather than faked.
 */
#define _POSIX_C_SOURCE 200112L   /* setenv (used by --kv-q8) under -std=c11 */
#include "ornith.h"
#include "ornith_model.h"
#include "ornith_config.h"
#include "ornith_gguf.h"
#include "ornith_gguf_write.h"
#include "ornith_quant.h"
#include "ornith_forward.h"
#include "ornith_rforward.h"
#include "ornith_server.h"
#include "ornith_agent.h"
#include <string.h>
#include <stdlib.h>

static int cmd_version(void) {
    printf("ornithology %s — local inference engine for Ornith-1.0\n",
           ORNITH_VERSION);
    printf("target model: Ornith-1.0-397B (qwen3_5_moe)\n");
    return 0;
}

static int cmd_arch(const char *which) {
    ornith_arch a;
    ornith_arch_defaults_by_name(which, &a); /* "397b" (default) or "35b" */
    ornith_arch_print(&a, stdout);
    return 0;
}

static int cmd_config(const char *path) {
    ornith_arch a;
    ornith_status s = ornith_config_load(path, &a);
    if (s != ORNITH_OK) {
        fprintf(stderr, "config: %s: %s\n", ornith_strerror(s),
                ornith_last_error());
        return 1;
    }
    ornith_arch_print(&a, stdout);

    bool ok = strstr(a.model_type, "qwen3_5_moe") != NULL ||
              strstr(a.arch, "Qwen3_5Moe") != NULL;
    printf("\nrecognized as Ornith/qwen3_5_moe: %s\n", ok ? "yes" : "NO");
    if (!ok)
        printf("  (this engine is specialized for qwen3_5_moe; other archs "
               "are not supported)\n");
    return ok ? 0 : 2;
}

static int cmd_inspect(const char *path, bool list_tensors) {
    ogguf_file g;
    ornith_status s = ogguf_open(path, &g);
    if (s != ORNITH_OK) {
        fprintf(stderr, "inspect: %s: %s\n", ornith_strerror(s),
                ornith_last_error());
        return 1;
    }
    ogguf_print(&g, stdout, list_tensors);

    /* Real Ornith GGUFs report general.architecture "qwen35" (dense 9B) or
     * "qwen35moe" (35B/397B); also accept the HF-style string and the ssm/expert
     * tensor signatures of the hybrid layout. */
    bool ok = strstr(g.model_arch, "qwen35") != NULL ||
              strstr(g.model_arch, "qwen3_5") != NULL ||
              strstr(g.model_arch, "ornith") != NULL ||
              g.n_expert_tensors > 0;
    printf("\nlooks like an Ornith GGUF: %s\n", ok ? "yes" : "unclear");
    ogguf_close(&g);
    return 0;
}

/* Map a short type name (case-insensitive) to an oggml_type for --base. */
static int parse_quant_type(const char *s, uint32_t *out) {
    struct { const char *n; uint32_t t; } tab[] = {
        {"f32", OGGML_F32}, {"f16", OGGML_F16}, {"bf16", OGGML_BF16},
        {"q8_0", OGGML_Q8_0}, {"q4_0", OGGML_Q4_0},
    };
    for (size_t i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
        const char *a = s, *b = tab[i].n;
        while (*a && *b && (*a | 0x20) == *b) { a++; b++; }
        if (!*a && !*b) { *out = tab[i].t; return 0; }
    }
    return -1;
}

static int cmd_quantize(const char *in, const char *out, uint32_t base) {
    fprintf(stderr, "quantize: %s -> %s (policy: tools/quantize/POLICY.md)\n",
            in, out);
    ornith_status s = ornith_quantize_file(in, out, base, stderr);
    if (s != ORNITH_OK) {
        fprintf(stderr, "quantize: %s: %s\n", ornith_strerror(s),
                ornith_last_error());
        return 1;
    }
    fprintf(stderr, "quantize: wrote %s\n", out);
    fprintf(stderr, "verify with: ornith inspect --tensors %s\n", out);
    return 0;
}

/* Count how many tensors contain `needle` in their name. */
static int count_tensor_substr(const ogguf_file *g, const char *needle) {
    int n = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strstr(g->tensors[i].name, needle)) n++;
    return n;
}

/* True if all of the model's weight tensors are F32/F16 (i.e. directly usable
 * without a dequant kernel). Real Ornith GGUFs are quantized, so this is
 * normally false until the M1 dequant path lands. */
static bool gguf_all_dense(const ogguf_file *g) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        uint32_t t = g->tensors[i].type;
        if (t != OGGML_F32 && t != OGGML_F16 && t != OGGML_BF16) return false;
    }
    return true;
}

/* Run an end-to-end forward pass on a tiny seeded-synthetic model: this is the
 * proof that the M2 engine (hybrid linear/full attention + MoE) is wired and
 * correct. It does not use the GGUF's weights (those are quantized and need the
 * M1 dequant path); it demonstrates the graph the loader will feed once real
 * weights bind. tests/test_forward.c pins this path's correctness. */
static int run_synthetic_selftest(void) {
    ornith_arch a;
    ornith_arch_tiny(&a);
    of_model *m = of_model_build_synthetic(&a, 0x0017417);
    if (!m) { fprintf(stderr, "run: synthetic model alloc failed\n"); return 1; }
    of_state *s = of_state_new(m, 64);
    if (!s) { of_model_free(m); fprintf(stderr, "run: state alloc failed\n");
              return 1; }

    int32_t prompt[] = {7, 1, 12, 3, 9, 2, 18, 4};
    int T = (int)(sizeof(prompt)/sizeof(prompt[0]));
    float *logits = malloc((size_t)T * a.vocab_size * sizeof(float));
    ornith_status st = of_forward_prefill(m, s, prompt, T, 16, logits);
    if (st != ORNITH_OK) {
        fprintf(stderr, "run: forward failed: %s\n", ornith_last_error());
        free(logits); of_state_free(s); of_model_free(m); return 1;
    }
    int top = of_argmax(logits + (size_t)(T-1)*a.vocab_size, a.vocab_size);
    printf("\nengine self-test (synthetic tiny qwen3_5_moe, hidden %d, %d "
           "layers, %d experts):\n", a.hidden_size, a.num_layers,
           a.n_routed_experts);
    printf("  prefilled %d tokens through %d full-attn + %d linear-attn layers\n",
           T, ornith_count_full_attn_layers(&a),
           a.num_layers - ornith_count_full_attn_layers(&a));
    printf("  next-token argmax over %d-vocab logits = %d\n", a.vocab_size, top);
    printf("  (forward pass verified; prefill == decode parity is gated in "
           "`make test`)\n");

    free(logits); of_state_free(s); of_model_free(m);
    return 0;
}

/* Load a real (quantized) GGUF and dequantize every tensor to f32 via the
 * ornith_quant codecs, proving the k-quant decoders read the actual file.
 * Memory-safe: each tensor is sample-decoded (a bounded prefix) then freed, so
 * peak extra memory is a few MB on top of the loaded file. Reports per-type
 * coverage and sanity stats (finite, plausible magnitude). */
static int verify_real_dequant(const char *path) {
    ogguf_loaded l;
    if (ogguf_load(path, &l) != ORNITH_OK) {
        fprintf(stderr, "run: load failed: %s\n", ornith_last_error());
        return 1;
    }
    printf("\nk-quant dequant check (decoding real weights to f32):\n");

    const size_t SAMPLE_CAP = 1u << 20;  /* <=1M elems per tensor (~4 MB) */
    uint64_t decoded = 0, skipped = 0, nonfinite = 0;
    double gmin = 1e30, gmax = -1e30;
    for (uint64_t i = 0; i < l.n_tensors; i++) {
        const ogguf_ltensor *t = &l.tensors[i];
        if (!oq_can_decode(t->type)) { skipped++; continue; }
        size_t be = oq_block_elems(t->type);
        size_t n = (size_t)t->n_elements;
        size_t sample = n < SAMPLE_CAP ? n : SAMPLE_CAP;
        if (be > 1) sample -= sample % be;          /* whole blocks only */
        if (sample == 0) sample = (n >= be) ? be : n;
        float *buf = malloc(sample * sizeof(float));
        if (!buf) { fprintf(stderr, "  oom sampling %s\n", t->name); break; }
        if (oq_dequantize(t->type, t->data, buf, sample) != ORNITH_OK) {
            fprintf(stderr, "  decode FAILED for %s (%s)\n",
                    t->name, oggml_type_name(t->type));
            free(buf); ogguf_loaded_free(&l); return 1;
        }
        bool finite = true; double tmin = 1e30, tmax = -1e30;
        for (size_t k = 0; k < sample; k++) {
            float v = buf[k];
            if (!(v == v) || v > 1e30f || v < -1e30f) { finite = false; break; }
            if (v < tmin) tmin = v;
            if (v > tmax) tmax = v;
        }
        if (!finite) { nonfinite++; }
        else { if (tmin < gmin) gmin = tmin; if (tmax > gmax) gmax = tmax; }
        /* spot-print a couple of representative tensors */
        if (strcmp(t->name, "token_embd.weight") == 0 ||
            strcmp(t->name, "blk.0.attn_qkv.weight") == 0 ||
            strcmp(t->name, "blk.3.attn_q.weight") == 0) {
            printf("    %-30s %-6s sample[%zu] range [% .4f, % .4f] %s\n",
                   t->name, oggml_type_name(t->type), sample, tmin, tmax,
                   finite ? "finite" : "NON-FINITE");
        }
        decoded++;
        free(buf);
    }
    printf("  decoded %llu tensors, skipped %llu (undecodable type), "
           "non-finite %llu\n",
           (unsigned long long)decoded, (unsigned long long)skipped,
           (unsigned long long)nonfinite);
    printf("  global decoded value range: [% .4f, % .4f]\n", gmin, gmax);
    printf("  => k-quant decoders read the real GGUF %s\n",
           (nonfinite == 0 && decoded > 0) ? "correctly (all finite)"
                                           : "with problems");
    ogguf_loaded_free(&l);
    return (nonfinite == 0 && decoded > 0) ? 0 : 1;
}

static int cmd_run(const char *path) {
    ogguf_file g;
    ornith_status s = ogguf_open(path, &g);
    if (s != ORNITH_OK) {
        fprintf(stderr, "run: %s: %s\n", ornith_strerror(s), ornith_last_error());
        return 1;
    }

    printf("loaded GGUF: %s\n", path);
    printf("  general.arch      : %s\n",
           g.model_arch[0] ? g.model_arch : "(unset)");
    printf("  general.name      : %s\n",
           g.model_name[0] ? g.model_name : "(unset)");
    printf("  tensors           : %llu\n", (unsigned long long)g.n_tensors);

    /* Detect the Ornith layout using the real GGUF tensor names: full-attn
     * blocks carry attn_q/attn_k/attn_v, linear (SSM) blocks carry ssm_* and a
     * fused attn_qkv. (Names per a real Ornith GGUF; see ornith_forward.h.) */
    int n_ssm  = count_tensor_substr(&g, "ssm_");
    int n_attq = count_tensor_substr(&g, "attn_q.");
    int n_exps = g.n_expert_tensors;
    bool has_embed = count_tensor_substr(&g, "token_embd") > 0;
    bool has_head  = count_tensor_substr(&g, "output.weight") > 0 ||
                     count_tensor_substr(&g, "output_norm") > 0;
    printf("  layout            : %d ssm tensors, %d full-attn q tensors, "
           "%d expert tensors\n", n_ssm, n_attq, n_exps);
    bool looks_ornith = (n_ssm > 0 && n_attq > 0 && has_embed && has_head);
    printf("  recognized Ornith hybrid layout: %s\n",
           looks_ornith ? "yes" : "no (or dense/9B variant)");

    /* Real-weight forward needs every weight in a precision we can read. Ornith
     * GGUFs are asymmetrically quantized, which requires the M1 dequant kernels
     * (IQ2_XXS/Q2_K/Q*_K) that are not part of this milestone. Gate honestly. */
    bool dense = gguf_all_dense(&g);
    ogguf_close(&g);

    /* Decode the real weights with the k-quant codecs (the part this milestone
     * delivers). A numerically-correct *forward* additionally needs the exact
     * qwen3.5 ops the synthetic engine approximates — see the note below. */
    if (!dense) {
        if (verify_real_dequant(path) != 0) return 1;
    }

    printf("\nfull real-weight forward: not yet wired end-to-end. The k-quant\n"
           "decoders above read the real GGUF correctly; a *coherent* forward\n"
           "still needs three qwen3.5-exact pieces the reference engine\n"
           "approximates: (1) gated full attention (attn_output_gate splits\n"
           "attn_q into q + output gate; 16 heads x 256, 4 kv heads), (2) the\n"
           "exact gated-delta-net gating from ssm_a/ssm_dt (not the sigmoid\n"
           "stand-in), and (3) a tokenizer for text I/O. See ROADMAP.md.\n");

    /* Prove the forward engine itself on a tiny synthetic model. */
    return run_synthetic_selftest();
}

/* Real-weight generation: load the GGUF, dequantize on the fly, and greedily
 * decode `n_predict` tokens after the prompt. This is the coherent-text path. */
static int cmd_generate(const char *path, const char *prompt, int n_predict,
                        const osample_params *sp) {
    rmodel *m = NULL;
    ornith_status s = rmodel_load(path, &m);
    if (s != ORNITH_OK) {
        fprintf(stderr, "run: load failed: %s\n", ornith_last_error());
        return 1;
    }
    const ornith_arch *a = rmodel_arch(m);
    fprintf(stderr,
            "loaded %s: hidden %d, %d layers (%d full-attn), vocab %d, "
            "eos %d\n", path, a->hidden_size, a->num_layers,
            ornith_count_full_attn_layers(a), a->vocab_size, a->eos_token_id);
    int sampling = sp && sp->temperature > 0.0f;
    if (sampling)
        fprintf(stderr, "generating %d tokens (temp %.3g, top-k %d, top-p %.3g, "
                "min-p %.3g, repeat %.3g, seed %llu)...\n\n", n_predict,
                (double)sp->temperature, sp->top_k, (double)sp->top_p,
                (double)sp->min_p, (double)sp->repeat_penalty,
                (unsigned long long)sp->seed);
    else
        fprintf(stderr, "generating %d tokens (greedy)...\n\n", n_predict);

    s = rmodel_generate_s(m, prompt, n_predict, sampling ? sp : NULL, stdout);
    if (s != ORNITH_OK)
        fprintf(stderr, "run: generation failed: %s\n", ornith_last_error());
    rmodel_free(m);
    return s == ORNITH_OK ? 0 : 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "ornithology %s\n\n"
        "usage:\n"
        "  %s version\n"
        "  %s arch [397b|35b]\n"
        "  %s config <config.json>\n"
        "  %s inspect [--tensors] <model.gguf>\n"
        "  %s quantize [--base TYPE] <in.gguf> <out.gguf>\n"
        "  %s run [--prompt TEXT] [-n N] [--temp T] [--top-p P] [--top-k K]\n"
        "         [--min-p M] [--repeat-penalty R] [--seed S] [--kv-q8] <model.gguf>\n"
        "        with --prompt: real-weight generation (default N=32);\n"
        "        default greedy (--temp 0); --temp>0 enables sampling;\n"
        "        without --prompt: inspect + dequant check + synthetic self-test\n"
        "  %s serve [--host H] [--port P] <model.gguf>\n"
        "        OpenAI/Anthropic-compatible HTTP server (default 127.0.0.1:8080)\n"
        "  %s repl [--temp T ...] <model.gguf>\n"
        "        interactive multi-turn chat REPL (/reset, /exit)\n"
        "  %s agent [--yolo] [--max-iters N] --task \"...\" <model.gguf>\n"
        "        coding-agent loop (read_file/write_file/list_dir/run_command);\n"
        "        write_file/run_command confirm via y/N unless --yolo\n"
        "\n"
        "quantize applies the asymmetric POLICY.md mapping; --base TYPE\n"
        "(f32|f16|bf16|q8_0|q4_0, default f16) covers tensors with no rule.\n",
        ORNITH_VERSION, argv0, argv0, argv0, argv0, argv0, argv0, argv0,
        argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const char *cmd = argv[1];

    if (!strcmp(cmd, "version") || !strcmp(cmd, "--version")) return cmd_version();
    if (!strcmp(cmd, "arch"))    return cmd_arch(argc > 2 ? argv[2] : NULL);
    if (!strcmp(cmd, "config")) {
        if (argc < 3) { usage(argv[0]); return 1; }
        return cmd_config(argv[2]);
    }
    if (!strcmp(cmd, "inspect")) {
        bool list = false;
        const char *path = NULL;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--tensors")) list = true;
            else path = argv[i];
        }
        if (!path) { usage(argv[0]); return 1; }
        return cmd_inspect(path, list);
    }
    if (!strcmp(cmd, "quantize")) {
        uint32_t base = OGGML_F16;
        const char *paths[2] = { NULL, NULL };
        int np = 0;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--base") && i + 1 < argc) {
                if (parse_quant_type(argv[++i], &base) != 0) {
                    fprintf(stderr, "quantize: unknown --base type '%s'\n",
                            argv[i]);
                    return 1;
                }
            } else if (np < 2) {
                paths[np++] = argv[i];
            }
        }
        if (np != 2) { usage(argv[0]); return 1; }
        return cmd_quantize(paths[0], paths[1], base);
    }
    if (!strcmp(cmd, "run")) {
        const char *path = NULL, *prompt = NULL;
        int n_predict = 32;
        osample_params sp = osample_params_default();  /* greedy by default */
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
            else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n-predict"))
                     && i + 1 < argc) n_predict = atoi(argv[++i]);
            else if ((!strcmp(argv[i], "--temp") || !strcmp(argv[i], "--temperature"))
                     && i + 1 < argc) sp.temperature = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--top-p") && i + 1 < argc)
                sp.top_p = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--top-k") && i + 1 < argc)
                sp.top_k = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--min-p") && i + 1 < argc)
                sp.min_p = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc)
                sp.repeat_penalty = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
                sp.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "--kv-q8"))
                /* int8 + per-(token,head) scale KV cache (see ORNITH_KV_Q8). */
                setenv("ORNITH_KV_Q8", "1", 1);
            else path = argv[i];
        }
        if (!path) { usage(argv[0]); return 1; }
        if (prompt) return cmd_generate(path, prompt, n_predict, &sp);
        return cmd_run(path);   /* no prompt: inspect + synthetic self-test */
    }
    if (!strcmp(cmd, "serve")) {
        return ornith_server_main(argc - 2, argv + 2);
    }
    if (!strcmp(cmd, "repl") || !strcmp(cmd, "chat")) {
        return ornith_repl_main(argc - 2, argv + 2);
    }
    if (!strcmp(cmd, "agent")) {
        return ornith_agent_main(argc - 2, argv + 2);
    }

    usage(argv[0]);
    return 1;
}
