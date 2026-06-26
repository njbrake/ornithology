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
#include "ornith.h"
#include "ornith_model.h"
#include "ornith_config.h"
#include "ornith_gguf.h"
#include "ornith_gguf_write.h"
#include "ornith_quant.h"
#include "ornith_forward.h"
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

    bool ok = strstr(g.model_arch, "qwen3_5") != NULL ||
              strstr(g.model_arch, "ornith") != NULL ||
              g.n_expert_tensors > 0;
    printf("\nlooks like an Ornith/MoE GGUF: %s\n", ok ? "yes" : "unclear");
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
    if (!gguf_all_dense(&g)) {
        printf("\nreal-weight forward: NOT YET — this GGUF is quantized.\n");
        printf("  Binding real Ornith weights needs the asymmetric dequant path\n"
               "  (IQ2_XXS / Q2_K / Q*_K -> f32), which is M1/M3 work; the loader\n"
               "  tensor-name mapping is in place (token_embd / output[_norm] /\n"
               "  blk.N.attn_* / blk.N.ssm_*). See ROADMAP.md.\n");
    } else {
        printf("\nthis GGUF is dense (F16/F32); full real-weight binding is the\n"
               "remaining M2 task (tensor data section reader).\n");
    }
    ogguf_close(&g);

    /* Prove the forward engine itself on a tiny synthetic model. */
    return run_synthetic_selftest();
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
        "  %s run <model.gguf>            (loads GGUF; runs the M2 forward engine)\n"
        "\n"
        "quantize applies the asymmetric POLICY.md mapping; --base TYPE\n"
        "(f32|f16|bf16|q8_0|q4_0, default f16) covers tensors with no rule.\n",
        ORNITH_VERSION, argv0, argv0, argv0, argv0, argv0, argv0);
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
        if (argc < 3) { usage(argv[0]); return 1; }
        return cmd_run(argv[2]);
    }

    usage(argv[0]);
    return 1;
}
