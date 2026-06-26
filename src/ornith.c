/* ornith.c — command-line entry point.
 *
 * Implemented today (the parts that don't need weights or a GPU):
 *   ornith version                 print version
 *   ornith arch                    print the Ornith-1.0-397B reference arch
 *   ornith config <config.json>    parse a HF config and print the arch
 *   ornith inspect <model.gguf>    parse GGUF header/metadata/tensor index
 *
 * Stubbed (return a clear "not implemented" until the backends land):
 *   ornith run <model.gguf>        interactive / one-shot generation
 *
 * The split is deliberate: everything that can be correct and tested without a
 * 397B checkpoint or a Metal/CUDA device works now and is covered by tests;
 * the inference path is scaffolded with an honest error rather than faked.
 */
#include "ornith.h"
#include "ornith_model.h"
#include "ornith_config.h"
#include "ornith_gguf.h"
#include "ornith_gguf_write.h"
#include "ornith_quant.h"
#include <string.h>

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

static int cmd_run(const char *path) {
    (void)path;
    fprintf(stderr,
        "run: not implemented yet.\n"
        "The forward pass (hybrid linear + full attention, MoE routing) is the\n"
        "next milestone. See ROADMAP.md M2/M3. Today you can use `inspect`,\n"
        "`config`, and `arch` to validate a model and plan memory.\n");
    return 64;
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
        "  %s run <model.gguf>            (not implemented yet)\n"
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
