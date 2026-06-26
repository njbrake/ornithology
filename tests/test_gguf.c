/* test_gguf.c — exercises the parts that work without weights or a GPU:
 * the JSON parser, the config loader, the GGUF reader, and the arch math.
 *
 * We synthesize a tiny but well-formed GGUF and config.json on disk, then read
 * them back. No external fixtures, no network. Run via `make test`.
 */
#include "ornith_json.h"
#include "ornith_config.h"
#include "ornith_gguf.h"
#include "ornith_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

/* ---- little-endian GGUF writer (test fixture only) -------------------- */

static void w_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w_str(FILE *f, const char *s) {
    uint64_t n = strlen(s);
    w_u64(f, n);
    fwrite(s, 1, n, f);
}

enum { T_STRING = 8, T_UINT32 = 4 };

static const char *write_fixture_gguf(void) {
    static const char *path = "build/_fixture.gguf";
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen fixture"); exit(2); }

    fwrite("GGUF", 1, 4, f);
    w_u32(f, 3);          /* version            */
    w_u64(f, 2);          /* n_tensors          */
    w_u64(f, 2);          /* n_kv               */

    /* kv: general.architecture = "qwen3_5_moe" */
    w_str(f, "general.architecture"); w_u32(f, T_STRING); w_str(f, "qwen3_5_moe");
    /* kv: general.name = "Ornith-1.0-397B" */
    w_str(f, "general.name"); w_u32(f, T_STRING); w_str(f, "Ornith-1.0-397B");

    /* tensor 0: a normal F16 tensor */
    w_str(f, "token_embd.weight");
    w_u32(f, 2); w_u64(f, 4096); w_u64(f, 256);
    w_u32(f, 1 /* F16 */); w_u64(f, 0);

    /* tensor 1: an IQ2_XXS expert tensor (name contains "exps") */
    w_str(f, "blk.0.ffn_gate_exps.weight");
    w_u32(f, 2); w_u64(f, 4096); w_u64(f, 1024);
    w_u32(f, 16 /* IQ2_XXS */); w_u64(f, 4096 * 256 * 2);

    fclose(f);
    return path;
}

static const char *write_fixture_config(void) {
    static const char *path = "build/_fixture_config.json";
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen config"); exit(2); }
    fputs(
      "{\n"
      "  \"model_type\": \"qwen3_5_moe\",\n"
      "  \"architectures\": [\"Qwen3_5MoeForConditionalGeneration\"],\n"
      "  \"hidden_size\": 4096,\n"
      "  \"num_hidden_layers\": 60,\n"
      "  \"vocab_size\": 248320,\n"
      "  \"max_position_embeddings\": 262144,\n"
      "  \"num_attention_heads\": 32,\n"
      "  \"num_key_value_heads\": 2,\n"
      "  \"head_dim\": 256,\n"
      "  \"full_attention_interval\": 4,\n"
      "  \"linear_num_key_heads\": 16,\n"
      "  \"linear_num_value_heads\": 64,\n"
      "  \"num_experts\": 512,\n"
      "  \"num_experts_per_tok\": 10,\n"
      "  \"moe_intermediate_size\": 1024,\n"
      "  \"shared_expert_intermediate_size\": 1024,\n"
      "  \"vision_config\": {\"depth\": 27},\n"
      "  \"eos_token_id\": 248044\n"
      "}\n", f);
    fclose(f);
    return path;
}

int main(void) {
    printf("== json ==\n");
    {
        const char *err = NULL;
        ojson *v = ojson_parse("{\"a\": 1, \"b\": [true, null, \"x\"]}", &err);
        CHECK(v != NULL, "parse object");
        CHECK(ojson_get_int(v, "a", -1) == 1, "int field");
        const ojson *arr = ojson_get(v, "b");
        CHECK(arr && arr->type == OJSON_ARRAY && arr->count == 3, "array len");
        ojson_free(v);
    }

    printf("== config ==\n");
    {
        const char *cpath = write_fixture_config();
        ornith_arch a;
        ornith_status s = ornith_config_load(cpath, &a);
        CHECK(s == ORNITH_OK, "config load");
        CHECK(strcmp(a.model_type, "qwen3_5_moe") == 0, "model_type");
        CHECK(a.hidden_size == 4096, "hidden_size");
        CHECK(a.num_layers == 60, "num_layers");
        CHECK(a.n_routed_experts == 512, "n_routed_experts");
        CHECK(a.experts_per_tok == 10, "experts_per_tok");
        CHECK(a.num_kv_heads == 2, "num_kv_heads");
        CHECK(a.has_vision == true, "vision detected");
        CHECK(ornith_count_full_attn_layers(&a) == 15, "15 full-attn layers");

        double pb = ornith_arch_param_billions(&a);
        CHECK(pb > 350 && pb < 440, "param count ~397B");

        /* KV per token: 15 full layers * 2 (k,v) * 2 kv-heads * 256 head_dim
         * * 2 bytes = 30720 bytes ~= 30 KB. Only full-attn layers cost KV;
         * the 45 linear-attn layers carry constant state instead. */
        double kv = ornith_arch_kv_bytes_per_token(&a, 2.0);
        CHECK(kv > 30000 && kv < 31500, "KV/token ~30KB fp16");
    }

    printf("== gguf ==\n");
    {
        const char *gpath = write_fixture_gguf();
        ogguf_file g;
        ornith_status s = ogguf_open(gpath, &g);
        CHECK(s == ORNITH_OK, "gguf open");
        CHECK(g.version == 3, "version 3");
        CHECK(g.n_tensors == 2, "2 tensors");
        CHECK(strcmp(g.model_arch, "qwen3_5_moe") == 0, "general.architecture");
        CHECK(strcmp(g.model_name, "Ornith-1.0-397B") == 0, "general.name");
        CHECK(g.n_expert_tensors == 1, "1 expert tensor");
        CHECK(g.type_elem_count[OGGML_IQ2_XXS] == 4096 * 1024, "IQ2_XXS elems");
        ogguf_close(&g);
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "TESTS FAILED" : "ALL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
