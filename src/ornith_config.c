/* ornith_config.c — map a Hugging Face config.json onto ornith_arch.
 *
 * Field names for qwen3_5_moe are not fully standardized across releases, so we
 * try a few aliases per value. Whatever is missing keeps the 397B default,
 * which means an abbreviated config still produces a coherent architecture. */
#include "ornith_config.h"
#include "ornith_json.h"
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, long *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { ornith_set_error("cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); ornith_set_error("ftell failed"); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); ornith_set_error("oom reading %s", path); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf); fclose(f); ornith_set_error("short read on %s", path);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    if (len_out) *len_out = n;
    return buf;
}

/* Try several keys in order; first numeric hit wins. */
static long get_int_alias(const ojson *o, long def, const char *const *keys) {
    for (; *keys; keys++) {
        const ojson *v = ojson_get(o, *keys);
        if (v && v->type == OJSON_NUMBER) return (long)v->num;
    }
    return def;
}

ornith_status ornith_config_load(const char *path, ornith_arch *out) {
    ornith_arch_defaults_397b(out); /* start from known-good 397B shape */

    long len = 0;
    char *text = read_file(path, &len);
    if (!text) return ORNITH_ERR_IO;

    const char *errpos = NULL;
    long erroff = -1;
    ojson *root = ojson_parse(text, &errpos);
    if (errpos) erroff = (long)(errpos - text); /* compute before free */
    free(text);
    if (!root) {
        ornith_set_error("JSON parse error in %s near offset %ld",
                         path, erroff);
        return ORNITH_ERR_FORMAT;
    }
    if (root->type != OJSON_OBJECT) {
        ojson_free(root);
        ornith_set_error("config.json is not a JSON object");
        return ORNITH_ERR_FORMAT;
    }

    const char *mt = ojson_get_str(root, "model_type", NULL);
    if (mt) snprintf(out->model_type, sizeof(out->model_type), "%s", mt);

    const ojson *arch_arr = ojson_get(root, "architectures");
    if (arch_arr && arch_arr->type == OJSON_ARRAY && arch_arr->count > 0 &&
        arch_arr->items[0]->type == OJSON_STRING)
        snprintf(out->arch, sizeof(out->arch), "%s", arch_arr->items[0]->str);

    out->hidden_size  = (int)ojson_get_int(root, "hidden_size", out->hidden_size);
    out->num_layers   = (int)get_int_alias(root, out->num_layers,
        (const char*[]){"num_hidden_layers", "num_layers", NULL});
    out->vocab_size   = (int)ojson_get_int(root, "vocab_size", out->vocab_size);
    out->max_position = (int)ojson_get_int(root, "max_position_embeddings",
                                           out->max_position);
    out->rms_norm_eps = (float)ojson_get_num(root, "rms_norm_eps",
                                             out->rms_norm_eps);
    out->rope_theta   = (float)ojson_get_num(root, "rope_theta", out->rope_theta);

    out->num_attn_heads = (int)ojson_get_int(root, "num_attention_heads",
                                             out->num_attn_heads);
    out->num_kv_heads   = (int)ojson_get_int(root, "num_key_value_heads",
                                             out->num_kv_heads);
    out->head_dim       = (int)ojson_get_int(root, "head_dim", out->head_dim);
    out->full_attn_interval = (int)get_int_alias(root, out->full_attn_interval,
        (const char*[]){"full_attention_interval", "full_attn_interval", NULL});

    out->lin_key_heads = (int)get_int_alias(root, out->lin_key_heads,
        (const char*[]){"linear_num_key_heads", NULL});
    out->lin_value_heads = (int)get_int_alias(root, out->lin_value_heads,
        (const char*[]){"linear_num_value_heads", NULL});
    out->lin_key_head_dim = (int)get_int_alias(root, out->lin_key_head_dim,
        (const char*[]){"linear_key_head_dim", NULL});
    out->lin_value_head_dim = (int)get_int_alias(root, out->lin_value_head_dim,
        (const char*[]){"linear_value_head_dim", NULL});
    out->lin_conv_kernel = (int)get_int_alias(root, out->lin_conv_kernel,
        (const char*[]){"linear_conv_kernel_dim", "conv_kernel", NULL});

    out->n_routed_experts = (int)get_int_alias(root, out->n_routed_experts,
        (const char*[]){"num_experts", "n_routed_experts", NULL});
    out->experts_per_tok = (int)get_int_alias(root, out->experts_per_tok,
        (const char*[]){"num_experts_per_tok", "experts_per_tok", NULL});
    out->moe_inter_size = (int)get_int_alias(root, out->moe_inter_size,
        (const char*[]){"moe_intermediate_size", NULL});
    out->shared_inter_size = (int)get_int_alias(root, out->shared_inter_size,
        (const char*[]){"shared_expert_intermediate_size", NULL});

    out->image_token_id = (int)ojson_get_int(root, "image_token_id",
                                             out->image_token_id);
    out->video_token_id = (int)ojson_get_int(root, "video_token_id",
                                             out->video_token_id);
    out->eos_token_id   = (int)ojson_get_int(root, "eos_token_id",
                                             out->eos_token_id);
    out->bos_token_id   = (int)ojson_get_int(root, "bos_token_id",
                                             out->bos_token_id);
    out->has_vision     = (ojson_get(root, "vision_config") != NULL);

    ojson_free(root);
    return ORNITH_OK;
}
