/* ornith_model.c — architecture helpers and back-of-envelope cost model. */
#include "ornith_model.h"
#include <string.h>

void ornith_arch_defaults_397b(ornith_arch *a) {
    memset(a, 0, sizeof(*a));
    snprintf(a->model_type, sizeof(a->model_type), "qwen3_5_moe");
    snprintf(a->arch, sizeof(a->arch), "Qwen3_5MoeForConditionalGeneration");

    a->hidden_size       = 4096;
    a->num_layers        = 60;
    a->vocab_size        = 248320;
    a->max_position      = 262144;
    a->rms_norm_eps      = 1e-6f;
    a->rope_theta        = 10000000.0f;

    a->num_attn_heads    = 32;
    a->num_kv_heads      = 2;
    a->head_dim          = 256;
    a->full_attn_interval= 4;

    a->lin_key_heads     = 16;
    a->lin_value_heads   = 64;
    a->lin_key_head_dim  = 128;
    a->lin_value_head_dim= 128;
    a->lin_conv_kernel   = 4;

    a->n_routed_experts  = 512;
    a->experts_per_tok   = 10;
    a->moe_inter_size    = 1024;
    a->shared_inter_size = 1024;

    a->has_vision        = true;
    a->image_token_id    = 248056;
    a->video_token_id    = 248057;
    a->eos_token_id      = 248044;
    a->bos_token_id      = -1;
}

void ornith_arch_defaults_35b(ornith_arch *a) {
    memset(a, 0, sizeof(*a));
    snprintf(a->model_type, sizeof(a->model_type), "qwen3_5_moe");
    snprintf(a->arch, sizeof(a->arch), "Qwen3_5MoeForConditionalGeneration");

    a->hidden_size       = 2048;
    a->num_layers        = 40;
    a->vocab_size        = 248320;
    a->max_position      = 262144;
    a->rms_norm_eps      = 1e-6f;
    a->rope_theta        = 10000000.0f;

    a->num_attn_heads    = 16;
    a->num_kv_heads      = 2;
    a->head_dim          = 256;
    a->full_attn_interval= 4;

    a->lin_key_heads     = 16;
    a->lin_value_heads   = 32;
    a->lin_key_head_dim  = 128;
    a->lin_value_head_dim= 128;
    a->lin_conv_kernel   = 4;

    a->n_routed_experts  = 256;
    a->experts_per_tok   = 8;
    a->moe_inter_size    = 512;
    a->shared_inter_size = 512;

    a->has_vision        = true;
    a->image_token_id    = 248056;
    a->video_token_id    = 248057;
    a->eos_token_id      = 248046;
    a->bos_token_id      = -1;
}

bool ornith_arch_defaults_by_name(const char *name, ornith_arch *a) {
    if (name && (strstr(name, "35") != NULL)) {
        ornith_arch_defaults_35b(a);
        return true;
    }
    if (name && (strstr(name, "397") != NULL)) {
        ornith_arch_defaults_397b(a);
        return true;
    }
    ornith_arch_defaults_397b(a);
    return name == NULL; /* default ok if no name asked */
}

int ornith_count_full_attn_layers(const ornith_arch *a) {
    int n = 0;
    for (int i = 0; i < a->num_layers; i++)
        if (ornith_layer_is_full_attn(a, i)) n++;
    return n;
}

double ornith_arch_param_billions(const ornith_arch *a) {
    /* Routed experts dominate; count the three projections per expert. */
    double experts = (double)a->n_routed_experts * a->num_layers * 3.0 *
                     a->hidden_size * a->moe_inter_size;

    double shared = (double)a->num_layers * 3.0 * a->hidden_size *
                    a->shared_inter_size;

    /* Embedding + (untied) lm_head. */
    double embed = 2.0 * (double)a->vocab_size * a->hidden_size;

    /* Attention is small relative to the MoE; approximate both flavors. */
    int full = ornith_count_full_attn_layers(a);
    int lin  = a->num_layers - full;
    double attn_full = (double)full * (
        /* q,k,v,o roughly */
        (double)a->hidden_size * a->num_attn_heads * a->head_dim +
        2.0 * a->hidden_size * a->num_kv_heads * a->head_dim +
        (double)a->num_attn_heads * a->head_dim * a->hidden_size);
    double attn_lin = (double)lin * (
        (double)a->hidden_size * a->lin_key_heads * a->lin_key_head_dim +
        (double)a->hidden_size * a->lin_value_heads * a->lin_value_head_dim);

    double total = experts + shared + embed + attn_full + attn_lin;
    return total / 1e9;
}

double ornith_arch_kv_bytes_per_token(const ornith_arch *a, double bytes_per_elem) {
    int full = ornith_count_full_attn_layers(a);
    /* K and V, per full-attn layer, num_kv_heads * head_dim each. */
    double elems = (double)full * 2.0 * a->num_kv_heads * a->head_dim;
    return elems * bytes_per_elem;
}

void ornith_arch_print(const ornith_arch *a, FILE *out) {
    int full = ornith_count_full_attn_layers(a);
    int lin  = a->num_layers - full;

    fprintf(out, "Ornith architecture\n");
    fprintf(out, "  model_type        : %s\n", a->model_type);
    fprintf(out, "  arch              : %s\n", a->arch);
    fprintf(out, "  hidden_size       : %d\n", a->hidden_size);
    fprintf(out, "  num_layers        : %d  (%d full-attn, %d linear-attn)\n",
            a->num_layers, full, lin);
    fprintf(out, "  vocab_size        : %d\n", a->vocab_size);
    fprintf(out, "  max_position      : %d\n", a->max_position);
    fprintf(out, "  full-attn (GQA)   : %d heads / %d kv-heads, head_dim %d, "
                 "1 per %d layers\n",
            a->num_attn_heads, a->num_kv_heads, a->head_dim,
            a->full_attn_interval);
    fprintf(out, "  linear-attn       : %d key-heads (d%d), %d value-heads (d%d),"
                 " conv k=%d\n",
            a->lin_key_heads, a->lin_key_head_dim,
            a->lin_value_heads, a->lin_value_head_dim, a->lin_conv_kernel);
    fprintf(out, "  MoE               : %d experts, top-%d, inter %d, "
                 "shared inter %d\n",
            a->n_routed_experts, a->experts_per_tok, a->moe_inter_size,
            a->shared_inter_size);
    fprintf(out, "  vision            : %s\n", a->has_vision ? "yes" : "no");

    double pb = ornith_arch_param_billions(a);
    fprintf(out, "\nDerived\n");
    fprintf(out, "  total params      : ~%.0fB\n", pb);

    /* Footprint at a ds4-style asymmetric ~2.24 bpw expert blend. */
    double experts_b = (double)a->n_routed_experts * a->num_layers * 3.0 *
                       a->hidden_size * a->moe_inter_size / 1e9;
    double expert_gb = experts_b * 1e9 * 2.24 / 8.0 / 1e9;
    double rest_gb   = (pb - experts_b) * 1e9 * 6.0 / 8.0 / 1e9;
    fprintf(out, "  ~weights @ 2.24bpw experts + 6bpw rest : ~%.0f GB\n",
            expert_gb + rest_gb);

    double kv16 = ornith_arch_kv_bytes_per_token(a, 2.0);
    double kv8  = ornith_arch_kv_bytes_per_token(a, 1.0);
    fprintf(out, "  KV per token      : %.0f KB (fp16) / %.0f KB (q8)\n",
            kv16 / 1024.0, kv8 / 1024.0);
    fprintf(out, "  KV @ 128K ctx     : %.1f GB (fp16) / %.1f GB (q8)\n",
            kv16 * 131072 / 1e9, kv8 * 131072 / 1e9);
    fprintf(out, "  KV @ %dK ctx (max): %.1f GB (fp16) / %.1f GB (q8)\n",
            a->max_position / 1024,
            kv16 * a->max_position / 1e9, kv8 * a->max_position / 1e9);
}
