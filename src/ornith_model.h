/* ornith_model.h — architecture description for the Ornith-1.0 family.
 *
 * These structs hold the parsed hyper-parameters of a qwen3_5_moe model.
 * Values are populated either from a Hugging Face config.json (ornith_config.c)
 * or from GGUF metadata (ornith_gguf.c). The defaults below are the
 * Ornith-1.0-397B reference values, kept here so the engine can sanity-check a
 * loaded model and so tools that don't have a config handy still know the shape.
 */
#ifndef ORNITH_MODEL_H
#define ORNITH_MODEL_H

#include "ornith.h"

/* Per-layer attention flavor. Ornith interleaves them: with
 * full_attention_interval == 4 the pattern is L L L F repeating, i.e. layers
 * (i % 4 == 3) are full attention, the rest are linear. */
typedef enum {
    ORNITH_ATTN_LINEAR = 0, /* gated-delta / short-conv recurrent attention  */
    ORNITH_ATTN_FULL   = 1, /* standard softmax attention with a GQA KV cache */
} ornith_attn_kind;

typedef struct {
    /* identity */
    char  model_type[64];      /* expect "qwen3_5_moe"                        */
    char  arch[96];            /* expect "Qwen3_5MoeForConditionalGeneration" */

    /* core dims */
    int32_t hidden_size;       /* 4096                                        */
    int32_t num_layers;        /* 60                                          */
    int32_t vocab_size;        /* 248320                                      */
    int32_t max_position;      /* 262144                                      */
    float   rms_norm_eps;
    float   rope_theta;

    /* full-attention layers (GQA) */
    int32_t num_attn_heads;    /* 32                                          */
    int32_t num_kv_heads;      /* 2                                           */
    int32_t head_dim;          /* 256                                         */
    int32_t rope_dim;          /* rotary dims per head (partial rope); 0=full */
    bool    attn_output_gate;  /* full-attn out gate (attn_q packs q+gate)    */
    int32_t full_attn_interval;/* 4  -> 1 full attn layer per 4 layers        */

    /* linear-attention layers (gated delta-net) */
    int32_t lin_key_heads;     /* linear_num_key_heads   = 16                 */
    int32_t lin_value_heads;   /* linear_num_value_heads = 64                 */
    int32_t lin_key_head_dim;  /* 128                                         */
    int32_t lin_value_head_dim;/* 128                                         */
    int32_t lin_conv_kernel;   /* 4                                           */

    /* mixture of experts */
    int32_t n_routed_experts;  /* 512                                         */
    int32_t experts_per_tok;   /* 10                                          */
    int32_t moe_inter_size;    /* 1024                                        */
    int32_t shared_inter_size; /* 1024 (0 if no shared expert)                */

    /* multimodal (optional; v1 is text-only) */
    bool    has_vision;
    int32_t image_token_id;
    int32_t video_token_id;

    /* tokens */
    int32_t eos_token_id;      /* 248044                                      */
    int32_t bos_token_id;
} ornith_arch;

/* Fill `a` with the Ornith-1.0-397B reference values (the flagship). */
void ornith_arch_defaults_397b(ornith_arch *a);

/* Fill `a` with the Ornith-1.0-35B reference values. Same qwen3_5_moe family,
 * smaller everywhere: hidden 2048, 40 layers, 256 experts (top-8). Both 397B
 * and 35B are supported targets. */
void ornith_arch_defaults_35b(ornith_arch *a);

/* Select reference defaults by short name ("397b" or "35b"). Returns false and
 * leaves `a` as 397B if the name is unknown. */
bool ornith_arch_defaults_by_name(const char *name, ornith_arch *a);

/* True if layer `i` (0-based) is a full-attention layer for arch `a`. */
static inline bool ornith_layer_is_full_attn(const ornith_arch *a, int i) {
    return a->full_attn_interval > 0 && (i % a->full_attn_interval) ==
           (a->full_attn_interval - 1);
}

/* Number of full vs linear attention layers. */
int ornith_count_full_attn_layers(const ornith_arch *a);

/* Pretty-print the architecture and derived facts (param counts, KV cost). */
void ornith_arch_print(const ornith_arch *a, FILE *out);

/* Rough total parameter count in billions, from the architecture alone. */
double ornith_arch_param_billions(const ornith_arch *a);

/* Per-token full-attention KV cache cost in bytes, at `bytes_per_elem`
 * (2 for FP16, 1 for Q8). Linear layers contribute ~0 (constant state). */
double ornith_arch_kv_bytes_per_token(const ornith_arch *a, double bytes_per_elem);

#endif /* ORNITH_MODEL_H */
