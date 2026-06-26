/* ornith_forward.c — the assembled qwen3_5_moe decoder forward pass.
 *
 * Pre-norm decoder: x += attn(attn_norm(x)); x += moe(post_attn_norm(x)). Full
 * attention layers use GQA with QK-norm + RoPE and a KV cache; linear layers use
 * the gated delta-net (fused qkv -> causal conv -> per-head delta rule -> gated
 * RMSNorm -> out proj). Weights here are seeded-synthetic; the tensor *shapes*
 * and dataflow mirror a real Ornith GGUF (see of_gguf tensor-name mapping).
 *
 * Decode (single token, step recurrence + KV append) and prefill (whole prompt,
 * chunked scan + causal pass) must agree to fp tolerance: that is the headline
 * parity property, asserted in tests/test_forward.c.
 */
#include "ornith_forward.h"
#include "ornith_tensor.h"
#include "ornith_attn.h"
#include "ornith_moe.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- weight container -------------------------------------------------- */

typedef struct {
    bool is_full;
    float *attn_norm;   /* [H] */
    /* full attention */
    float *wq, *wk, *wv, *wo;   /* projections */
    float *q_norm, *k_norm;     /* QK-norm, [head_dim] each */
    /* linear attention (gated delta-net) */
    float *lq, *lk, *lv;        /* projections (the fused attn_qkv, split out) */
    float *la, *lb;             /* a (ssm_alpha) / beta (ssm_beta) projs [nv,H]*/
    float *ssm_a;               /* A_log per value head [nv]                   */
    float *ssm_dt;              /* dt bias per value head [nv]                 */
    float *conv_w, *conv_b;     /* depthwise conv [Cc, K], bias [Cc] */
    float *lgate;               /* output gate proj [n_v*dv, H] */
    float *lnorm;               /* per-head out RMSNorm weight [dv] (ssm_norm) */
    float *lo;                  /* out proj [H, n_v*dv] */
    /* MoE FFN */
    float *ffn_norm;            /* [H] */
    float *router;              /* [n_exp, H] */
    float *e_gate, *e_up, *e_down;
    float *s_gate, *s_up, *s_down;
} of_layer;

struct of_model {
    ornith_arch a;
    float *tok_embd;    /* [vocab, H] */
    float *final_norm;  /* [H] */
    float *lm_head;     /* [vocab, H] */
    of_layer *layers;
};

/* dims helpers */
static int lin_qdim(const ornith_arch *a){ return a->lin_key_heads*a->lin_key_head_dim; }
static int lin_vdim(const ornith_arch *a){ return a->lin_value_heads*a->lin_value_head_dim; }
static int lin_cc(const ornith_arch *a){ return 2*lin_qdim(a) + lin_vdim(a); }
static int full_qdim(const ornith_arch *a){ return a->num_attn_heads*a->head_dim; }
static int full_kvdim(const ornith_arch *a){ return a->num_kv_heads*a->head_dim; }
/* attn_q output width: when output-gated, packs [query|gate] per head. */
static int full_qproj(const ornith_arch *a){
    return (a->attn_output_gate ? 2 : 1) * full_qdim(a);
}

/* ---- synthetic model construction ------------------------------------- */

static float *rng_alloc(ot_rng *r, int64_t n, float lo, float hi) {
    float *p = malloc((size_t)n * sizeof(float));
    if (!p) abort();
    ot_rng_fill(r, p, n, lo, hi);
    return p;
}

void ornith_arch_tiny(ornith_arch *a) {
    memset(a, 0, sizeof(*a));
    snprintf(a->model_type, sizeof(a->model_type), "qwen3_5_moe");
    snprintf(a->arch, sizeof(a->arch), "Qwen3_5MoeForConditionalGeneration");
    a->hidden_size = 32;
    a->num_layers = 4;          /* L L L F — the interleave appears once */
    a->vocab_size = 24;
    a->max_position = 4096;
    a->rms_norm_eps = 1e-6f;
    a->rope_theta = 10000.0f;
    a->num_attn_heads = 4;
    a->num_kv_heads = 2;
    a->head_dim = 8;
    a->rope_dim = 4;              /* partial rope (exercise the partial path)   */
    a->attn_output_gate = true;  /* exercise the full-attn output gate          */
    a->full_attn_interval = 4;
    a->lin_key_heads = 2;
    a->lin_value_heads = 4;
    a->lin_key_head_dim = 8;
    a->lin_value_head_dim = 8;
    a->lin_conv_kernel = 4;
    a->n_routed_experts = 8;
    a->experts_per_tok = 2;
    a->moe_inter_size = 16;
    a->shared_inter_size = 16;
    a->eos_token_id = -1;
    a->bos_token_id = -1;
}

of_model *of_model_build_synthetic(const ornith_arch *a, uint64_t seed) {
    of_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->a = *a;
    int H = a->hidden_size, V = a->vocab_size;
    ot_rng r = ot_rng_seed(seed);

    const float W = 0.08f;   /* projection weight range */
    m->tok_embd   = rng_alloc(&r, (int64_t)V * H, -W, W);
    m->final_norm = rng_alloc(&r, H, 0.7f, 1.3f);
    m->lm_head    = rng_alloc(&r, (int64_t)V * H, -W, W);

    m->layers = calloc((size_t)a->num_layers, sizeof(of_layer));
    if (!m->layers) { of_model_free(m); return NULL; }

    int nv = a->lin_value_heads, dv = a->lin_value_head_dim, hd = a->head_dim;
    int I = a->moe_inter_size, sI = a->shared_inter_size, ne = a->n_routed_experts;

    for (int L = 0; L < a->num_layers; L++) {
        of_layer *ly = &m->layers[L];
        ly->is_full = ornith_layer_is_full_attn(a, L);
        ly->attn_norm = rng_alloc(&r, H, 0.7f, 1.3f);
        if (ly->is_full) {
            ly->wq = rng_alloc(&r, (int64_t)full_qproj(a) * H, -W, W);
            ly->wk = rng_alloc(&r, (int64_t)full_kvdim(a) * H, -W, W);
            ly->wv = rng_alloc(&r, (int64_t)full_kvdim(a) * H, -W, W);
            ly->wo = rng_alloc(&r, (int64_t)H * full_qdim(a), -W, W);
            ly->q_norm = rng_alloc(&r, hd, 0.7f, 1.3f);
            ly->k_norm = rng_alloc(&r, hd, 0.7f, 1.3f);
        } else {
            ly->lq = rng_alloc(&r, (int64_t)lin_qdim(a) * H, -W, W);
            ly->lk = rng_alloc(&r, (int64_t)lin_qdim(a) * H, -W, W);
            ly->lv = rng_alloc(&r, (int64_t)lin_vdim(a) * H, -W, W);
            ly->la = rng_alloc(&r, (int64_t)nv * H, -W, W);
            ly->lb = rng_alloc(&r, (int64_t)nv * H, -W, W);
            ly->ssm_a = rng_alloc(&r, nv, -W, W);
            ly->ssm_dt = rng_alloc(&r, nv, -W, W);
            ly->conv_w = rng_alloc(&r, (int64_t)lin_cc(a) * a->lin_conv_kernel, -W, W);
            ly->conv_b = rng_alloc(&r, lin_cc(a), -W, W);
            ly->lgate = rng_alloc(&r, (int64_t)lin_vdim(a) * H, -W, W);
            ly->lnorm = rng_alloc(&r, dv, 0.7f, 1.3f);
            ly->lo = rng_alloc(&r, (int64_t)H * lin_vdim(a), -W, W);
        }
        ly->ffn_norm = rng_alloc(&r, H, 0.7f, 1.3f);
        ly->router   = rng_alloc(&r, (int64_t)ne * H, -W, W);
        ly->e_gate   = rng_alloc(&r, (int64_t)ne * I * H, -W, W);
        ly->e_up     = rng_alloc(&r, (int64_t)ne * I * H, -W, W);
        ly->e_down   = rng_alloc(&r, (int64_t)ne * H * I, -W, W);
        if (sI > 0) {
            ly->s_gate = rng_alloc(&r, (int64_t)sI * H, -W, W);
            ly->s_up   = rng_alloc(&r, (int64_t)sI * H, -W, W);
            ly->s_down = rng_alloc(&r, (int64_t)H * sI, -W, W);
        }
    }
    return m;
}

void of_model_free(of_model *m) {
    if (!m) return;
    free(m->tok_embd); free(m->final_norm); free(m->lm_head);
    if (m->layers) {
        for (int L = 0; L < m->a.num_layers; L++) {
            of_layer *ly = &m->layers[L];
            free(ly->attn_norm);
            free(ly->wq); free(ly->wk); free(ly->wv); free(ly->wo);
            free(ly->q_norm); free(ly->k_norm);
            free(ly->lq); free(ly->lk); free(ly->lv);
            free(ly->la); free(ly->lb);
            free(ly->ssm_a); free(ly->ssm_dt);
            free(ly->conv_w); free(ly->conv_b);
            free(ly->lgate); free(ly->lnorm); free(ly->lo);
            free(ly->ffn_norm); free(ly->router);
            free(ly->e_gate); free(ly->e_up); free(ly->e_down);
            free(ly->s_gate); free(ly->s_up); free(ly->s_down);
        }
        free(m->layers);
    }
    free(m);
}

const ornith_arch *of_model_arch(const of_model *m) { return &m->a; }

/* ---- per-sequence state ------------------------------------------------ */

struct of_state {
    ornith_arch a;
    int capacity;
    ornith_kv_cache *kv;     /* [num_layers], used only for full layers */
    float **delta_S;         /* [num_layers], [n_v*dv*dk] for linear layers */
    ornith_conv_state *conv; /* [num_layers], used only for linear layers */
    int pos;
};

of_state *of_state_new(const of_model *m, int capacity) {
    const ornith_arch *a = &m->a;
    of_state *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->a = *a; s->capacity = capacity;
    s->kv      = calloc((size_t)a->num_layers, sizeof(ornith_kv_cache));
    s->conv    = calloc((size_t)a->num_layers, sizeof(ornith_conv_state));
    s->delta_S = calloc((size_t)a->num_layers, sizeof(float*));
    if (!s->kv || !s->conv || !s->delta_S) { of_state_free(s); return NULL; }

    int dk = a->lin_key_head_dim, dv = a->lin_value_head_dim, nv = a->lin_value_heads;
    for (int L = 0; L < a->num_layers; L++) {
        if (ornith_layer_is_full_attn(a, L)) {
            if (ornith_kv_init(&s->kv[L], capacity, a->num_kv_heads,
                               a->head_dim) != ORNITH_OK) {
                of_state_free(s); return NULL;
            }
        } else {
            if (ornith_conv_init(&s->conv[L], a->lin_conv_kernel,
                                 lin_cc(a)) != ORNITH_OK) {
                of_state_free(s); return NULL;
            }
            s->delta_S[L] = calloc((size_t)nv * dv * dk, sizeof(float));
            if (!s->delta_S[L]) { of_state_free(s); return NULL; }
        }
    }
    return s;
}

void of_state_free(of_state *s) {
    if (!s) return;
    for (int L = 0; L < s->a.num_layers; L++) {
        if (s->kv) ornith_kv_free(&s->kv[L]);
        if (s->conv) ornith_conv_free(&s->conv[L]);
        if (s->delta_S) free(s->delta_S[L]);
    }
    free(s->kv); free(s->conv); free(s->delta_S);
    free(s);
}

void of_state_reset(of_state *s) {
    for (int L = 0; L < s->a.num_layers; L++) {
        if (ornith_layer_is_full_attn(&s->a, L)) ornith_kv_reset(&s->kv[L]);
        else {
            ornith_conv_reset(&s->conv[L]);
            int dk=s->a.lin_key_head_dim, dv=s->a.lin_value_head_dim,
                nv=s->a.lin_value_heads;
            memset(s->delta_S[L], 0, (size_t)nv*dv*dk*sizeof(float));
        }
    }
    s->pos = 0;
}

/* ---- shared layer math ------------------------------------------------- */

/* MoE FFN over a single token, accumulating into x (residual). */
static void moe_residual(const of_model *m, const of_layer *ly,
                         const float *x_in, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size;
    float *xn = malloc((size_t)H * sizeof(float));
    ot_rmsnorm(x_in, ly->ffn_norm, xn, H, a->rms_norm_eps);

    ornith_moe mo = {0};
    mo.hidden = H; mo.inter = a->moe_inter_size;
    mo.n_experts = a->n_routed_experts; mo.top_k = a->experts_per_tok;
    mo.shared_inter = a->shared_inter_size;
    mo.w_router = ly->router; mo.w_gate = ly->e_gate;
    mo.w_up = ly->e_up; mo.w_down = ly->e_down;
    mo.sw_gate = ly->s_gate; mo.sw_up = ly->s_up; mo.sw_down = ly->s_down;

    int maxI = a->moe_inter_size > a->shared_inter_size ?
               a->moe_inter_size : a->shared_inter_size;
    float *out = malloc((size_t)H * sizeof(float));
    float *scr = malloc((size_t)(H + 2*maxI) * sizeof(float));
    ornith_moe_forward(&mo, xn, out, scr);
    ot_add_(x_acc, out, H);
    free(out); free(scr); free(xn);
}

/* Apply QK-norm (per-head RMSNorm with weight) then (partial) RoPE to a
 * full-attn q/k laid out as n_heads contiguous head_dim vectors. */
static void qk_norm_rope(float *vec, int n_heads, int hd, int rope_dim,
                         const float *norm_w, int pos, float theta, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *vh = vec + (size_t)h * hd;
        ot_rmsnorm(vh, norm_w, vh, hd, eps);
        ot_rope_partial(vh, hd, rope_dim, pos, theta);
    }
}

/* gated-delta-net decay: alpha = exp(-exp(A_log) * softplus(a_in + dt_bias)). */
static float delta_decay(float a_in, float A_log, float dt_bias) {
    return expf(-expf(A_log) * ot_softplus(a_in + dt_bias));
}

/* Split the gated attn_q projection [n_heads*(2*hd)] into contiguous queries
 * q[n_heads*hd] and gates g[n_heads*hd] (head h: [q(hd)|gate(hd)]). When the
 * model is not output-gated, q is just a copy and g is left untouched. */
static void split_q_gate(const ornith_arch *a, const float *qproj,
                         float *q, float *g) {
    int nh = a->num_attn_heads, hd = a->head_dim;
    if (!a->attn_output_gate) {
        memcpy(q, qproj, (size_t)nh * hd * sizeof(float));
        return;
    }
    for (int h = 0; h < nh; h++) {
        const float *src = qproj + (size_t)h * 2 * hd;
        memcpy(q + (size_t)h * hd, src,      (size_t)hd * sizeof(float));
        memcpy(g + (size_t)h * hd, src + hd, (size_t)hd * sizeof(float));
    }
}

/* Apply the output gate o_h *= sigmoid(gate_h) elementwise per query head. */
static void apply_out_gate(const ornith_arch *a, float *o, const float *g) {
    if (!a->attn_output_gate) return;
    int n = a->num_attn_heads * a->head_dim;
    for (int i = 0; i < n; i++) o[i] *= ot_sigmoid(g[i]);
}

/* Gated delta-net output post-processing for one token: per-head RMSNorm of the
 * delta output, multiply by SiLU(gate), then out-projection accumulated into x. */
static void delta_finish(const of_model *m, const of_layer *ly,
                         const float *xn, const float *o_all, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, nv = a->lin_value_heads, dv = a->lin_value_head_dim;
    int vd = lin_vdim(a);
    float *g = malloc((size_t)vd * sizeof(float));
    float *o = malloc((size_t)vd * sizeof(float));
    ot_linear(ly->lgate, xn, g, vd, H);
    ot_silu_(g, vd);
    for (int hv = 0; hv < nv; hv++)
        ot_rmsnorm(o_all + (size_t)hv*dv, ly->lnorm, o + (size_t)hv*dv, dv,
                   a->rms_norm_eps);
    ot_mul_(o, g, vd);
    float *outp = malloc((size_t)H * sizeof(float));
    ot_linear(ly->lo, o, outp, H, vd);
    ot_add_(x_acc, outp, H);
    free(g); free(o); free(outp);
}

/* ---- decode: one token ------------------------------------------------- */

static void full_attn_decode(const of_model *m, const of_layer *ly,
                             of_state *s, int L, const float *xn, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nh=a->num_attn_heads, nkv=a->num_kv_heads, hd=a->head_dim;
    float *qproj = malloc((size_t)full_qproj(a)*sizeof(float));
    float *q = malloc((size_t)full_qdim(a)*sizeof(float));
    float *g = malloc((size_t)full_qdim(a)*sizeof(float));
    float *k = malloc((size_t)full_kvdim(a)*sizeof(float));
    float *v = malloc((size_t)full_kvdim(a)*sizeof(float));
    ot_linear(ly->wq, xn, qproj, full_qproj(a), H);
    ot_linear(ly->wk, xn, k, full_kvdim(a), H);
    ot_linear(ly->wv, xn, v, full_kvdim(a), H);
    split_q_gate(a, qproj, q, g);
    qk_norm_rope(q, nh, hd, a->rope_dim, ly->q_norm, s->pos, a->rope_theta, a->rms_norm_eps);
    qk_norm_rope(k, nkv, hd, a->rope_dim, ly->k_norm, s->pos, a->rope_theta, a->rms_norm_eps);
    float *o = malloc((size_t)full_qdim(a)*sizeof(float));
    ornith_gqa_step(&s->kv[L], q, k, v, nh, o, 0.0f);
    apply_out_gate(a, o, g);
    float *outp = malloc((size_t)H*sizeof(float));
    ot_linear(ly->wo, o, outp, H, full_qdim(a));
    ot_add_(x_acc, outp, H);
    free(qproj); free(q); free(g); free(k); free(v); free(o); free(outp);
}

static void linear_attn_decode(const of_model *m, const of_layer *ly,
                               of_state *s, int L, const float *xn, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nk=a->lin_key_heads, nv=a->lin_value_heads;
    int dk=a->lin_key_head_dim, dv=a->lin_value_head_dim, Cc=lin_cc(a);
    int qd=lin_qdim(a), vd=lin_vdim(a), group = nv/nk;

    float *cat = malloc((size_t)Cc*sizeof(float));
    ot_linear(ly->lq, xn, cat,            qd, H);
    ot_linear(ly->lk, xn, cat+qd,         qd, H);
    ot_linear(ly->lv, xn, cat+2*qd,       vd, H);
    float *conv = malloc((size_t)Cc*sizeof(float));
    ornith_conv_step(&s->conv[L], cat, ly->conv_w, ly->conv_b, conv);
    ot_silu_(conv, Cc);
    float *q = conv, *k = conv+qd, *v = conv+2*qd;

    float *a_in = malloc((size_t)nv*sizeof(float));
    float *b_in = malloc((size_t)nv*sizeof(float));
    ot_linear(ly->la, xn, a_in, nv, H);
    ot_linear(ly->lb, xn, b_in, nv, H);

    float *o_all = malloc((size_t)vd*sizeof(float));
    float *knorm = malloc((size_t)dk*sizeof(float));
    float *qnorm = malloc((size_t)dk*sizeof(float));
    for (int hv = 0; hv < nv; hv++) {
        int kh = hv / group;
        ot_l2norm_eps(k + (size_t)kh*dk, knorm, dk, 1e-6f);
        ot_l2norm_eps(q + (size_t)kh*dk, qnorm, dk, 1e-6f);
        float alpha = delta_decay(a_in[hv], ly->ssm_a[hv], ly->ssm_dt[hv]);
        ornith_delta_state st = { s->delta_S[L] + (size_t)hv*dv*dk, dk, dv };
        ornith_delta_step(&st, qnorm, knorm, v + (size_t)hv*dv,
                          alpha, ot_sigmoid(b_in[hv]),
                          o_all + (size_t)hv*dv);
    }
    delta_finish(m, ly, xn, o_all, x_acc);
    free(cat); free(conv); free(a_in); free(b_in);
    free(o_all); free(knorm); free(qnorm);
}

ornith_status of_forward_decode(const of_model *m, of_state *s,
                                int32_t token, float *logits) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, V = a->vocab_size;
    if (token < 0 || token >= V) {
        ornith_set_error("token %d out of range [0,%d)", token, V);
        return ORNITH_ERR_FORMAT;
    }
    if (s->pos >= s->capacity) {
        ornith_set_error("decode past capacity %d", s->capacity);
        return ORNITH_ERR_UNSUPPORTED;
    }
    float *x  = malloc((size_t)H*sizeof(float));
    float *xn = malloc((size_t)H*sizeof(float));
    memcpy(x, m->tok_embd + (size_t)token*H, (size_t)H*sizeof(float));

    for (int L = 0; L < a->num_layers; L++) {
        of_layer *ly = &m->layers[L];
        ot_rmsnorm(x, ly->attn_norm, xn, H, a->rms_norm_eps);
        if (ly->is_full) full_attn_decode(m, ly, s, L, xn, x);
        else             linear_attn_decode(m, ly, s, L, xn, x);
        moe_residual(m, ly, x, x);
    }
    s->pos++;

    ot_rmsnorm(x, m->final_norm, xn, H, a->rms_norm_eps);
    ot_linear(m->lm_head, xn, logits, V, H);
    free(x); free(xn);
    return ORNITH_OK;
}

/* ---- prefill: whole prompt -------------------------------------------- */

static void full_attn_prefill(const of_model *m, const of_layer *ly,
                              of_state *s, int L, const float *XN, int T,
                              float *X_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nh=a->num_attn_heads, nkv=a->num_kv_heads, hd=a->head_dim;
    int qd=full_qdim(a), qp=full_qproj(a), kvd=full_kvdim(a);
    float *QP= malloc((size_t)T*qp*sizeof(float));
    float *Q = malloc((size_t)T*qd*sizeof(float));
    float *G = malloc((size_t)T*qd*sizeof(float));
    float *K = malloc((size_t)T*kvd*sizeof(float));
    float *Vv= malloc((size_t)T*kvd*sizeof(float));
    ot_linear_batch(ly->wq, XN, QP, T, qp, H);
    ot_linear_batch(ly->wk, XN, K, T, kvd, H);
    ot_linear_batch(ly->wv, XN, Vv,T, kvd, H);
    for (int t = 0; t < T; t++) {
        int pos = s->pos + t;
        split_q_gate(a, QP + (size_t)t*qp, Q + (size_t)t*qd, G + (size_t)t*qd);
        qk_norm_rope(Q + (size_t)t*qd, nh, hd, a->rope_dim, ly->q_norm, pos,
                     a->rope_theta, a->rms_norm_eps);
        qk_norm_rope(K + (size_t)t*kvd, nkv, hd, a->rope_dim, ly->k_norm, pos,
                     a->rope_theta, a->rms_norm_eps);
    }
    /* fresh prefill assumes empty cache (pos==0); fill cache and attend. */
    float *O = malloc((size_t)T*qd*sizeof(float));
    ornith_gqa_reference(Q, K, Vv, T, nh, nkv, hd, O, 0.0f);
    /* persist K,V into the cache so later decode continues correctly. */
    memcpy(s->kv[L].K, K, (size_t)T*kvd*sizeof(float));
    memcpy(s->kv[L].V, Vv,(size_t)T*kvd*sizeof(float));
    s->kv[L].len = T;
    for (int t = 0; t < T; t++) {
        apply_out_gate(a, O + (size_t)t*qd, G + (size_t)t*qd);
        float *outp = malloc((size_t)H*sizeof(float));
        ot_linear(ly->wo, O + (size_t)t*qd, outp, H, qd);
        ot_add_(X_acc + (size_t)t*H, outp, H);
        free(outp);
    }
    free(QP); free(Q); free(G); free(K); free(Vv); free(O);
}

static void linear_attn_prefill(const of_model *m, const of_layer *ly,
                                of_state *s, int L, const float *XN, int T,
                                int chunk, float *X_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nk=a->lin_key_heads, nv=a->lin_value_heads;
    int dk=a->lin_key_head_dim, dv=a->lin_value_head_dim, Cc=lin_cc(a);
    int qd=lin_qdim(a), vd=lin_vdim(a), group = nv/nk;

    /* projections + fused qkv conv over the whole prompt */
    float *CAT  = malloc((size_t)T*Cc*sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *xn = XN + (size_t)t*H;
        ot_linear(ly->lq, xn, CAT + (size_t)t*Cc,          qd, H);
        ot_linear(ly->lk, xn, CAT + (size_t)t*Cc + qd,     qd, H);
        ot_linear(ly->lv, xn, CAT + (size_t)t*Cc + 2*qd,   vd, H);
    }
    float *CONV = malloc((size_t)T*Cc*sizeof(float));
    ornith_conv_prefill(&s->conv[L], CAT, ly->conv_w, ly->conv_b, T, CONV);
    ot_silu_(CONV, (int)((size_t)T*Cc));  /* fine: T*Cc small for reference */

    /* gates */
    float *A = malloc((size_t)T*nv*sizeof(float));
    float *B = malloc((size_t)T*nv*sizeof(float));
    ot_linear_batch(ly->la, XN, A, T, nv, H);
    ot_linear_batch(ly->lb, XN, B, T, nv, H);

    /* per value-head chunked delta scan */
    float *Qh = malloc((size_t)T*dk*sizeof(float));
    float *Kh = malloc((size_t)T*dk*sizeof(float));
    float *Vh = malloc((size_t)T*dv*sizeof(float));
    float *al = malloc((size_t)T*sizeof(float));
    float *be = malloc((size_t)T*sizeof(float));
    float *Oh = malloc((size_t)T*dv*sizeof(float));
    float *O_all = malloc((size_t)T*vd*sizeof(float));
    for (int hv = 0; hv < nv; hv++) {
        int kh = hv / group;
        for (int t = 0; t < T; t++) {
            const float *q = CONV + (size_t)t*Cc + kh*dk;
            const float *k = CONV + (size_t)t*Cc + qd + kh*dk;
            const float *v = CONV + (size_t)t*Cc + 2*qd + hv*dv;
            ot_l2norm_eps(q, Qh + (size_t)t*dk, dk, 1e-6f);
            ot_l2norm_eps(k, Kh + (size_t)t*dk, dk, 1e-6f);
            memcpy(Vh + (size_t)t*dv, v, (size_t)dv*sizeof(float));
            al[t] = delta_decay(A[(size_t)t*nv + hv], ly->ssm_a[hv], ly->ssm_dt[hv]);
            be[t] = ot_sigmoid(B[(size_t)t*nv + hv]);
        }
        ornith_delta_state st = { s->delta_S[L] + (size_t)hv*dv*dk, dk, dv };
        ornith_delta_prefill(&st, Qh, Kh, Vh, al, be, T, chunk, Oh);
        for (int t = 0; t < T; t++)
            memcpy(O_all + (size_t)t*vd + hv*dv, Oh + (size_t)t*dv,
                   (size_t)dv*sizeof(float));
    }
    for (int t = 0; t < T; t++)
        delta_finish(m, ly, XN + (size_t)t*H, O_all + (size_t)t*vd,
                     X_acc + (size_t)t*H);

    free(CAT); free(CONV); free(A); free(B);
    free(Qh); free(Kh); free(Vh); free(al); free(be); free(Oh); free(O_all);
}

ornith_status of_forward_prefill(const of_model *m, of_state *s,
                                 const int32_t *tokens, int T, int chunk,
                                 float *logits_all) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, V = a->vocab_size;
    if (chunk <= 0) chunk = 16;
    if (s->pos != 0) {
        ornith_set_error("prefill requires a fresh state");
        return ORNITH_ERR_UNSUPPORTED;
    }
    if (T > s->capacity) {
        ornith_set_error("prefill T=%d exceeds capacity %d", T, s->capacity);
        return ORNITH_ERR_UNSUPPORTED;
    }
    for (int t = 0; t < T; t++)
        if (tokens[t] < 0 || tokens[t] >= V) {
            ornith_set_error("token[%d]=%d out of range", t, tokens[t]);
            return ORNITH_ERR_FORMAT;
        }

    float *X  = malloc((size_t)T*H*sizeof(float));
    float *XN = malloc((size_t)T*H*sizeof(float));
    for (int t = 0; t < T; t++)
        memcpy(X + (size_t)t*H, m->tok_embd + (size_t)tokens[t]*H,
               (size_t)H*sizeof(float));

    for (int L = 0; L < a->num_layers; L++) {
        of_layer *ly = &m->layers[L];
        for (int t = 0; t < T; t++)
            ot_rmsnorm(X + (size_t)t*H, ly->attn_norm, XN + (size_t)t*H, H,
                       a->rms_norm_eps);
        if (ly->is_full) full_attn_prefill(m, ly, s, L, XN, T, X);
        else             linear_attn_prefill(m, ly, s, L, XN, T, chunk, X);
        for (int t = 0; t < T; t++)
            moe_residual(m, ly, X + (size_t)t*H, X + (size_t)t*H);
    }
    s->pos = T;

    for (int t = 0; t < T; t++) {
        ot_rmsnorm(X + (size_t)t*H, m->final_norm, XN + (size_t)t*H, H,
                   a->rms_norm_eps);
        ot_linear(m->lm_head, XN + (size_t)t*H, logits_all + (size_t)t*V, V, H);
    }
    free(X); free(XN);
    return ORNITH_OK;
}

int of_argmax(const float *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
}
