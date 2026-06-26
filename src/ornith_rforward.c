/* ornith_rforward.c — real-weight qwen3.5 forward with on-the-fly dequant. */
#include "ornith_rforward.h"
#include "ornith_gguf_write.h"
#include "ornith_quant.h"
#include "ornith_imatrix.h"
#include "ornith_qdot.h"
#include "ornith_tensor.h"
#include "ornith_sample.h"
#include "ornith_attn.h"
#include "ornith_moe.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

static int32_t argmax_f(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* ---- model container --------------------------------------------------- */

struct rmodel {
    ogguf_loaded l;
    ornith_arch  a;
    otokenizer   tok;
    int          nthreads;
    char        *name;   /* general.name, or NULL */
};

/* ---- KV metadata helpers ----------------------------------------------- */

enum { GT_U32 = 4, GT_I32 = 5, GT_F32 = 6 };

static const ogguf_lkv *kv_find(const ogguf_loaded *l, const char *key) {
    for (uint64_t i = 0; i < l->n_kv; i++)
        if (strcmp(l->kv[i].key, key) == 0) return &l->kv[i];
    return NULL;
}
static uint32_t kv_u32(const ogguf_loaded *l, const char *key, uint32_t def) {
    const ogguf_lkv *e = kv_find(l, key);
    if (e && (e->vtype == GT_U32 || e->vtype == GT_I32) && e->payload_len >= 4) {
        uint32_t v; memcpy(&v, e->payload, 4); return v;
    }
    return def;
}
static float kv_f32(const ogguf_loaded *l, const char *key, float def) {
    const ogguf_lkv *e = kv_find(l, key);
    if (e && e->vtype == GT_F32 && e->payload_len >= 4) {
        float v; memcpy(&v, e->payload, 4); return v;
    }
    return def;
}
/* A STRING KV payload is u64 length + bytes (the type tag is stored in vtype).
 * Returns a malloc'd, NUL-terminated copy, or NULL if absent/not a string. */
enum { GT_STR = 8 };
static char *kv_str(const ogguf_loaded *l, const char *key) {
    const ogguf_lkv *e = kv_find(l, key);
    if (!e || e->vtype != GT_STR || e->payload_len < 8) return NULL;
    uint64_t len; memcpy(&len, e->payload, 8);
    if (len > e->payload_len - 8) return NULL;
    char *s = malloc((size_t)len + 1);
    if (!s) return NULL;
    memcpy(s, e->payload + 8, (size_t)len);
    s[len] = '\0';
    return s;
}

/* ---- tensor lookup ----------------------------------------------------- */

static const ogguf_ltensor *T(const rmodel *m, const char *name) {
    for (uint64_t i = 0; i < m->l.n_tensors; i++)
        if (strcmp(m->l.tensors[i].name, name) == 0) return &m->l.tensors[i];
    return NULL;
}
static const ogguf_ltensor *Tb(const rmodel *m, int blk, const char *suffix) {
    char nm[128];
    snprintf(nm, sizeof(nm), "blk.%d.%s", blk, suffix);
    return T(m, nm);
}
/* F32 tensor payload as floats (norms, ssm_a/dt/norm, conv). */
static const float *f32(const ogguf_ltensor *t) { return (const float *)t->data; }

/* ---- threaded fused dequant-matvec ------------------------------------- */
/* W (out=dims[1], in=dims[0]) row-major [out,in]; y[out] = W @ x[in].
 * For each output row, the row is decoded to f32 then dotted with x. */

typedef struct {
    const ogguf_ltensor *W;
    const float *x;   /* [in] (matvec) or [T*in] (batch) — f32 fallback path  */
    const oq8k_block *xq; /* Tn columns of Q8_K-quantized x (NULL => f32 path) */
    float *y;         /* [out] or [T*out] */
    int in, out, Tn;  /* Tn columns of x to apply (1 for plain matvec) */
    int o0, o1;
} mv_job;

static void mv_run(mv_job *j) {
    const ogguf_ltensor *W = j->W;
    int in = j->in, Tn = j->Tn;
    size_t rb = oq_row_bytes(W->type, (size_t)in);

    /* Quant-aware integer path: the activation was already quantized to Q8_K
     * once (per column); dot the quantized weight bytes directly, no f32 row
     * materialization. */
    if (j->xq) {
        size_t nblk = (size_t)in / OQDOT_QK_K;   /* Q8_K blocks per column */
        for (int o = j->o0; o < j->o1; o++) {
            const void *wrow = (const uint8_t *)W->data + (size_t)o * rb;
            for (int t = 0; t < Tn; t++)
                qdot_vec_dot(W->type, in, &j->y[(size_t)t * j->out + o],
                             wrow, j->xq + (size_t)t * nblk);
        }
        return;
    }

    /* F32 fallback: dequantize the row (if needed) then float-dot. */
    int is_f32 = (W->type == OGGML_F32);
    float *row = is_f32 ? NULL : malloc((size_t)in * sizeof(float));
    for (int o = j->o0; o < j->o1; o++) {
        const float *w;
        if (is_f32) w = (const float *)W->data + (size_t)o * in;
        else {
            oq_dequantize(W->type, (const uint8_t *)W->data + (size_t)o * rb,
                          row, (size_t)in);
            w = row;
        }
        for (int t = 0; t < Tn; t++)
            j->y[(size_t)t * j->out + o] = ot_dot(w, j->x + (size_t)t * in, in);
    }
    free(row);
}

static void *mv_thread(void *p) { mv_run((mv_job *)p); return NULL; }

static void matmul_batch_q(const rmodel *m, const ogguf_ltensor *W,
                           const float *X, float *Y, int Tn) {
    int in  = (int)W->dims[0];
    int out = (int)W->dims[1];

    /* imatrix collection hook: when a collector is active, accumulate the
     * per-input-channel sum of squared activations for this weight. Cheap
     * (one pointer compare when off) and single-threaded (runs before the
     * worker threads spawn below). */
    oimatrix_on_matmul(W->name, X, in, Tn);

    /* If we have an integer vec_dot for this weight type, quantize each of the
     * Tn activation columns to Q8_K ONCE and reuse it across all output rows
     * (the big win for the per-token lm_head over ~248k rows). Otherwise fall
     * back to the dequant + f32 dot path. */
    /* ORNITH_NO_QDOT=1 forces the old dequant + f32-dot path (A/B parity /
     * logit-delta measurement against the quant-aware integer kernels). */
    static int qdot_off = -1;
    if (qdot_off < 0) {
        const char *e = getenv("ORNITH_NO_QDOT");
        qdot_off = (e && e[0] == '1') ? 1 : 0;
    }

    oq8k_block *XQ = NULL;
    if (!qdot_off && qdot_can(W->type, in)) {
        size_t nblk = (size_t)in / OQDOT_QK_K;
        XQ = malloc((size_t)Tn * nblk * sizeof(oq8k_block));
        if (XQ) {
            for (int t = 0; t < Tn; t++)
                qdot_quantize_row_q8_K(X + (size_t)t * in,
                                       XQ + (size_t)t * nblk, in);
        }
    }

    int nth = m->nthreads;
    if (nth > out) nth = out;
    if (nth < 1) nth = 1;
    pthread_t th[64];
    mv_job jobs[64];
    int per = (out + nth - 1) / nth;
    int n = 0;
    for (int i = 0; i < nth; i++) {
        int o0 = i * per, o1 = o0 + per;
        if (o0 >= out) break;
        if (o1 > out) o1 = out;
        jobs[n] = (mv_job){ W, X, XQ, Y, in, out, Tn, o0, o1 };
        n++;
    }
    if (n == 1) mv_run(&jobs[0]);
    else {
        for (int i = 0; i < n; i++) pthread_create(&th[i], NULL, mv_thread, &jobs[i]);
        for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    }
    free(XQ);
}
static void matvec_q(const rmodel *m, const ogguf_ltensor *W,
                     const float *x, float *y) {
    matmul_batch_q(m, W, x, y, 1);
}

/* gather one row (token embedding) from a [in, n_rows] tensor. */
static void embed_row(const ogguf_ltensor *W, int idx, float *out, int in) {
    if (W->type == OGGML_F32) {
        memcpy(out, (const float *)W->data + (size_t)idx * in,
               (size_t)in * sizeof(float));
    } else {
        size_t rb = oq_row_bytes(W->type, (size_t)in);
        oq_dequantize(W->type, (const uint8_t *)W->data + (size_t)idx * rb,
                      out, (size_t)in);
    }
}

/* ---- per-sequence state ------------------------------------------------ */

typedef struct {
    const ornith_arch *a;
    ornith_kv_cache   *kv;
    float            **delta_S;
    ornith_conv_state *conv;
    int pos;
} rstate;

static int lin_qd(const ornith_arch *a){ return a->lin_key_heads*a->lin_key_head_dim; }
static int lin_vd(const ornith_arch *a){ return a->lin_value_heads*a->lin_value_head_dim; }
static int lin_cc_(const ornith_arch *a){ return 2*lin_qd(a)+lin_vd(a); }

/* ORNITH_KV_Q8=1 stores the full-attention KV cache as int8 + per-(token,head)
 * scale instead of fp32, roughly quartering its footprint (the linear layers'
 * recurrent state is unaffected). Default (unset/0) keeps the exact fp32 path. */
static int kv_q8_enabled(void) {
    const char *e = getenv("ORNITH_KV_Q8");
    return e && e[0] && e[0] != '0';
}

/* Allocate the rstate container and its per-layer arrays, but leave the
 * individual KV caches / conv states / delta states uninitialized (zeroed). The
 * caller fills them (rstate_new for a fresh run; the loader for a restore).
 * rstate_free is safe on a partially-initialized skeleton (freeing zeroed
 * structs / NULL pointers is a no-op). */
static rstate *rstate_skeleton(const ornith_arch *a) {
    rstate *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->a = a; s->pos = 0;
    s->kv = calloc((size_t)a->num_layers, sizeof(ornith_kv_cache));
    s->conv = calloc((size_t)a->num_layers, sizeof(ornith_conv_state));
    s->delta_S = calloc((size_t)a->num_layers, sizeof(float *));
    if (!s->kv || !s->conv || !s->delta_S) {
        free(s->kv); free(s->conv); free(s->delta_S); free(s);
        return NULL;
    }
    return s;
}

static rstate *rstate_new(const ornith_arch *a, int cap) {
    rstate *s = rstate_skeleton(a);
    if (!s) return NULL;
    int kv_q8 = kv_q8_enabled();
    int dk=a->lin_key_head_dim, dv=a->lin_value_head_dim, nv=a->lin_value_heads;
    for (int L = 0; L < a->num_layers; L++) {
        if (ornith_layer_is_full_attn(a, L))
            ornith_kv_init_ex(&s->kv[L], cap, a->num_kv_heads, a->head_dim, kv_q8);
        else {
            ornith_conv_init(&s->conv[L], a->lin_conv_kernel, lin_cc_(a));
            s->delta_S[L] = calloc((size_t)nv*dv*dk, sizeof(float));
        }
    }
    return s;
}
static void rstate_free(rstate *s) {
    if (!s) return;
    for (int L = 0; L < s->a->num_layers; L++) {
        ornith_kv_free(&s->kv[L]);
        ornith_conv_free(&s->conv[L]);
        free(s->delta_S[L]);
    }
    free(s->kv); free(s->conv); free(s->delta_S); free(s);
}

/* ---- layer math -------------------------------------------------------- */

/* gated-delta decay. NOTE: in the GGUF the ssm_a tensor is stored ALREADY as
 * -exp(A_log) (the negative coefficient), so the decay is exp(a_coef *
 * softplus(a_in + dt_bias)) using a_coef directly (no second exp/negate). This
 * matches llama.cpp's qwen35 graph (gate = softplus(alpha+dt) * ssm_a). */
static float decay_alpha(float a_in, float a_coef, float dt_bias) {
    return expf(a_coef * ot_softplus(a_in + dt_bias));
}

/* Default chunk size for the linear-attention chunked parallel-scan prefill.
 * Modest so intra-chunk buffers stay small (the Metal kernel mirrors this). */
#define RFWD_PREFILL_CHUNK 32

/* Gated delta-net output post-processing for ONE token, shared by the step
 * (decode) path and the chunked (prefill) path so the tail math can never
 * drift between them:
 *   1) scale the raw per-head delta output by 1/sqrt(dk) (NOT a no-op: the
 *      gated RMSNorm's eps regularizes small heads, so absolute scale matters,
 *      matching llama.cpp ggml_gated_delta_net);
 *   2) per-head RMSNorm with ssm_norm;
 *   3) multiply by SiLU(attn_gate(xn));
 *   4) out-projection (ssm_out), accumulated into x_acc.
 * o_all is [vd] for this token and is overwritten in place by the scale. */
static void linear_finish(const rmodel *m, int L, const float *xn,
                          float *o_all, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, nv = a->lin_value_heads;
    int dv = a->lin_value_head_dim, dk = a->lin_key_head_dim, vd = lin_vd(a);
    ot_scale_(o_all, 1.0f / sqrtf((float)dk), vd);
    float *gate = malloc((size_t)vd*sizeof(float));
    matvec_q(m, Tb(m, L, "attn_gate.weight"), xn, gate);
    ot_silu_(gate, vd);
    float *o = malloc((size_t)vd*sizeof(float));
    const float *snorm = f32(Tb(m, L, "ssm_norm.weight"));
    for (int hv = 0; hv < nv; hv++)
        ot_rmsnorm(o_all + (size_t)hv*dv, snorm, o + (size_t)hv*dv, dv,
                   a->rms_norm_eps);
    ot_mul_(o, gate, vd);
    float *outp = malloc((size_t)H*sizeof(float));
    matvec_q(m, Tb(m, L, "ssm_out.weight"), o, outp);
    ot_add_(x_acc, outp, H);
    free(gate); free(o); free(outp);
}

/* dense SwiGLU FFN (pre-norm = post_attention_norm), accumulate into x. */
static void ffn_dense(const rmodel *m, int L, const float *x, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, I = a->shared_inter_size;
    float *xn = malloc((size_t)H*sizeof(float));
    ot_rmsnorm(x, f32(Tb(m, L, "post_attention_norm.weight")), xn, H, a->rms_norm_eps);
    float *g = malloc((size_t)I*sizeof(float));
    float *u = malloc((size_t)I*sizeof(float));
    matvec_q(m, Tb(m, L, "ffn_gate.weight"), xn, g);
    matvec_q(m, Tb(m, L, "ffn_up.weight"),   xn, u);
    for (int i = 0; i < I; i++) g[i] = ot_silu1(g[i]) * u[i];
    float *o = malloc((size_t)H*sizeof(float));
    matvec_q(m, Tb(m, L, "ffn_down.weight"), g, o);
    ot_add_(x_acc, o, H);
    free(xn); free(g); free(u); free(o);
}

/* One expert's 2D weight as a stack tensor view into a 3D expert stack.
 * The stack is [.., out, in] with `in` (dims[0]) contiguous; expert e begins at
 * e * out rows of oq_row_bytes(type, in) bytes. */
static ogguf_ltensor expert_view(const ogguf_ltensor *W, int e, int in, int out) {
    ogguf_ltensor v = *W;
    v.n_dims = 2; v.dims[0] = (uint64_t)in; v.dims[1] = (uint64_t)out;
    size_t rb = oq_row_bytes(W->type, (size_t)in);
    v.data = (uint8_t *)W->data + (size_t)e * (size_t)out * rb;
    return v;
}

/* MoE SwiGLU FFN: router top-k over routed experts + always-on shared expert.
 * Mirrors the tested ornith_moe logic (router -> top-k softmax -> per-expert
 * SwiGLU -> weighted sum -> shared expert) but on real quantized weights via the
 * quant-aware matvec. Expert tensors are 3D [hidden, inter, n_expert] (gate/up)
 * and [inter, hidden, n_expert] (down); sliced per expert with expert_view. */
static void ffn_moe(const rmodel *m, int L, const float *x, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, I = a->moe_inter_size;
    int ne = a->n_routed_experts, K = a->experts_per_tok;
    float *xn = malloc((size_t)H*sizeof(float));
    ot_rmsnorm(x, f32(Tb(m, L, "post_attention_norm.weight")), xn, H, a->rms_norm_eps);

    const ogguf_ltensor *router = Tb(m, L, "ffn_gate_inp.weight");  /* [H, ne] */
    float *logits = malloc((size_t)ne*sizeof(float));
    matvec_q(m, router, xn, logits);
    int *idx = malloc((size_t)K*sizeof(int));
    float *w = malloc((size_t)K*sizeof(float));
    ornith_moe_route(logits, ne, K, idx, w);

    const ogguf_ltensor *ge = Tb(m, L, "ffn_gate_exps.weight");
    const ogguf_ltensor *ue = Tb(m, L, "ffn_up_exps.weight");
    const ogguf_ltensor *de = Tb(m, L, "ffn_down_exps.weight");
    float *g = malloc((size_t)I*sizeof(float));
    float *u = malloc((size_t)I*sizeof(float));
    float *o = malloc((size_t)H*sizeof(float));
    float *acc = calloc((size_t)H, sizeof(float));
    for (int s = 0; s < K; s++) {
        int e = idx[s];
        ogguf_ltensor gv = expert_view(ge, e, H, I);
        ogguf_ltensor uv = expert_view(ue, e, H, I);
        ogguf_ltensor dv = expert_view(de, e, I, H);
        matvec_q(m, &gv, xn, g);
        matvec_q(m, &uv, xn, u);
        for (int i = 0; i < I; i++) g[i] = ot_silu1(g[i]) * u[i];
        matvec_q(m, &dv, g, o);
        ot_addscaled_(acc, o, w[s], H);
    }

    /* always-on shared expert (+ optional sigmoid gate ffn_gate_inp_shexp) */
    const ogguf_ltensor *sgate_w = Tb(m, L, "ffn_gate_shexp.weight");
    if (sgate_w) {
        int sI = a->shared_inter_size;
        float *sg = malloc((size_t)sI*sizeof(float));
        float *su = malloc((size_t)sI*sizeof(float));
        float *so = malloc((size_t)H*sizeof(float));
        matvec_q(m, sgate_w, xn, sg);
        matvec_q(m, Tb(m, L, "ffn_up_shexp.weight"), xn, su);
        for (int i = 0; i < sI; i++) sg[i] = ot_silu1(sg[i]) * su[i];
        matvec_q(m, Tb(m, L, "ffn_down_shexp.weight"), sg, so);
        const ogguf_ltensor *sgi = Tb(m, L, "ffn_gate_inp_shexp.weight"); /* [H]->1 */
        if (sgi) {
            ogguf_ltensor gv = *sgi; gv.n_dims = 2; gv.dims[0] = (uint64_t)H; gv.dims[1] = 1;
            float gl = 0.0f; matvec_q(m, &gv, xn, &gl);
            float gs = ot_sigmoid(gl);
            for (int i = 0; i < H; i++) so[i] *= gs;
        }
        ot_add_(acc, so, H);
        free(sg); free(su); free(so);
    }

    ot_add_(x_acc, acc, H);
    free(xn); free(logits); free(idx); free(w); free(g); free(u); free(o); free(acc);
}

/* FFN dispatch: MoE when the model has routed experts, else dense. */
static void ffn_layer(const rmodel *m, int L, const float *x, float *x_acc) {
    if (m->a.n_routed_experts > 0) ffn_moe(m, L, x, x_acc);
    else                           ffn_dense(m, L, x, x_acc);
}

static void full_attn(const rmodel *m, rstate *s, int L,
                      const float *xn, float *x_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nh=a->num_attn_heads, nkv=a->num_kv_heads, hd=a->head_dim;
    int qd=nh*hd, kvd=nkv*hd, qp=2*qd;  /* attn_output_gate: q packs [q|gate] */
    float *qproj = malloc((size_t)qp*sizeof(float));
    float *q = malloc((size_t)qd*sizeof(float));
    float *g = malloc((size_t)qd*sizeof(float));
    float *k = malloc((size_t)kvd*sizeof(float));
    float *v = malloc((size_t)kvd*sizeof(float));
    matvec_q(m, Tb(m, L, "attn_q.weight"), xn, qproj);
    matvec_q(m, Tb(m, L, "attn_k.weight"), xn, k);
    matvec_q(m, Tb(m, L, "attn_v.weight"), xn, v);
    for (int h = 0; h < nh; h++) {
        memcpy(q + (size_t)h*hd, qproj + (size_t)h*2*hd,      (size_t)hd*sizeof(float));
        memcpy(g + (size_t)h*hd, qproj + (size_t)h*2*hd + hd, (size_t)hd*sizeof(float));
    }
    const float *qn = f32(Tb(m, L, "attn_q_norm.weight"));
    const float *kn = f32(Tb(m, L, "attn_k_norm.weight"));
    for (int h = 0; h < nh; h++) {
        float *vh = q + (size_t)h*hd;
        ot_rmsnorm(vh, qn, vh, hd, a->rms_norm_eps);
        ot_rope_partial(vh, hd, a->rope_dim, s->pos, a->rope_theta);
    }
    for (int h = 0; h < nkv; h++) {
        float *vh = k + (size_t)h*hd;
        ot_rmsnorm(vh, kn, vh, hd, a->rms_norm_eps);
        ot_rope_partial(vh, hd, a->rope_dim, s->pos, a->rope_theta);
    }
    float *o = malloc((size_t)qd*sizeof(float));
    ornith_gqa_step(&s->kv[L], q, k, v, nh, o, 0.0f);
    for (int i = 0; i < qd; i++) o[i] *= ot_sigmoid(g[i]);
    float *outp = malloc((size_t)H*sizeof(float));
    matvec_q(m, Tb(m, L, "attn_output.weight"), o, outp);
    ot_add_(x_acc, outp, H);
    free(qproj); free(q); free(g); free(k); free(v); free(o); free(outp);
}

static void linear_attn(const rmodel *m, rstate *s, int L,
                        const float *xn, float *x_acc) {
    const ornith_arch *a = &m->a;
    int nk=a->lin_key_heads, nv=a->lin_value_heads;
    int dk=a->lin_key_head_dim, dv=a->lin_value_head_dim;
    int qd=lin_qd(a), vd=lin_vd(a), Cc=lin_cc_(a), group=nv/nk;

    float *cat  = malloc((size_t)Cc*sizeof(float));
    matvec_q(m, Tb(m, L, "attn_qkv.weight"), xn, cat);
    /* causal depthwise conv (no bias) over Cc channels, then SiLU */
    float *conv = malloc((size_t)Cc*sizeof(float));
    ornith_conv_step(&s->conv[L], cat, f32(Tb(m, L, "ssm_conv1d.weight")), NULL, conv);
    ot_silu_(conv, Cc);
    float *q = conv, *k = conv + qd, *v = conv + 2*qd;

    /* L2-normalize each key head's q and k over the head dim */
    float *qn = malloc((size_t)qd*sizeof(float));
    float *kn = malloc((size_t)qd*sizeof(float));
    for (int h = 0; h < nk; h++) {
        ot_l2norm_eps(q + (size_t)h*dk, qn + (size_t)h*dk, dk, 1e-6f);
        ot_l2norm_eps(k + (size_t)h*dk, kn + (size_t)h*dk, dk, 1e-6f);
    }
    /* gating */
    float *a_in = malloc((size_t)nv*sizeof(float));
    float *b_in = malloc((size_t)nv*sizeof(float));
    matvec_q(m, Tb(m, L, "ssm_alpha.weight"), xn, a_in);
    matvec_q(m, Tb(m, L, "ssm_beta.weight"),  xn, b_in);
    const float *A_log = f32(Tb(m, L, "ssm_a"));
    const float *dtb   = f32(Tb(m, L, "ssm_dt.bias"));

    float *o_all = malloc((size_t)vd*sizeof(float));
    for (int hv = 0; hv < nv; hv++) {
        /* llama.cpp repeats k-heads to v-heads with ggml_repeat (tiling), so
         * value head hv pairs with key head (hv % n_k_heads). */
        int kh = hv % nk;
        (void)group;
        float alpha = decay_alpha(a_in[hv], A_log[hv], dtb[hv]);
        ornith_delta_state st = { s->delta_S[L] + (size_t)hv*dv*dk, dk, dv };
        ornith_delta_step(&st, qn + (size_t)kh*dk, kn + (size_t)kh*dk,
                          v + (size_t)hv*dv, alpha, ot_sigmoid(b_in[hv]),
                          o_all + (size_t)hv*dv);
    }
    /* gated RMSNorm + SiLU gate + out-projection (shared with prefill). */
    linear_finish(m, L, xn, o_all, x_acc);
    free(cat); free(conv); free(qn); free(kn); free(a_in); free(b_in);
    free(o_all);
}

/* ======================================================================= */
/*  Whole-prompt prefill (batched projections; chunked scan for SSM layers) */
/* ======================================================================= */

/* Full-attention prefill: identical math to T sequential decode steps (each
 * token's K,V is appended and it attends causally over the cache), but the
 * projections and the output projection are batched matvecs. `base` is the
 * starting position for RoPE (== s->pos at entry). */
static void full_attn_prefill(const rmodel *m, rstate *s, int L,
                              const float *XN, int T, int base, float *X_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nh=a->num_attn_heads, nkv=a->num_kv_heads, hd=a->head_dim;
    int qd=nh*hd, kvd=nkv*hd, qp=2*qd;
    float *QP = malloc((size_t)T*qp*sizeof(float));
    float *K  = malloc((size_t)T*kvd*sizeof(float));
    float *Vv = malloc((size_t)T*kvd*sizeof(float));
    matmul_batch_q(m, Tb(m, L, "attn_q.weight"), XN, QP, T);
    matmul_batch_q(m, Tb(m, L, "attn_k.weight"), XN, K,  T);
    matmul_batch_q(m, Tb(m, L, "attn_v.weight"), XN, Vv, T);
    const float *qn = f32(Tb(m, L, "attn_q_norm.weight"));
    const float *kn = f32(Tb(m, L, "attn_k_norm.weight"));
    float *q = malloc((size_t)qd*sizeof(float));
    float *G = malloc((size_t)T*qd*sizeof(float));   /* per-token output gates */
    float *O = malloc((size_t)T*qd*sizeof(float));
    for (int t = 0; t < T; t++) {
        int pos = base + t;
        const float *qpt = QP + (size_t)t*qp;
        float *gt = G + (size_t)t*qd;
        for (int h = 0; h < nh; h++) {
            memcpy(q  + (size_t)h*hd, qpt + (size_t)h*2*hd,      (size_t)hd*sizeof(float));
            memcpy(gt + (size_t)h*hd, qpt + (size_t)h*2*hd + hd, (size_t)hd*sizeof(float));
        }
        for (int h = 0; h < nh; h++) {
            float *vh = q + (size_t)h*hd;
            ot_rmsnorm(vh, qn, vh, hd, a->rms_norm_eps);
            ot_rope_partial(vh, hd, a->rope_dim, pos, a->rope_theta);
        }
        float *kt = K + (size_t)t*kvd;
        for (int h = 0; h < nkv; h++) {
            float *vh = kt + (size_t)h*hd;
            ot_rmsnorm(vh, kn, vh, hd, a->rms_norm_eps);
            ot_rope_partial(vh, hd, a->rope_dim, pos, a->rope_theta);
        }
        ornith_gqa_step(&s->kv[L], q, kt, Vv + (size_t)t*kvd, nh,
                        O + (size_t)t*qd, 0.0f);
    }
    for (size_t i = 0; i < (size_t)T*qd; i++) O[i] *= ot_sigmoid(G[i]);
    float *OUTP = malloc((size_t)T*H*sizeof(float));
    matmul_batch_q(m, Tb(m, L, "attn_output.weight"), O, OUTP, T);
    for (int t = 0; t < T; t++) ot_add_(X_acc + (size_t)t*H, OUTP + (size_t)t*H, H);
    free(QP); free(K); free(Vv); free(q); free(G); free(O); free(OUTP);
}

/* Linear (SSM gated-delta-net) prefill via the chunked parallel-scan. The conv
 * and the delta recurrence advance their state across the whole prompt exactly
 * as T decode steps would (conv_prefill == T conv_steps; delta_prefill == T
 * delta_steps within fp tolerance). Gating is computed with the SAME helpers as
 * decode (decay_alpha, ot_sigmoid, ot_l2norm_eps, kh = hv % nk). */
static void linear_attn_prefill(const rmodel *m, rstate *s, int L,
                                const float *XN, int T, int chunk, float *X_acc) {
    const ornith_arch *a = &m->a;
    int H=a->hidden_size, nk=a->lin_key_heads, nv=a->lin_value_heads;
    int dk=a->lin_key_head_dim, dv=a->lin_value_head_dim, Cc=lin_cc_(a);
    int qd=lin_qd(a), vd=lin_vd(a);

    /* fused qkv projection over the whole prompt, then causal conv + SiLU */
    float *CAT  = malloc((size_t)T*Cc*sizeof(float));
    matmul_batch_q(m, Tb(m, L, "attn_qkv.weight"), XN, CAT, T);
    float *CONV = malloc((size_t)T*Cc*sizeof(float));
    ornith_conv_prefill(&s->conv[L], CAT, f32(Tb(m, L, "ssm_conv1d.weight")),
                        NULL, T, CONV);
    ot_silu_(CONV, (int)((size_t)T*Cc));

    /* gating projections over the whole prompt */
    float *A = malloc((size_t)T*nv*sizeof(float));
    float *B = malloc((size_t)T*nv*sizeof(float));
    matmul_batch_q(m, Tb(m, L, "ssm_alpha.weight"), XN, A, T);
    matmul_batch_q(m, Tb(m, L, "ssm_beta.weight"),  XN, B, T);
    const float *A_log = f32(Tb(m, L, "ssm_a"));
    const float *dtb   = f32(Tb(m, L, "ssm_dt.bias"));

    /* per value-head chunked delta scan */
    float *Qh   = malloc((size_t)T*dk*sizeof(float));
    float *Kh   = malloc((size_t)T*dk*sizeof(float));
    float *Vh   = malloc((size_t)T*dv*sizeof(float));
    float *al   = malloc((size_t)T*sizeof(float));
    float *be   = malloc((size_t)T*sizeof(float));
    float *Oh   = malloc((size_t)T*dv*sizeof(float));
    float *O_all= malloc((size_t)T*vd*sizeof(float));
    for (int hv = 0; hv < nv; hv++) {
        int kh = hv % nk;   /* k-heads repeated/tiled onto v-heads (llama.cpp) */
        for (int t = 0; t < T; t++) {
            const float *q = CONV + (size_t)t*Cc + kh*dk;
            const float *k = CONV + (size_t)t*Cc + qd + kh*dk;
            const float *v = CONV + (size_t)t*Cc + 2*qd + hv*dv;
            ot_l2norm_eps(q, Qh + (size_t)t*dk, dk, 1e-6f);
            ot_l2norm_eps(k, Kh + (size_t)t*dk, dk, 1e-6f);
            memcpy(Vh + (size_t)t*dv, v, (size_t)dv*sizeof(float));
            al[t] = decay_alpha(A[(size_t)t*nv + hv], A_log[hv], dtb[hv]);
            be[t] = ot_sigmoid(B[(size_t)t*nv + hv]);
        }
        ornith_delta_state st = { s->delta_S[L] + (size_t)hv*dv*dk, dk, dv };
        ornith_delta_prefill(&st, Qh, Kh, Vh, al, be, T, chunk, Oh);
        for (int t = 0; t < T; t++)
            memcpy(O_all + (size_t)t*vd + hv*dv, Oh + (size_t)t*dv,
                   (size_t)dv*sizeof(float));
    }
    for (int t = 0; t < T; t++)
        linear_finish(m, L, XN + (size_t)t*H, O_all + (size_t)t*vd,
                      X_acc + (size_t)t*H);

    free(CAT); free(CONV); free(A); free(B);
    free(Qh); free(Kh); free(Vh); free(al); free(be); free(Oh); free(O_all);
}

/* Dense SwiGLU FFN over T tokens (batched matvecs). X and X_acc are the same
 * residual buffer; equals T calls to ffn_dense. */
static void ffn_dense_prefill(const rmodel *m, int L, const float *X, int T,
                              float *X_acc) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size, I = a->shared_inter_size;
    const float *pn = f32(Tb(m, L, "post_attention_norm.weight"));
    float *XN = malloc((size_t)T*H*sizeof(float));
    for (int t = 0; t < T; t++)
        ot_rmsnorm(X + (size_t)t*H, pn, XN + (size_t)t*H, H, a->rms_norm_eps);
    float *Gp = malloc((size_t)T*I*sizeof(float));
    float *Up = malloc((size_t)T*I*sizeof(float));
    matmul_batch_q(m, Tb(m, L, "ffn_gate.weight"), XN, Gp, T);
    matmul_batch_q(m, Tb(m, L, "ffn_up.weight"),   XN, Up, T);
    for (size_t i = 0; i < (size_t)T*I; i++) Gp[i] = ot_silu1(Gp[i]) * Up[i];
    float *O = malloc((size_t)T*H*sizeof(float));
    matmul_batch_q(m, Tb(m, L, "ffn_down.weight"), Gp, O, T);
    for (int t = 0; t < T; t++) ot_add_(X_acc + (size_t)t*H, O + (size_t)t*H, H);
    free(XN); free(Gp); free(Up); free(O);
}

/* Prefill the whole prompt layer-by-layer. Linear layers use the chunked
 * parallel-scan; full-attention layers use the (batched) sequential pass.
 * Writes the final-position logits into `logits` if non-NULL. Numerically
 * equal (within fp tolerance) to feeding the tokens one at a time through
 * rforward_token, because tokens interact only through the sequential conv /
 * delta / KV state, which is built in token order either way. */
static void rforward_prefill(rmodel *m, rstate *s, const int32_t *tokens,
                             int nt, int chunk, float *logits) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size;
    int base = s->pos;
    const ogguf_ltensor *embd = T(m, "token_embd.weight");
    float *X  = malloc((size_t)nt*H*sizeof(float));
    float *XN = malloc((size_t)nt*H*sizeof(float));
    for (int t = 0; t < nt; t++)
        embed_row(embd, tokens[t], X + (size_t)t*H, H);

    for (int L = 0; L < a->num_layers; L++) {
        const float *an = f32(Tb(m, L, "attn_norm.weight"));
        for (int t = 0; t < nt; t++)
            ot_rmsnorm(X + (size_t)t*H, an, XN + (size_t)t*H, H, a->rms_norm_eps);
        if (ornith_layer_is_full_attn(a, L))
            full_attn_prefill(m, s, L, XN, nt, base, X);
        else
            linear_attn_prefill(m, s, L, XN, nt, chunk, X);
        if (a->n_routed_experts > 0)            /* MoE routes per token */
            for (int t = 0; t < nt; t++) ffn_moe(m, L, X + (size_t)t*H, X + (size_t)t*H);
        else
            ffn_dense_prefill(m, L, X, nt, X);
    }
    s->pos = base + nt;

    if (logits) {
        const ogguf_ltensor *onorm = T(m, "output_norm.weight");
        const ogguf_ltensor *ohead = T(m, "output.weight");
        float *xn = malloc((size_t)H*sizeof(float));
        ot_rmsnorm(X + (size_t)(nt-1)*H, f32(onorm), xn, H, a->rms_norm_eps);
        matvec_q(m, ohead, xn, logits);
        free(xn);
    }
    free(X); free(XN);
}

/* One token through the whole stack. If logits != NULL, also computes the
 * final norm + lm_head for this position. */
static void rforward_token(rmodel *m, rstate *s, int32_t token, float *logits) {
    const ornith_arch *a = &m->a;
    int H = a->hidden_size;
    float *x  = malloc((size_t)H*sizeof(float));
    float *xn = malloc((size_t)H*sizeof(float));
    embed_row(T(m, "token_embd.weight"), token, x, H);

    for (int L = 0; L < a->num_layers; L++) {
        ot_rmsnorm(x, f32(Tb(m, L, "attn_norm.weight")), xn, H, a->rms_norm_eps);
        if (ornith_layer_is_full_attn(a, L)) full_attn(m, s, L, xn, x);
        else                                 linear_attn(m, s, L, xn, x);
        ffn_layer(m, L, x, x);
    }
    s->pos++;

    if (logits) {
        ot_rmsnorm(x, f32(T(m, "output_norm.weight")), xn, H, a->rms_norm_eps);
        matvec_q(m, T(m, "output.weight"), xn, logits);
    }
    free(x); free(xn);
}

/* ---- load / free ------------------------------------------------------- */

static long ncpus(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#else
    return 4;
#endif
}

ornith_status rmodel_load(const char *path, rmodel **out) {
    rmodel *m = calloc(1, sizeof(*m));
    if (!m) { ornith_set_error("oom"); return ORNITH_ERR_OOM; }
    ornith_status st = ogguf_load(path, &m->l);
    if (st != ORNITH_OK) { free(m); return st; }

    ornith_arch *a = &m->a;
    memset(a, 0, sizeof(*a));

    /* Metadata key prefix follows general.architecture: "qwen35" (dense 9B) or
     * "qwen35moe" (MoE 35B/397B). Structural dims are DERIVED FROM TENSOR SHAPES
     * (robust against metadata-key drift between the dense and MoE archs);
     * scalars (rope/eps) come from metadata under the detected prefix. */
    char *archs = kv_str(&m->l, "general.architecture");
    snprintf(a->model_type, sizeof(a->model_type), "%s", archs ? archs : "qwen35");
    char pfx[64]; snprintf(pfx, sizeof(pfx), "%s", archs ? archs : "qwen35");
    free(archs);
    char kb[160];
    #define MK(suffix) (snprintf(kb, sizeof(kb), "%s.%s", pfx, suffix), kb)

    const ogguf_ltensor *te = T(m, "token_embd.weight");
    a->hidden_size = te ? (int)te->dims[0] : (int)kv_u32(&m->l, MK("embedding_length"), 4096);
    a->vocab_size  = te ? (int)te->dims[1] : 248320;

    /* layer count: highest blk.N with an attn_norm */
    { char nm[64]; int n = 0;
      for (;;) { snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", n); if (!T(m, nm)) break; n++; }
      a->num_layers = n ? n : (int)kv_u32(&m->l, MK("block_count"), 32); }

    /* full-attention interval: first layer carrying attn_q (pattern i%I==I-1) */
    { char nm[64]; int ff = -1;
      for (int i = 0; i < a->num_layers; i++) {
          snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", i);
          if (T(m, nm)) { ff = i; break; }
      }
      a->full_attn_interval = ff >= 0 ? ff + 1
                                      : (int)kv_u32(&m->l, MK("full_attention_interval"), 4); }

    a->rope_theta   = kv_f32(&m->l, MK("rope.freq_base"), 1e7f);
    a->rms_norm_eps = kv_f32(&m->l, MK("attention.layer_norm_rms_epsilon"), 1e-6f);
    a->rope_dim     = (int)kv_u32(&m->l, MK("rope.dimension_count"), 64);
    a->attn_output_gate = true;

    /* full-attention head geometry from the first full layer (attn_q packs
     * [query|gate] when attn_output_gate, so q_out = 2*nh*hd). */
    { int ff = a->full_attn_interval - 1;
      const ogguf_ltensor *aqn = Tb(m, ff, "attn_q_norm.weight");
      const ogguf_ltensor *aq  = Tb(m, ff, "attn_q.weight");
      const ogguf_ltensor *ak  = Tb(m, ff, "attn_k.weight");
      a->head_dim = aqn ? (int)aqn->dims[0] : 256;
      int q_out = aq ? (int)aq->dims[1] : a->head_dim * 16;
      a->num_attn_heads = (q_out / (a->attn_output_gate ? 2 : 1)) / a->head_dim;
      a->num_kv_heads = ak ? (int)ak->dims[1] / a->head_dim : 4; }

    /* linear (SSM) geometry from the first linear layer */
    { int l0 = 0;
      while (l0 < a->num_layers && ornith_layer_is_full_attn(a, l0)) l0++;
      const ogguf_ltensor *ssa = Tb(m, l0, "ssm_a");
      const ogguf_ltensor *ssn = Tb(m, l0, "ssm_norm.weight");
      const ogguf_ltensor *qkv = Tb(m, l0, "attn_qkv.weight");
      const ogguf_ltensor *cv  = Tb(m, l0, "ssm_conv1d.weight");
      a->lin_value_heads = ssa ? (int)ssa->dims[0] : 32;
      a->lin_value_head_dim = ssn ? (int)ssn->dims[0] : 128;
      a->lin_key_head_dim   = a->lin_value_head_dim;
      a->lin_conv_kernel = cv ? (int)cv->dims[0] : 4;
      int Cc = qkv ? (int)qkv->dims[1] : 8192;
      int vd = a->lin_value_heads * a->lin_value_head_dim;
      int qd = (Cc - vd) / 2;
      a->lin_key_heads = a->lin_key_head_dim ? qd / a->lin_key_head_dim : 16; }

    /* MoE vs dense: presence of routed-expert tensors. */
    const ogguf_ltensor *ge0 = Tb(m, 0, "ffn_gate_exps.weight");
    if (ge0) {
        a->n_routed_experts = (int)ge0->dims[2];   /* [hidden, inter, n_expert] */
        a->moe_inter_size   = (int)ge0->dims[1];
        a->experts_per_tok  = (int)kv_u32(&m->l, MK("expert_used_count"), 8);
        const ogguf_ltensor *sh = Tb(m, 0, "ffn_gate_shexp.weight");
        a->shared_inter_size = sh ? (int)sh->dims[1] : 0;
    } else {
        a->n_routed_experts = 0;
        a->experts_per_tok  = 0;
        const ogguf_ltensor *fg = Tb(m, 0, "ffn_gate.weight");
        a->shared_inter_size = fg ? (int)fg->dims[1]
                                  : (int)kv_u32(&m->l, MK("feed_forward_length"), 12288);
    }
    #undef MK

    st = otok_load_from_gguf(&m->l, &m->tok);
    if (st != ORNITH_OK) { ogguf_loaded_free(&m->l); free(m); return st; }
    a->eos_token_id = m->tok.eos_id;
    a->bos_token_id = m->tok.bos_id;

    m->nthreads = (int)ncpus();
    if (m->nthreads > 64) m->nthreads = 64;
    m->name = kv_str(&m->l, "general.name");
    *out = m;
    return ORNITH_OK;
}

void rmodel_free(rmodel *m) {
    if (!m) return;
    free(m->name);
    otok_free(&m->tok);
    ogguf_loaded_free(&m->l);
    free(m);
}

const char *rmodel_name(const rmodel *m) { return m->name; }

const ornith_arch *rmodel_arch(const rmodel *m) { return &m->a; }
const otokenizer  *rmodel_tokenizer(const rmodel *m) { return &m->tok; }

/* ---- generation -------------------------------------------------------- */

/* Prompt prefill dispatch. Default: the chunked parallel-scan path (linear
 * layers use ornith_delta_prefill / ornith_conv_prefill). Set
 * ORNITH_PREFILL_SERIAL=1 to fall back to the old token-by-token recurrence
 * (for A/B parity checks). Set ORNITH_PROFILE to log prefill timing. Only the
 * last prompt position needs logits. */
static void rprefill(rmodel *m, rstate *s, const int32_t *toks, int np,
                     float *logits) {
    const char *serial = getenv("ORNITH_PREFILL_SERIAL");
    int use_serial = serial && serial[0] == '1';
    int prof = getenv("ORNITH_PROFILE") != NULL;
    struct timespec t0, t1;
    if (prof) clock_gettime(CLOCK_MONOTONIC, &t0);
    if (use_serial) {
        for (int i = 0; i < np; i++)
            rforward_token(m, s, toks[i], (i == np - 1) ? logits : NULL);
    } else {
        rforward_prefill(m, s, toks, np, RFWD_PREFILL_CHUNK, logits);
    }
    const char *dump = getenv("ORNITH_DUMP_LOGITS");
    if (dump && logits) {
        FILE *f = fopen(dump, "wb");
        if (f) { fwrite(logits, sizeof(float), (size_t)m->a.vocab_size, f); fclose(f); }
    }
    if (prof) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "[profile] prefill %d tok %s: %.1f ms (%.3f ms/tok)\n",
                np, use_serial ? "serial" : "chunked", ms, ms / np);
        if (logits)
            fprintf(stderr, "[profile] last-pos argmax token=%d logit=%.6f\n",
                    argmax_f(logits, m->a.vocab_size),
                    logits[argmax_f(logits, m->a.vocab_size)]);
    }
}

/* Run prefill over `tokens` (nt of them) for side effects only (e.g. imatrix
 * collection via the global hook); logits are discarded. Each call uses a fresh
 * recurrent/KV state, so chunks are independent — fine for calibration. */
ornith_status rmodel_prefill_only(rmodel *m, const int32_t *tokens, int nt) {
    if (nt <= 0) { ornith_set_error("empty chunk"); return ORNITH_ERR_FORMAT; }
    rstate *s = rstate_new(&m->a, nt + 4);
    if (!s) { ornith_set_error("oom"); return ORNITH_ERR_OOM; }
    rprefill(m, s, tokens, nt, NULL);
    rstate_free(s);
    return ORNITH_OK;
}

/* Pick the next token: greedy argmax when sp is NULL or greedy, else sample.
 * `hist`/`nhist` is the recent-token history for the repeat penalty. */
static int32_t rpick(const float *logits, int V, const osample_params *sp,
                     ot_rng *rng, const int32_t *hist, int nhist,
                     float *scratch) {
    if (!sp) return argmax_f(logits, V);
    /* osample mutates logits (penalty/temperature); work on a copy so the
     * caller's buffer is reusable and greedy paths stay pure. */
    memcpy(scratch, logits, (size_t)V * sizeof(float));
    return osample(scratch, V, sp, rng, hist, nhist);
}

/* ---- persistent / resumable sessions ----------------------------------- */

/* A session owns the recurrent/KV state for one sequence plus the bookkeeping
 * needed to (a) drive the repeat penalty and (b) snapshot/restore: the token
 * prefix consumed so far and the next-token logits at the current position. */
struct rsession {
    rmodel  *m;
    rstate  *st;
    int      cap;          /* max total positions (KV capacity)              */
    int32_t *hist;         /* [cap] token prefix (prompt + generated so far) */
    int      nhist;
    float   *logits;       /* [vocab] next-token logits, valid iff have_logits */
    int      have_logits;
};

rsession *rmodel_session_new(rmodel *m, int cap) {
    if (cap < 1) cap = 1;
    rsession *s = calloc(1, sizeof(*s));
    if (!s) { ornith_set_error("oom"); return NULL; }
    s->m = m; s->cap = cap;
    s->st     = rstate_new(&m->a, cap);
    s->hist   = malloc((size_t)cap * sizeof(int32_t));
    s->logits = malloc((size_t)m->a.vocab_size * sizeof(float));
    if (!s->st || !s->hist || !s->logits) {
        rmodel_session_free(s); ornith_set_error("oom"); return NULL;
    }
    return s;
}

void rmodel_session_free(rsession *s) {
    if (!s) return;
    if (s->st) rstate_free(s->st);
    free(s->hist); free(s->logits); free(s);
}

const float *rmodel_session_logits(const rsession *s) {
    return (s && s->have_logits) ? s->logits : NULL;
}
int rmodel_session_pos(const rsession *s) { return s ? s->st->pos : 0; }

ornith_status rmodel_session_eval(rsession *s, const int32_t *tokens, int n) {
    if (n <= 0) return ORNITH_OK;
    if (s->st->pos + n > s->cap) {
        ornith_set_error("session capacity %d exceeded (pos %d + %d tokens)",
                         s->cap, s->st->pos, n);
        return ORNITH_ERR_UNSUPPORTED;
    }
    rprefill(s->m, s->st, tokens, n, s->logits);
    s->have_logits = 1;
    for (int i = 0; i < n && s->nhist < s->cap; i++) s->hist[s->nhist++] = tokens[i];
    return ORNITH_OK;
}

ornith_status rmodel_session_generate(rsession *s, int n_predict,
                                      const int32_t *stop_ids, int n_stop,
                                      const osample_params *sp,
                                      void (*on_token)(int32_t, const char *,
                                                       void *),
                                      void *ud, int *out_finish) {
    rmodel *m = s->m; const ornith_arch *a = &m->a; int V = a->vocab_size;
    if (!s->have_logits) {
        ornith_set_error("session has no logits; eval a prompt or load first");
        return ORNITH_ERR_FORMAT;
    }
    if (n_predict < 0) n_predict = 0;
    float *scratch = sp ? malloc((size_t)V * sizeof(float)) : NULL;
    if (sp && !scratch) { ornith_set_error("oom"); return ORNITH_ERR_OOM; }
    ot_rng rng = ot_rng_seed(sp ? sp->seed : 0);

    int finish = 1;            /* default: hit the length cap */
    char piece[512];
    int32_t next = rpick(s->logits, V, sp, &rng, s->hist, s->nhist, scratch);
    for (int step = 0; step < n_predict; step++) {
        int stop = (next == a->eos_token_id);
        for (int k = 0; !stop && k < n_stop; k++)
            if (next == stop_ids[k]) stop = 1;
        if (stop) { finish = 0; break; }
        if (s->st->pos >= s->cap) { finish = 1; break; }  /* out of room */

        otok_detok_token(&m->tok, next, piece, sizeof(piece));
        if (on_token) on_token(next, piece, ud);

        if (s->nhist < s->cap) s->hist[s->nhist++] = next;
        rforward_token(m, s->st, next, s->logits);
        next = rpick(s->logits, V, sp, &rng, s->hist, s->nhist, scratch);
    }
    if (out_finish) *out_finish = finish;
    free(scratch);
    return ORNITH_OK;
}

/* ---- session serialization --------------------------------------------- */

#define SESS_MAGIC   "ORNSESS\0"   /* 8 bytes incl. the trailing NUL        */
#define SESS_VERSION 1u

/* Arch fingerprint embedded in a snapshot so it can't be restored into a
 * structurally different model. */
typedef struct {
    int32_t hidden, num_layers, num_attn_heads, num_kv_heads, head_dim;
    int32_t full_attn_interval, lin_key_heads, lin_value_heads;
    int32_t lin_key_head_dim, lin_value_head_dim, lin_conv_kernel;
    int32_t n_routed_experts, vocab_size;
} sess_fp;

static void sess_fp_fill(sess_fp *fp, const ornith_arch *a) {
    memset(fp, 0, sizeof(*fp));
    fp->hidden = a->hidden_size; fp->num_layers = a->num_layers;
    fp->num_attn_heads = a->num_attn_heads; fp->num_kv_heads = a->num_kv_heads;
    fp->head_dim = a->head_dim; fp->full_attn_interval = a->full_attn_interval;
    fp->lin_key_heads = a->lin_key_heads; fp->lin_value_heads = a->lin_value_heads;
    fp->lin_key_head_dim = a->lin_key_head_dim;
    fp->lin_value_head_dim = a->lin_value_head_dim;
    fp->lin_conv_kernel = a->lin_conv_kernel;
    fp->n_routed_experts = a->n_routed_experts; fp->vocab_size = a->vocab_size;
}

/* FNV-1a over the token prefix; the snapshot records which prefix it stands for. */
static uint64_t sess_prefix_hash(const int32_t *t, int n) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)t;
    for (size_t i = 0; i < (size_t)n * sizeof(int32_t); i++) {
        h ^= p[i]; h *= 1099511628211ULL;
    }
    return h;
}

ornith_status rmodel_session_save(const rsession *s, const char *path) {
    const ornith_arch *a = &s->m->a;
    FILE *f = fopen(path, "wb");
    if (!f) { ornith_set_error("session save: cannot open %s", path); return ORNITH_ERR_IO; }

    int ok = 1;
    ok = ok && fwrite(SESS_MAGIC, 1, 8, f) == 8;
    uint32_t ver = SESS_VERSION; ok = ok && fwrite(&ver, 4, 1, f) == 1;
    sess_fp fp; sess_fp_fill(&fp, a);
    ok = ok && fwrite(&fp, sizeof(fp), 1, f) == 1;
    int32_t pos = s->st->pos, nhist = s->nhist, hl = s->have_logits;
    uint64_t ph = sess_prefix_hash(s->hist, s->nhist);
    ok = ok && fwrite(&pos, 4, 1, f) == 1;
    ok = ok && fwrite(&nhist, 4, 1, f) == 1;
    ok = ok && fwrite(&ph, 8, 1, f) == 1;
    ok = ok && fwrite(&hl, 4, 1, f) == 1;

    for (int L = 0; ok && L < a->num_layers; L++) {
        if (ornith_layer_is_full_attn(a, L)) {
            const ornith_kv_cache *c = &s->st->kv[L];
            int32_t q = c->quantized, len = c->len,
                    nkv = c->n_kv_heads, hd = c->head_dim;
            ok = ok && fwrite(&q, 4, 1, f) == 1 && fwrite(&len, 4, 1, f) == 1
                    && fwrite(&nkv, 4, 1, f) == 1 && fwrite(&hd, 4, 1, f) == 1;
            size_t hv = (size_t)len * nkv * hd, scl = (size_t)len * nkv;
            if (q) {
                ok = ok && fwrite(c->Kq, 1, hv, f) == hv
                        && fwrite(c->Vq, 1, hv, f) == hv
                        && fwrite(c->Ks, sizeof(float), scl, f) == scl
                        && fwrite(c->Vs, sizeof(float), scl, f) == scl;
            } else {
                ok = ok && fwrite(c->K, sizeof(float), hv, f) == hv
                        && fwrite(c->V, sizeof(float), hv, f) == hv;
            }
        } else {
            int nv = a->lin_value_heads, dv = a->lin_value_head_dim,
                dk = a->lin_key_head_dim;
            size_t ds = (size_t)nv * dv * dk;
            ok = ok && fwrite(s->st->delta_S[L], sizeof(float), ds, f) == ds;
            const ornith_conv_state *cv = &s->st->conv[L];
            int32_t K = cv->K, C = cv->C;
            size_t cb = (size_t)(cv->K > 0 ? cv->K - 1 : 0) * cv->C;
            ok = ok && fwrite(&K, 4, 1, f) == 1 && fwrite(&C, 4, 1, f) == 1
                    && fwrite(cv->buf, sizeof(float), cb, f) == cb;
        }
    }
    if (ok && s->nhist > 0)
        ok = ok && fwrite(s->hist, sizeof(int32_t), (size_t)s->nhist, f)
                   == (size_t)s->nhist;
    if (ok && s->have_logits)
        ok = ok && fwrite(s->logits, sizeof(float), (size_t)a->vocab_size, f)
                   == (size_t)a->vocab_size;

    if (fclose(f) != 0) ok = 0;
    if (!ok) { ornith_set_error("session save: write to %s failed", path); return ORNITH_ERR_IO; }
    return ORNITH_OK;
}

ornith_status rmodel_session_load(rmodel *m, const char *path, int extra_cap,
                                  rsession **out) {
    const ornith_arch *a = &m->a;
    FILE *f = fopen(path, "rb");
    if (!f) { ornith_set_error("session load: cannot open %s", path); return ORNITH_ERR_IO; }

    char magic[8]; uint32_t ver;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, SESS_MAGIC, 8) != 0) {
        fclose(f); ornith_set_error("session load: not a session file"); return ORNITH_ERR_FORMAT;
    }
    if (fread(&ver, 4, 1, f) != 1 || ver != SESS_VERSION) {
        fclose(f); ornith_set_error("session load: unsupported version"); return ORNITH_ERR_FORMAT;
    }
    sess_fp fp, cur; sess_fp_fill(&cur, a);
    if (fread(&fp, sizeof(fp), 1, f) != 1) {
        fclose(f); ornith_set_error("session load: truncated header"); return ORNITH_ERR_FORMAT;
    }
    if (memcmp(&fp, &cur, sizeof(fp)) != 0) {
        fclose(f);
        ornith_set_error("session load: arch fingerprint mismatch "
                         "(snapshot is for a different model)");
        return ORNITH_ERR_FORMAT;
    }
    int32_t pos, nhist, hl; uint64_t ph;
    if (fread(&pos, 4, 1, f) != 1 || fread(&nhist, 4, 1, f) != 1 ||
        fread(&ph, 8, 1, f) != 1 || fread(&hl, 4, 1, f) != 1) {
        fclose(f); ornith_set_error("session load: truncated body"); return ORNITH_ERR_FORMAT;
    }

    int cap = pos + (extra_cap > 0 ? extra_cap : 0);
    if (cap < pos)   cap = pos;
    if (cap < nhist) cap = nhist;
    if (cap < 1)     cap = 1;

    rsession *s = calloc(1, sizeof(*s));
    if (!s) { fclose(f); ornith_set_error("oom"); return ORNITH_ERR_OOM; }
    s->m = m; s->cap = cap;
    s->st     = rstate_skeleton(a);
    s->hist   = malloc((size_t)cap * sizeof(int32_t));
    s->logits = malloc((size_t)a->vocab_size * sizeof(float));
    if (!s->st || !s->hist || !s->logits) {
        fclose(f); rmodel_session_free(s); ornith_set_error("oom"); return ORNITH_ERR_OOM;
    }

    int ok = 1;
    for (int L = 0; ok && L < a->num_layers; L++) {
        if (ornith_layer_is_full_attn(a, L)) {
            int32_t q, len, nkv, hd;
            if (fread(&q, 4, 1, f) != 1 || fread(&len, 4, 1, f) != 1 ||
                fread(&nkv, 4, 1, f) != 1 || fread(&hd, 4, 1, f) != 1) { ok = 0; break; }
            if (nkv != a->num_kv_heads || hd != a->head_dim || len > cap) { ok = 0; break; }
            if (ornith_kv_init_ex(&s->st->kv[L], cap, nkv, hd, q) != ORNITH_OK) { ok = 0; break; }
            ornith_kv_cache *c = &s->st->kv[L];
            c->len = len;
            size_t hv = (size_t)len * nkv * hd, scl = (size_t)len * nkv;
            if (q) {
                ok = fread(c->Kq, 1, hv, f) == hv && fread(c->Vq, 1, hv, f) == hv
                  && fread(c->Ks, sizeof(float), scl, f) == scl
                  && fread(c->Vs, sizeof(float), scl, f) == scl;
            } else {
                ok = fread(c->K, sizeof(float), hv, f) == hv
                  && fread(c->V, sizeof(float), hv, f) == hv;
            }
        } else {
            int nv = a->lin_value_heads, dv = a->lin_value_head_dim,
                dk = a->lin_key_head_dim;
            size_t ds = (size_t)nv * dv * dk;
            s->st->delta_S[L] = malloc(ds * sizeof(float));
            if (!s->st->delta_S[L]) { ok = 0; break; }
            if (fread(s->st->delta_S[L], sizeof(float), ds, f) != ds) { ok = 0; break; }
            int32_t K, C;
            if (fread(&K, 4, 1, f) != 1 || fread(&C, 4, 1, f) != 1) { ok = 0; break; }
            if (ornith_conv_init(&s->st->conv[L], K, C) != ORNITH_OK) { ok = 0; break; }
            size_t cb = (size_t)(K > 0 ? K - 1 : 0) * C;
            if (fread(s->st->conv[L].buf, sizeof(float), cb, f) != cb) { ok = 0; break; }
        }
    }
    if (ok && nhist > 0)
        ok = fread(s->hist, sizeof(int32_t), (size_t)nhist, f) == (size_t)nhist;
    if (ok && hl)
        ok = fread(s->logits, sizeof(float), (size_t)a->vocab_size, f)
             == (size_t)a->vocab_size;
    fclose(f);

    if (!ok) { rmodel_session_free(s); ornith_set_error("session load: truncated/corrupt"); return ORNITH_ERR_FORMAT; }
    if (sess_prefix_hash(s->hist, nhist) != ph) {
        rmodel_session_free(s);
        ornith_set_error("session load: prefix hash mismatch (corrupt snapshot)");
        return ORNITH_ERR_FORMAT;
    }
    s->st->pos = pos; s->nhist = nhist; s->have_logits = hl ? 1 : 0;
    *out = s;
    return ORNITH_OK;
}

/* ---- generation (over a transient session) ----------------------------- */

/* Stream callback for rmodel_generate_s: write each token's bytes to a FILE. */
struct ostream_ud { FILE *out; };
static void ostream_cb(int32_t id, const char *piece, void *ud) {
    (void)id;
    struct ostream_ud *u = ud;
    fwrite(piece, 1, strlen(piece), u->out);
    fflush(u->out);
}

ornith_status rmodel_generate_s(rmodel *m, const char *prompt, int n_predict,
                                const osample_params *sp, FILE *out) {
    int32_t *ptoks = NULL; int np = 0;
    otok_encode(&m->tok, prompt, &ptoks, &np);
    if (np == 0) { free(ptoks); ornith_set_error("empty prompt"); return ORNITH_ERR_FORMAT; }
    if (n_predict < 0) n_predict = 0;

    rsession *s = rmodel_session_new(m, np + n_predict + 4);
    if (!s) { free(ptoks); return ORNITH_ERR_OOM; }

    fprintf(out, "%s", prompt);
    fflush(out);

    ornith_status st = rmodel_session_eval(s, ptoks, np);
    if (st == ORNITH_OK) {
        struct ostream_ud ud = { out };
        st = rmodel_session_generate(s, n_predict, NULL, 0, sp, ostream_cb, &ud, NULL);
    }
    fprintf(out, "\n");

    free(ptoks); rmodel_session_free(s);
    return st;
}

ornith_status rmodel_generate(rmodel *m, const char *prompt, int n_predict,
                              FILE *out) {
    return rmodel_generate_s(m, prompt, n_predict, NULL, out);
}

ornith_status rmodel_generate_ids_s(rmodel *m,
                                    const int32_t *prompt_ids, int n_prompt,
                                    int n_predict,
                                    const int32_t *stop_ids, int n_stop,
                                    const osample_params *sp,
                                    void (*on_token)(int32_t id,
                                                     const char *piece, void *ud),
                                    void *ud, int *out_finish) {
    if (n_prompt <= 0) { ornith_set_error("empty prompt"); return ORNITH_ERR_FORMAT; }
    if (n_predict < 0) n_predict = 0;

    rsession *s = rmodel_session_new(m, n_prompt + n_predict + 4);
    if (!s) { ornith_set_error("oom"); return ORNITH_ERR_OOM; }

    ornith_status st = rmodel_session_eval(s, prompt_ids, n_prompt);
    if (st == ORNITH_OK)
        st = rmodel_session_generate(s, n_predict, stop_ids, n_stop, sp,
                                     on_token, ud, out_finish);
    rmodel_session_free(s);
    return st;
}

ornith_status rmodel_generate_ids(rmodel *m,
                                  const int32_t *prompt_ids, int n_prompt,
                                  int n_predict,
                                  const int32_t *stop_ids, int n_stop,
                                  void (*on_token)(int32_t id, const char *piece,
                                                   void *ud),
                                  void *ud, int *out_finish) {
    return rmodel_generate_ids_s(m, prompt_ids, n_prompt, n_predict,
                                 stop_ids, n_stop, NULL,
                                 on_token, ud, out_finish);
}
