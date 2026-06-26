/* ornith_rforward.c — real-weight qwen3.5 forward with on-the-fly dequant. */
#include "ornith_rforward.h"
#include "ornith_gguf_write.h"
#include "ornith_quant.h"
#include "ornith_tensor.h"
#include "ornith_attn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

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
    const float *x;   /* [in] (matvec) or [T*in] (batch) */
    float *y;         /* [out] or [T*out] */
    int in, out, Tn;  /* Tn columns of x to apply (1 for plain matvec) */
    int o0, o1;
} mv_job;

static void mv_run(mv_job *j) {
    const ogguf_ltensor *W = j->W;
    int in = j->in, Tn = j->Tn;
    size_t rb = oq_row_bytes(W->type, (size_t)in);
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
        jobs[n] = (mv_job){ W, X, Y, in, out, Tn, o0, o1 };
        n++;
    }
    if (n == 1) { mv_run(&jobs[0]); return; }
    for (int i = 0; i < n; i++) pthread_create(&th[i], NULL, mv_thread, &jobs[i]);
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
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

static rstate *rstate_new(const ornith_arch *a, int cap) {
    rstate *s = calloc(1, sizeof(*s));
    s->a = a; s->pos = 0;
    s->kv = calloc((size_t)a->num_layers, sizeof(ornith_kv_cache));
    s->conv = calloc((size_t)a->num_layers, sizeof(ornith_conv_state));
    s->delta_S = calloc((size_t)a->num_layers, sizeof(float *));
    int dk=a->lin_key_head_dim, dv=a->lin_value_head_dim, nv=a->lin_value_heads;
    for (int L = 0; L < a->num_layers; L++) {
        if (ornith_layer_is_full_attn(a, L))
            ornith_kv_init(&s->kv[L], cap, a->num_kv_heads, a->head_dim);
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
    int H=a->hidden_size, nk=a->lin_key_heads, nv=a->lin_value_heads;
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
    /* scale the delta output by 1/sqrt(head_dim) BEFORE the gated RMSNorm — this
     * is NOT a no-op: the gated norm's eps regularizes small-magnitude heads, so
     * the absolute scale matters (matches llama.cpp ggml_gated_delta_net). */
    ot_scale_(o_all, 1.0f / sqrtf((float)dk), vd);
    /* gated RMSNorm (Qwen3NextRMSNormGated): per-head RMSNorm with ssm_norm
     * weight, THEN multiply by SiLU(gate), then out-projection. */
    float *gate = malloc((size_t)vd*sizeof(float));
    matvec_q(m, Tb(m, L, "attn_gate.weight"), xn, gate);
    ot_silu_(gate, vd);
    float *o = malloc((size_t)vd*sizeof(float));
    const float *snorm = f32(Tb(m, L, "ssm_norm.weight"));
    for (int hv = 0; hv < nv; hv++)
        ot_rmsnorm(o_all + (size_t)hv*dv, snorm, o + (size_t)hv*dv, dv, a->rms_norm_eps);
    ot_mul_(o, gate, vd);
    float *outp = malloc((size_t)H*sizeof(float));
    matvec_q(m, Tb(m, L, "ssm_out.weight"), o, outp);
    ot_add_(x_acc, outp, H);
    free(cat); free(conv); free(qn); free(kn); free(a_in); free(b_in);
    free(o_all); free(gate); free(o); free(outp);
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
        ffn_dense(m, L, x, x);
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
    snprintf(a->model_type, sizeof(a->model_type), "qwen35");
    a->hidden_size  = (int)kv_u32(&m->l, "qwen35.embedding_length", 4096);
    a->num_layers   = (int)kv_u32(&m->l, "qwen35.block_count", 32);
    a->num_attn_heads = (int)kv_u32(&m->l, "qwen35.attention.head_count", 16);
    a->num_kv_heads = (int)kv_u32(&m->l, "qwen35.attention.head_count_kv", 4);
    a->head_dim     = (int)kv_u32(&m->l, "qwen35.attention.key_length", 256);
    a->rope_dim     = (int)kv_u32(&m->l, "qwen35.rope.dimension_count", 64);
    a->rope_theta   = kv_f32(&m->l, "qwen35.rope.freq_base", 1e7f);
    a->rms_norm_eps = kv_f32(&m->l, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    a->full_attn_interval = (int)kv_u32(&m->l, "qwen35.full_attention_interval", 4);
    a->attn_output_gate = true;
    a->lin_key_heads   = (int)kv_u32(&m->l, "qwen35.ssm.group_count", 16);
    a->lin_value_heads = (int)kv_u32(&m->l, "qwen35.ssm.time_step_rank", 32);
    a->lin_key_head_dim   = (int)kv_u32(&m->l, "qwen35.ssm.state_size", 128);
    a->lin_value_head_dim = a->lin_key_head_dim;
    a->lin_conv_kernel = (int)kv_u32(&m->l, "qwen35.ssm.conv_kernel", 4);
    a->shared_inter_size = (int)kv_u32(&m->l, "qwen35.feed_forward_length", 12288);
    a->n_routed_experts = 0;
    a->experts_per_tok = 0;
    /* vocab from token_embd dims[1] */
    const ogguf_ltensor *te = T(m, "token_embd.weight");
    a->vocab_size = te ? (int)te->dims[1] : 248320;

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

ornith_status rmodel_generate(rmodel *m, const char *prompt, int n_predict,
                              FILE *out) {
    const ornith_arch *a = &m->a;
    int V = a->vocab_size;
    int32_t *ptoks = NULL; int np = 0;
    otok_encode(&m->tok, prompt, &ptoks, &np);
    if (np == 0) { free(ptoks); ornith_set_error("empty prompt"); return ORNITH_ERR_FORMAT; }

    int cap = np + n_predict + 4;
    rstate *s = rstate_new(a, cap);
    float *logits = malloc((size_t)V*sizeof(float));

    fprintf(out, "%s", prompt);
    fflush(out);

    /* prefill: run all prompt tokens; only the last needs logits */
    for (int i = 0; i < np; i++)
        rforward_token(m, s, ptoks[i], (i == np - 1) ? logits : NULL);

    char piece[256];
    int produced = 0;
    int32_t next = argmax_f(logits, V);
    for (int step = 0; step < n_predict; step++) {
        if (next == a->eos_token_id) break;
        size_t pl = otok_detok_token(&m->tok, next, piece, sizeof(piece));
        fwrite(piece, 1, pl, out); fflush(out);
        produced++;
        rforward_token(m, s, next, logits);
        next = argmax_f(logits, V);
    }
    fprintf(out, "\n");

    free(ptoks); free(logits); rstate_free(s);
    return ORNITH_OK;
}

ornith_status rmodel_generate_ids(rmodel *m,
                                  const int32_t *prompt_ids, int n_prompt,
                                  int n_predict,
                                  const int32_t *stop_ids, int n_stop,
                                  void (*on_token)(int32_t id, const char *piece,
                                                   void *ud),
                                  void *ud, int *out_finish) {
    const ornith_arch *a = &m->a;
    int V = a->vocab_size;
    if (n_prompt <= 0) { ornith_set_error("empty prompt"); return ORNITH_ERR_FORMAT; }
    if (n_predict < 0) n_predict = 0;

    int cap = n_prompt + n_predict + 4;
    rstate *s = rstate_new(a, cap);
    if (!s) { ornith_set_error("oom"); return ORNITH_ERR_OOM; }
    float *logits = malloc((size_t)V * sizeof(float));
    if (!logits) { rstate_free(s); ornith_set_error("oom"); return ORNITH_ERR_OOM; }

    /* prefill: only the last prompt token needs logits */
    for (int i = 0; i < n_prompt; i++)
        rforward_token(m, s, prompt_ids[i], (i == n_prompt - 1) ? logits : NULL);

    int finish = 1;  /* default: hit the length cap */
    char piece[512];
    int32_t next = argmax_f(logits, V);
    for (int step = 0; step < n_predict; step++) {
        int stop = (next == a->eos_token_id);
        for (int k = 0; !stop && k < n_stop; k++)
            if (next == stop_ids[k]) stop = 1;
        if (stop) { finish = 0; break; }

        otok_detok_token(&m->tok, next, piece, sizeof(piece));
        if (on_token) on_token(next, piece, ud);

        rforward_token(m, s, next, logits);
        next = argmax_f(logits, V);
    }

    if (out_finish) *out_finish = finish;
    free(logits); rstate_free(s);
    return ORNITH_OK;
}
