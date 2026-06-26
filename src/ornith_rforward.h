/* ornith_rforward.h — real-weight forward + greedy generation for an Ornith
 * (qwen3.5 / "qwen35") GGUF, with on-the-fly dequantization.
 *
 * The quantized tensors stay resident (ogguf_load, ~file size); each weight is
 * decoded to f32 a row at a time inside a fused, threaded matvec right before
 * its dot product, so peak extra memory is a few small buffers (never the whole
 * model or even a whole tensor in f32). The recurrent linear-attention state and
 * the full-attention KV cache are the same fp32 structures the reference engine
 * uses; generation is token-by-token through the exact decode recurrence.
 */
#ifndef ORNITH_RFORWARD_H
#define ORNITH_RFORWARD_H

#include "ornith.h"
#include "ornith_model.h"
#include "ornith_tokenizer.h"
#include "ornith_sample.h"
#include <stdio.h>

typedef struct rmodel rmodel;

/* Load a GGUF: read metadata into an arch, index tensors, build the tokenizer,
 * keep quantized weights resident. */
ornith_status rmodel_load(const char *path, rmodel **out);
void          rmodel_free(rmodel *m);

const ornith_arch *rmodel_arch(const rmodel *m);
const otokenizer  *rmodel_tokenizer(const rmodel *m);

/* Tokenize `prompt`, prefill it, then greedily decode up to `n_predict` tokens,
 * streaming detokenized text to `out`. Stops early on EOS. */
ornith_status rmodel_generate(rmodel *m, const char *prompt, int n_predict,
                              FILE *out);

/* Same as rmodel_generate but with a sampler config (temperature / top-k /
 * top-p / min-p / repeat-penalty / seed). Pass `sp == NULL` for greedy. */
ornith_status rmodel_generate_s(rmodel *m, const char *prompt, int n_predict,
                                const osample_params *sp, FILE *out);

/* The model id: general.name from the GGUF if present, else NULL. */
const char *rmodel_name(const rmodel *m);

/* Prefill `tokens` (nt of them) for side effects only (imatrix collection via
 * the global hook in ornith_imatrix.h); discards logits. */
ornith_status rmodel_prefill_only(rmodel *m, const int32_t *tokens, int nt);

/* Token-level, stop-aware greedy generation for the server. Prefills
 * `prompt_ids` (n_prompt of them), then greedily decodes up to `n_predict`
 * tokens. For each produced token it invokes `on_token(id, piece, ud)` where
 * `piece` is that token's detokenized bytes (NUL-terminated; empty for special
 * tokens). Generation stops, WITHOUT emitting the token, when the next token is
 * the GGUF eos id or appears in `stop_ids` (n_stop entries). `*out_finish` is
 * set to 0 if it stopped on a stop/eos token, 1 if it hit the n_predict length
 * cap. The old rmodel_generate keeps working unchanged. */
ornith_status rmodel_generate_ids(rmodel *m,
                                  const int32_t *prompt_ids, int n_prompt,
                                  int n_predict,
                                  const int32_t *stop_ids, int n_stop,
                                  void (*on_token)(int32_t id, const char *piece,
                                                   void *ud),
                                  void *ud, int *out_finish);

/* Same as rmodel_generate_ids but with a sampler config (temperature / top-k /
 * top-p / min-p / repeat-penalty / seed). Pass `sp == NULL` for greedy, which is
 * identical to rmodel_generate_ids. A recent-token window (seeded with the
 * prompt and grown with each emitted token) drives the repeat penalty. */
ornith_status rmodel_generate_ids_s(rmodel *m,
                                    const int32_t *prompt_ids, int n_prompt,
                                    int n_predict,
                                    const int32_t *stop_ids, int n_stop,
                                    const osample_params *sp,
                                    void (*on_token)(int32_t id,
                                                     const char *piece, void *ud),
                                    void *ud, int *out_finish);

/* ---- persistent / resumable sessions (on-disk KV cache) ---------------- *
 *
 * A `rsession` wraps the full per-sequence recurrent state (every full-attn
 * layer's KV cache, every linear layer's gated-delta state + conv state, and
 * the stream position) for one chat/sequence on one model. It can be fed tokens
 * incrementally, saved to disk, and later restored into a fresh session so a
 * conversation resumes WITHOUT re-prefilling the prompt that was already seen.
 *
 * The on-disk snapshot carries an arch fingerprint (hidden/layers/heads/...)
 * and a hash of the token prefix it represents, so it can't be loaded into a
 * mismatched model. fp32 and Q8 (ORNITH_KV_Q8) KV caches both round-trip
 * exactly: the saved bytes are the cache's exact contents, so a restored
 * session produces bit-identical next-token logits to the original.
 *
 * rmodel_generate* still work unchanged; they now run over a transient session
 * internally. */
typedef struct rsession rsession;

/* New empty session on `m` with room for up to `cap` total positions (prompt +
 * generated). Returns NULL on OOM. */
rsession *rmodel_session_new(rmodel *m, int cap);
void      rmodel_session_free(rsession *s);

/* Feed `n` tokens, advancing the recurrent/KV state from the current position
 * and computing the next-token logits (retrievable via rmodel_session_logits).
 * n == 0 is a no-op that keeps any existing logits. Fails if it would exceed the
 * session capacity. */
ornith_status rmodel_session_eval(rsession *s, const int32_t *tokens, int n);

/* Logits for the next token after the most recent eval/generate (V entries), or
 * NULL if nothing has been fed yet. The buffer is owned by the session. */
const float *rmodel_session_logits(const rsession *s);

/* Current stream position (number of tokens consumed). */
int rmodel_session_pos(const rsession *s);

/* Serialize the whole session (all layers' state + position + token prefix +
 * the next-token logits) to `path`, and restore it into a fresh session. On
 * load, the session capacity is (saved position + `extra_cap`) so the caller can
 * keep generating; pass extra_cap for the tokens still to come. */
ornith_status rmodel_session_save(const rsession *s, const char *path);
ornith_status rmodel_session_load(rmodel *m, const char *path, int extra_cap,
                                  rsession **out);

/* Greedily/sampled decode up to `n_predict` tokens starting from the session's
 * current next-token logits (so eval a prompt, or load a session, first).
 * Semantics mirror rmodel_generate_ids_s: stop (without emitting) on eos or any
 * id in stop_ids; `*out_finish` is 1 on the length cap, 0 on stop/eos. */
ornith_status rmodel_session_generate(rsession *s, int n_predict,
                                      const int32_t *stop_ids, int n_stop,
                                      const osample_params *sp,
                                      void (*on_token)(int32_t id,
                                                       const char *piece,
                                                       void *ud),
                                      void *ud, int *out_finish);

#endif /* ORNITH_RFORWARD_H */
