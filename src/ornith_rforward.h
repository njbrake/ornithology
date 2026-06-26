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

#endif /* ORNITH_RFORWARD_H */
