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

#endif /* ORNITH_RFORWARD_H */
