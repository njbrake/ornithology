/* ornith_eval.h — perplexity evaluation over a text corpus (ds4 parity:
 * ds4_eval.c).
 *
 * `ornith eval <model.gguf> <corpus.txt> [--max-tokens N]` tokenizes the corpus,
 * advances the model one token at a time through the public session API, and at
 * each position accumulates the negative log-likelihood of the actual next token
 * (a numerically-stable log-softmax of that position's logits). It reports mean
 * NLL (nats/token), perplexity = exp(mean NLL), and the token count. This is a
 * real quality metric: a coherent corpus scores a low perplexity, and it lets
 * you compare quant levels of the same model.
 *
 * The NLL math is factored out (ornith_eval_token_nll) and the whole id-sequence
 * pipeline (ornith_eval_ids) is exposed so tests can drive it deterministically.
 */
#ifndef ORNITH_EVAL_H
#define ORNITH_EVAL_H

#include "ornith.h"
#include "ornith_rforward.h"
#include <stdint.h>

/* Negative log-likelihood (nats) of token id `next` under `vocab` raw logits:
 * -log(softmax(logits)[next]), computed via a stable log-sum-exp so the logits
 * are never mutated and large magnitudes don't overflow. Always >= 0. Returns
 * NaN if `next` is out of [0, vocab). */
double ornith_eval_token_nll(const float *logits, int vocab, int32_t next);

/* Run model `m` over the token sequence `ids` (n of them, n >= 2) using the
 * session API and accumulate per-position NLL of the actual next token. Writes
 * the mean NLL, perplexity (exp of mean), and the number of scored predictions
 * (= n-1) to the out-params (any may be NULL). */
ornith_status ornith_eval_ids(rmodel *m, const int32_t *ids, int n,
                              double *out_mean_nll, double *out_ppl,
                              long *out_count);

/* Argv is the args AFTER the `eval` subcommand. Returns 0 on success. */
int ornith_eval_main(int argc, char **argv);

#endif /* ORNITH_EVAL_H */
