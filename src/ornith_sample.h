/* ornith_sample.h — a self-contained logits sampler (temperature / top-k /
 * top-p / min-p / repeat-penalty / seed), matching the llama.cpp-style chain.
 *
 * The sampler is a pure function over a logits[vocab] array plus a parameter
 * struct, a seeded RNG (ot_rng from ornith_tensor.h), and a recent-token window
 * used for the repetition penalty. Same seed + same logits + same params + same
 * history => same token, on every platform. With temperature <= 0 it degenerates
 * to greedy argmax so the existing reproducible behaviour is preserved.
 *
 * Pipeline (applied in this order, llama.cpp convention):
 *   repeat penalty over recent tokens
 *     -> temperature scale
 *     -> top-k truncate
 *     -> top-p (nucleus) truncate
 *     -> min-p truncate
 *     -> softmax
 *     -> sample with the seeded RNG
 */
#ifndef ORNITH_SAMPLE_H
#define ORNITH_SAMPLE_H

#include "ornith_tensor.h"   /* ot_rng */
#include <stdint.h>

typedef struct {
    float    temperature;     /* <= 0  => greedy/argmax (penalty still applies) */
    int      top_k;           /* 0     => disabled                              */
    float    top_p;           /* 1.0   => disabled (nucleus)                    */
    float    min_p;           /* 0.0   => disabled                              */
    float    repeat_penalty;  /* 1.0   => disabled                             */
    int      repeat_last_n;   /* recent-token window for the penalty (<=0: all) */
    uint64_t seed;            /* RNG seed for reproducibility                  */
} osample_params;

/* Greedy defaults: temperature 0 (=> argmax), every other knob disabled. This
 * keeps generation deterministic unless a caller asks for sampling. */
osample_params osample_params_default(void);

/* Pick a token from `logits` (length `n_vocab`). The array is modified in place
 * (penalty + temperature scaling), which is fine because callers recompute fresh
 * logits each step. `recent`/`n_recent` is the recent-token history used for the
 * repeat penalty (may be NULL/0). `rng` is advanced; pass a seeded ot_rng for
 * reproducibility. Returns the chosen token id in [0, n_vocab). */
int32_t osample(float *logits, int n_vocab, const osample_params *p,
                ot_rng *rng, const int32_t *recent, int n_recent);

#endif /* ORNITH_SAMPLE_H */
