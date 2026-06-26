/* ornith_forward.h — the assembled qwen3_5_moe decoder forward pass.
 *
 * Ties together the tensor kernels, both attention flavors, and the MoE FFN into
 * a full pre-norm decoder stack parameterized by an ornith_arch. Layers are
 * interleaved L L L F (linear-attn except every full_attn_interval-th, which is
 * full GQA attention); every layer has a MoE FFN.
 *
 * Two entry points share one set of caches/states:
 *   - of_forward_prefill: process a whole prompt at once (full attention runs a
 *     causal pass; linear attention runs the chunked parallel scan).
 *   - of_forward_decode:  process one token, continuing from the cached state
 *     (full attention appends to its KV cache; linear attention runs the step
 *     recurrence + conv step).
 * Running a prompt through prefill must yield the same per-position logits as
 * feeding the same tokens one-by-one through decode. tests/test_forward.c asserts
 * exactly that — the milestone's correctness property at the system level.
 *
 * For M2 the weights are small SEEDED-SYNTHETIC tensors (no real checkpoint is
 * needed to prove the math). Real GGUF weight loading is the remaining M2 item.
 */
#ifndef ORNITH_FORWARD_H
#define ORNITH_FORWARD_H

#include "ornith.h"
#include "ornith_model.h"
#include "ornith_attn.h"

typedef struct of_model of_model;
typedef struct of_state of_state;

/* Build a tiny model with deterministic seeded-random weights matching `a`.
 * Use ornith_arch_tiny() (below) for a fast, test-sized architecture. Returns
 * NULL on allocation failure. */
of_model *of_model_build_synthetic(const ornith_arch *a, uint64_t seed);
void      of_model_free(of_model *m);
const ornith_arch *of_model_arch(const of_model *m);

/* Fill `a` with a small but structurally complete qwen3_5_moe (4 layers so the
 * L L L F pattern appears once, tiny dims). For tests and `run` demos. */
void ornith_arch_tiny(ornith_arch *a);

/* Per-sequence caches/states (KV cache per full layer, fp32 delta state + conv
 * state per linear layer). `capacity` bounds the full-attention KV cache. */
of_state *of_state_new(const of_model *m, int capacity);
void      of_state_free(of_state *s);
void      of_state_reset(of_state *s);

/* Prefill T tokens. Writes logits for every position to logits_all
 * [T * vocab_size] (caller-allocated). `chunk` is the linear-attn scan chunk
 * size (>=1); pass 0 for a sensible default. Advances state. */
ornith_status of_forward_prefill(const of_model *m, of_state *s,
                                 const int32_t *tokens, int T, int chunk,
                                 float *logits_all);

/* Decode one token, writing logits[vocab_size]. Advances state. */
ornith_status of_forward_decode(const of_model *m, of_state *s,
                                int32_t token, float *logits);

/* argmax over logits[n]. */
int of_argmax(const float *logits, int n);

#endif /* ORNITH_FORWARD_H */
