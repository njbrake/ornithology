/* ornithology — a local inference engine for the Ornith-1.0 family.
 *
 * Like antirez/ds4 (DwarfStar) is to DeepSeek V4, ornithology is to
 * Ornith-1.0: a narrow, GGML-free engine specialized for one model family
 * so that it can run the 397B flagship on a single high-memory machine.
 *
 * Ornith-1.0-397B is NOT a DeepSeek derivative. Its config.json reports:
 *   model_type:    qwen3_5_moe
 *   architectures: ["Qwen3_5MoeForConditionalGeneration"]
 * It is a hybrid linear/full-attention MoE post-trained on Gemma 4 + Qwen 3.5.
 * That single fact is why this is a sibling of ds4, not a fork: ds4 is built
 * around DeepSeek MLA + a compressed KV cache, while Ornith gets its
 * long-context efficiency from gated-delta linear attention on 3 of every 4
 * layers, with a normal (GQA) KV cache only on the 1-in-4 full-attention
 * layers. See DESIGN.md.
 *
 * This header declares the public surface shared across the CLI, the GGUF
 * loader, the config parser, and (eventually) the backends.
 *
 * License: MIT.
 */
#ifndef ORNITH_H
#define ORNITH_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define ORNITH_VERSION "0.0.1"

/* ---- error handling ---------------------------------------------------- */

typedef enum {
    ORNITH_OK = 0,
    ORNITH_ERR_IO,         /* file open/read/seek failure                   */
    ORNITH_ERR_FORMAT,     /* malformed GGUF / JSON                         */
    ORNITH_ERR_UNSUPPORTED,/* recognized but not implemented yet            */
    ORNITH_ERR_OOM,        /* allocation failure                            */
    ORNITH_ERR_NOTFOUND,   /* requested key/tensor missing                  */
} ornith_status;

const char *ornith_strerror(ornith_status s);

/* Last human-readable error detail, set by the lower layers. */
const char *ornith_last_error(void);
void ornith_set_error(const char *fmt, ...);

#endif /* ORNITH_H */
