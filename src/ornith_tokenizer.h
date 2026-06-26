/* ornith_tokenizer.h — byte-level (GPT-2/Qwen "gpt2") BPE tokenizer.
 *
 * Loads the vocab + merges from GGUF metadata (tokenizer.ggml.tokens /
 * .merges / .token_type / .eos_token_id) and implements:
 *   - decode: token id -> piece -> GPT-2 byte-level-unicode inverse -> raw bytes
 *   - encode: text -> bytes -> byte-level unicode -> greedy lowest-rank BPE
 * The pre-tokenizer is a Qwen2/GPT-2-style splitter (good enough for plain
 * ASCII prompts; decode is exact). Vocab and merge lookups use open-addressing
 * hash tables built once at load.
 */
#ifndef ORNITH_TOKENIZER_H
#define ORNITH_TOKENIZER_H

#include "ornith.h"
#include "ornith_gguf_write.h"   /* ogguf_loaded */

typedef struct { char **keys; int *vals; size_t cap, count; } otok_smap;

typedef struct {
    char    **tokens;      /* [n_tokens] owned piece strings (byte-level utf8) */
    int       n_tokens;
    int32_t   bos_id, eos_id;

    otok_smap vocab;       /* piece string -> id                              */
    otok_smap merges;      /* "A\x01B"     -> rank                            */
    char    **merge_keys;  /* owned merge key strings (to free)               */
    int       n_merges;

    char     *byte2uni[256]; /* byte -> utf8 of its byte-level codepoint       */
    int       uni2byte[1024]; /* codepoint -> byte (-1 if none)                */
} otokenizer;

/* Build directly from arrays (used by tests). Copies what it needs; the caller
 * keeps ownership of its inputs. `types` may be NULL. merges are "A B" strings
 * in rank order. */
ornith_status otok_init(otokenizer *t, const char *const *tokens,
                        const int32_t *types, int n_tokens,
                        const char *const *merges, int n_merges,
                        int32_t bos, int32_t eos);

/* Build from a loaded GGUF's tokenizer metadata. */
ornith_status otok_load_from_gguf(const ogguf_loaded *l, otokenizer *t);

void otok_free(otokenizer *t);

/* Encode text -> token ids. *ids is malloc'd (caller frees); *n is the count. */
ornith_status otok_encode(const otokenizer *t, const char *text,
                          int32_t **ids, int *n);

/* Look up the id of an exact vocab piece (e.g. a special token like
 * "<|im_start|>"). Returns the id, or -1 if the piece is not in the vocab.
 * This does NOT run BPE; it is an exact piece->id match, which is what special
 * tokens need (they are single atomic vocab entries). */
int32_t otok_id_of(const otokenizer *t, const char *piece);

/* Append the raw bytes of one token id to buf (NUL-terminated, bounded by cap).
 * Returns the number of bytes appended. */
size_t otok_detok_token(const otokenizer *t, int32_t id, char *buf, size_t cap);

/* Decode a sequence of ids into a NUL-terminated string in `out` (bounded). */
void otok_decode(const otokenizer *t, const int32_t *ids, int n,
                 char *out, size_t cap);

#endif /* ORNITH_TOKENIZER_H */
