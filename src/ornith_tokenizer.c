/* ornith_tokenizer.c — byte-level BPE (see header). */
#include "ornith_tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

/* ---- small string->int open-addressing hash map ----------------------- */

static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= *p; h *= 1099511628211ULL;
    }
    return h;
}

static void smap_init(otok_smap *m, size_t want) {
    size_t cap = 16;
    while (cap < want * 2) cap *= 2;
    m->keys = calloc(cap, sizeof(char *));
    m->vals = calloc(cap, sizeof(int));
    m->cap = cap; m->count = 0;
}
static void smap_free(otok_smap *m) {
    free(m->keys); free(m->vals);
    m->keys = NULL; m->vals = NULL; m->cap = m->count = 0;
}
/* keys are borrowed (owned by the caller's storage). */
static void smap_put(otok_smap *m, char *key, int val) {
    size_t mask = m->cap - 1;
    size_t i = fnv1a(key) & mask;
    while (m->keys[i]) {
        if (strcmp(m->keys[i], key) == 0) { m->vals[i] = val; return; }
        i = (i + 1) & mask;
    }
    m->keys[i] = key; m->vals[i] = val; m->count++;
}
static int smap_get(const otok_smap *m, const char *key, int *out) {
    if (!m->cap) return 0;
    size_t mask = m->cap - 1;
    size_t i = fnv1a(key) & mask;
    while (m->keys[i]) {
        if (strcmp(m->keys[i], key) == 0) { *out = m->vals[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

/* ---- GPT-2 byte-level alphabet ---------------------------------------- */

static void utf8_encode(int cp, char *out, size_t *len) {
    if (cp < 0x80) { out[0] = (char)cp; *len = 1; }
    else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        *len = 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        *len = 3;
    }
}
/* decode one codepoint from s; returns bytes consumed (0 on end). */
static int utf8_decode(const char *s, int *cp) {
    unsigned char c = (unsigned char)s[0];
    if (!c) return 0;
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) {
        *cp = ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        *cp = ((c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) |
              ((unsigned char)s[2] & 0x3F);
        return 3;
    }
    *cp = c; return 1;
}

static void build_byte_alphabet(otokenizer *t) {
    for (int i = 0; i < 1024; i++) t->uni2byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= '!' && b <= '~') ||
                        (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        int cp = printable ? b : (256 + n);
        if (!printable) n++;
        char buf[4]; size_t len;
        utf8_encode(cp, buf, &len);
        char *s = malloc(len + 1);
        memcpy(s, buf, len); s[len] = '\0';
        t->byte2uni[b] = s;
        if (cp < 1024) t->uni2byte[cp] = b;
    }
}

/* ---- init / free ------------------------------------------------------- */

ornith_status otok_init(otokenizer *t, const char *const *tokens,
                        const int32_t *types, int n_tokens,
                        const char *const *merges, int n_merges,
                        int32_t bos, int32_t eos) {
    memset(t, 0, sizeof(*t));
    (void)types;
    t->n_tokens = n_tokens;
    t->bos_id = bos; t->eos_id = eos;
    t->tokens = calloc((size_t)n_tokens, sizeof(char *));
    if (!t->tokens) { ornith_set_error("tokenizer oom"); return ORNITH_ERR_OOM; }
    for (int i = 0; i < n_tokens; i++) {
        size_t len = strlen(tokens[i]);
        char *s = malloc(len + 1);
        memcpy(s, tokens[i], len + 1);
        t->tokens[i] = s;
    }
    /* vocab map */
    smap_init(&t->vocab, (size_t)n_tokens);
    for (int i = 0; i < n_tokens; i++) smap_put(&t->vocab, t->tokens[i], i);

    /* merges: store "A\x01B" -> rank */
    t->n_merges = n_merges;
    t->merge_keys = calloc((size_t)(n_merges > 0 ? n_merges : 1), sizeof(char *));
    smap_init(&t->merges, (size_t)(n_merges > 0 ? n_merges : 1));
    for (int i = 0; i < n_merges; i++) {
        const char *m = merges[i];
        const char *sp = strchr(m, ' ');
        if (!sp) { t->merge_keys[i] = NULL; continue; }
        size_t la = (size_t)(sp - m), lb = strlen(sp + 1);
        char *key = malloc(la + 1 + lb + 1);
        memcpy(key, m, la);
        key[la] = '\x01';
        memcpy(key + la + 1, sp + 1, lb + 1);
        t->merge_keys[i] = key;
        smap_put(&t->merges, key, i);   /* rank = i (lower is better) */
    }

    build_byte_alphabet(t);
    return ORNITH_OK;
}

void otok_free(otokenizer *t) {
    if (!t) return;
    if (t->tokens) {
        for (int i = 0; i < t->n_tokens; i++) free(t->tokens[i]);
        free(t->tokens);
    }
    if (t->merge_keys) {
        for (int i = 0; i < t->n_merges; i++) free(t->merge_keys[i]);
        free(t->merge_keys);
    }
    smap_free(&t->vocab);
    smap_free(&t->merges);
    for (int b = 0; b < 256; b++) free(t->byte2uni[b]);
    memset(t, 0, sizeof(*t));
}

/* ---- GGUF loader ------------------------------------------------------- */

enum { GT_INT32 = 5, GT_STRING = 8, GT_ARRAY = 9 };

static const ogguf_lkv *find_kv(const ogguf_loaded *l, const char *key) {
    for (uint64_t i = 0; i < l->n_kv; i++)
        if (strcmp(l->kv[i].key, key) == 0) return &l->kv[i];
    return NULL;
}

/* Parse a STRING-array payload into a malloc'd array of NUL-terminated copies. */
static char **parse_str_array(const uint8_t *p, size_t len, uint64_t *out_n) {
    if (len < 12) return NULL;
    uint32_t et; uint64_t n;
    memcpy(&et, p, 4); memcpy(&n, p + 4, 8);
    if (et != GT_STRING) return NULL;
    const uint8_t *q = p + 12, *end = p + len;
    char **arr = malloc((size_t)n * sizeof(char *));
    if (!arr) return NULL;
    for (uint64_t i = 0; i < n; i++) {
        if (q + 8 > end) { free(arr); return NULL; }
        uint64_t sl; memcpy(&sl, q, 8); q += 8;
        if (q + sl > end) { free(arr); return NULL; }
        char *s = malloc(sl + 1);
        memcpy(s, q, sl); s[sl] = '\0';
        arr[i] = s; q += sl;
    }
    *out_n = n;
    return arr;
}

ornith_status otok_load_from_gguf(const ogguf_loaded *l, otokenizer *t) {
    const ogguf_lkv *ktok = find_kv(l, "tokenizer.ggml.tokens");
    const ogguf_lkv *kmrg = find_kv(l, "tokenizer.ggml.merges");
    const ogguf_lkv *keos = find_kv(l, "tokenizer.ggml.eos_token_id");
    const ogguf_lkv *kbos = find_kv(l, "tokenizer.ggml.bos_token_id");
    if (!ktok) { ornith_set_error("no tokenizer.ggml.tokens"); return ORNITH_ERR_FORMAT; }

    uint64_t nt = 0, nm = 0;
    char **tokens = parse_str_array(ktok->payload, ktok->payload_len, &nt);
    if (!tokens) { ornith_set_error("bad tokens array"); return ORNITH_ERR_FORMAT; }
    char **merges = NULL;
    if (kmrg) merges = parse_str_array(kmrg->payload, kmrg->payload_len, &nm);

    int32_t eos = -1, bos = -1;
    /* id may be stored as INT32 (5) or UINT32 (4); both are 4 LE bytes. */
    if (keos && keos->payload_len >= 4) memcpy(&eos, keos->payload, 4);
    if (kbos && kbos->payload_len >= 4) memcpy(&bos, kbos->payload, 4);

    ornith_status st = otok_init(t, (const char *const *)tokens, NULL, (int)nt,
                                 (const char *const *)merges, (int)nm, bos, eos);

    for (uint64_t i = 0; i < nt; i++) free(tokens[i]);
    free(tokens);
    if (merges) { for (uint64_t i = 0; i < nm; i++) free(merges[i]); free(merges); }
    return st;
}

/* ---- decode ------------------------------------------------------------ */

size_t otok_detok_token(const otokenizer *t, int32_t id, char *buf, size_t cap) {
    size_t w = 0;
    if (id < 0 || id >= t->n_tokens) { if (cap) buf[0] = '\0'; return 0; }
    const char *piece = t->tokens[id];
    int i = 0, cp;
    int adv;
    while ((adv = utf8_decode(piece + i, &cp)) > 0) {
        i += adv;
        int b = (cp >= 0 && cp < 1024) ? t->uni2byte[cp] : -1;
        if (b < 0) continue;             /* non-byte-level char (special tok) */
        if (w + 1 < cap) buf[w] = (char)b;
        w++;
    }
    if (cap) buf[w < cap ? w : cap - 1] = '\0';
    return w;
}

void otok_decode(const otokenizer *t, const int32_t *ids, int n,
                 char *out, size_t cap) {
    size_t w = 0;
    for (int i = 0; i < n && w + 1 < cap; i++)
        w += otok_detok_token(t, ids[i], out + w, cap - w);
    if (cap) out[w < cap ? w : cap - 1] = '\0';
}

/* ---- encode ------------------------------------------------------------ */

/* Append the byte-level-unicode form of `chunk` (raw bytes) as the symbol list,
 * then run greedy BPE, emitting ids into the dynamic array (*ids,*n,*cap). */
static void push_id(int32_t **ids, int *n, int *cap, int32_t v) {
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 64;
                      *ids = realloc(*ids, (size_t)*cap * sizeof(int32_t)); }
    (*ids)[(*n)++] = v;
}

static void bpe_chunk(const otokenizer *t, const char *bytes, size_t blen,
                      int32_t **ids, int *n, int *cap) {
    if (blen == 0) return;
    /* symbols: one per input byte, each the byte-level-unicode string */
    char **sym = malloc(blen * sizeof(char *));
    size_t nsym = blen;
    for (size_t i = 0; i < blen; i++) {
        const char *u = t->byte2uni[(unsigned char)bytes[i]];
        size_t ul = strlen(u);
        char *s = malloc(ul + 1);
        memcpy(s, u, ul + 1);
        sym[i] = s;
    }

    /* greedy: repeatedly merge the adjacent pair with the lowest rank */
    for (;;) {
        int best_rank = -1; size_t best_i = 0;
        for (size_t i = 0; i + 1 < nsym; i++) {
            size_t la = strlen(sym[i]), lb = strlen(sym[i + 1]);
            char key[256];
            if (la + 1 + lb + 1 > sizeof(key)) continue;
            memcpy(key, sym[i], la);
            key[la] = '\x01';
            memcpy(key + la + 1, sym[i + 1], lb + 1);
            int rank;
            if (smap_get(&t->merges, key, &rank)) {
                if (best_rank < 0 || rank < best_rank) { best_rank = rank; best_i = i; }
            }
        }
        if (best_rank < 0) break;
        /* merge sym[best_i] + sym[best_i+1] */
        size_t la = strlen(sym[best_i]), lb = strlen(sym[best_i + 1]);
        char *m = malloc(la + lb + 1);
        memcpy(m, sym[best_i], la);
        memcpy(m + la, sym[best_i + 1], lb + 1);
        free(sym[best_i]); free(sym[best_i + 1]);
        sym[best_i] = m;
        for (size_t j = best_i + 1; j + 1 < nsym; j++) sym[j] = sym[j + 1];
        nsym--;
    }

    for (size_t i = 0; i < nsym; i++) {
        int id;
        if (smap_get(&t->vocab, sym[i], &id)) push_id(ids, n, cap, id);
        /* if a symbol isn't in vocab (shouldn't happen for byte-level), drop */
        free(sym[i]);
    }
    free(sym);
}

/* classifiers on raw bytes */
static int is_sp(int c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
static int is_letter(int c){ return isalpha(c) || c >= 0x80; }
static int is_digit_(int c){ return c >= '0' && c <= '9'; }

/* Qwen2/GPT-2-style pre-tokenizer: returns the end index of the chunk that
 * starts at `i` (always advances by >= 1). */
static size_t next_chunk(const char *s, size_t n, size_t i) {
    unsigned char c = (unsigned char)s[i];
    /* contractions: 'x */
    if (c == '\'' && i + 1 < n) {
        const char *r = s + i + 1;
        size_t rem = n - i - 1;
        const char *two[] = {"re","ve","ll"};
        const char *one[] = {"s","t","m","d"};
        for (int k = 0; k < 3; k++)
            if (rem >= 2 && (tolower(r[0])==two[k][0]) && (tolower(r[1])==two[k][1]))
                return i + 3;
        for (int k = 0; k < 4; k++)
            if (rem >= 1 && tolower(r[0]) == one[k][0]) return i + 2;
    }
    /* [^\r\n\p{L}\p{N}]? \p{L}+  : optional single non-alnum lead, then letters */
    {
        size_t j = i;
        if (!is_letter(c) && !is_digit_(c) && c != '\n' && c != '\r') {
            if (i + 1 < n && is_letter((unsigned char)s[i + 1])) j = i + 1;
        }
        if (is_letter((unsigned char)s[j])) {
            j++;
            while (j < n && is_letter((unsigned char)s[j])) j++;
            return j;
        }
    }
    /* \p{N} : single digit */
    if (is_digit_(c)) return i + 1;
    /* ' '? [^\s\p{L}\p{N}]+ : optional single space, then punctuation run */
    {
        size_t j = i;
        if (c == ' ' && i + 1 < n) {
            unsigned char d = (unsigned char)s[i + 1];
            if (!is_sp(d) && !is_letter(d) && !is_digit_(d)) j = i + 1;
        }
        if (!is_sp((unsigned char)s[j]) && !is_letter((unsigned char)s[j]) &&
            !is_digit_((unsigned char)s[j])) {
            j++;
            while (j < n && !is_sp((unsigned char)s[j]) &&
                   !is_letter((unsigned char)s[j]) && !is_digit_((unsigned char)s[j]))
                j++;
            return j;
        }
    }
    /* whitespace: \s+(?!\S) leaves the last space for the following word */
    if (is_sp(c)) {
        size_t j = i;
        while (j < n && is_sp((unsigned char)s[j])) j++;
        /* if a non-space follows the run, the last space attaches to it */
        if (j < n && j - i > 1) j--;
        return j;
    }
    return i + 1;
}

ornith_status otok_encode(const otokenizer *t, const char *text,
                          int32_t **ids_out, int *n_out) {
    size_t n = strlen(text);
    int32_t *ids = NULL; int cnt = 0, cap = 0;
    size_t i = 0;
    while (i < n) {
        size_t j = next_chunk(text, n, i);
        if (j <= i) j = i + 1;
        bpe_chunk(t, text + i, j - i, &ids, &cnt, &cap);
        i = j;
    }
    *ids_out = ids; *n_out = cnt;
    return ORNITH_OK;
}
