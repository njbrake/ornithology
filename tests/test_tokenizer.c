/* test_tokenizer.c — byte-level BPE round-trip on a small in-memory vocab.
 *
 * Builds a tiny GPT-2-style tokenizer (single-byte base tokens for the ASCII
 * letters used, plus a few merges) and asserts encode->decode reproduces the
 * input exactly for several ASCII strings. Also checks that merges actually fire
 * (so a merged piece collapses multiple symbols into one id). */
#include "ornith_tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c,msg) do{ if(!(c)){printf("  FAIL: %s\n",msg);failures++;} \
                         else printf("  ok  : %s\n",msg);}while(0)

int main(void) {
    /* Single-char tokens for the letters we use (printable ASCII map to
     * themselves in the byte-level alphabet), plus a couple of merged tokens. */
    const char *tokens[] = {
        "a","b","c","e","h","i","l","o","r","s","t","w","d","g","n","u","p",
        "he","ll","hello"
    };
    int n_tokens = (int)(sizeof(tokens)/sizeof(tokens[0]));
    /* rank-ordered merges: "h"+"e"->"he", "l"+"l"->"ll", "he"+"ll"->"hell"(n/a),
     * "hell"+"o" n/a; we add the ones whose result is in the vocab. */
    const char *merges[] = { "h e", "l l", "he llo" };
    int n_merges = (int)(sizeof(merges)/sizeof(merges[0]));

    otokenizer t;
    ornith_status st = otok_init(&t, tokens, NULL, n_tokens, merges, n_merges,
                                 -1, 99);
    CHECK(st == ORNITH_OK, "otok_init");

    const char *cases[] = { "hello", "abc", "test", "world" };
    for (int i = 0; i < 4; i++) {
        int32_t *ids = NULL; int n = 0;
        otok_encode(&t, cases[i], &ids, &n);
        char out[64];
        otok_decode(&t, ids, n, out, sizeof(out));
        char msg[96];
        snprintf(msg, sizeof(msg), "round-trip \"%s\" -> %d ids -> \"%s\"",
                 cases[i], n, out);
        CHECK(strcmp(out, cases[i]) == 0, msg);
        free(ids);
    }

    /* "hello" should collapse via merges to fewer than 5 ids (BPE fired). */
    {
        int32_t *ids = NULL; int n = 0;
        otok_encode(&t, "hello", &ids, &n);
        CHECK(n < 5, "BPE merges reduce 'hello' below 5 symbols");
        free(ids);
    }

    /* byte-level decode of an arbitrary id sequence (single chars). */
    {
        /* find the single-char token ids that spell "cat" */
        int ic=-1,ia=-1,it=-1;
        for (int i = 0; i < n_tokens; i++) {
            if(!strcmp(t.tokens[i],"c"))ic=i;
            if(!strcmp(t.tokens[i],"a"))ia=i;
            if(!strcmp(t.tokens[i],"t"))it=i;
        }
        int32_t cat[3] = { ic, ia, it };
        char out[16];
        otok_decode(&t, cat, 3, out, sizeof(out));
        CHECK(strcmp(out, "cat") == 0, "decode id sequence -> \"cat\"");
    }

    otok_free(&t);
    printf("\n%s (%d failure%s)\n",
           failures ? "TOKENIZER TESTS FAILED" : "ALL TOKENIZER TESTS PASSED",
           failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
