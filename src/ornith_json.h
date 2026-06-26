/* ornith_json.h — a tiny, dependency-free JSON parser.
 *
 * Just enough to read a Hugging Face config.json: objects, arrays, strings,
 * numbers, booleans, null. Not a general-purpose library (no unicode escape
 * decoding beyond the basics, no streaming). Builds a small DOM you free with
 * ojson_free().
 */
#ifndef ORNITH_JSON_H
#define ORNITH_JSON_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    OJSON_NULL,
    OJSON_BOOL,
    OJSON_NUMBER,
    OJSON_STRING,
    OJSON_ARRAY,
    OJSON_OBJECT,
} ojson_type;

typedef struct ojson ojson;

struct ojson {
    ojson_type type;
    /* scalars */
    double      num;
    bool        boolean;
    char       *str;        /* OJSON_STRING: owned                          */
    /* containers */
    ojson     **items;      /* ARRAY/OBJECT values                          */
    char      **keys;       /* OBJECT keys (parallel to items), NULL for ARRAY */
    size_t      count;
};

/* Parse a NUL-terminated buffer. Returns NULL on error (see *errpos). */
ojson *ojson_parse(const char *text, const char **errpos);
void   ojson_free(ojson *v);

/* Object lookups. Return NULL if absent or wrong container type. */
const ojson *ojson_get(const ojson *obj, const char *key);

/* Typed convenience getters with defaults. */
long        ojson_get_int(const ojson *obj, const char *key, long def);
double      ojson_get_num(const ojson *obj, const char *key, double def);
bool        ojson_get_bool(const ojson *obj, const char *key, bool def);
const char *ojson_get_str(const ojson *obj, const char *key, const char *def);

#endif /* ORNITH_JSON_H */
