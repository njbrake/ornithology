/* ornith_json.c — recursive-descent JSON parser. See ornith_json.h. */
#include "ornith_json.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p; const char *err; } P;

static ojson *parse_value(P *s);

static void skip_ws(P *s) {
    while (*s->p && (*s->p == ' ' || *s->p == '\t' ||
                     *s->p == '\n' || *s->p == '\r'))
        s->p++;
}

static ojson *new_node(ojson_type t) {
    ojson *v = calloc(1, sizeof(ojson));
    if (v) v->type = t;
    return v;
}

void ojson_free(ojson *v) {
    if (!v) return;
    if (v->type == OJSON_STRING) free(v->str);
    if (v->type == OJSON_ARRAY || v->type == OJSON_OBJECT) {
        for (size_t i = 0; i < v->count; i++) {
            ojson_free(v->items[i]);
            if (v->keys) free(v->keys[i]);
        }
        free(v->items);
        free(v->keys);
    }
    free(v);
}

/* Parse a JSON string literal (leading quote already at *s->p). */
static char *parse_string_raw(P *s) {
    if (*s->p != '"') { s->err = s->p; return NULL; }
    s->p++;
    size_t cap = 16, len = 0;
    char *out = malloc(cap);
    if (!out) { s->err = s->p; return NULL; }
    while (*s->p && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\') {
            char e = *s->p++;
            switch (e) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case '/': c = '/';  break;
            case '\\':c = '\\'; break;
            case '"': c = '"';  break;
            case 'u': {
                /* Minimal: decode \uXXXX, emit as-is for ASCII, '?' otherwise. */
                int code = 0;
                for (int i = 0; i < 4 && isxdigit((unsigned char)*s->p); i++) {
                    char h = *s->p++;
                    code = code * 16 + (h <= '9' ? h - '0' :
                          (tolower(h) - 'a' + 10));
                }
                c = code < 128 ? (char)code : '?';
                break;
            }
            default: c = e; break;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *n = realloc(out, cap);
            if (!n) { free(out); s->err = s->p; return NULL; }
            out = n;
        }
        out[len++] = c;
    }
    if (*s->p != '"') { free(out); s->err = s->p; return NULL; }
    s->p++;
    out[len] = '\0';
    return out;
}

static ojson *parse_object(P *s) {
    ojson *v = new_node(OJSON_OBJECT);
    if (!v) { s->err = s->p; return NULL; }
    s->p++; /* { */
    skip_ws(s);
    if (*s->p == '}') { s->p++; return v; }
    for (;;) {
        skip_ws(s);
        char *key = parse_string_raw(s);
        if (!key) { ojson_free(v); return NULL; }
        skip_ws(s);
        if (*s->p != ':') { free(key); ojson_free(v); s->err = s->p; return NULL; }
        s->p++;
        ojson *val = parse_value(s);
        if (!val) { free(key); ojson_free(v); return NULL; }

        ojson **ni = realloc(v->items, (v->count + 1) * sizeof(*ni));
        char  **nk = realloc(v->keys,  (v->count + 1) * sizeof(*nk));
        if (!ni || !nk) {
            free(key); ojson_free(val);
            if (ni) v->items = ni;
            if (nk) v->keys = nk;
            ojson_free(v); s->err = s->p; return NULL;
        }
        v->items = ni; v->keys = nk;
        v->items[v->count] = val;
        v->keys[v->count]  = key;
        v->count++;

        skip_ws(s);
        if (*s->p == ',') { s->p++; continue; }
        if (*s->p == '}') { s->p++; break; }
        ojson_free(v); s->err = s->p; return NULL;
    }
    return v;
}

static ojson *parse_array(P *s) {
    ojson *v = new_node(OJSON_ARRAY);
    if (!v) { s->err = s->p; return NULL; }
    s->p++; /* [ */
    skip_ws(s);
    if (*s->p == ']') { s->p++; return v; }
    for (;;) {
        ojson *val = parse_value(s);
        if (!val) { ojson_free(v); return NULL; }
        ojson **ni = realloc(v->items, (v->count + 1) * sizeof(*ni));
        if (!ni) { ojson_free(val); ojson_free(v); s->err = s->p; return NULL; }
        v->items = ni;
        v->items[v->count++] = val;
        skip_ws(s);
        if (*s->p == ',') { s->p++; continue; }
        if (*s->p == ']') { s->p++; break; }
        ojson_free(v); s->err = s->p; return NULL;
    }
    return v;
}

static ojson *parse_value(P *s) {
    skip_ws(s);
    char c = *s->p;
    if (c == '{') return parse_object(s);
    if (c == '[') return parse_array(s);
    if (c == '"') {
        char *str = parse_string_raw(s);
        if (!str) return NULL;
        ojson *v = new_node(OJSON_STRING);
        if (!v) { free(str); s->err = s->p; return NULL; }
        v->str = str;
        return v;
    }
    if (c == 't' || c == 'f') {
        bool b = (c == 't');
        const char *lit = b ? "true" : "false";
        if (strncmp(s->p, lit, strlen(lit)) != 0) { s->err = s->p; return NULL; }
        s->p += strlen(lit);
        ojson *v = new_node(OJSON_BOOL);
        if (!v) { s->err = s->p; return NULL; }
        v->boolean = b;
        return v;
    }
    if (c == 'n') {
        if (strncmp(s->p, "null", 4) != 0) { s->err = s->p; return NULL; }
        s->p += 4;
        return new_node(OJSON_NULL);
    }
    if (c == '-' || isdigit((unsigned char)c)) {
        char *end = NULL;
        double d = strtod(s->p, &end);
        if (end == s->p) { s->err = s->p; return NULL; }
        s->p = end;
        ojson *v = new_node(OJSON_NUMBER);
        if (!v) { s->err = s->p; return NULL; }
        v->num = d;
        return v;
    }
    s->err = s->p;
    return NULL;
}

ojson *ojson_parse(const char *text, const char **errpos) {
    P s = { text, NULL };
    ojson *v = parse_value(&s);
    if (!v) { if (errpos) *errpos = s.err; return NULL; }
    skip_ws(&s);
    /* trailing junk is tolerated; config.json is a single object */
    if (errpos) *errpos = NULL;
    return v;
}

const ojson *ojson_get(const ojson *obj, const char *key) {
    if (!obj || obj->type != OJSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->count; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

long ojson_get_int(const ojson *obj, const char *key, long def) {
    const ojson *v = ojson_get(obj, key);
    return (v && v->type == OJSON_NUMBER) ? (long)v->num : def;
}
double ojson_get_num(const ojson *obj, const char *key, double def) {
    const ojson *v = ojson_get(obj, key);
    return (v && v->type == OJSON_NUMBER) ? v->num : def;
}
bool ojson_get_bool(const ojson *obj, const char *key, bool def) {
    const ojson *v = ojson_get(obj, key);
    return (v && v->type == OJSON_BOOL) ? v->boolean : def;
}
const char *ojson_get_str(const ojson *obj, const char *key, const char *def) {
    const ojson *v = ojson_get(obj, key);
    return (v && v->type == OJSON_STRING) ? v->str : def;
}
