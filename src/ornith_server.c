/* ornith_server.c — OpenAI/Anthropic-compatible HTTP/1.1 server for Ornith.
 *
 * HTTP server adapted from antirez/ds4 (ds4_server.c), MIT licensed. The
 * connection-per-thread model, the unify-three-protocols-into-one-chat-history
 * design, and the graceful-shutdown / IO-timeout patterns come from ds4 and are
 * re-pointed at the Ornith engine. ds4's DeepSeek/DSML/Metal specifics are not
 * used: rendering uses our ChatML template and parsing uses our ornith_json.
 *
 * Dependency-free (POSIX sockets only). Each accepted connection runs in its
 * own pthread that parses one request; the heavy generation runs under a single
 * global mutex so the one loaded model / KV path is used serially. Three request
 * protocols (OpenAI Chat, OpenAI Responses, Anthropic Messages) collapse to one
 * internal chat_msg[] history, rendered to a ChatML prompt, generated once via
 * rmodel_generate_ids() (token-level, stop-aware), then formatted back per
 * protocol. Clean <|im_end|>/eos stop -> "stop"/"end_turn"; cap -> "length"/
 * "max_tokens".
 */
#define _GNU_SOURCE   /* memmem, strncasecmp */
#include "ornith_server.h"
#include "ornith_rforward.h"
#include "ornith_tokenizer.h"
#include "ornith_json.h"
#include "ornith.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* ======================================================================== */
/* growable byte buffer                                                     */
/* ======================================================================== */

void sbuf_init(sbuf *b) { b->data = malloc(64); b->data[0] = '\0'; b->len = 0; b->cap = 64; }
void sbuf_free(sbuf *b) { free(b->data); b->data = NULL; b->len = b->cap = 0; }

static void sbuf_grow(sbuf *b, size_t need) {
    if (b->len + need + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + need + 1) cap *= 2;
    b->data = realloc(b->data, cap);
    b->cap = cap;
}
void sbuf_append(sbuf *b, const char *s, size_t n) {
    sbuf_grow(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}
void sbuf_puts(sbuf *b, const char *s) { sbuf_append(b, s, strlen(s)); }

static void sbuf_printf(sbuf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) { sbuf_append(b, tmp, (size_t)n); return; }
    char *big = malloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sbuf_append(b, big, (size_t)n);
    free(big);
}

/* ======================================================================== */
/* JSON escaping (rigorous: a model that writes code must not corrupt JSON) */
/* ======================================================================== */

void srv_json_escape(sbuf *b, const char *s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '\\': sbuf_append(b, "\\\\", 2); break;
        case '"':  sbuf_append(b, "\\\"", 2); break;
        case '\n': sbuf_append(b, "\\n", 2);  break;
        case '\r': sbuf_append(b, "\\r", 2);  break;
        case '\t': sbuf_append(b, "\\t", 2);  break;
        case '\b': sbuf_append(b, "\\b", 2);  break;
        case '\f': sbuf_append(b, "\\f", 2);  break;
        default:
            if (c < 0x20) {
                char u[6] = { '\\', 'u', '0', '0', hex[(c >> 4) & 0xF], hex[c & 0xF] };
                sbuf_append(b, u, 6);
            } else {
                sbuf_append(b, (const char *)&c, 1);
            }
        }
    }
}
static void sbuf_jstr(sbuf *b, const char *s) {
    sbuf_append(b, "\"", 1);
    srv_json_escape(b, s, strlen(s));
    sbuf_append(b, "\"", 1);
}

/* ======================================================================== */
/* unified chat history                                                     */
/* ======================================================================== */

void chat_msgs_init(chat_msgs *m) { m->v = NULL; m->len = m->cap = 0; }
void chat_msgs_add(chat_msgs *m, const char *role, const char *content) {
    if (m->len == m->cap) { m->cap = m->cap ? m->cap * 2 : 8;
                            m->v = realloc(m->v, (size_t)m->cap * sizeof(chat_msg)); }
    m->v[m->len].role = strdup(role ? role : "user");
    m->v[m->len].content = strdup(content ? content : "");
    m->len++;
}
void chat_msgs_free(chat_msgs *m) {
    for (int i = 0; i < m->len; i++) { free(m->v[i].role); free(m->v[i].content); }
    free(m->v); m->v = NULL; m->len = m->cap = 0;
}

/* ======================================================================== */
/* ChatML rendering                                                         */
/* ======================================================================== */

char *srv_render_chatml(const chat_msgs *m) {
    sbuf b; sbuf_init(&b);
    for (int i = 0; i < m->len; i++) {
        sbuf_puts(&b, "<|im_start|>");
        sbuf_puts(&b, m->v[i].role);
        sbuf_puts(&b, "\n");
        sbuf_puts(&b, m->v[i].content);
        sbuf_puts(&b, "<|im_end|>\n");
    }
    sbuf_puts(&b, "<|im_start|>assistant\n");
    return b.data;
}

/* ======================================================================== */
/* content collapsing: a content value is a bare string OR an array of typed   */
/* blocks ({type, text}); collapse to a single concatenated text string.      */
/* ======================================================================== */

static char *collapse_content(const ojson *v) {
    if (!v) return strdup("");
    if (v->type == OJSON_STRING) return strdup(v->str ? v->str : "");
    if (v->type == OJSON_ARRAY) {
        sbuf b; sbuf_init(&b);
        for (size_t i = 0; i < v->count; i++) {
            const ojson *blk = v->items[i];
            if (!blk) continue;
            if (blk->type == OJSON_STRING) { sbuf_puts(&b, blk->str ? blk->str : ""); continue; }
            if (blk->type == OJSON_OBJECT) {
                /* input_text/output_text/text all carry the payload in "text" */
                const char *t = ojson_get_str(blk, "text", NULL);
                if (t) sbuf_puts(&b, t);
            }
        }
        return b.data;
    }
    return strdup("");
}

/* ======================================================================== */
/* protocol parsers: each fills the unified history + options               */
/* ======================================================================== */

const char *srv_parse_openai_chat(const ojson *root, chat_msgs *out, gen_opts *o) {
    chat_msgs_init(out);
    o->max_tokens = (int)ojson_get_int(root, "max_tokens", 256);
    if (o->max_tokens <= 0) o->max_tokens = 256;
    o->stream = ojson_get_bool(root, "stream", false);
    const ojson *msgs = ojson_get(root, "messages");
    if (!msgs || msgs->type != OJSON_ARRAY || msgs->count == 0)
        return "missing or empty 'messages' array";
    for (size_t i = 0; i < msgs->count; i++) {
        const ojson *msg = msgs->items[i];
        if (!msg || msg->type != OJSON_OBJECT) return "malformed message object";
        const char *role = ojson_get_str(msg, "role", "user");
        char *content = collapse_content(ojson_get(msg, "content"));
        chat_msgs_add(out, role, content);
        free(content);
    }
    return NULL;
}

const char *srv_parse_responses(const ojson *root, chat_msgs *out, gen_opts *o) {
    chat_msgs_init(out);
    o->max_tokens = (int)ojson_get_int(root, "max_output_tokens", 256);
    if (o->max_tokens <= 0) o->max_tokens = 256;
    o->stream = ojson_get_bool(root, "stream", false);
    const char *instr = ojson_get_str(root, "instructions", NULL);
    if (instr) chat_msgs_add(out, "system", instr);
    const ojson *input = ojson_get(root, "input");
    if (!input) return "missing 'input'";
    if (input->type == OJSON_STRING) {
        chat_msgs_add(out, "user", input->str ? input->str : "");
    } else if (input->type == OJSON_ARRAY) {
        /* array of items; each may be a content block or a {role,content} msg */
        for (size_t i = 0; i < input->count; i++) {
            const ojson *it = input->items[i];
            if (!it) continue;
            if (it->type == OJSON_STRING) { chat_msgs_add(out, "user", it->str); continue; }
            if (it->type != OJSON_OBJECT) continue;
            const ojson *c = ojson_get(it, "content");
            const char *role = ojson_get_str(it, "role", "user");
            if (c) { char *t = collapse_content(c); chat_msgs_add(out, role, t); free(t); }
            else {
                /* a bare typed content block: {type, text} */
                const char *t = ojson_get_str(it, "text", NULL);
                if (t) chat_msgs_add(out, "user", t);
            }
        }
    } else {
        return "'input' must be a string or array";
    }
    if (out->len == (instr ? 1 : 0)) return "empty 'input'";
    return NULL;
}

const char *srv_parse_anthropic(const ojson *root, chat_msgs *out, gen_opts *o) {
    chat_msgs_init(out);
    o->max_tokens = (int)ojson_get_int(root, "max_tokens", 256);
    if (o->max_tokens <= 0) o->max_tokens = 256;
    o->stream = ojson_get_bool(root, "stream", false);
    const ojson *sys = ojson_get(root, "system");
    if (sys) { char *t = collapse_content(sys); if (t[0]) chat_msgs_add(out, "system", t); free(t); }
    const ojson *msgs = ojson_get(root, "messages");
    if (!msgs || msgs->type != OJSON_ARRAY || msgs->count == 0)
        return "missing or empty 'messages' array";
    for (size_t i = 0; i < msgs->count; i++) {
        const ojson *msg = msgs->items[i];
        if (!msg || msg->type != OJSON_OBJECT) return "malformed message object";
        const char *role = ojson_get_str(msg, "role", "user");
        char *content = collapse_content(ojson_get(msg, "content"));
        chat_msgs_add(out, role, content);
        free(content);
    }
    return NULL;
}

/* ======================================================================== */
/* non-streaming response builders (pure, unit-tested)                      */
/* ======================================================================== */

char *srv_build_chat_response(const char *id, const char *model, long created,
                              const char *content, bool length,
                              int prompt_tok, int compl_tok) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"id\":"); sbuf_jstr(&b, id);
    sbuf_printf(&b, ",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", created);
    sbuf_jstr(&b, model);
    sbuf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sbuf_jstr(&b, content);
    sbuf_puts(&b, "},\"finish_reason\":");
    sbuf_jstr(&b, length ? "length" : "stop");
    sbuf_printf(&b, "}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                "\"total_tokens\":%d}}", prompt_tok, compl_tok, prompt_tok + compl_tok);
    return b.data;
}

char *srv_build_responses_response(const char *id, const char *model,
                                   const char *content, bool length,
                                   int prompt_tok, int compl_tok) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"id\":"); sbuf_jstr(&b, id);
    sbuf_puts(&b, ",\"object\":\"response\",\"status\":\"completed\",\"model\":");
    sbuf_jstr(&b, model);
    sbuf_puts(&b, ",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":"
                  "[{\"type\":\"output_text\",\"text\":");
    sbuf_jstr(&b, content);
    sbuf_puts(&b, "}]}]");
    (void)length; /* responses has no top-level stop field; status is "completed" */
    sbuf_printf(&b, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d,"
                "\"total_tokens\":%d}}", prompt_tok, compl_tok, prompt_tok + compl_tok);
    return b.data;
}

char *srv_build_anthropic_response(const char *id, const char *model,
                                   const char *content, bool length,
                                   int prompt_tok, int compl_tok) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"id\":"); sbuf_jstr(&b, id);
    sbuf_puts(&b, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    sbuf_jstr(&b, model);
    sbuf_puts(&b, ",\"content\":[{\"type\":\"text\",\"text\":");
    sbuf_jstr(&b, content);
    sbuf_puts(&b, "}],\"stop_reason\":");
    sbuf_jstr(&b, length ? "max_tokens" : "end_turn");
    sbuf_printf(&b, ",\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,"
                "\"output_tokens\":%d}}", prompt_tok, compl_tok);
    return b.data;
}

/* ======================================================================== */
/* below here: networking + the live engine (not unit-tested)               */
/* ======================================================================== */

#define ORNITH_IO_TIMEOUT_SEC_DEFAULT 10

static rmodel       *g_model;
static const char   *g_model_id;
static int32_t       g_im_start, g_im_end;
static pthread_mutex_t g_gen_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_id_lock  = PTHREAD_MUTEX_INITIALIZER;
static unsigned long g_req_counter;
static int           g_io_timeout = ORNITH_IO_TIMEOUT_SEC_DEFAULT;

/* ---- prompt id building (interleave special tokens with BPE) ----------- */

typedef struct { int32_t *ids; int n, cap; } idvec;
static void idvec_push(idvec *v, int32_t id) {
    if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 256;
                          v->ids = realloc(v->ids, (size_t)v->cap * sizeof(int32_t)); }
    v->ids[v->n++] = id;
}
static void idvec_encode(idvec *v, const otokenizer *t, const char *text, size_t len) {
    if (len == 0) return;
    char *z = malloc(len + 1);
    memcpy(z, text, len); z[len] = '\0';
    int32_t *ids = NULL; int n = 0;
    otok_encode(t, z, &ids, &n);
    for (int i = 0; i < n; i++) idvec_push(v, ids[i]);
    free(ids); free(z);
}
static int32_t *build_prompt_ids(const otokenizer *t, const char *chatml, int *out_n) {
    static const char *S = "<|im_start|>";
    static const char *E = "<|im_end|>";
    size_t ls = strlen(S), le = strlen(E);
    idvec v = {0};
    const char *p = chatml;
    while (*p) {
        if (g_im_start >= 0 && strncmp(p, S, ls) == 0) { idvec_push(&v, g_im_start); p += ls; }
        else if (g_im_end >= 0 && strncmp(p, E, le) == 0) { idvec_push(&v, g_im_end); p += le; }
        else {
            const char *q = p;
            while (*q) {
                if (g_im_start >= 0 && strncmp(q, S, ls) == 0) break;
                if (g_im_end   >= 0 && strncmp(q, E, le) == 0) break;
                q++;
            }
            idvec_encode(&v, t, p, (size_t)(q - p));
            p = q;
        }
    }
    *out_n = v.n;
    return v.ids;
}

/* ---- socket write + responses ------------------------------------------ */

static int write_all(int fd, const char *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, buf + off, n - off);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        off += (size_t)w;
    }
    return 0;
}
static void send_response(int fd, int status, const char *status_text,
                          const char *ctype, const char *body, size_t blen) {
    sbuf h; sbuf_init(&h);
    sbuf_printf(&h, "HTTP/1.1 %d %s\r\n", status, status_text);
    sbuf_printf(&h, "Content-Type: %s\r\n", ctype);
    sbuf_printf(&h, "Content-Length: %zu\r\n", blen);
    sbuf_puts(&h, "Connection: close\r\n\r\n");
    write_all(fd, h.data, h.len);
    if (body && blen) write_all(fd, body, blen);
    sbuf_free(&h);
}
static void send_json(int fd, int status, const char *status_text, const char *body) {
    send_response(fd, status, status_text, "application/json", body, strlen(body));
}
static void send_error(int fd, int status, const char *status_text,
                       const char *type, const char *message) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"error\":{\"message\":"); sbuf_jstr(&b, message);
    sbuf_puts(&b, ",\"type\":"); sbuf_jstr(&b, type);
    sbuf_puts(&b, "}}");
    send_json(fd, status, status_text, b.data);
    sbuf_free(&b);
}
static void send_sse_headers(int fd) {
    const char *h = "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: close\r\n\r\n";
    write_all(fd, h, strlen(h));
}

static void make_id(char *buf, size_t cap, const char *prefix) {
    pthread_mutex_lock(&g_id_lock);
    unsigned long c = g_req_counter++;
    pthread_mutex_unlock(&g_id_lock);
    snprintf(buf, cap, "%s-%ld%03lu", prefix, (long)time(NULL), c % 1000);
}

/* ======================================================================== */
/* generation drivers                                                       */
/* ======================================================================== */

typedef enum { API_CHAT, API_RESPONSES, API_ANTHROPIC } api_style;

/* non-stream collector */
typedef struct { sbuf text; int count; } collect_ctx;
static void on_token_collect(int32_t id, const char *piece, void *ud) {
    (void)id; collect_ctx *c = ud;
    sbuf_append(&c->text, piece, strlen(piece));
    c->count++;
}

/* stream emitter: per-protocol delta event */
typedef struct { int fd; api_style api; int count; } stream_ctx;
static void stream_delta(stream_ctx *c, const char *piece) {
    sbuf b; sbuf_init(&b);
    if (c->api == API_CHAT) {
        sbuf_puts(&b, "data: {\"object\":\"chat.completion.chunk\",\"choices\":"
                      "[{\"index\":0,\"delta\":{\"content\":");
        sbuf_jstr(&b, piece);
        sbuf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    } else if (c->api == API_RESPONSES) {
        sbuf_puts(&b, "event: response.output_text.delta\n"
                      "data: {\"type\":\"response.output_text.delta\",\"delta\":");
        sbuf_jstr(&b, piece);
        sbuf_puts(&b, "}\n\n");
    } else { /* anthropic */
        sbuf_puts(&b, "event: content_block_delta\n"
                      "data: {\"type\":\"content_block_delta\",\"index\":0,"
                      "\"delta\":{\"type\":\"text_delta\",\"text\":");
        sbuf_jstr(&b, piece);
        sbuf_puts(&b, "}}\n\n");
    }
    write_all(c->fd, b.data, b.len);
    sbuf_free(&b);
}
static void on_token_stream(int32_t id, const char *piece, void *ud) {
    (void)id; stream_ctx *c = ud;
    stream_delta(c, piece);
    c->count++;
}

/* Run greedy generation under the global lock, collecting text + count. */
static int generate_collect(const int32_t *pids, int n_prompt, int max_tokens,
                            sbuf *out_text, int *out_count, int *out_finish) {
    int32_t stop_ids[1]; int n_stop = 0;
    if (g_im_end >= 0) stop_ids[n_stop++] = g_im_end;
    collect_ctx cc; sbuf_init(&cc.text); cc.count = 0;
    int finish = 1;
    pthread_mutex_lock(&g_gen_lock);
    ornith_status st = rmodel_generate_ids(g_model, pids, n_prompt, max_tokens,
                                           stop_ids, n_stop, on_token_collect,
                                           &cc, &finish);
    pthread_mutex_unlock(&g_gen_lock);
    if (st != ORNITH_OK) { sbuf_free(&cc.text); return -1; }
    *out_text = cc.text; *out_count = cc.count; *out_finish = finish;
    return 0;
}

/* ======================================================================== */
/* request dispatch (one chat history -> one generation -> per-protocol fmt)*/
/* ======================================================================== */

static void run_request(int fd, api_style api, const ojson *root) {
    chat_msgs hist; gen_opts opt = {256, false};
    const char *perr = NULL;
    if (api == API_CHAT)            perr = srv_parse_openai_chat(root, &hist, &opt);
    else if (api == API_RESPONSES)  perr = srv_parse_responses(root, &hist, &opt);
    else                            perr = srv_parse_anthropic(root, &hist, &opt);
    if (perr) {
        chat_msgs_free(&hist);
        send_error(fd, 400, "Bad Request", "invalid_request_error", perr);
        return;
    }

    char *chatml = srv_render_chatml(&hist);
    chat_msgs_free(&hist);
    int n_prompt = 0;
    int32_t *pids = build_prompt_ids(rmodel_tokenizer(g_model), chatml, &n_prompt);
    free(chatml);
    if (n_prompt == 0) {
        free(pids);
        send_error(fd, 400, "Bad Request", "invalid_request_error", "empty prompt");
        return;
    }

    const char *prefix = api == API_CHAT ? "chatcmpl"
                       : api == API_RESPONSES ? "resp" : "msg";
    char id[64]; make_id(id, sizeof(id), prefix);
    long created = (long)time(NULL);

    if (opt.stream) {
        send_sse_headers(fd);
        stream_ctx sc = { fd, api, 0 };
        /* opening events per protocol */
        if (api == API_CHAT) {
            sbuf b; sbuf_init(&b);
            sbuf_puts(&b, "data: {\"id\":"); sbuf_jstr(&b, id);
            sbuf_printf(&b, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", created);
            sbuf_jstr(&b, g_model_id);
            sbuf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n");
            write_all(fd, b.data, b.len); sbuf_free(&b);
        } else if (api == API_RESPONSES) {
            sbuf b; sbuf_init(&b);
            sbuf_puts(&b, "event: response.created\ndata: {\"type\":\"response.created\","
                          "\"response\":{\"id\":"); sbuf_jstr(&b, id);
            sbuf_puts(&b, ",\"object\":\"response\",\"status\":\"in_progress\",\"model\":");
            sbuf_jstr(&b, g_model_id); sbuf_puts(&b, "}}\n\n");
            write_all(fd, b.data, b.len); sbuf_free(&b);
        } else { /* anthropic */
            sbuf b; sbuf_init(&b);
            sbuf_puts(&b, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":");
            sbuf_jstr(&b, id);
            sbuf_puts(&b, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
            sbuf_jstr(&b, g_model_id);
            sbuf_printf(&b, ",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}\n\n", n_prompt);
            sbuf_puts(&b, "event: content_block_start\ndata: {\"type\":\"content_block_start\","
                          "\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
            write_all(fd, b.data, b.len); sbuf_free(&b);
        }

        int32_t stop_ids[1]; int n_stop = 0;
        if (g_im_end >= 0) stop_ids[n_stop++] = g_im_end;
        int finish = 1;
        pthread_mutex_lock(&g_gen_lock);
        rmodel_generate_ids(g_model, pids, n_prompt, opt.max_tokens, stop_ids, n_stop,
                            on_token_stream, &sc, &finish);
        pthread_mutex_unlock(&g_gen_lock);
        free(pids);

        /* closing events per protocol */
        if (api == API_CHAT) {
            sbuf b; sbuf_init(&b);
            sbuf_puts(&b, "data: {\"object\":\"chat.completion.chunk\",\"choices\":"
                          "[{\"index\":0,\"delta\":{},\"finish_reason\":");
            sbuf_jstr(&b, finish ? "length" : "stop");
            sbuf_puts(&b, "}]}\n\ndata: [DONE]\n\n");
            write_all(fd, b.data, b.len); sbuf_free(&b);
        } else if (api == API_RESPONSES) {
            sbuf b; sbuf_init(&b);
            sbuf_puts(&b, "event: response.completed\ndata: {\"type\":\"response.completed\","
                          "\"response\":{\"id\":"); sbuf_jstr(&b, id);
            sbuf_puts(&b, ",\"object\":\"response\",\"status\":\"completed\",\"model\":");
            sbuf_jstr(&b, g_model_id);
            sbuf_printf(&b, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d,"
                        "\"total_tokens\":%d}}}\n\n", n_prompt, sc.count, n_prompt + sc.count);
            write_all(fd, b.data, b.len); sbuf_free(&b);
        } else { /* anthropic */
            sbuf b; sbuf_init(&b);
            sbuf_puts(&b, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
            sbuf_puts(&b, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
            sbuf_jstr(&b, finish ? "max_tokens" : "end_turn");
            sbuf_printf(&b, ",\"stop_sequence\":null},\"usage\":{\"output_tokens\":%d}}\n\n", sc.count);
            sbuf_puts(&b, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
            write_all(fd, b.data, b.len); sbuf_free(&b);
        }
        return;
    }

    /* non-streaming */
    sbuf text; int count = 0, finish = 1;
    if (generate_collect(pids, n_prompt, opt.max_tokens, &text, &count, &finish) != 0) {
        free(pids);
        send_error(fd, 500, "Internal Server Error", "server_error", "generation failed");
        return;
    }
    free(pids);
    char *body = NULL;
    bool length = finish != 0;
    if (api == API_CHAT)
        body = srv_build_chat_response(id, g_model_id, created, text.data, length, n_prompt, count);
    else if (api == API_RESPONSES)
        body = srv_build_responses_response(id, g_model_id, text.data, length, n_prompt, count);
    else
        body = srv_build_anthropic_response(id, g_model_id, text.data, length, n_prompt, count);
    send_json(fd, 200, "OK", body);
    free(body);
    sbuf_free(&text);
}

/* ---- simple GET handlers ----------------------------------------------- */

static void handle_health(int fd) { send_json(fd, 200, "OK", "{\"status\":\"ok\"}"); }
static void handle_models(int fd) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":"); sbuf_jstr(&b, g_model_id);
    sbuf_puts(&b, ",\"object\":\"model\",\"owned_by\":\"ornithology\"}]}");
    send_json(fd, 200, "OK", b.data);
    sbuf_free(&b);
}

/* ======================================================================== */
/* HTTP read + per-connection thread                                        */
/* ======================================================================== */

static int read_request(int fd, sbuf *req, size_t *body_off) {
    sbuf_init(req);
    char tmp[4096];
    size_t hdr_end = 0;
    for (;;) {
        if (req->len >= 4) {
            char *m = memmem(req->data, req->len, "\r\n\r\n", 4);
            if (m) { hdr_end = (size_t)(m - req->data) + 4; break; }
        }
        ssize_t r = read(fd, tmp, sizeof(tmp));
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        sbuf_append(req, tmp, (size_t)r);
        if (req->len > (64u << 20)) return -1;   /* 64 MB header guard */
    }
    *body_off = hdr_end;
    long clen = 0;
    for (size_t i = 0; i + 15 < hdr_end; i++)
        if (strncasecmp(req->data + i, "Content-Length:", 15) == 0) {
            clen = strtol(req->data + i + 15, NULL, 10); break;
        }
    while ((long)(req->len - hdr_end) < clen) {
        ssize_t r = read(fd, tmp, sizeof(tmp));
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) break;
        sbuf_append(req, tmp, (size_t)r);
    }
    return 0;
}

static void handle_connection(int fd) {
    /* IO timeouts so a slow/stuck client can't hold a worker thread. */
    struct timeval tv = { g_io_timeout, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sbuf req; size_t body_off = 0;
    if (read_request(fd, &req, &body_off) != 0) { sbuf_free(&req); return; }

    char method[16] = {0}, path[1024] = {0};
    sscanf(req.data, "%15s %1023s", method, path);
    int is_get = strcmp(method, "GET") == 0, is_post = strcmp(method, "POST") == 0;

    if (is_get && strcmp(path, "/health") == 0)    { handle_health(fd); sbuf_free(&req); return; }
    if (is_get && strcmp(path, "/v1/models") == 0) { handle_models(fd); sbuf_free(&req); return; }

    api_style api;
    int matched = 1;
    if (is_post && strcmp(path, "/v1/chat/completions") == 0) api = API_CHAT;
    else if (is_post && strcmp(path, "/v1/responses") == 0)   api = API_RESPONSES;
    else if (is_post && strcmp(path, "/v1/messages") == 0)    api = API_ANTHROPIC;
    else matched = 0;

    if (!matched) {
        send_error(fd, 404, "Not Found", "invalid_request_error", "unknown route");
        sbuf_free(&req);
        return;
    }

    const char *errpos = NULL;
    ojson *root = ojson_parse(req.data + body_off, &errpos);
    if (!root || root->type != OJSON_OBJECT) {
        if (root) ojson_free(root);
        send_error(fd, 400, "Bad Request", "invalid_request_error", "malformed JSON body");
        sbuf_free(&req);
        return;
    }
    run_request(fd, api, root);
    ojson_free(root);
    sbuf_free(&req);
}

static void *conn_thread(void *arg) {
    int fd = (int)(intptr_t)arg;
    handle_connection(fd);
    close(fd);
    return NULL;
}

/* ======================================================================== */
/* server main                                                              */
/* ======================================================================== */

static volatile sig_atomic_t g_stop;
static int g_listen_fd = -1;
static void on_signal(int s) {
    (void)s;
    if (g_stop) _exit(130);     /* second signal: hard exit */
    g_stop = 1;
    if (g_listen_fd >= 0) close(g_listen_fd);  /* unblock accept() */
}

int ornith_server_main(int argc, char **argv) {
    const char *host = "127.0.0.1";
    int port = 8080;
    const char *model_path = NULL;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else model_path = argv[i];
    }
    if (!model_path) {
        fprintf(stderr, "serve: usage: ornith serve [--host H] [--port P] <model.gguf>\n");
        return 1;
    }
    const char *to = getenv("DS4_SERVER_IO_TIMEOUT_SEC");
    if (!to) to = getenv("ORNITH_SERVER_IO_TIMEOUT_SEC");
    if (to) { int v = atoi(to); if (v > 0) g_io_timeout = v; }

    fprintf(stderr, "serve: loading %s ...\n", model_path);
    ornith_status st = rmodel_load(model_path, &g_model);
    if (st != ORNITH_OK) {
        fprintf(stderr, "serve: load failed: %s\n", ornith_last_error());
        return 1;
    }
    const otokenizer *t = rmodel_tokenizer(g_model);
    g_im_start = otok_id_of(t, "<|im_start|>");
    g_im_end   = otok_id_of(t, "<|im_end|>");
    const char *name = rmodel_name(g_model);
    g_model_id = name ? name : model_path;
    const ornith_arch *a = rmodel_arch(g_model);
    fprintf(stderr, "serve: model '%s' loaded (hidden %d, %d layers, vocab %d, "
            "eos %d, im_start %d, im_end %d)\n", g_model_id, a->hidden_size,
            a->num_layers, a->vocab_size, a->eos_token_id, g_im_start, g_im_end);
    if (g_im_start < 0 || g_im_end < 0)
        fprintf(stderr, "serve: WARNING: ChatML special tokens missing from vocab\n");

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); rmodel_free(g_model); return 1; }
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(srv); rmodel_free(g_model); return 1;
    }
    if (listen(srv, 16) < 0) {
        perror("listen"); close(srv); rmodel_free(g_model); return 1;
    }
    g_listen_fd = srv;
    fprintf(stderr, "serve: listening on http://%s:%d  (greedy, CPU, generation "
            "serialized; Ctrl-C to stop)\n", host, port);

    while (!g_stop) {
        int cfd = accept(srv, NULL, NULL);
        if (cfd < 0) { if (g_stop) break; if (errno == EINTR) continue; perror("accept"); continue; }
        pthread_t th;
        if (pthread_create(&th, NULL, conn_thread, (void *)(intptr_t)cfd) != 0) {
            close(cfd); continue;
        }
        pthread_detach(th);
    }

    fprintf(stderr, "\nserve: shutting down\n");
    if (g_listen_fd >= 0) close(g_listen_fd);
    rmodel_free(g_model);
    return 0;
}
