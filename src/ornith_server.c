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

/* Compact-serialize a parsed ojson DOM back to a JSON string. Used to render a
 * tool's `parameters` schema into the prompt and to re-emit a tool call's
 * `arguments`/`input` object. Numbers that are exact integers print without a
 * decimal point so schema-ish JSON round-trips cleanly. */
static void ojson_stringify(sbuf *b, const ojson *v) {
    if (!v) { sbuf_puts(b, "null"); return; }
    switch (v->type) {
    case OJSON_NULL:   sbuf_puts(b, "null"); break;
    case OJSON_BOOL:   sbuf_puts(b, v->boolean ? "true" : "false"); break;
    case OJSON_NUMBER: {
        double d = v->num;
        char tmp[40];
        if (d == (double)(long long)d && d < 1e15 && d > -1e15)
            snprintf(tmp, sizeof tmp, "%lld", (long long)d);
        else
            snprintf(tmp, sizeof tmp, "%.17g", d);
        sbuf_puts(b, tmp);
        break;
    }
    case OJSON_STRING: sbuf_jstr(b, v->str ? v->str : ""); break;
    case OJSON_ARRAY:
        sbuf_append(b, "[", 1);
        for (size_t i = 0; i < v->count; i++) {
            if (i) sbuf_append(b, ",", 1);
            ojson_stringify(b, v->items[i]);
        }
        sbuf_append(b, "]", 1);
        break;
    case OJSON_OBJECT:
        sbuf_append(b, "{", 1);
        for (size_t i = 0; i < v->count; i++) {
            if (i) sbuf_append(b, ",", 1);
            sbuf_jstr(b, v->keys[i] ? v->keys[i] : "");
            sbuf_append(b, ":", 1);
            ojson_stringify(b, v->items[i]);
        }
        sbuf_append(b, "}", 1);
        break;
    }
}

/* ======================================================================== */
/* tool / function calling: data structures                                 */
/* ======================================================================== */

void srv_tools_init(srv_tools *t) { t->v = NULL; t->len = t->cap = 0; }
static void srv_tools_push(srv_tools *t, char *name, char *desc, char *params) {
    if (t->len == t->cap) { t->cap = t->cap ? t->cap * 2 : 4;
                            t->v = realloc(t->v, (size_t)t->cap * sizeof(srv_tool)); }
    t->v[t->len].name = name;
    t->v[t->len].description = desc;
    t->v[t->len].parameters = params;
    t->len++;
}
void srv_tools_free(srv_tools *t) {
    for (int i = 0; i < t->len; i++) {
        free(t->v[i].name); free(t->v[i].description); free(t->v[i].parameters);
    }
    free(t->v); t->v = NULL; t->len = t->cap = 0;
}

void srv_toolcalls_init(srv_toolcalls *t) { t->v = NULL; t->len = t->cap = 0; }
static void srv_toolcalls_push(srv_toolcalls *t, char *id, char *name, char *args) {
    if (t->len == t->cap) { t->cap = t->cap ? t->cap * 2 : 4;
                            t->v = realloc(t->v, (size_t)t->cap * sizeof(srv_toolcall)); }
    t->v[t->len].id = id;
    t->v[t->len].name = name;
    t->v[t->len].arguments = args;
    t->len++;
}
void srv_toolcalls_free(srv_toolcalls *t) {
    for (int i = 0; i < t->len; i++) {
        free(t->v[i].id); free(t->v[i].name); free(t->v[i].arguments);
    }
    free(t->v); t->v = NULL; t->len = t->cap = 0;
}

void gen_opts_free(gen_opts *o) {
    if (!o) return;
    srv_tools_free(&o->tools);
    free(o->tool_choice_name);
    o->tool_choice_name = NULL;
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

/* Render the Qwen-style tools section (without the surrounding system block).
 * One JSON object per line inside <tools>...</tools>, then the <tool_call>
 * usage instructions, plus an optional REQUIRED/NAMED directive. */
static void append_tools_section(sbuf *b, const srv_tools *tools,
                                 tool_choice_kind tool_choice,
                                 const char *tool_choice_name) {
    sbuf_puts(b, "# Tools\n\nYou may call one or more functions to assist with "
                 "the user query.\n\nYou are provided with function signatures "
                 "within <tools></tools> XML tags:\n<tools>\n");
    for (int i = 0; i < tools->len; i++) {
        sbuf_puts(b, "{\"type\": \"function\", \"function\": {\"name\": ");
        sbuf_jstr(b, tools->v[i].name ? tools->v[i].name : "");
        if (tools->v[i].description) {
            sbuf_puts(b, ", \"description\": ");
            sbuf_jstr(b, tools->v[i].description);
        }
        sbuf_puts(b, ", \"parameters\": ");
        sbuf_puts(b, tools->v[i].parameters ? tools->v[i].parameters : "{}");
        sbuf_puts(b, "}}\n");
    }
    sbuf_puts(b, "</tools>\n\nFor each function call, return a json object with "
                 "function name and arguments within <tool_call></tool_call> XML "
                 "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
                 "<args-json-object>}\n</tool_call>");
    if (tool_choice == TOOL_CHOICE_REQUIRED)
        sbuf_puts(b, "\n\nYou must call at least one of the functions above.");
    else if (tool_choice == TOOL_CHOICE_NAMED && tool_choice_name) {
        sbuf_puts(b, "\n\nYou must call the function named ");
        sbuf_jstr(b, tool_choice_name);
        sbuf_puts(b, ".");
    }
}

char *srv_render_chatml_tools(const chat_msgs *m, const srv_tools *tools,
                             tool_choice_kind tool_choice,
                             const char *tool_choice_name) {
    sbuf b; sbuf_init(&b);
    bool have_tools = tools && tools->len > 0;
    int start = 0;
    if (have_tools) {
        bool first_is_system = m->len > 0 && strcmp(m->v[0].role, "system") == 0;
        sbuf_puts(&b, "<|im_start|>system\n");
        if (first_is_system) {
            sbuf_puts(&b, m->v[0].content);
            sbuf_puts(&b, "\n\n");
            start = 1;
        }
        append_tools_section(&b, tools, tool_choice, tool_choice_name);
        sbuf_puts(&b, "<|im_end|>\n");
    }
    for (int i = start; i < m->len; i++) {
        sbuf_puts(&b, "<|im_start|>");
        sbuf_puts(&b, m->v[i].role);
        sbuf_puts(&b, "\n");
        sbuf_puts(&b, m->v[i].content);
        sbuf_puts(&b, "<|im_end|>\n");
    }
    sbuf_puts(&b, "<|im_start|>assistant\n");
    return b.data;
}

char *srv_render_chatml(const chat_msgs *m) {
    return srv_render_chatml_tools(m, NULL, TOOL_CHOICE_AUTO, NULL);
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
/* tool / function calling: request parsing helpers                         */
/* ======================================================================== */

/* OpenAI tools: [{type:"function", function:{name, description, parameters}}].
 * (We tolerate a flat {name, description, parameters} too.) */
static void parse_tools_openai(const ojson *arr, srv_tools *out) {
    if (!arr || arr->type != OJSON_ARRAY) return;
    for (size_t i = 0; i < arr->count; i++) {
        const ojson *it = arr->items[i];
        if (!it || it->type != OJSON_OBJECT) continue;
        const ojson *fn = ojson_get(it, "function");
        if (!fn) fn = it;
        const char *name = ojson_get_str(fn, "name", NULL);
        if (!name) continue;
        const char *desc = ojson_get_str(fn, "description", NULL);
        const ojson *params = ojson_get(fn, "parameters");
        char *ps = NULL;
        if (params) { sbuf b; sbuf_init(&b); ojson_stringify(&b, params); ps = b.data; }
        srv_tools_push(out, strdup(name), desc ? strdup(desc) : NULL, ps);
    }
}

/* Anthropic tools: [{name, description, input_schema}]. */
static void parse_tools_anthropic(const ojson *arr, srv_tools *out) {
    if (!arr || arr->type != OJSON_ARRAY) return;
    for (size_t i = 0; i < arr->count; i++) {
        const ojson *it = arr->items[i];
        if (!it || it->type != OJSON_OBJECT) continue;
        const char *name = ojson_get_str(it, "name", NULL);
        if (!name) continue;
        const char *desc = ojson_get_str(it, "description", NULL);
        const ojson *params = ojson_get(it, "input_schema");
        char *ps = NULL;
        if (params) { sbuf b; sbuf_init(&b); ojson_stringify(&b, params); ps = b.data; }
        srv_tools_push(out, strdup(name), desc ? strdup(desc) : NULL, ps);
    }
}

/* OpenAI tool_choice: "auto"|"none"|"required" or {type:"function",function:{name}}. */
static void parse_tool_choice_openai(const ojson *root, gen_opts *o) {
    const ojson *tc = ojson_get(root, "tool_choice");
    if (!tc) return;
    if (tc->type == OJSON_STRING) {
        if      (!strcmp(tc->str, "none"))     o->tool_choice = TOOL_CHOICE_NONE;
        else if (!strcmp(tc->str, "required")) o->tool_choice = TOOL_CHOICE_REQUIRED;
        else                                   o->tool_choice = TOOL_CHOICE_AUTO;
    } else if (tc->type == OJSON_OBJECT) {
        const ojson *fn = ojson_get(tc, "function");
        const char *nm = fn ? ojson_get_str(fn, "name", NULL) : NULL;
        if (nm) { o->tool_choice = TOOL_CHOICE_NAMED;
                  free(o->tool_choice_name); o->tool_choice_name = strdup(nm); }
    }
}

/* Anthropic tool_choice: {type:"auto"|"any"|"tool"|"none", name?}. */
static void parse_tool_choice_anthropic(const ojson *root, gen_opts *o) {
    const ojson *tc = ojson_get(root, "tool_choice");
    if (!tc || tc->type != OJSON_OBJECT) return;
    const char *type = ojson_get_str(tc, "type", NULL);
    if (!type) return;
    if      (!strcmp(type, "none")) o->tool_choice = TOOL_CHOICE_NONE;
    else if (!strcmp(type, "any"))  o->tool_choice = TOOL_CHOICE_REQUIRED;
    else if (!strcmp(type, "tool")) {
        const char *nm = ojson_get_str(tc, "name", NULL);
        if (nm) { o->tool_choice = TOOL_CHOICE_NAMED;
                  free(o->tool_choice_name); o->tool_choice_name = strdup(nm); }
        else o->tool_choice = TOOL_CHOICE_REQUIRED;
    } else o->tool_choice = TOOL_CHOICE_AUTO;
}

/* Append one ChatML <tool_call> block for a call with the given name and a
 * pre-serialized arguments JSON object string. */
static void append_tool_call_block(sbuf *b, const char *name, const char *args_json) {
    if (b->len) sbuf_puts(b, "\n");
    sbuf_puts(b, "<tool_call>\n{\"name\": ");
    sbuf_jstr(b, name ? name : "");
    sbuf_puts(b, ", \"arguments\": ");
    sbuf_puts(b, (args_json && args_json[0]) ? args_json : "{}");
    sbuf_puts(b, "}\n</tool_call>");
}

/* Render a prior OpenAI assistant message that carried tool_calls into ChatML
 * assistant content: any text, then a <tool_call> block per call. The OpenAI
 * wire form keeps arguments as a JSON string, which we inline verbatim. */
static char *render_openai_assistant_toolcalls(const ojson *msg) {
    sbuf b; sbuf_init(&b);
    const ojson *c = ojson_get(msg, "content");
    if (c) { char *t = collapse_content(c); if (t[0]) sbuf_puts(&b, t); free(t); }
    const ojson *tcs = ojson_get(msg, "tool_calls");
    for (size_t i = 0; i < tcs->count; i++) {
        const ojson *tc = tcs->items[i];
        if (!tc || tc->type != OJSON_OBJECT) continue;
        const ojson *fn = ojson_get(tc, "function");
        if (!fn) continue;
        const char *nm = ojson_get_str(fn, "name", NULL);
        const ojson *args = ojson_get(fn, "arguments");
        char *as = NULL;
        if (args && args->type == OJSON_STRING) as = strdup(args->str ? args->str : "{}");
        else if (args) { sbuf ab; sbuf_init(&ab); ojson_stringify(&ab, args); as = ab.data; }
        append_tool_call_block(&b, nm, as);
        free(as);
    }
    return b.data;
}

/* Add one Anthropic message (whose content may be an array of typed blocks) to
 * the unified history. text blocks accumulate into the message's own role;
 * tool_use blocks (assistant) become <tool_call> blocks appended to it;
 * tool_result blocks (user) are flushed as separate `tool` role messages. */
static void anthropic_add_message(chat_msgs *out, const char *role, const ojson *content) {
    if (!content) { chat_msgs_add(out, role, ""); return; }
    if (content->type == OJSON_STRING) {
        chat_msgs_add(out, role, content->str ? content->str : ""); return;
    }
    if (content->type != OJSON_ARRAY) {
        char *t = collapse_content(content); chat_msgs_add(out, role, t); free(t); return;
    }
    sbuf text; sbuf_init(&text);
    sbuf calls; sbuf_init(&calls);
    for (size_t i = 0; i < content->count; i++) {
        const ojson *blk = content->items[i];
        if (!blk || blk->type != OJSON_OBJECT) continue;
        const char *type = ojson_get_str(blk, "type", NULL);
        if (type && !strcmp(type, "tool_use")) {
            const char *nm = ojson_get_str(blk, "name", NULL);
            const ojson *input = ojson_get(blk, "input");
            char *as = NULL;
            if (input) { sbuf ab; sbuf_init(&ab); ojson_stringify(&ab, input); as = ab.data; }
            append_tool_call_block(&calls, nm, as);
            free(as);
        } else if (type && !strcmp(type, "tool_result")) {
            char *rt = collapse_content(ojson_get(blk, "content"));
            chat_msgs_add(out, "tool", rt);
            free(rt);
        } else {
            const char *t = ojson_get_str(blk, "text", NULL);
            if (t) sbuf_puts(&text, t);
        }
    }
    if (calls.len) {
        sbuf merged; sbuf_init(&merged);
        if (text.len) { sbuf_puts(&merged, text.data); sbuf_puts(&merged, "\n"); }
        sbuf_puts(&merged, calls.data);
        chat_msgs_add(out, role, merged.data);
        sbuf_free(&merged);
    } else if (text.len) {
        chat_msgs_add(out, role, text.data);
    }
    /* else: only tool_result blocks -> already added as `tool` messages */
    sbuf_free(&text);
    sbuf_free(&calls);
}

/* ======================================================================== */
/* protocol parsers: each fills the unified history + options               */
/* ======================================================================== */

/* Parse the sampler knobs common to all three protocols into o->samp. Absent
 * fields keep the greedy default (temperature 0), so behaviour stays
 * deterministic unless the client explicitly asks for sampling. The field names
 * line up across OpenAI/Anthropic (temperature, top_p, top_k); seed/min_p/
 * repeat_penalty are accepted as extensions wherever a client sends them. */
static void srv_parse_sampling(const ojson *root, gen_opts *o) {
    osample_params s = osample_params_default();
    s.temperature    = (float)ojson_get_num(root, "temperature",    s.temperature);
    s.top_p          = (float)ojson_get_num(root, "top_p",          s.top_p);
    s.top_k          = (int)  ojson_get_int(root, "top_k",          s.top_k);
    s.min_p          = (float)ojson_get_num(root, "min_p",          s.min_p);
    s.repeat_penalty = (float)ojson_get_num(root, "repeat_penalty", s.repeat_penalty);
    s.seed           = (uint64_t)ojson_get_int(root, "seed",        (long)s.seed);
    o->samp = s;
}

const char *srv_parse_openai_chat(const ojson *root, chat_msgs *out, gen_opts *o) {
    chat_msgs_init(out);
    o->max_tokens = (int)ojson_get_int(root, "max_tokens", 256);
    if (o->max_tokens <= 0) o->max_tokens = 256;
    o->stream = ojson_get_bool(root, "stream", false);
    srv_parse_sampling(root, o);
    parse_tools_openai(ojson_get(root, "tools"), &o->tools);
    parse_tool_choice_openai(root, o);
    const ojson *msgs = ojson_get(root, "messages");
    if (!msgs || msgs->type != OJSON_ARRAY || msgs->count == 0)
        return "missing or empty 'messages' array";
    for (size_t i = 0; i < msgs->count; i++) {
        const ojson *msg = msgs->items[i];
        if (!msg || msg->type != OJSON_OBJECT) return "malformed message object";
        const char *role = ojson_get_str(msg, "role", "user");
        /* a prior tool result -> a `tool` role message in the history */
        if (!strcmp(role, "tool")) {
            char *content = collapse_content(ojson_get(msg, "content"));
            chat_msgs_add(out, "tool", content);
            free(content);
            continue;
        }
        /* a prior assistant turn that called tools -> render the <tool_call> blocks */
        const ojson *tcs = ojson_get(msg, "tool_calls");
        if (!strcmp(role, "assistant") && tcs && tcs->type == OJSON_ARRAY && tcs->count > 0) {
            char *content = render_openai_assistant_toolcalls(msg);
            chat_msgs_add(out, "assistant", content);
            free(content);
            continue;
        }
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
    srv_parse_sampling(root, o);
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
    srv_parse_sampling(root, o);
    const ojson *sys = ojson_get(root, "system");
    if (sys) { char *t = collapse_content(sys); if (t[0]) chat_msgs_add(out, "system", t); free(t); }
    parse_tools_anthropic(ojson_get(root, "tools"), &o->tools);
    parse_tool_choice_anthropic(root, o);
    const ojson *msgs = ojson_get(root, "messages");
    if (!msgs || msgs->type != OJSON_ARRAY || msgs->count == 0)
        return "missing or empty 'messages' array";
    for (size_t i = 0; i < msgs->count; i++) {
        const ojson *msg = msgs->items[i];
        if (!msg || msg->type != OJSON_OBJECT) return "malformed message object";
        const char *role = ojson_get_str(msg, "role", "user");
        /* content may interleave text, tool_use (assistant) and tool_result
         * (user) blocks; anthropic_add_message splits them into the history. */
        anthropic_add_message(out, role, ojson_get(msg, "content"));
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
/* tool-call output parsing + tool-call response builders (unit-tested)     */
/* ======================================================================== */

int srv_parse_tool_calls_from_text(const char *text, srv_toolcalls *out) {
    srv_toolcalls_init(out);
    if (!text) return 0;
    static const char *OPEN = "<tool_call>", *CLOSE = "</tool_call>";
    size_t lo = strlen(OPEN), lc = strlen(CLOSE);
    const char *p = text;
    while ((p = strstr(p, OPEN)) != NULL) {
        p += lo;
        const char *end = strstr(p, CLOSE);
        const char *inner_end = end ? end : p + strlen(p);
        size_t n = (size_t)(inner_end - p);
        char *inner = malloc(n + 1);
        memcpy(inner, p, n); inner[n] = '\0';
        const char *errpos = NULL;
        ojson *obj = ojson_parse(inner, &errpos);
        if (obj && obj->type == OJSON_OBJECT) {
            const char *nm = ojson_get_str(obj, "name", NULL);
            const ojson *args = ojson_get(obj, "arguments");
            if (nm) {
                char *as = NULL;
                if (args && args->type == OJSON_STRING)
                    as = strdup(args->str ? args->str : "{}");
                else if (args) { sbuf b; sbuf_init(&b); ojson_stringify(&b, args); as = b.data; }
                else as = strdup("{}");
                srv_toolcalls_push(out, NULL, strdup(nm), as);
            }
        }
        if (obj) ojson_free(obj);
        free(inner);
        if (!end) break;
        p = end + lc;
    }
    return out->len;
}

char *srv_build_chat_response_tools(const char *id, const char *model, long created,
                                    const char *content, const srv_toolcalls *calls,
                                    int prompt_tok, int compl_tok) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"id\":"); sbuf_jstr(&b, id);
    sbuf_printf(&b, ",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", created);
    sbuf_jstr(&b, model);
    sbuf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    if (content && content[0]) sbuf_jstr(&b, content); else sbuf_puts(&b, "null");
    sbuf_puts(&b, ",\"tool_calls\":[");
    for (int i = 0; i < calls->len; i++) {
        if (i) sbuf_puts(&b, ",");
        sbuf_puts(&b, "{\"id\":"); sbuf_jstr(&b, calls->v[i].id ? calls->v[i].id : "");
        sbuf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
        sbuf_jstr(&b, calls->v[i].name ? calls->v[i].name : "");
        sbuf_puts(&b, ",\"arguments\":");
        /* OpenAI carries arguments as a JSON *string* */
        sbuf_jstr(&b, calls->v[i].arguments ? calls->v[i].arguments : "{}");
        sbuf_puts(&b, "}}");
    }
    sbuf_puts(&b, "]},\"finish_reason\":\"tool_calls\"}],");
    sbuf_printf(&b, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                "\"total_tokens\":%d}}", prompt_tok, compl_tok, prompt_tok + compl_tok);
    return b.data;
}

char *srv_build_anthropic_response_tools(const char *id, const char *model,
                                         const char *text, const srv_toolcalls *calls,
                                         int prompt_tok, int compl_tok) {
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, "{\"id\":"); sbuf_jstr(&b, id);
    sbuf_puts(&b, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    sbuf_jstr(&b, model);
    sbuf_puts(&b, ",\"content\":[");
    bool wrote = false;
    if (text && text[0]) {
        sbuf_puts(&b, "{\"type\":\"text\",\"text\":"); sbuf_jstr(&b, text); sbuf_puts(&b, "}");
        wrote = true;
    }
    for (int i = 0; i < calls->len; i++) {
        if (wrote) sbuf_puts(&b, ",");
        wrote = true;
        sbuf_puts(&b, "{\"type\":\"tool_use\",\"id\":");
        sbuf_jstr(&b, calls->v[i].id ? calls->v[i].id : "");
        sbuf_puts(&b, ",\"name\":"); sbuf_jstr(&b, calls->v[i].name ? calls->v[i].name : "");
        /* Anthropic carries the arguments as the JSON object `input` inline */
        sbuf_puts(&b, ",\"input\":");
        sbuf_puts(&b, (calls->v[i].arguments && calls->v[i].arguments[0])
                       ? calls->v[i].arguments : "{}");
        sbuf_puts(&b, "}");
    }
    sbuf_puts(&b, "],\"stop_reason\":\"tool_use\",\"stop_sequence\":null,");
    sbuf_printf(&b, "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}",
                prompt_tok, compl_tok);
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

/* Tool-call ids use the per-protocol prefix OpenAI ("call_") / Anthropic
 * ("toolu_") clients expect, with a request-unique suffix. */
static void make_tool_id(char *buf, size_t cap, bool anthropic, int idx) {
    pthread_mutex_lock(&g_id_lock);
    unsigned long c = g_req_counter++;
    pthread_mutex_unlock(&g_id_lock);
    snprintf(buf, cap, "%s%ld%03lu%d", anthropic ? "toolu_" : "call_",
             (long)time(NULL), c % 1000, idx);
}

/* Return a malloc'd copy of `text` with every <tool_call>...</tool_call> block
 * removed (used to recover any plain prose around tool calls). */
static char *strip_tool_calls(const char *text) {
    sbuf b; sbuf_init(&b);
    static const char *OPEN = "<tool_call>", *CLOSE = "</tool_call>";
    size_t lc = strlen(CLOSE);
    const char *p = text;
    while (*p) {
        const char *o = strstr(p, OPEN);
        if (!o) { sbuf_puts(&b, p); break; }
        sbuf_append(&b, p, (size_t)(o - p));
        const char *e = strstr(o, CLOSE);
        if (!e) break;          /* unterminated: drop the rest */
        p = e + lc;
    }
    /* trim leading/trailing whitespace left behind */
    char *s = b.data;
    size_t len = strlen(s);
    while (len && (s[len-1]=='\n'||s[len-1]==' '||s[len-1]=='\t'||s[len-1]=='\r')) s[--len]='\0';
    size_t lead = 0;
    while (s[lead]=='\n'||s[lead]==' '||s[lead]=='\t'||s[lead]=='\r') lead++;
    if (lead) memmove(s, s + lead, strlen(s + lead) + 1);
    return b.data;
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
                            const osample_params *sp,
                            sbuf *out_text, int *out_count, int *out_finish) {
    int32_t stop_ids[1]; int n_stop = 0;
    if (g_im_end >= 0) stop_ids[n_stop++] = g_im_end;
    collect_ctx cc; sbuf_init(&cc.text); cc.count = 0;
    int finish = 1;
    pthread_mutex_lock(&g_gen_lock);
    ornith_status st = rmodel_generate_ids_s(g_model, pids, n_prompt, max_tokens,
                                             stop_ids, n_stop, sp, on_token_collect,
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
    chat_msgs hist;
    gen_opts opt = { .max_tokens = 256, .stream = false,
                     .samp = {0} };
    const char *perr = NULL;
    if (api == API_CHAT)            perr = srv_parse_openai_chat(root, &hist, &opt);
    else if (api == API_RESPONSES)  perr = srv_parse_responses(root, &hist, &opt);
    else                            perr = srv_parse_anthropic(root, &hist, &opt);
    if (perr) {
        chat_msgs_free(&hist);
        gen_opts_free(&opt);
        send_error(fd, 400, "Bad Request", "invalid_request_error", perr);
        return;
    }

    /* Tool calls are detected by scanning the full generation, so when tools are
     * in play we serve a single non-streaming response (can't tag a tool_call
     * mid-stream without buffering the whole output). */
    if (opt.tools.len > 0) opt.stream = false;

    char *chatml = srv_render_chatml_tools(&hist, &opt.tools,
                                           opt.tool_choice, opt.tool_choice_name);
    chat_msgs_free(&hist);
    int n_prompt = 0;
    int32_t *pids = build_prompt_ids(rmodel_tokenizer(g_model), chatml, &n_prompt);
    free(chatml);
    if (n_prompt == 0) {
        free(pids);
        gen_opts_free(&opt);
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
        rmodel_generate_ids_s(g_model, pids, n_prompt, opt.max_tokens, stop_ids,
                              n_stop, &opt.samp, on_token_stream, &sc, &finish);
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
        gen_opts_free(&opt);
        return;
    }

    /* non-streaming */
    sbuf text; int count = 0, finish = 1;
    if (generate_collect(pids, n_prompt, opt.max_tokens, &opt.samp, &text, &count, &finish) != 0) {
        free(pids);
        gen_opts_free(&opt);
        send_error(fd, 500, "Internal Server Error", "server_error", "generation failed");
        return;
    }
    free(pids);
    char *body = NULL;
    bool length = finish != 0;

    /* If tools were offered and the model emitted <tool_call> block(s), surface
     * them as native tool_calls/tool_use (OpenAI Chat + Anthropic). Otherwise
     * fall through to a plain text completion exactly as before. */
    srv_toolcalls calls;
    int ncalls = srv_parse_tool_calls_from_text(text.data, &calls);
    bool emit_tools = ncalls > 0 && opt.tools.len > 0 &&
                      opt.tool_choice != TOOL_CHOICE_NONE &&
                      (api == API_CHAT || api == API_ANTHROPIC);
    if (emit_tools) {
        for (int i = 0; i < calls.len; i++) {
            char tid[80];
            make_tool_id(tid, sizeof tid, api == API_ANTHROPIC, i);
            calls.v[i].id = strdup(tid);
        }
        if (api == API_CHAT) {
            body = srv_build_chat_response_tools(id, g_model_id, created, NULL,
                                                 &calls, n_prompt, count);
        } else {
            char *clean = strip_tool_calls(text.data);
            body = srv_build_anthropic_response_tools(id, g_model_id, clean,
                                                      &calls, n_prompt, count);
            free(clean);
        }
    } else if (api == API_CHAT)
        body = srv_build_chat_response(id, g_model_id, created, text.data, length, n_prompt, count);
    else if (api == API_RESPONSES)
        body = srv_build_responses_response(id, g_model_id, text.data, length, n_prompt, count);
    else
        body = srv_build_anthropic_response(id, g_model_id, text.data, length, n_prompt, count);

    srv_toolcalls_free(&calls);
    send_json(fd, 200, "OK", body);
    free(body);
    sbuf_free(&text);
    gen_opts_free(&opt);
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
