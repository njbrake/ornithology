/* ornith_server.h — OpenAI/Anthropic-compatible HTTP server for Ornith.
 *
 * HTTP server adapted from antirez/ds4 (ds4_server.c), MIT licensed: the
 * connection-per-thread model, the unify-three-protocols-into-one-chat-history
 * design, and the graceful-shutdown / IO-timeout patterns are taken from there
 * and re-pointed at the Ornith engine (ds4's DeepSeek/DSML/Metal specifics are
 * not used; rendering uses our ChatML template and our ornith_json parser).
 *
 * ornith_server_main() runs an HTTP/1.1 server (POSIX sockets, no external
 * deps). Each connection is handled by a small pthread that parses one request;
 * generation is serialized through the single loaded model behind a mutex.
 *
 * Three request protocols all collapse to one internal chat_msg[] history,
 * which is rendered to a ChatML prompt, generated once, then formatted back in
 * whichever protocol was asked for:
 *   POST /v1/chat/completions   (OpenAI Chat)
 *   POST /v1/responses          (OpenAI Responses)
 *   POST /v1/messages           (Anthropic Messages)
 *   GET  /health, GET /v1/models
 *
 * The pure helpers below (history rendering, the three request parsers, the
 * three response builders, JSON escaping) take no socket and no model, so
 * tests/test_server.c exercises them directly.
 */
#ifndef ORNITH_SERVER_H
#define ORNITH_SERVER_H

#include "ornith_json.h"
#include "ornith_sample.h"
#include <stdbool.h>
#include <stddef.h>

/* CLI entry: `ornith serve [--host H] [--port P] <model.gguf>`. */
int ornith_server_main(int argc, char **argv);

/* ---- growable byte buffer --------------------------------------------- */

typedef struct { char *data; size_t len, cap; } sbuf;
void sbuf_init(sbuf *b);
void sbuf_free(sbuf *b);
void sbuf_append(sbuf *b, const char *s, size_t n);
void sbuf_puts(sbuf *b, const char *s);

/* ---- unified internal chat history ------------------------------------ */

typedef struct { char *role; char *content; } chat_msg;
typedef struct { chat_msg *v; int len, cap; } chat_msgs;
void chat_msgs_init(chat_msgs *m);
void chat_msgs_add(chat_msgs *m, const char *role, const char *content);
void chat_msgs_free(chat_msgs *m);

/* ---- tool / function calling -----------------------------------------
 *
 * Tool use follows the Qwen/ChatML convention: the available tools are
 * described in the system prompt (as one JSON object per line inside a
 * <tools>...</tools> section), the model emits a call wrapped in
 * <tool_call>{"name":..,"arguments":{..}}</tool_call> tags, and tool results
 * come back as a `tool` role (OpenAI) or a tool_result content block
 * (Anthropic), which we render back into the ChatML history on later turns. */

/* One tool/function definition extracted from a request. `parameters` is the
 * raw JSON schema string (a JSON object), or NULL if the request omitted it. */
typedef struct { char *name; char *description; char *parameters; } srv_tool;
typedef struct { srv_tool *v; int len, cap; } srv_tools;
void srv_tools_init(srv_tools *t);
void srv_tools_free(srv_tools *t);

/* How the client constrains tool selection. AUTO (0) is the default. */
typedef enum {
    TOOL_CHOICE_AUTO = 0,   /* model may or may not call a tool        */
    TOOL_CHOICE_NONE,       /* never call a tool                       */
    TOOL_CHOICE_REQUIRED,   /* must call at least one tool             */
    TOOL_CHOICE_NAMED,      /* must call the tool in tool_choice_name  */
} tool_choice_kind;

/* One tool call parsed out of the model's generated text. `arguments` is a
 * compact JSON string (an object); `id` is assigned by the server when emitting
 * the response (NULL until then). */
typedef struct { char *id; char *name; char *arguments; } srv_toolcall;
typedef struct { srv_toolcall *v; int len, cap; } srv_toolcalls;
void srv_toolcalls_init(srv_toolcalls *t);
void srv_toolcalls_free(srv_toolcalls *t);

/* Sampling/request options shared by all three protocols. `samp` carries the
 * sampler config (temperature / top-k / top-p / min-p / repeat-penalty / seed);
 * it defaults to greedy (temperature 0) unless the client sends those fields.
 * `tools`/`tool_choice` carry parsed tool-calling state (empty by default). */
typedef struct {
    int max_tokens; bool stream; osample_params samp;
    srv_tools tools;
    tool_choice_kind tool_choice;
    char *tool_choice_name;   /* owned; set when tool_choice == NAMED */
} gen_opts;
/* Free the owned tool state inside a gen_opts (safe on a zero-inited struct). */
void gen_opts_free(gen_opts *o);

/* ---- pure helpers (unit-tested) --------------------------------------- */

/* Append `s` (length `n`) to `b` JSON-escaped (no surrounding quotes): \\, \",
 * \n, \r, \t, \b, \f and \u00XX for any other control byte < 0x20. */
void srv_json_escape(sbuf *b, const char *s, size_t n);

/* Render a chat history into a ChatML prompt:
 *   <|im_start|>{role}\n{content}<|im_end|>\n  (per message)
 *   <|im_start|>assistant\n                    (generation primer)
 * Returns a malloc'd string (caller frees). */
char *srv_render_chatml(const chat_msgs *m);

/* Like srv_render_chatml but, when `tools` is non-empty, injects a Qwen-style
 * tools section (the <tools>...</tools> listing and the <tool_call> usage
 * instructions) into the leading system block, merging with an existing system
 * message if present. `tool_choice`/`tool_choice_name` add a "you must call ..."
 * instruction for REQUIRED/NAMED. With no tools this is identical to
 * srv_render_chatml. Returns a malloc'd string (caller frees). */
char *srv_render_chatml_tools(const chat_msgs *m, const srv_tools *tools,
                             tool_choice_kind tool_choice,
                             const char *tool_choice_name);

/* Scan generated assistant text for <tool_call>{...}</tool_call> block(s). Each
 * block's inner JSON is parsed for "name" + "arguments"; valid calls are pushed
 * onto `out` (arguments serialized to a compact JSON string). Returns the number
 * of calls found. `out` is initialized by this call (free with
 * srv_toolcalls_free). */
int srv_parse_tool_calls_from_text(const char *text, srv_toolcalls *out);

/* Parse each protocol's JSON body into the unified history + options. Returns
 * NULL on success, or a static human-readable error string on bad input. */
const char *srv_parse_openai_chat(const ojson *root, chat_msgs *out, gen_opts *o);
const char *srv_parse_responses (const ojson *root, chat_msgs *out, gen_opts *o);
const char *srv_parse_anthropic (const ojson *root, chat_msgs *out, gen_opts *o);

/* Build the non-streaming response body for each protocol. `length` is true if
 * generation hit the token cap (-> "length"/"max_tokens"), false if it stopped
 * cleanly on <|im_end|>/eos (-> "stop"/"end_turn"). Returns a malloc'd string. */
char *srv_build_chat_response(const char *id, const char *model, long created,
                              const char *content, bool length,
                              int prompt_tok, int compl_tok);
char *srv_build_responses_response(const char *id, const char *model,
                                   const char *content, bool length,
                                   int prompt_tok, int compl_tok);
char *srv_build_anthropic_response(const char *id, const char *model,
                                   const char *content, bool length,
                                   int prompt_tok, int compl_tok);

/* Tool-calling response builders. The OpenAI form sets content to null and
 * emits choices[0].message.tool_calls[] with finish_reason "tool_calls"; each
 * call's function.arguments is the JSON string from the parsed call. The
 * Anthropic form emits an optional leading text block followed by tool_use
 * content blocks (input is the JSON object inline) with stop_reason "tool_use".
 * Each call must already carry an `id`. Returns a malloc'd string. */
char *srv_build_chat_response_tools(const char *id, const char *model, long created,
                                    const char *content, const srv_toolcalls *calls,
                                    int prompt_tok, int compl_tok);
char *srv_build_anthropic_response_tools(const char *id, const char *model,
                                         const char *text, const srv_toolcalls *calls,
                                         int prompt_tok, int compl_tok);

/* Responses-protocol tool-call builder. Emits output:[{type:"function_call",
 * name, arguments:"<json string>", call_id, id, status:"completed"}, ...] with
 * top-level status "completed". Each call must already carry an `id` (used as
 * call_id). Returns a malloc'd string. */
char *srv_build_responses_response_tools(const char *id, const char *model,
                                         const srv_toolcalls *calls,
                                         int prompt_tok, int compl_tok);

#endif /* ORNITH_SERVER_H */
