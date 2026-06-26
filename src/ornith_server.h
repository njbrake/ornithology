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

/* Sampling/request options shared by all three protocols. */
typedef struct { int max_tokens; bool stream; } gen_opts;

/* ---- pure helpers (unit-tested) --------------------------------------- */

/* Append `s` (length `n`) to `b` JSON-escaped (no surrounding quotes): \\, \",
 * \n, \r, \t, \b, \f and \u00XX for any other control byte < 0x20. */
void srv_json_escape(sbuf *b, const char *s, size_t n);

/* Render a chat history into a ChatML prompt:
 *   <|im_start|>{role}\n{content}<|im_end|>\n  (per message)
 *   <|im_start|>assistant\n                    (generation primer)
 * Returns a malloc'd string (caller frees). */
char *srv_render_chatml(const chat_msgs *m);

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

#endif /* ORNITH_SERVER_H */
