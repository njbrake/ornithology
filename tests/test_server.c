/* test_server.c — pure-logic tests for the HTTP server (no socket, no model).
 *
 * Covers: ChatML rendering from a unified chat history, the three request
 * parsers (OpenAI Chat, OpenAI Responses, Anthropic Messages) extracting the
 * history + max_tokens + stream, the three non-streaming response builders, and
 * rigorous JSON string escaping (quotes, backslashes, newlines, control chars).
 */
#include "ornith_server.h"
#include "ornith_json.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c,msg) do{ if(!(c)){printf("  FAIL: %s\n",msg);failures++;} \
                         else printf("  ok  : %s\n",msg);}while(0)

static int has(const char *hay, const char *needle) {
    return hay && strstr(hay, needle) != NULL;
}

static ojson *parse(const char *s) {
    const char *err = NULL;
    ojson *v = ojson_parse(s, &err);
    if (!v) printf("  (json parse failed near: %s)\n", err ? err : "?");
    return v;
}

static void test_escape(void) {
    printf("== JSON escaping ==\n");
    sbuf b; sbuf_init(&b);
    const char *in = "a\"b\\c\nd\te\rf\bg\fh";
    srv_json_escape(&b, in, strlen(in));
    CHECK(strcmp(b.data, "a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh") == 0,
          "escapes quote/backslash/newline/tab/cr/bs/ff");
    sbuf_free(&b);

    /* control char < 0x20 with no named escape -> \u00XX */
    sbuf c; sbuf_init(&c);
    char ctrl[2] = { 0x01, 0 };
    srv_json_escape(&c, ctrl, 1);
    CHECK(strcmp(c.data, "\\u0001") == 0, "control char 0x01 -> \\u0001");
    sbuf_free(&c);

    /* code-like content with braces/quotes must pass through but escape quotes */
    sbuf d; sbuf_init(&d);
    const char *code = "printf(\"%d\\n\", x);";
    srv_json_escape(&d, code, strlen(code));
    CHECK(has(d.data, "\\\"%d\\\\n\\\""), "code string escaped without corruption");
    sbuf_free(&d);
}

static void test_render(void) {
    printf("== ChatML render ==\n");
    chat_msgs m; chat_msgs_init(&m);
    chat_msgs_add(&m, "system", "You are terse.");
    chat_msgs_add(&m, "user", "Hi");
    char *s = srv_render_chatml(&m);
    CHECK(has(s, "<|im_start|>system\nYou are terse.<|im_end|>\n"), "system block");
    CHECK(has(s, "<|im_start|>user\nHi<|im_end|>\n"), "user block");
    CHECK(has(s, "<|im_start|>assistant\n"), "trailing assistant primer");
    /* the primer must be at the very end of the rendered prompt */
    const char *primer = "<|im_start|>assistant\n";
    size_t L = strlen(s), PL = strlen(primer);
    CHECK(L >= PL && strcmp(s + L - PL, primer) == 0, "primer is at end");
    free(s);
    chat_msgs_free(&m);
}

static void test_parse_chat(void) {
    printf("== parse OpenAI chat ==\n");
    ojson *r = parse("{\"messages\":[{\"role\":\"system\",\"content\":\"S\"},"
                     "{\"role\":\"user\",\"content\":\"hi\"}],"
                     "\"max_tokens\":42,\"stream\":true,\"temperature\":0.7}");
    chat_msgs m; gen_opts o = {0,0};
    const char *err = srv_parse_openai_chat(r, &m, &o);
    CHECK(err == NULL, "parse ok");
    CHECK(m.len == 2, "2 messages");
    CHECK(strcmp(m.v[0].role, "system") == 0 && strcmp(m.v[0].content, "S") == 0, "msg0 system/S");
    CHECK(strcmp(m.v[1].role, "user") == 0 && strcmp(m.v[1].content, "hi") == 0, "msg1 user/hi");
    CHECK(o.max_tokens == 42, "max_tokens 42");
    CHECK(o.stream == true, "stream true");
    chat_msgs_free(&m); ojson_free(r);

    /* missing messages -> error */
    ojson *r2 = parse("{\"max_tokens\":5}");
    chat_msgs m2; gen_opts o2 = {0,0};
    const char *err2 = srv_parse_openai_chat(r2, &m2, &o2);
    CHECK(err2 != NULL, "missing messages -> error");
    CHECK(o2.max_tokens == 5, "default-applied max_tokens read");
    chat_msgs_free(&m2); ojson_free(r2);

    /* content as array of blocks collapses to text */
    ojson *r3 = parse("{\"messages\":[{\"role\":\"user\",\"content\":"
                      "[{\"type\":\"text\",\"text\":\"ab\"},{\"type\":\"text\",\"text\":\"cd\"}]}]}");
    chat_msgs m3; gen_opts o3 = {0,0};
    srv_parse_openai_chat(r3, &m3, &o3);
    CHECK(m3.len == 1 && strcmp(m3.v[0].content, "abcd") == 0, "content blocks collapse to 'abcd'");
    CHECK(o3.max_tokens == 256, "default max_tokens 256");
    chat_msgs_free(&m3); ojson_free(r3);
}

static void test_parse_responses(void) {
    printf("== parse OpenAI responses ==\n");
    ojson *r = parse("{\"input\":\"hello\",\"instructions\":\"be brief\","
                     "\"max_output_tokens\":10}");
    chat_msgs m; gen_opts o = {0,0};
    const char *err = srv_parse_responses(r, &m, &o);
    CHECK(err == NULL, "parse ok");
    CHECK(m.len == 2, "instructions + input -> 2 messages");
    CHECK(strcmp(m.v[0].role, "system") == 0 && strcmp(m.v[0].content, "be brief") == 0,
          "instructions -> system");
    CHECK(strcmp(m.v[1].role, "user") == 0 && strcmp(m.v[1].content, "hello") == 0,
          "input string -> user");
    CHECK(o.max_tokens == 10, "max_output_tokens 10");
    chat_msgs_free(&m); ojson_free(r);

    /* input as array of typed blocks */
    ojson *r2 = parse("{\"input\":[{\"type\":\"input_text\",\"text\":\"abc\"}]}");
    chat_msgs m2; gen_opts o2 = {0,0};
    srv_parse_responses(r2, &m2, &o2);
    CHECK(m2.len == 1 && strcmp(m2.v[0].content, "abc") == 0, "input block -> 'abc'");
    chat_msgs_free(&m2); ojson_free(r2);
}

static void test_parse_anthropic(void) {
    printf("== parse Anthropic messages ==\n");
    ojson *r = parse("{\"model\":\"ornith\",\"max_tokens\":7,\"system\":\"sys\","
                     "\"messages\":[{\"role\":\"user\",\"content\":\"q\"}]}");
    chat_msgs m; gen_opts o = {0,0};
    const char *err = srv_parse_anthropic(r, &m, &o);
    CHECK(err == NULL, "parse ok");
    CHECK(m.len == 2, "system + user -> 2 messages");
    CHECK(strcmp(m.v[0].role, "system") == 0 && strcmp(m.v[0].content, "sys") == 0, "system from top-level");
    CHECK(strcmp(m.v[1].content, "q") == 0, "user content q");
    CHECK(o.max_tokens == 7, "max_tokens 7");
    chat_msgs_free(&m); ojson_free(r);

    /* content as block array + system as block array */
    ojson *r2 = parse("{\"max_tokens\":5,\"system\":[{\"type\":\"text\",\"text\":\"S\"}],"
                      "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"blk\"}]}]}");
    chat_msgs m2; gen_opts o2 = {0,0};
    srv_parse_anthropic(r2, &m2, &o2);
    CHECK(m2.len == 2 && strcmp(m2.v[0].content, "S") == 0 && strcmp(m2.v[1].content, "blk") == 0,
          "block-array system + content collapse");
    chat_msgs_free(&m2); ojson_free(r2);
}

static void test_builders(void) {
    printf("== response builders ==\n");
    /* content with a quote and newline must be escaped in every protocol */
    const char *content = "Paris\n\"x\"";
    char *chat = srv_build_chat_response("chatcmpl-1", "ornith", 123, content, false, 10, 3);
    CHECK(has(chat, "\"object\":\"chat.completion\""), "chat object type");
    CHECK(has(chat, "\"content\":\"Paris\\n\\\"x\\\"\""), "chat content escaped");
    CHECK(has(chat, "\"finish_reason\":\"stop\""), "chat finish stop");
    CHECK(has(chat, "\"prompt_tokens\":10") && has(chat, "\"completion_tokens\":3")
          && has(chat, "\"total_tokens\":13"), "chat usage");
    free(chat);

    char *chatL = srv_build_chat_response("id", "m", 1, "x", true, 1, 1);
    CHECK(has(chatL, "\"finish_reason\":\"length\""), "chat finish length when capped");
    free(chatL);

    char *resp = srv_build_responses_response("resp-1", "ornith", content, false, 4, 2);
    CHECK(has(resp, "\"object\":\"response\"") && has(resp, "\"status\":\"completed\""),
          "responses object/status");
    CHECK(has(resp, "\"type\":\"output_text\",\"text\":\"Paris\\n\\\"x\\\"\""),
          "responses output_text escaped");
    CHECK(has(resp, "\"input_tokens\":4") && has(resp, "\"output_tokens\":2"), "responses usage");
    free(resp);

    char *ant = srv_build_anthropic_response("msg-1", "ornith", content, false, 4, 2);
    CHECK(has(ant, "\"type\":\"message\"") && has(ant, "\"role\":\"assistant\""), "anthropic shape");
    CHECK(has(ant, "\"type\":\"text\",\"text\":\"Paris\\n\\\"x\\\"\""), "anthropic text escaped");
    CHECK(has(ant, "\"stop_reason\":\"end_turn\""), "anthropic end_turn on clean stop");
    CHECK(has(ant, "\"input_tokens\":4") && has(ant, "\"output_tokens\":2"), "anthropic usage");
    free(ant);

    char *antL = srv_build_anthropic_response("id", "m", "x", true, 1, 1);
    CHECK(has(antL, "\"stop_reason\":\"max_tokens\""), "anthropic max_tokens when capped");
    free(antL);
}

int main(void) {
    test_escape();
    test_render();
    test_parse_chat();
    test_parse_responses();
    test_parse_anthropic();
    test_builders();
    printf("\n%s (%d failure%s)\n",
           failures ? "SERVER TESTS FAILED" : "ALL SERVER TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
