/* test_server.c — pure-logic tests for the HTTP server (no socket, no model).
 *
 * Covers: ChatML rendering from a unified chat history, the three request
 * parsers (OpenAI Chat, OpenAI Responses, Anthropic Messages) extracting the
 * history + max_tokens + stream, the three non-streaming response builders, and
 * rigorous JSON string escaping (quotes, backslashes, newlines, control chars).
 */
#define _POSIX_C_SOURCE 200809L   /* strdup under -std=c11 */
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
    chat_msgs m; gen_opts o = {0};
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
    chat_msgs m2; gen_opts o2 = {0};
    const char *err2 = srv_parse_openai_chat(r2, &m2, &o2);
    CHECK(err2 != NULL, "missing messages -> error");
    CHECK(o2.max_tokens == 5, "default-applied max_tokens read");
    chat_msgs_free(&m2); ojson_free(r2);

    /* content as array of blocks collapses to text */
    ojson *r3 = parse("{\"messages\":[{\"role\":\"user\",\"content\":"
                      "[{\"type\":\"text\",\"text\":\"ab\"},{\"type\":\"text\",\"text\":\"cd\"}]}]}");
    chat_msgs m3; gen_opts o3 = {0};
    srv_parse_openai_chat(r3, &m3, &o3);
    CHECK(m3.len == 1 && strcmp(m3.v[0].content, "abcd") == 0, "content blocks collapse to 'abcd'");
    CHECK(o3.max_tokens == 256, "default max_tokens 256");
    chat_msgs_free(&m3); ojson_free(r3);
}

static void test_parse_responses(void) {
    printf("== parse OpenAI responses ==\n");
    ojson *r = parse("{\"input\":\"hello\",\"instructions\":\"be brief\","
                     "\"max_output_tokens\":10}");
    chat_msgs m; gen_opts o = {0};
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
    chat_msgs m2; gen_opts o2 = {0};
    srv_parse_responses(r2, &m2, &o2);
    CHECK(m2.len == 1 && strcmp(m2.v[0].content, "abc") == 0, "input block -> 'abc'");
    chat_msgs_free(&m2); ojson_free(r2);
}

static void test_parse_anthropic(void) {
    printf("== parse Anthropic messages ==\n");
    ojson *r = parse("{\"model\":\"ornith\",\"max_tokens\":7,\"system\":\"sys\","
                     "\"messages\":[{\"role\":\"user\",\"content\":\"q\"}]}");
    chat_msgs m; gen_opts o = {0};
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
    chat_msgs m2; gen_opts o2 = {0};
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

static void test_parse_tools(void) {
    printf("== parse tools (both protocols) ==\n");
    /* OpenAI chat: tools[].function{name,description,parameters} + tool_choice */
    ojson *r = parse(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"weather in Paris?\"}],"
        "\"tool_choice\":\"required\","
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get the weather for a city\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}}}]}");
    chat_msgs m; gen_opts o = {0};
    const char *err = srv_parse_openai_chat(r, &m, &o);
    CHECK(err == NULL, "openai parse ok");
    CHECK(o.tools.len == 1, "one tool parsed");
    CHECK(o.tools.len == 1 && strcmp(o.tools.v[0].name, "get_weather") == 0, "tool name");
    CHECK(o.tools.len == 1 && o.tools.v[0].description
          && strcmp(o.tools.v[0].description, "Get the weather for a city") == 0, "tool description");
    CHECK(o.tools.len == 1 && o.tools.v[0].parameters
          && has(o.tools.v[0].parameters, "\"city\""), "tool parameters schema captured");
    CHECK(o.tool_choice == TOOL_CHOICE_REQUIRED, "tool_choice required");

    /* render tools into the prompt: name/desc/usage instructions must appear */
    char *prompt = srv_render_chatml_tools(&m, &o.tools, o.tool_choice, o.tool_choice_name);
    CHECK(has(prompt, "get_weather"), "prompt lists tool name");
    CHECK(has(prompt, "Get the weather for a city"), "prompt lists tool description");
    CHECK(has(prompt, "<tools>") && has(prompt, "</tools>"), "prompt has <tools> section");
    CHECK(has(prompt, "<tool_call>"), "prompt instructs <tool_call> format");
    CHECK(has(prompt, "must call at least one"), "required directive present");
    CHECK(has(prompt, "<|im_start|>system\n"), "tools fold into a system block");
    free(prompt);
    chat_msgs_free(&m); gen_opts_free(&o); ojson_free(r);

    /* Anthropic: tools[].input_schema + tool_choice{type:"tool",name} */
    ojson *r2 = parse(
        "{\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"tool_choice\":{\"type\":\"tool\",\"name\":\"lookup\"},"
        "\"tools\":[{\"name\":\"lookup\",\"description\":\"Look something up\","
        "\"input_schema\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}]}");
    chat_msgs m2; gen_opts o2 = {0};
    const char *err2 = srv_parse_anthropic(r2, &m2, &o2);
    CHECK(err2 == NULL, "anthropic parse ok");
    CHECK(o2.tools.len == 1 && strcmp(o2.tools.v[0].name, "lookup") == 0, "anthropic tool name");
    CHECK(o2.tools.len == 1 && o2.tools.v[0].parameters
          && has(o2.tools.v[0].parameters, "\"q\""), "anthropic input_schema captured");
    CHECK(o2.tool_choice == TOOL_CHOICE_NAMED, "anthropic tool_choice named");
    CHECK(o2.tool_choice_name && strcmp(o2.tool_choice_name, "lookup") == 0, "named tool is 'lookup'");
    char *p2 = srv_render_chatml_tools(&m2, &o2.tools, o2.tool_choice, o2.tool_choice_name);
    CHECK(has(p2, "lookup") && has(p2, "must call the function named"), "anthropic prompt + named directive");
    free(p2);
    chat_msgs_free(&m2); gen_opts_free(&o2); ojson_free(r2);
}

static void test_tool_result_roundtrip(void) {
    printf("== tool result round-trip into prompt ==\n");
    /* OpenAI: assistant tool_call turn + a role:"tool" result must render back */
    ojson *r = parse(
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"weather in Paris?\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_1\","
        "\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"18C and sunny\"}]}");
    chat_msgs m; gen_opts o = {0};
    const char *err = srv_parse_openai_chat(r, &m, &o);
    CHECK(err == NULL, "openai history parse ok");
    CHECK(m.len == 3, "user + assistant + tool -> 3 messages");
    CHECK(m.len == 3 && strcmp(m.v[1].role, "assistant") == 0
          && has(m.v[1].content, "<tool_call>")
          && has(m.v[1].content, "get_weather")
          && has(m.v[1].content, "\"city\":\"Paris\""), "assistant tool_call rendered to <tool_call>");
    CHECK(m.len == 3 && strcmp(m.v[2].role, "tool") == 0
          && strcmp(m.v[2].content, "18C and sunny") == 0, "tool result -> tool role message");
    char *prompt = srv_render_chatml(&m);
    CHECK(has(prompt, "<|im_start|>tool\n18C and sunny<|im_end|>"), "tool result in ChatML prompt");
    CHECK(has(prompt, "<tool_call>"), "assistant tool_call in ChatML prompt");
    free(prompt);
    chat_msgs_free(&m); gen_opts_free(&o); ojson_free(r);

    /* Anthropic: assistant tool_use block + user tool_result block */
    ojson *r2 = parse(
        "{\"max_tokens\":32,\"messages\":["
        "{\"role\":\"user\",\"content\":\"weather?\"},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\","
        "\"name\":\"get_weather\",\"input\":{\"city\":\"Paris\"}}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_1\","
        "\"content\":\"18C and sunny\"}]}]}");
    chat_msgs m2; gen_opts o2 = {0};
    const char *err2 = srv_parse_anthropic(r2, &m2, &o2);
    CHECK(err2 == NULL, "anthropic history parse ok");
    CHECK(m2.len == 3, "user + assistant(tool_use) + tool_result -> 3 messages");
    CHECK(m2.len == 3 && has(m2.v[1].content, "<tool_call>")
          && has(m2.v[1].content, "get_weather")
          && has(m2.v[1].content, "\"city\":\"Paris\""), "anthropic tool_use -> <tool_call>");
    CHECK(m2.len == 3 && strcmp(m2.v[2].role, "tool") == 0
          && strcmp(m2.v[2].content, "18C and sunny") == 0, "anthropic tool_result -> tool role");
    chat_msgs_free(&m2); gen_opts_free(&o2); ojson_free(r2);
}

static void test_tool_call_output(void) {
    printf("== parse model <tool_call> output ==\n");
    const char *out =
        "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", "
        "\"arguments\": {\"city\": \"Paris\", \"unit\": \"c\"}}\n</tool_call>";
    srv_toolcalls calls;
    int n = srv_parse_tool_calls_from_text(out, &calls);
    CHECK(n == 1, "one tool call parsed from text");
    CHECK(n == 1 && strcmp(calls.v[0].name, "get_weather") == 0, "parsed name");
    CHECK(n == 1 && has(calls.v[0].arguments, "\"city\":\"Paris\"")
          && has(calls.v[0].arguments, "\"unit\":\"c\""), "parsed arguments object");

    /* OpenAI emission: content null, finish_reason tool_calls, arguments as a
     * JSON *string* (so the inner quotes are escaped). */
    free(calls.v[0].id); calls.v[0].id = strdup("call_abc");
    char *chat = srv_build_chat_response_tools("chatcmpl-9", "ornith", 1, NULL, &calls, 5, 4);
    CHECK(has(chat, "\"content\":null"), "openai content null with tool call");
    CHECK(has(chat, "\"finish_reason\":\"tool_calls\""), "openai finish_reason tool_calls");
    CHECK(has(chat, "\"id\":\"call_abc\""), "openai tool call id");
    CHECK(has(chat, "\"name\":\"get_weather\""), "openai tool name");
    CHECK(has(chat, "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\""),
          "openai arguments is an escaped JSON string");
    free(chat);

    /* Anthropic emission: tool_use block, input is the JSON object inline. */
    char *ant = srv_build_anthropic_response_tools("msg-9", "ornith", NULL, &calls, 5, 4);
    CHECK(has(ant, "\"type\":\"tool_use\""), "anthropic tool_use block");
    CHECK(has(ant, "\"stop_reason\":\"tool_use\""), "anthropic stop_reason tool_use");
    CHECK(has(ant, "\"input\":{\"city\":\"Paris\""), "anthropic input is inline object (not a string)");
    CHECK(has(ant, "\"id\":\"call_abc\""), "anthropic tool_use id");
    free(ant);
    srv_toolcalls_free(&calls);

    /* two calls in one output; argument escaping survives quotes/braces */
    const char *out2 =
        "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":\"q\\\"q\"}}</tool_call>"
        "<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
    srv_toolcalls c2;
    int n2 = srv_parse_tool_calls_from_text(out2, &c2);
    CHECK(n2 == 2, "two tool calls parsed");
    CHECK(n2 == 2 && strcmp(c2.v[0].name, "a") == 0 && strcmp(c2.v[1].name, "b") == 0, "names a,b");
    CHECK(n2 == 2 && has(c2.v[0].arguments, "q\\\"q"), "embedded quote preserved in arguments");
    char *chat2 = srv_build_chat_response_tools("id", "m", 1, NULL, &c2, 1, 1);
    CHECK(has(chat2, "\"name\":\"a\"") && has(chat2, "\"name\":\"b\""), "both calls emitted");
    free(chat2);
    srv_toolcalls_free(&c2);

    /* no tool call -> zero, plain completion path unaffected */
    srv_toolcalls c3;
    int n3 = srv_parse_tool_calls_from_text("just a normal answer", &c3);
    CHECK(n3 == 0, "plain text -> no tool calls");
    srv_toolcalls_free(&c3);
}

static void test_responses_tools(void) {
    printf("== responses tools (parse + emit) ==\n");
    /* Responses function tools are flat: {type:"function", name, description,
     * parameters}. tool_choice {type:"function", name} -> NAMED. The input array
     * carries a prior function_call (assistant) + function_call_output (result)
     * that must fold back into the rendered prompt. */
    ojson *r = parse(
        "{\"max_output_tokens\":32,"
        "\"tool_choice\":{\"type\":\"function\",\"name\":\"get_weather\"},"
        "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\","
        "\"description\":\"Get the weather for a city\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}],"
        "\"input\":[{\"role\":\"user\",\"content\":\"weather in Paris?\"},"
        "{\"type\":\"function_call\",\"name\":\"get_weather\","
        "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\",\"call_id\":\"call_1\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_1\","
        "\"output\":\"18C and sunny\"}]}");
    chat_msgs m; gen_opts o = {0};
    const char *err = srv_parse_responses(r, &m, &o);
    CHECK(err == NULL, "parse ok");
    CHECK(o.tools.len == 1 && strcmp(o.tools.v[0].name, "get_weather") == 0,
          "responses function tool name (flat form)");
    CHECK(o.tools.len == 1 && o.tools.v[0].description
          && strcmp(o.tools.v[0].description, "Get the weather for a city") == 0,
          "responses tool description");
    CHECK(o.tools.len == 1 && o.tools.v[0].parameters
          && has(o.tools.v[0].parameters, "\"city\""), "responses tool parameters captured");
    CHECK(o.tool_choice == TOOL_CHOICE_NAMED, "responses tool_choice named");
    CHECK(o.tool_choice_name && strcmp(o.tool_choice_name, "get_weather") == 0,
          "responses named tool is 'get_weather'");
    /* history: user + assistant(function_call) + tool(function_call_output) */
    CHECK(m.len == 3, "user + function_call + function_call_output -> 3 messages");
    CHECK(m.len == 3 && strcmp(m.v[1].role, "assistant") == 0
          && has(m.v[1].content, "<tool_call>")
          && has(m.v[1].content, "\"city\":\"Paris\""),
          "function_call -> assistant <tool_call> block");
    CHECK(m.len == 3 && strcmp(m.v[2].role, "tool") == 0
          && strcmp(m.v[2].content, "18C and sunny") == 0,
          "function_call_output -> tool role message");
    char *prompt = srv_render_chatml_tools(&m, &o.tools, o.tool_choice, o.tool_choice_name);
    CHECK(has(prompt, "<tools>") && has(prompt, "get_weather"), "tools rendered into prompt");
    CHECK(has(prompt, "<|im_start|>tool\n18C and sunny<|im_end|>"), "tool result in ChatML prompt");
    free(prompt);
    chat_msgs_free(&m); gen_opts_free(&o); ojson_free(r);

    /* emit: a model <tool_call> output -> Responses function_call output item. */
    const char *out =
        "<tool_call>\n{\"name\": \"get_weather\", "
        "\"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    srv_toolcalls calls;
    int n = srv_parse_tool_calls_from_text(out, &calls);
    CHECK(n == 1, "one tool call parsed from text");
    free(calls.v[0].id); calls.v[0].id = strdup("call_xyz");
    char *resp = srv_build_responses_response_tools("resp-9", "ornith", &calls, 6, 5);
    CHECK(has(resp, "\"object\":\"response\"") && has(resp, "\"status\":\"completed\""),
          "responses object/status completed");
    CHECK(has(resp, "\"type\":\"function_call\""), "responses function_call output item");
    CHECK(has(resp, "\"name\":\"get_weather\""), "responses function_call name");
    CHECK(has(resp, "\"call_id\":\"call_xyz\""), "responses function_call call_id");
    CHECK(has(resp, "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\""),
          "responses arguments is an escaped JSON string");
    CHECK(has(resp, "\"input_tokens\":6") && has(resp, "\"output_tokens\":5"),
          "responses usage");
    free(resp);
    srv_toolcalls_free(&calls);
}

int main(void) {
    test_escape();
    test_render();
    test_parse_chat();
    test_parse_responses();
    test_parse_anthropic();
    test_builders();
    test_parse_tools();
    test_tool_result_roundtrip();
    test_tool_call_output();
    test_responses_tools();
    printf("\n%s (%d failure%s)\n",
           failures ? "SERVER TESTS FAILED" : "ALL SERVER TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
