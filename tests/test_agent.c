/* test_agent.c — pure-logic tests for the coding agent (no model, no socket).
 *
 * Covers: the built-in tool table, the confirmation gate, <tool_call> parsing
 * (via the shared server helper), the tool dispatcher executing read_file /
 * write_file / list_dir / run_command against a temp dir under build/, the
 * safety behaviour (write_file/run_command never fire without approval), the
 * tool-result formatter, and the agent loop driven by a stub generator (final
 * answer path + max-iterations guard).
 */
#define _POSIX_C_SOURCE 200809L   /* strdup under -std=c11 */
#include "ornith_agent.h"
#include "ornith_server.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(c,msg) do{ if(!(c)){printf("  FAIL: %s\n",msg);failures++;} \
                         else printf("  ok  : %s\n",msg);}while(0)

static int has(const char *hay, const char *needle) {
    return hay && strstr(hay, needle) != NULL;
}

/* run a shell command, checking the result (keeps -Wunused-result quiet). */
static void sh(const char *cmd) {
    int rc = system(cmd);
    if (rc != 0) printf("  (warning: '%s' returned %d)\n", cmd, rc);
}

/* A tmp workspace under build/ (the Makefile creates build/ before tests). */
#define TMPDIR "build/agent_tmp"
#define TMPFILE TMPDIR "/hello.txt"

/* confirm hooks for the safety tests */
static int confirm_yes(const char *n, const char *a, void *ud) {
    (void)n; (void)a; (void)ud; return 1;
}
static int confirm_no(const char *n, const char *a, void *ud) {
    (void)n; (void)a; (void)ud; return 0;
}

static void test_tool_table(void) {
    printf("== built-in tool table ==\n");
    srv_tools t;
    agent_builtin_tools(&t);
    CHECK(t.len == 4, "four built-in tools");
    int seen_read = 0, seen_write = 0, seen_list = 0, seen_run = 0;
    for (int i = 0; i < t.len; i++) {
        if (!strcmp(t.v[i].name, "read_file"))   seen_read = 1;
        if (!strcmp(t.v[i].name, "write_file"))  seen_write = 1;
        if (!strcmp(t.v[i].name, "list_dir"))    seen_list = 1;
        if (!strcmp(t.v[i].name, "run_command")) seen_run = 1;
        CHECK(t.v[i].parameters && t.v[i].parameters[0] == '{',
              "tool has a JSON-schema parameters object");
    }
    CHECK(seen_read && seen_write && seen_list && seen_run,
          "read_file/write_file/list_dir/run_command all present");
    srv_tools_free(&t);

    CHECK(agent_tool_needs_confirm("write_file"), "write_file needs confirm");
    CHECK(agent_tool_needs_confirm("run_command"), "run_command needs confirm");
    CHECK(!agent_tool_needs_confirm("read_file"), "read_file is safe");
    CHECK(!agent_tool_needs_confirm("list_dir"), "list_dir is safe");
}

static void test_tool_call_parsing(void) {
    printf("== <tool_call> parsing ==\n");
    const char *out =
        "I'll do that.\n"
        "<tool_call>\n"
        "{\"name\": \"write_file\", \"arguments\": {\"path\": \"x.txt\", "
        "\"content\": \"banana\"}}\n"
        "</tool_call>";
    srv_toolcalls calls;
    int n = srv_parse_tool_calls_from_text(out, &calls);
    CHECK(n == 1, "one tool call parsed");
    CHECK(n == 1 && !strcmp(calls.v[0].name, "write_file"), "name is write_file");
    CHECK(n == 1 && has(calls.v[0].arguments, "banana"),
          "arguments carry the content");
    srv_toolcalls_free(&calls);

    /* no tool call -> zero */
    srv_toolcalls c2;
    int n2 = srv_parse_tool_calls_from_text("just a plain answer", &c2);
    CHECK(n2 == 0, "plain text yields no tool calls");
    srv_toolcalls_free(&c2);
}

static void test_dispatch_fs(void) {
    printf("== dispatcher: file tools ==\n");
    sh("rm -rf " TMPDIR " && mkdir -p " TMPDIR);

    /* write_file (auto-approve) */
    bool denied = true;
    char *w = agent_execute_tool(
        "write_file",
        "{\"path\": \"" TMPFILE "\", \"content\": \"banana\"}",
        AGENT_CONFIRM_AUTO, NULL, NULL, &denied);
    CHECK(!denied, "write_file not denied under --auto");
    CHECK(has(w, "wrote"), "write_file reports bytes written");
    free(w);

    /* read_file back */
    char *r = agent_execute_tool("read_file",
                                 "{\"path\": \"" TMPFILE "\"}",
                                 AGENT_CONFIRM_AUTO, NULL, NULL, NULL);
    CHECK(!strcmp(r, "banana"), "read_file returns exactly what we wrote");
    free(r);

    /* list_dir shows the file */
    char *l = agent_execute_tool("list_dir", "{\"path\": \"" TMPDIR "\"}",
                                 AGENT_CONFIRM_AUTO, NULL, NULL, NULL);
    CHECK(has(l, "hello.txt"), "list_dir lists the new file");
    free(l);

    /* read_file on a missing path -> error string, no crash */
    char *e = agent_execute_tool("read_file",
                                 "{\"path\": \"" TMPDIR "/nope\"}",
                                 AGENT_CONFIRM_AUTO, NULL, NULL, NULL);
    CHECK(has(e, "error"), "read_file on missing path returns an error");
    free(e);

    /* unknown tool */
    char *u = agent_execute_tool("frobnicate", "{}",
                                 AGENT_CONFIRM_AUTO, NULL, NULL, NULL);
    CHECK(has(u, "unknown tool"), "unknown tool reported");
    free(u);
}

static void test_dispatch_run_command(void) {
    printf("== dispatcher: run_command ==\n");
    char *r = agent_execute_tool("run_command",
                                 "{\"command\": \"echo hi-from-agent\"}",
                                 AGENT_CONFIRM_AUTO, NULL, NULL, NULL);
    CHECK(has(r, "hi-from-agent"), "run_command captures stdout");
    CHECK(has(r, "exit code 0"), "run_command reports exit code 0");
    free(r);

    char *f = agent_execute_tool("run_command",
                                 "{\"command\": \"exit 3\"}",
                                 AGENT_CONFIRM_AUTO, NULL, NULL, NULL);
    CHECK(has(f, "exit code 3"), "run_command reports a non-zero exit code");
    free(f);
}

static void test_safety_gate(void) {
    printf("== safety: confirmation gate ==\n");
    sh("rm -rf " TMPDIR " && mkdir -p " TMPDIR);
    const char *args = "{\"path\": \"" TMPDIR "/guard.txt\", "
                       "\"content\": \"nope\"}";

    /* PROMPT mode + deny callback -> denied, file NOT created */
    bool denied = false;
    char *d = agent_execute_tool("write_file", args,
                                 AGENT_CONFIRM_PROMPT, confirm_no, NULL,
                                 &denied);
    CHECK(denied, "write_file denied when user says no");
    CHECK(has(d, "denied"), "denial message returned");
    CHECK(access(TMPDIR "/guard.txt", 0) != 0, "denied write created no file");
    free(d);

    /* PROMPT mode + NULL callback -> denied (never fire blind) */
    bool denied2 = false;
    char *d2 = agent_execute_tool("write_file", args,
                                  AGENT_CONFIRM_PROMPT, NULL, NULL, &denied2);
    CHECK(denied2, "write_file denied with no confirm hook");
    CHECK(access(TMPDIR "/guard.txt", 0) != 0, "still no file after blind deny");
    free(d2);

    /* run_command must not execute when denied */
    char *rmd = agent_execute_tool(
        "run_command",
        "{\"command\": \"touch " TMPDIR "/should_not_exist\"}",
        AGENT_CONFIRM_PROMPT, confirm_no, NULL, &denied);
    CHECK(denied, "run_command denied when user says no");
    CHECK(access(TMPDIR "/should_not_exist", 0) != 0,
          "denied run_command executed nothing");
    free(rmd);

    /* PROMPT mode + approve callback -> file IS created */
    bool denied3 = true;
    char *ok = agent_execute_tool("write_file", args,
                                  AGENT_CONFIRM_PROMPT, confirm_yes, NULL,
                                  &denied3);
    CHECK(!denied3, "write_file allowed when user approves");
    CHECK(access(TMPDIR "/guard.txt", 0) == 0, "approved write created the file");
    free(ok);
}

static void test_result_format(void) {
    printf("== tool-result formatting ==\n");
    sbuf b; sbuf_init(&b);
    agent_format_tool_result(&b, "read_file", "banana");
    CHECK(has(b.data, "<tool_response>"), "result wrapped in <tool_response>");
    CHECK(has(b.data, "</tool_response>"), "result has closing tag");
    CHECK(has(b.data, "read_file"), "result names the tool");
    CHECK(has(b.data, "banana"), "result carries the payload");
    sbuf_free(&b);
}

/* ---- stub generators for the loop ------------------------------------- */

typedef struct { int calls; const char *fixed; } stub_ctx;

/* Always emits a tool call (drives the max-iters guard). */
static int gen_always_tool(void *ud, const char *chatml, char **out) {
    (void)chatml;
    stub_ctx *s = ud;
    s->calls++;
    *out = strdup("<tool_call>\n{\"name\": \"list_dir\", "
                  "\"arguments\": {\"path\": \"" TMPDIR "\"}}\n</tool_call>");
    return 0;
}

/* First call: a tool call; afterwards: a plain final answer. */
static int gen_then_answer(void *ud, const char *chatml, char **out) {
    (void)chatml;
    stub_ctx *s = ud;
    if (s->calls++ == 0)
        *out = strdup("<tool_call>\n{\"name\": \"list_dir\", "
                      "\"arguments\": {\"path\": \"" TMPDIR "\"}}\n</tool_call>");
    else
        *out = strdup("All done: the directory was listed.");
    return 0;
}

static char *slurp(FILE *f) {
    fflush(f); rewind(f);
    sbuf b; sbuf_init(&b);
    char chunk[1024]; size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) sbuf_append(&b, chunk, n);
    return b.data;
}

static void test_loop(void) {
    printf("== agent loop ==\n");
    sh("rm -rf " TMPDIR " && mkdir -p " TMPDIR);

    /* max-iters guard: a generator that never stops should cap out. */
    {
        stub_ctx s = { 0, NULL };
        FILE *out = tmpfile();
        int rc = agent_loop(gen_always_tool, &s, NULL, "loop forever",
                            3, AGENT_CONFIRM_AUTO, NULL, NULL, out);
        CHECK(rc == 1, "loop returns 1 when it hits the max-iters cap");
        CHECK(s.calls == 3, "generator invoked exactly max_iters times");
        char *txt = slurp(out);
        CHECK(has(txt, "max iterations"), "loop reports the cap");
        free(txt); fclose(out);
    }

    /* final-answer path: tool call then a plain answer. */
    {
        stub_ctx s = { 0, NULL };
        FILE *out = tmpfile();
        int rc = agent_loop(gen_then_answer, &s, NULL, "list then answer",
                            10, AGENT_CONFIRM_AUTO, NULL, NULL, out);
        CHECK(rc == 0, "loop returns 0 when the model gives a final answer");
        CHECK(s.calls == 2, "generator ran twice (tool turn + answer turn)");
        char *txt = slurp(out);
        CHECK(has(txt, "[tool call] list_dir"), "tool call printed");
        CHECK(has(txt, "[tool result]"), "tool result printed");
        CHECK(has(txt, "All done"), "final answer printed");
        free(txt); fclose(out);
    }
}

int main(void) {
    printf("=== test_agent ===\n");
    test_tool_table();
    test_tool_call_parsing();
    test_dispatch_fs();
    test_dispatch_run_command();
    test_safety_gate();
    test_result_format();
    test_loop();
    if (failures) { printf("\n%d CHECK(s) FAILED\n", failures); return 1; }
    printf("\nall agent tests passed\n");
    return 0;
}
