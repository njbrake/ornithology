/* ornith_agent.c — integrated coding agent + interactive REPL.
 *
 * Ornith-1.0 is post-trained for agentic coding, so a local agent loop that can
 * read/write files and run commands is the marquee feature. This mirrors the
 * structure of ds4's ds4_agent.c, re-pointed at the Ornith engine, and reuses
 * the server's ChatML + Qwen tool-calling conventions (ornith_server.h) instead
 * of duplicating them.
 *
 * Three entry points:
 *   ornith_agent_main  — one-shot `ornith agent --task "..." <model.gguf>`
 *   ornith_repl_main   — interactive `ornith repl|chat <model.gguf>`
 *   (the loop + tool dispatcher are pure and unit-tested in test_agent.c)
 *
 * Safety: write_file and run_command are gated behind an interactive y/n
 * confirmation by default; --yolo / --auto-approve skips it. With no confirm
 * hook the dispatcher denies them rather than firing blind.
 */
#define _POSIX_C_SOURCE 200809L
#include "ornith_agent.h"
#include "ornith_rforward.h"
#include "ornith_tokenizer.h"
#include "ornith_sample.h"
#include "ornith.h"

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>

/* malloc a formatted string (small, bounded). Used for tool result messages. */
static char *agent_strdup_fmt(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return strdup(buf);
}

/* Bound a single file read / command output so a runaway tool result cannot
 * blow up the model's context (the small 9B has a modest window). */
#define AGENT_READ_CAP   (64u * 1024u)
#define AGENT_OUTPUT_CAP (32u * 1024u)

/* The agent's standing instructions. The tools section (the <tools> listing and
 * <tool_call> usage rules) is appended automatically by srv_render_chatml_tools,
 * so this only sets the role + working style. */
static const char AGENT_SYSTEM_PROMPT[] =
    "You are Ornith, a coding agent running locally in the user's workspace. "
    "You can read and write files, list directories, and run shell commands by "
    "calling the provided tools. Work step by step: call a tool, look at its "
    "result, then decide the next action. Prefer making the change with a tool "
    "over describing it. When the task is complete, reply with a short final "
    "answer and do not emit any further tool call.";

/* ======================================================================== */
/* built-in tool definitions                                                */
/* ======================================================================== */

void agent_builtin_tools(srv_tools *out) {
    srv_tools_init(out);
    struct { const char *name, *desc, *params; } defs[] = {
        { "read_file",
          "Read a UTF-8 text file and return its contents.",
          "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\","
          "\"description\":\"Path to the file to read.\"}},"
          "\"required\":[\"path\"]}" },
        { "write_file",
          "Create or overwrite a text file with the given content.",
          "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\","
          "\"description\":\"Path to the file to write.\"},"
          "\"content\":{\"type\":\"string\",\"description\":\"Full file "
          "contents to write.\"}},\"required\":[\"path\",\"content\"]}" },
        { "list_dir",
          "List the entries in a directory.",
          "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\","
          "\"description\":\"Directory to list (default '.').\"}},"
          "\"required\":[]}" },
        { "run_command",
          "Run a shell command and return its combined stdout/stderr and exit "
          "code.",
          "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":"
          "\"string\",\"description\":\"Shell command line to execute.\"}},"
          "\"required\":[\"command\"]}" },
    };
    for (size_t i = 0; i < sizeof(defs) / sizeof(defs[0]); i++) {
        srv_tool t;
        t.name = strdup(defs[i].name);
        t.description = strdup(defs[i].desc);
        t.parameters = strdup(defs[i].params);
        if (out->len == out->cap) {
            out->cap = out->cap ? out->cap * 2 : 8;
            out->v = realloc(out->v, (size_t)out->cap * sizeof(srv_tool));
        }
        out->v[out->len++] = t;
    }
}

bool agent_tool_needs_confirm(const char *name) {
    if (!name) return false;
    return strcmp(name, "write_file") == 0 || strcmp(name, "run_command") == 0;
}

/* ======================================================================== */
/* tool implementations                                                     */
/* ======================================================================== */

/* All tool impls return a malloc'd, NUL-terminated result string. */

static char *tool_read_file(const char *path) {
    if (!path || !path[0]) return strdup("error: read_file requires 'path'");
    FILE *f = fopen(path, "rb");
    if (!f) return agent_strdup_fmt("error: cannot open '%s': %s", path,
                                      strerror(errno));
    sbuf b; sbuf_init(&b);
    char chunk[8192];
    size_t total = 0, n;
    bool truncated = false;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        if (total + n > AGENT_READ_CAP) {
            sbuf_append(&b, chunk, AGENT_READ_CAP - total);
            truncated = true;
            break;
        }
        sbuf_append(&b, chunk, n);
        total += n;
    }
    fclose(f);
    if (truncated)
        sbuf_puts(&b, "\n... [truncated: file exceeds read cap]");
    return b.data;
}

static char *tool_write_file(const char *path, const char *content) {
    if (!path || !path[0]) return strdup("error: write_file requires 'path'");
    FILE *f = fopen(path, "wb");
    if (!f) return agent_strdup_fmt("error: cannot create '%s': %s", path,
                                     strerror(errno));
    size_t len = content ? strlen(content) : 0;
    size_t wrote = len ? fwrite(content, 1, len, f) : 0;
    int cerr = fclose(f);
    if (wrote != len || cerr != 0)
        return agent_strdup_fmt("error: write to '%s' failed", path);
    return agent_strdup_fmt("wrote %zu byte%s to '%s'", len,
                             len == 1 ? "" : "s", path);
}

/* dirent name compare for a stable, readable listing. */
static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static char *tool_list_dir(const char *path) {
    const char *p = (path && path[0]) ? path : ".";
    DIR *d = opendir(p);
    if (!d) return agent_strdup_fmt("error: cannot open dir '%s': %s", p,
                                     strerror(errno));
    char **names = NULL;
    size_t n = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        if (n == cap) { cap = cap ? cap * 2 : 32;
                        names = realloc(names, cap * sizeof(char *)); }
        names[n++] = strdup(de->d_name);
    }
    closedir(d);
    qsort(names, n, sizeof(char *), cmp_str);
    sbuf b; sbuf_init(&b);
    sbuf_puts(&b, p);
    sbuf_puts(&b, ":\n");
    for (size_t i = 0; i < n; i++) {
        /* mark directories with a trailing slash */
        char full[4096];
        snprintf(full, sizeof(full), "%s/%s", p, names[i]);
        struct stat st;
        bool isdir = (stat(full, &st) == 0) && S_ISDIR(st.st_mode);
        sbuf_puts(&b, "  ");
        sbuf_puts(&b, names[i]);
        if (isdir) sbuf_puts(&b, "/");
        sbuf_puts(&b, "\n");
        free(names[i]);
    }
    free(names);
    if (n == 0) sbuf_puts(&b, "  (empty)\n");
    return b.data;
}

static char *tool_run_command(const char *command) {
    if (!command || !command[0])
        return strdup("error: run_command requires 'command'");
    /* Merge stderr into stdout so the model sees diagnostics too. */
    sbuf cmd; sbuf_init(&cmd);
    sbuf_puts(&cmd, command);
    sbuf_puts(&cmd, " 2>&1");
    FILE *p = popen(cmd.data, "r");
    sbuf_free(&cmd);
    if (!p) return agent_strdup_fmt("error: failed to run '%s': %s", command,
                                     strerror(errno));
    sbuf b; sbuf_init(&b);
    char chunk[8192];
    size_t total = 0, n;
    bool truncated = false;
    while ((n = fread(chunk, 1, sizeof(chunk), p)) > 0) {
        if (total + n > AGENT_OUTPUT_CAP) {
            sbuf_append(&b, chunk, AGENT_OUTPUT_CAP - total);
            truncated = true;
            break;
        }
        sbuf_append(&b, chunk, n);
        total += n;
    }
    if (truncated) {
        /* drain the rest so pclose gets the real exit status */
        while (fread(chunk, 1, sizeof(chunk), p) > 0) { }
        sbuf_puts(&b, "\n... [truncated: output exceeds cap]");
    }
    int status = pclose(p);
    int code = -1;
    if (status != -1) {
        if (WIFEXITED(status)) code = WEXITSTATUS(status);
        else code = status;
    }
    char hdr[64];
    snprintf(hdr, sizeof(hdr), "exit code %d\n", code);
    sbuf out; sbuf_init(&out);
    sbuf_puts(&out, hdr);
    sbuf_puts(&out, b.data);
    sbuf_free(&b);
    return out.data;
}

/* ======================================================================== */
/* tool dispatcher + result formatting                                      */
/* ======================================================================== */

/* Pull a string field out of a compact-JSON arguments object. Returns a
 * malloc'd copy or NULL if absent. */
static char *args_get_str(const char *args_json, const char *key) {
    if (!args_json) return NULL;
    const char *errpos = NULL;
    ojson *o = ojson_parse(args_json, &errpos);
    char *out = NULL;
    if (o && o->type == OJSON_OBJECT) {
        const char *v = ojson_get_str(o, key, NULL);
        if (v) out = strdup(v);
    }
    if (o) ojson_free(o);
    return out;
}

char *agent_execute_tool(const char *name, const char *args_json,
                         agent_confirm_mode mode,
                         agent_confirm_fn confirm, void *confirm_ud,
                         bool *out_denied) {
    if (out_denied) *out_denied = false;
    if (!name) return strdup("error: missing tool name");

    /* Safety gate for the mutating / executing tools. */
    if (agent_tool_needs_confirm(name) && mode == AGENT_CONFIRM_PROMPT) {
        int ok = confirm ? confirm(name, args_json ? args_json : "{}",
                                   confirm_ud)
                         : 0;     /* no hook => deny, never fire blind */
        if (!ok) {
            if (out_denied) *out_denied = true;
            return agent_strdup_fmt("denied: user did not approve %s", name);
        }
    }

    if (strcmp(name, "read_file") == 0) {
        char *path = args_get_str(args_json, "path");
        char *r = tool_read_file(path);
        free(path);
        return r;
    }
    if (strcmp(name, "write_file") == 0) {
        char *path = args_get_str(args_json, "path");
        char *content = args_get_str(args_json, "content");
        char *r = tool_write_file(path, content ? content : "");
        free(path); free(content);
        return r;
    }
    if (strcmp(name, "list_dir") == 0) {
        char *path = args_get_str(args_json, "path");
        char *r = tool_list_dir(path);
        free(path);
        return r;
    }
    if (strcmp(name, "run_command") == 0) {
        char *command = args_get_str(args_json, "command");
        char *r = tool_run_command(command);
        free(command);
        return r;
    }
    return agent_strdup_fmt("error: unknown tool '%s'", name);
}

void agent_format_tool_result(sbuf *b, const char *name, const char *result) {
    sbuf_puts(b, "<tool_response>\n");
    sbuf_puts(b, name ? name : "tool");
    sbuf_puts(b, ": ");
    sbuf_puts(b, result ? result : "");
    sbuf_puts(b, "\n</tool_response>\n");
}

/* ======================================================================== */
/* the agent loop (generator-pluggable, model-agnostic)                     */
/* ======================================================================== */

int agent_loop(agent_generate_fn gen, void *gctx,
               const char *system_prompt, const char *task,
               int max_iters, agent_confirm_mode mode,
               agent_confirm_fn confirm, void *confirm_ud,
               FILE *out) {
    if (!out) out = stdout;
    chat_msgs hist;
    chat_msgs_init(&hist);
    chat_msgs_add(&hist, "system",
                  system_prompt ? system_prompt : AGENT_SYSTEM_PROMPT);
    chat_msgs_add(&hist, "user", task ? task : "");

    srv_tools tools;
    agent_builtin_tools(&tools);

    int rc = 0;
    for (int iter = 0; ; iter++) {
        if (iter >= max_iters) {
            fprintf(out,
                    "\n[agent] stopped: reached max iterations (%d) without a "
                    "final answer\n", max_iters);
            rc = 1;
            break;
        }

        char *chatml = srv_render_chatml_tools(&hist, &tools,
                                               TOOL_CHOICE_AUTO, NULL);
        char *reply = NULL;
        int g = gen(gctx, chatml, &reply);
        free(chatml);
        if (g != 0) {
            fprintf(out, "\n[agent] generation error\n");
            free(reply);
            rc = -1;
            break;
        }
        if (!reply) reply = strdup("");

        srv_toolcalls calls;
        int ncalls = srv_parse_tool_calls_from_text(reply, &calls);

        if (ncalls == 0) {
            /* No tool call: this is the final answer. */
            fprintf(out, "%s\n", reply);
            chat_msgs_add(&hist, "assistant", reply);
            srv_toolcalls_free(&calls);
            free(reply);
            rc = 0;
            break;
        }

        /* Record the assistant turn verbatim (it carries the <tool_call>). */
        chat_msgs_add(&hist, "assistant", reply);

        /* Execute every call and assemble the tool-response user turn. */
        sbuf resp; sbuf_init(&resp);
        for (int i = 0; i < ncalls; i++) {
            const char *nm = calls.v[i].name ? calls.v[i].name : "(null)";
            const char *as = calls.v[i].arguments ? calls.v[i].arguments : "{}";
            fprintf(out, "\n[tool call] %s %s\n", nm, as);
            fflush(out);

            bool denied = false;
            char *r = agent_execute_tool(nm, as, mode, confirm, confirm_ud,
                                         &denied);
            fprintf(out, "[tool result] %s\n", r ? r : "");
            fflush(out);
            agent_format_tool_result(&resp, nm, r ? r : "");
            free(r);
        }
        chat_msgs_add(&hist, "user", resp.data);
        sbuf_free(&resp);

        srv_toolcalls_free(&calls);
        free(reply);
    }

    srv_tools_free(&tools);
    chat_msgs_free(&hist);
    return rc;
}

/* ======================================================================== */
/* model-backed generator + prompt-id building (mirrors the server)         */
/* ======================================================================== */

typedef struct { int32_t *ids; int n, cap; } idvec;
static void idvec_push(idvec *v, int32_t id) {
    if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 256;
                          v->ids = realloc(v->ids,
                                           (size_t)v->cap * sizeof(int32_t)); }
    v->ids[v->n++] = id;
}
static void idvec_encode(idvec *v, const otokenizer *t, const char *text,
                         size_t len) {
    if (len == 0) return;
    char *z = malloc(len + 1);
    memcpy(z, text, len); z[len] = '\0';
    int32_t *ids = NULL; int n = 0;
    otok_encode(t, z, &ids, &n);
    for (int i = 0; i < n; i++) idvec_push(v, ids[i]);
    free(ids); free(z);
}

/* Tokenize a ChatML string, keeping <|im_start|>/<|im_end|> as their atomic
 * special-token ids (same approach as the server's build_prompt_ids). */
static int32_t *agent_prompt_ids(const otokenizer *t, const char *chatml,
                                 int32_t im_start, int32_t im_end, int *out_n) {
    static const char *S = "<|im_start|>";
    static const char *E = "<|im_end|>";
    size_t ls = strlen(S), le = strlen(E);
    idvec v = {0};
    const char *p = chatml;
    while (*p) {
        if (im_start >= 0 && strncmp(p, S, ls) == 0) {
            idvec_push(&v, im_start); p += ls;
        } else if (im_end >= 0 && strncmp(p, E, le) == 0) {
            idvec_push(&v, im_end); p += le;
        } else {
            const char *q = p;
            while (*q) {
                if (im_start >= 0 && strncmp(q, S, ls) == 0) break;
                if (im_end   >= 0 && strncmp(q, E, le) == 0) break;
                q++;
            }
            idvec_encode(&v, t, p, (size_t)(q - p));
            p = q;
        }
    }
    *out_n = v.n;
    return v.ids;
}

typedef struct {
    rmodel             *m;
    const otokenizer   *tok;
    int32_t             im_start, im_end;
    int                 max_tokens;
    const osample_params *sp;     /* NULL => greedy */
} agent_gen_ctx;

typedef struct { sbuf text; int count; } collect_ctx;
static void on_token_collect(int32_t id, const char *piece, void *ud) {
    (void)id;
    collect_ctx *c = ud;
    sbuf_append(&c->text, piece, strlen(piece));
    c->count++;
}

static int agent_model_generate(void *gctx, const char *chatml,
                                char **out_reply) {
    agent_gen_ctx *c = gctx;
    int n_prompt = 0;
    int32_t *pids = agent_prompt_ids(c->tok, chatml, c->im_start, c->im_end,
                                     &n_prompt);
    if (!pids || n_prompt == 0) { free(pids); *out_reply = NULL; return -1; }

    int32_t stop_ids[1]; int n_stop = 0;
    if (c->im_end >= 0) stop_ids[n_stop++] = c->im_end;

    collect_ctx cc; sbuf_init(&cc.text); cc.count = 0;
    int finish = 1;
    ornith_status st = rmodel_generate_ids_s(c->m, pids, n_prompt,
                                             c->max_tokens, stop_ids, n_stop,
                                             c->sp, on_token_collect, &cc,
                                             &finish);
    free(pids);
    if (st != ORNITH_OK) { sbuf_free(&cc.text); *out_reply = NULL; return -1; }
    *out_reply = cc.text.data;   /* always a valid NUL-terminated buffer */
    return 0;
}

/* ======================================================================== */
/* interactive confirmation (stdin y/n)                                     */
/* ======================================================================== */

static int confirm_stdin(const char *name, const char *args, void *ud) {
    (void)ud;
    fprintf(stderr, "\n[confirm] allow %s %s ? [y/N] ", name, args);
    fflush(stderr);
    char line[64];
    if (!fgets(line, sizeof(line), stdin)) return 0;
    return (line[0] == 'y' || line[0] == 'Y');
}

/* ======================================================================== */
/* CLI: `ornith agent ...`                                                  */
/* ======================================================================== */

static void agent_usage(void) {
    fprintf(stderr,
        "usage: ornith agent [--yolo|--auto-approve] [--max-iters N]\n"
        "                    [--temp T] [--top-p P] [--top-k K] [--min-p M]\n"
        "                    [--repeat-penalty R] [--seed S] [-n MAXTOK]\n"
        "                    --task \"...\" <model.gguf>\n"
        "\n"
        "  Runs a coding-agent loop: the model may call read_file, write_file,\n"
        "  list_dir, and run_command until it answers without a tool call.\n"
        "  write_file/run_command prompt for y/N confirmation unless --yolo.\n");
}

int ornith_agent_main(int argc, char **argv) {
    const char *path = NULL, *task = NULL;
    int max_iters = 10;
    int max_tokens = 512;
    agent_confirm_mode mode = AGENT_CONFIRM_PROMPT;
    osample_params sp = osample_params_default();   /* greedy by default */

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--yolo") || !strcmp(argv[i], "--auto-approve"))
            mode = AGENT_CONFIRM_AUTO;
        else if (!strcmp(argv[i], "--max-iters") && i + 1 < argc)
            max_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--task") && i + 1 < argc)
            task = argv[++i];
        else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n-predict"))
                 && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--temp") || !strcmp(argv[i], "--temperature"))
                 && i + 1 < argc)
            sp.temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc)
            sp.top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc)
            sp.top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc)
            sp.min_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc)
            sp.repeat_penalty = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            sp.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            agent_usage(); return 0;
        } else
            path = argv[i];
    }
    if (!path || !task) { agent_usage(); return 1; }
    if (max_iters <= 0) max_iters = 10;
    if (max_tokens <= 0) max_tokens = 512;

    rmodel *m = NULL;
    if (rmodel_load(path, &m) != ORNITH_OK) {
        fprintf(stderr, "agent: load failed: %s\n", ornith_last_error());
        return 1;
    }
    const ornith_arch *a = rmodel_arch(m);
    const otokenizer *t = rmodel_tokenizer(m);
    fprintf(stderr, "loaded %s: hidden %d, %d layers, vocab %d (agent mode%s)\n",
            path, a->hidden_size, a->num_layers, a->vocab_size,
            mode == AGENT_CONFIRM_AUTO ? ", --yolo" : "");

    int sampling = sp.temperature > 0.0f;
    agent_gen_ctx ctx = {
        .m = m, .tok = t,
        .im_start = otok_id_of(t, "<|im_start|>"),
        .im_end   = otok_id_of(t, "<|im_end|>"),
        .max_tokens = max_tokens,
        .sp = sampling ? &sp : NULL,
    };
    if (ctx.im_start < 0 || ctx.im_end < 0)
        fprintf(stderr, "agent: WARNING: ChatML special tokens missing from "
                "vocab\n");

    fprintf(stderr, "task: %s\n", task);
    int rc = agent_loop(agent_model_generate, &ctx, NULL, task, max_iters, mode,
                        confirm_stdin, NULL, stdout);
    rmodel_free(m);
    return rc < 0 ? 1 : 0;
}

/* ======================================================================== */
/* CLI: `ornith repl|chat ...` — multi-turn interactive chat                */
/* ======================================================================== */

/* Streaming printer that also collects the reply text for the history: write
 * each token to stdout as it is produced and append it to the collect buffer. */
static void on_token_stream_collect(int32_t id, const char *piece, void *ud) {
    (void)id;
    collect_ctx *c = ud;
    fputs(piece, stdout);
    fflush(stdout);
    sbuf_append(&c->text, piece, strlen(piece));
    c->count++;
}

static void repl_usage(void) {
    fprintf(stderr,
        "usage: ornith repl [--temp T] [--top-p P] [--top-k K] [--min-p M]\n"
        "                   [--repeat-penalty R] [--seed S] [-n MAXTOK]\n"
        "                   <model.gguf>\n"
        "\n"
        "  Interactive multi-turn chat. Commands: /reset (clear history),\n"
        "  /exit or /quit (leave). Ctrl-D also exits.\n");
}

int ornith_repl_main(int argc, char **argv) {
    const char *path = NULL;
    int max_tokens = 512;
    osample_params sp = osample_params_default();

    for (int i = 0; i < argc; i++) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n-predict"))
            && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--temp") || !strcmp(argv[i], "--temperature"))
                 && i + 1 < argc)
            sp.temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc)
            sp.top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc)
            sp.top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc)
            sp.min_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc)
            sp.repeat_penalty = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            sp.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            repl_usage(); return 0;
        } else
            path = argv[i];
    }
    if (!path) { repl_usage(); return 1; }
    if (max_tokens <= 0) max_tokens = 512;

    rmodel *m = NULL;
    if (rmodel_load(path, &m) != ORNITH_OK) {
        fprintf(stderr, "repl: load failed: %s\n", ornith_last_error());
        return 1;
    }
    const ornith_arch *a = rmodel_arch(m);
    const otokenizer *t = rmodel_tokenizer(m);
    int32_t im_start = otok_id_of(t, "<|im_start|>");
    int32_t im_end   = otok_id_of(t, "<|im_end|>");
    int sampling = sp.temperature > 0.0f;

    fprintf(stderr, "loaded %s: hidden %d, %d layers, vocab %d\n",
            path, a->hidden_size, a->num_layers, a->vocab_size);
    fprintf(stderr, "interactive chat. /reset clears history, /exit quits.\n");

    chat_msgs hist;
    chat_msgs_init(&hist);

    char line[8192];
    for (;;) {
        fputs("\n> ", stdout);
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) { fputs("\n", stdout); break; }
        /* strip trailing newline */
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = '\0';
        if (n == 0) continue;

        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
        if (!strcmp(line, "/reset")) {
            chat_msgs_free(&hist);
            chat_msgs_init(&hist);
            fputs("[history cleared]\n", stdout);
            continue;
        }

        chat_msgs_add(&hist, "user", line);
        char *chatml = srv_render_chatml(&hist);
        int n_prompt = 0;
        int32_t *pids = agent_prompt_ids(t, chatml, im_start, im_end, &n_prompt);
        free(chatml);
        if (!pids || n_prompt == 0) { free(pids); continue; }

        int32_t stop_ids[1]; int n_stop = 0;
        if (im_end >= 0) stop_ids[n_stop++] = im_end;

        /* collect the assistant reply (for history) while streaming it out */
        collect_ctx cc; sbuf_init(&cc.text); cc.count = 0;

        fputs("\n", stdout);
        int finish = 1;
        ornith_status st = rmodel_generate_ids_s(
            m, pids, n_prompt, max_tokens, stop_ids, n_stop,
            sampling ? &sp : NULL,
            on_token_stream_collect, &cc, &finish);
        free(pids);
        fputs("\n", stdout);
        if (st != ORNITH_OK) {
            fprintf(stderr, "repl: generation failed: %s\n",
                    ornith_last_error());
            sbuf_free(&cc.text);
            continue;
        }
        chat_msgs_add(&hist, "assistant", cc.text.data);
        sbuf_free(&cc.text);
    }

    chat_msgs_free(&hist);
    rmodel_free(m);
    return 0;
}
