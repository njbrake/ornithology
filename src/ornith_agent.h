/* ornith_agent.h — integrated coding agent + interactive REPL.
 *
 * Ornith-1.0 is post-trained for agentic coding, so a built-in agent loop that
 * can read/write files and run commands is a first-class feature, not an
 * afterthought. This is the local counterpart to ds4's ds4_agent.c.
 *
 * The agent reuses the server's ChatML + tool conventions (ornith_server.h):
 * tools are described in the system block (Qwen-style <tools>...</tools>), the
 * model emits <tool_call>{"name":..,"arguments":{..}}</tool_call>, and tool
 * results are fed back as a <tool_response> user turn. The loop repeats until
 * the model answers without a tool call or a max-iterations cap is hit.
 *
 * The pure pieces (built-in tool list, the tool dispatcher, the result
 * formatter, and the loop driven by a pluggable generator) take no model and no
 * socket, so tests/test_agent.c exercises them directly.
 */
#ifndef ORNITH_AGENT_H
#define ORNITH_AGENT_H

#include "ornith_server.h"   /* sbuf, chat_msgs, srv_tools, srv_toolcalls */
#include <stdbool.h>
#include <stdio.h>

/* Confirmation policy for the dangerous tools (write_file, run_command). */
typedef enum {
    AGENT_CONFIRM_PROMPT = 0,  /* ask y/n before a mutating/exec tool runs   */
    AGENT_CONFIRM_AUTO,        /* --yolo / --auto-approve: never ask         */
} agent_confirm_mode;

/* A confirmation hook: return non-zero to allow, zero to deny. `name` is the
 * tool, `args` its compact-JSON arguments. NULL ud is fine. The CLI passes a
 * stdin y/n prompt; tests pass a stub. */
typedef int (*agent_confirm_fn)(const char *name, const char *args, void *ud);

/* Fill `out` with the agent's built-in tool definitions (read_file, write_file,
 * list_dir, run_command), each with a JSON-schema `parameters`. The caller frees
 * with srv_tools_free. */
void agent_builtin_tools(srv_tools *out);

/* True if `name` is a built-in tool that mutates state / executes code and so
 * requires confirmation in AGENT_CONFIRM_PROMPT mode. */
bool agent_tool_needs_confirm(const char *name);

/* Execute one tool call. `name` is the tool, `args_json` its arguments (a JSON
 * object string; may be NULL/empty). Returns a malloc'd human-readable result
 * string (caller frees). For write_file/run_command, in AGENT_CONFIRM_PROMPT
 * mode the `confirm` hook is consulted first (NULL hook => denied for safety);
 * if denied, *out_denied is set true and a "denied" message is returned without
 * touching the filesystem or running anything. AGENT_CONFIRM_AUTO skips the
 * check. Unknown tools return an error string. `out_denied` may be NULL. */
char *agent_execute_tool(const char *name, const char *args_json,
                         agent_confirm_mode mode,
                         agent_confirm_fn confirm, void *confirm_ud,
                         bool *out_denied);

/* Append a Qwen-style tool-result turn body for one call to `b`:
 *   <tool_response>
 *   {name}: {result}
 *   </tool_response>
 * (No surrounding role tags; the caller wraps it in a user/tool message.) */
void agent_format_tool_result(sbuf *b, const char *name, const char *result);

/* A generator hook: given the fully rendered ChatML prompt, produce the
 * assistant's reply text (malloc'd, caller takes ownership via *out_reply).
 * Returns 0 on success, non-zero on failure. The real agent wires this to the
 * model; tests pass a stub so the loop is exercised without weights. */
typedef int (*agent_generate_fn)(void *gctx, const char *chatml, char **out_reply);

/* Run the agent loop with a pluggable generator. Builds the ChatML history
 * (optional `system_prompt`, then the user `task`), renders it with the
 * built-in tools, calls `gen`, parses + executes any <tool_call>s, appends the
 * results, and repeats until the model replies with no tool call or `max_iters`
 * iterations have run. Tool calls and results are printed to `out`.
 *
 * Returns 0 if the model produced a final answer, 1 if it hit the max-iters cap
 * without one, and -1 on a generator error. */
int agent_loop(agent_generate_fn gen, void *gctx,
               const char *system_prompt, const char *task,
               int max_iters, agent_confirm_mode mode,
               agent_confirm_fn confirm, void *confirm_ud,
               FILE *out);

/* CLI entry points (wired from ornith.c). */
int ornith_agent_main(int argc, char **argv);   /* `ornith agent ...`        */
int ornith_repl_main(int argc, char **argv);     /* `ornith repl|chat ...`    */

#endif /* ORNITH_AGENT_H */
