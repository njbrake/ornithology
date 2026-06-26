/* ornith_server.c — OpenAI/Anthropic-compatible HTTP server.
 *
 * STATUS: stub. Once the forward pass exists (ROADMAP M2/M3), this exposes it
 * over HTTP with three request shapes, matching ds4's surface:
 *   - OpenAI   /v1/chat/completions      (+ streaming SSE)
 *   - Anthropic /v1/messages             (+ streaming SSE)
 *   - a /health and /props introspection endpoint
 *
 * It also owns on-disk KV persistence (resume a session without reprompting)
 * and the Ornith chat template / tool-calling format. None of that needs to be
 * invented from scratch, but it needs the engine first, so it is parked here.
 */
int ornith_server_main(int argc, char **argv) {
    (void)argc; (void)argv;
    return 64; /* not implemented */
}
