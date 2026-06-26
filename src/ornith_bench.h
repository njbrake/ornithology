/* ornith_bench.h — a throughput/footprint benchmark harness for a real Ornith
 * GGUF (ds4 parity: ds4_bench.c).
 *
 * `ornith bench <model.gguf> [--prompt-len P] [--gen N] [--reps R]` measures the
 * four numbers people actually ask about: model load time, prefill throughput
 * (tok/s) on a P-token synthetic prompt, decode throughput (tok/s) over N
 * greedily-decoded tokens, and peak resident set size (ru_maxrss). A fixed
 * synthetic prompt (deterministic in-vocab ids) is used so the result depends on
 * nothing but the model + this machine, never a corpus.
 *
 * Uses only the public rforward session API (rmodel_load / rmodel_session_new /
 * rmodel_session_eval / rmodel_session_logits).
 */
#ifndef ORNITH_BENCH_H
#define ORNITH_BENCH_H

/* Argv is the args AFTER the `bench` subcommand. Returns 0 on success. */
int ornith_bench_main(int argc, char **argv);

#endif /* ORNITH_BENCH_H */
