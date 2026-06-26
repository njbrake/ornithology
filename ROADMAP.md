# Roadmap

Milestones in dependency order. The split is deliberate: everything that can be
correct without a 397B checkpoint or a GPU is done first and tested, then the
inference path is built bottom-up and gated on logit parity.

## M0 — scaffold (done)
- [x] GGUF v2/v3 reader (header, metadata, tensor index)
- [x] HF `config.json` parser -> `ornith_arch`
- [x] `qwen3_5_moe` architecture + memory cost model for **397B and 35B**
- [x] CLI: `version`, `arch`, `config`, `inspect`
- [x] Test suite (`make test`) over JSON/config/GGUF/arch
- [x] Build system with Metal/CUDA targets; backend/server/agent seams

## M1 — quant pipeline (produce a runnable 397B)
There is no official 397B GGUF, so this is on the critical path.
- [ ] GGUF writer + llama.cpp-compatible quant block layouts (IQ2_XXS, Q2_K, Q*_K)
- [ ] imatrix collector on a code-heavy calibration set
- [ ] asymmetric quant tool honoring `tools/quantize/POLICY.md`
- [ ] per-expert mixed precision from router hit-frequency
- [ ] convert 35B first (validate against the official 35B GGUF), then 397B

## M2 — forward pass on CPU (reference, 35B)
Correctness before speed. 35B is the bring-up target.
- [x] core f32 tensor kernels: matmul/linear, RMSNorm, RoPE, softmax,
      SiLU/SwiGLU, elementwise (`src/ornith_tensor.*`, unit-tested)
- [x] embeddings + RMSNorm + RoPE (NeoX/rotate-half) wired into the stack
- [x] full-attention layer (GQA, QK-norm, head_dim 256) with a simple f32 KV
      cache; step path verified == naive O(T^2) reference (`src/ornith_attn.*`)
- [x] **linear-attention layer**: gated delta-net chunked-scan prefill + step
      decode sharing one fp32 state, plus the short causal depthwise conv.
      **Prefill == decode to ~5e-7 for every chunk size** — the headline test
      (`tests/test_attn.c`, `tests/test_forward.c`)
- [x] MoE router (top-k argmax + softmax-over-selected) + grouped SwiGLU expert
      FFN + always-on shared expert (`src/ornith_moe.*`)
- [x] full decoder assembled (L L L F interleave) producing logits on a tiny
      seeded synthetic model; determinism + finiteness + golden-token regression
      (`src/ornith_forward.*`, `tests/test_forward.c`)
- [x] `ornith run`: loads a GGUF, detects the Ornith hybrid layout via the real
      tensor names (`token_embd` / `output[_norm]` / `blk.N.attn_*` /
      `blk.N.ssm_*`), and runs the forward engine
- [ ] real *quantized*-weight binding from GGUF (needs the M1 asymmetric dequant
      kernels: IQ2_XXS / Q2_K / Q*_K -> f32); `run` gates this path honestly
- [ ] a sampling / generation loop (today `run` reports prefill + argmax only)
- [ ] logit parity vs HF transformers on the real 35B (ship `tests/test-vectors/`)

## M3 — Metal backend (primary)
- [ ] dequant + expert GEMM kernels for the asymmetric quant mix
- [ ] GQA attention kernel + paged KV cache
- [ ] gated delta-net + causal conv kernels (highest risk)
- [ ] MoE dispatch over a unified-memory expert pool
- [ ] run 397B on a 128GB+ Mac; measure tok/s and memory

## M4 — CUDA backend
- [ ] port kernels; `cuda` + `cuda-spark` (sm_121a) targets
- [ ] DGX Spark / GB10 validation

## M5 — serving + agent
- [ ] HTTP server: OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, SSE
- [ ] Ornith chat template + tool-calling format
- [ ] on-disk KV persistence (session resume)
- [ ] integrated coding agent loop

## M6 — scale + polish
- [ ] Q8 / on-disk KV spill for full 262K context on 128GB
- [ ] SSD weight streaming for sub-128GB machines
- [ ] distributed layer-split inference across machines
- [ ] (later) the vision tower for true multimodal input

## Non-goals
- Generic GGUF/any-arch runner. This is specialized for Ornith `qwen3_5_moe`.
- Training / fine-tuning.
- The dense 9B variant.
