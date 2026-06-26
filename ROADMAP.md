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
- [ ] weight loading / mmap from GGUF into typed tensors
- [ ] embeddings + RMSNorm + RoPE
- [ ] full-attention layer (GQA, head_dim 256) with a simple KV cache
- [ ] **linear-attention layer**: chunked-scan prefill + step decode, fp32 state
- [ ] MoE router (top-k) + grouped expert GEMM + shared expert
- [ ] logit parity vs HF transformers (ship `tests/test-vectors/`)

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
