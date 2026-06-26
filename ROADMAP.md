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
- [x] GGUF v3 **writer** (`src/ornith_gguf_write.{h,c}`): metadata KV, tensor
      index, 32-byte-aligned data section; round-trips through the existing
      reader and is read by the Python converter's output too.
- [x] Quant block codecs (`src/ornith_quant.{h,c}`): `F32`, `F16`, `BF16`,
      `Q8_0`, `Q4_0` with **ggml-exact** block layout (Q8_0 = f16 d + 32 int8;
      Q4_0 = f16 d + 16 nibbles). Quant/dequant round-trip tested with bounded
      error. f16/bf16 conversions included.
- [x] asymmetric quant tool honoring `tools/quantize/POLICY.md`: the policy
      table is encoded as `oq_policy_target()` and driven by the
      `ornith quantize [--base TYPE] <in> <out>` subcommand
      (`ornith_quantize_file`). Metadata passes through; unmatched tensors take
      the base type.
- [x] HF -> BF16 GGUF converter scaffold
      (`tools/quantize/convert_hf_to_gguf.py`): concrete Ornith tensor naming +
      metadata (arch `qwen35moe`), pure-Python GGUF writer validated against the
      C reader. Runs end to end once a checkpoint + numpy/safetensors are present.
- [ ] **stubbed:** k-/i-quant encoders (`Q2_K`, `Q5_K`, `Q6_K`, `IQ2_XXS`,
      `IQ3_S`). The policy targets them; `ornith quantize` falls back to `F16`
      for these tensors and logs it honestly. This is the main remaining encoder
      work to hit the ~113GB asymmetric footprint.
- [ ] imatrix collector on a code-heavy calibration set
- [ ] per-expert mixed precision from router hit-frequency (depends on imatrix)
- [ ] convert 35B first (validate against the official 35B GGUF), then 397B —
      converter is ready; needs the weights + the k-quant encoders above

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
