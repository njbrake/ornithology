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
- [x] k-quant codecs (`src/ornith_quant.c`, ggml-exact super-block layouts,
      QK_K=256): **decoders** for `Q2_K`/`Q4_K`/`Q5_K`/`Q6_K` and **encoders**
      for `Q4_K`/`Q6_K` (round-to-nearest). Round-trip tested; the decoders are
      verified against the **real official 9B GGUF** — `ornith run` dequantizes
      all 427 tensors to f32, all finite, weight ranges sane.
- [ ] remaining encoders: `Q2_K`/`Q5_K` encode (decode works) and the i-quants
      `IQ2_XXS`/`IQ3_S` (the cold-expert types). `ornith quantize` still falls
      back to `F16` for not-yet-encoded targets and logs it. These close the gap
      to the ~113GB asymmetric 397B footprint.
- [ ] imatrix collector on a code-heavy calibration set
- [ ] per-expert mixed precision from router hit-frequency (depends on imatrix)
- [ ] convert 35B first (validate against the official 35B GGUF), then 397B —
      converter is ready; needs the weights + the k-quant encoders above

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
- [x] real-weight **dequant** from GGUF via the k-quant codecs: `ornith run`
      loads the real (quantized) 9B and decodes every tensor to f32 (memory-safe
      sample-decode), proving the dequant path on real weights.
- [ ] **coherent** forward on real weights — needs three qwen3.5-exact ops the
      reference engine currently approximates, all surfaced by loading the real
      9B/config:
      (1) **gated full attention**: `attn_output_gate=true` means `attn_q`
          emits `[q | output_gate]` (so 9B is 16 q-heads x 256 + a 4096-wide
          gate, 4 kv-heads); the attention output is multiplied by
          `sigmoid(gate)` before `attn_output`.
      (2) **exact gated-delta-net gating** from `ssm_a` (A_log) + `ssm_dt.bias`
          + `ssm_alpha`/`ssm_beta` (the reference uses a sigmoid stand-in).
      (3) the **IQ2_XXS/Q2_K** expert dequant for the MoE 35B/397B (9B is dense).
- [ ] a tokenizer (read tokens/merges from GGUF metadata) for text I/O, then a
      sampling / generation loop (today `run` reports prefill + argmax only)
- [ ] logit parity vs HF transformers on the real 9B/35B (ship `tests/test-vectors/`)

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
