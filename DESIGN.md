# ornithology — design

This document is the engineering source of truth for how the engine runs
Ornith-1.0. It is intentionally specific: the make-or-break part of this project
is the hybrid attention forward pass, and that has no analogue in ds4.

## 1. The model

Ornith-1.0 is a `qwen3_5_moe` family (`Qwen3_5MoeForConditionalGeneration`),
post-trained on top of Gemma 4 + Qwen 3.5, MIT licensed, multimodal
(image-text-to-text). We target the two MoE sizes:

| | 397B | 35B |
|---|---|---|
| hidden_size | 4096 | 2048 |
| num_hidden_layers | 60 | 40 |
| full_attention_interval | 4 | 4 |
| -> full-attn layers | 15 | 10 |
| -> linear-attn layers | 45 | 30 |
| num_attention_heads / kv_heads | 32 / 2 | 16 / 2 |
| head_dim | 256 | 256 |
| linear_num_key_heads | 16 | 16 |
| linear_num_value_heads | 64 | 32 |
| linear key/value_head_dim | 128 / 128 | 128 / 128 |
| linear_conv_kernel_dim | 4 | 4 |
| num_experts / top-k | 512 / 10 | 256 / 8 |
| moe_intermediate_size | 1024 | 512 |
| shared_expert_intermediate_size | 1024 | 512 |
| vocab_size | 248320 | 248320 |
| max_position_embeddings | 262144 | 262144 |
| rope_theta | 1e7 | 1e7 |

Both are validated by `make test` against parsed config values, and `./ornith
arch 397b|35b` prints these plus a derived memory model.

## 2. Layer structure

Each decoder layer is one of two kinds, interleaved `L L L F` (full attention
where `layer_idx % 4 == 3`):

### 2.1 Linear-attention layer (¾ of layers) — the hard, novel part

A gated delta-net ("linear attention") block with a short causal depthwise
convolution, in the Qwen3-Next / Gated-DeltaNet lineage. Per layer it maintains
a **constant-size recurrent state** independent of sequence length:

- A short conv state: `conv_kernel_dim (4) - 1` past inputs per channel.
- A delta-net state matrix per value head: roughly
  `linear_num_value_heads x value_head_dim x key_head_dim`.
  For 397B: `64 x 128 x 128 ≈ 1.05M` elems/layer; x45 layers ≈ 47M elems
  (~0.1-0.2 GB total at fp16/fp32). **This does not grow with context.** That
  is why Ornith does long context cheaply without MLA.

Forward (per step, conceptually):
1. project x -> q, k, v (multi-head, linear heads), plus gates (a, b / beta).
2. causal depthwise conv over the last `k=4` positions of q/k/v.
3. delta rule state update: `S_t = S_{t-1} * decay + beta_t * (k_t outer v_t)`
   with a gated correction term; output `o_t = q_t · S_t`.
4. output gate + norm + out-projection.

The two correctness traps:
- **prefill vs decode parity.** Prefill processes the prompt in a chunked
  parallel scan; decode is the step recurrence above. The two must produce
  identical state. This is the single most common place to be subtly wrong.
- **state precision.** Errors in `S` compound over up to 262K steps. Keep `S`
  and the conv state in fp32 even when weights are quantized. Do not store the
  recurrent state in low precision.

Reference to crib: llama.cpp's `qwen3next` graph (the chunked scan + conv) and
the HF `modeling_qwen3_5_moe` linear-attention path.

### 2.2 Full-attention layer (¼ of layers)

Standard causal softmax attention with **GQA**: 32 query heads share 2 KV heads
(16:1 for 397B), `head_dim = 256`, RoPE with `theta = 1e7`. Only these layers
hold a growing KV cache (see §4). Straightforward to implement; the only notable
parameter is the large head_dim (256).

### 2.3 MoE FFN (every layer)

Token-choice routing, **top-10 of 512** experts (397B) / top-8 of 256 (35B),
plus an always-on **shared expert**. `moe_intermediate_size` is small (1024 /
512), i.e. fine-grained experts. Implementation: router GEMM -> top-k softmax ->
grouped/batched expert GEMM over the selected experts -> weighted sum + shared
expert. The shared expert and router run for every token and must stay high
precision.

## 3. Quantization (asymmetric, ds4-style)

The routed experts are ~97% of the 397B parameters and each is cold (any token
touches 10/512), so we quantize them hard and leave the always-on path clean.

| Component | Quant | Rationale |
|---|---|---|
| Routed expert gate/up | **IQ2_XXS** (~2.06 bpw) | cold; error averages over experts |
| Routed expert down | **Q2_K / IQ3_S** (~2.6-3.1) | writes to residual; more sensitive |
| Shared expert | **Q5_K / Q6_K** | every token |
| Router / gate | **F16** | a bad route is unrecoverable |
| Linear-attn conv/delta weights | **Q8_0 / BF16** | recurrent; error compounds |
| Full-attn Q/K/V/O | **Q6_K** | every token |
| Embedding / lm_head | **Q5_K / Q6_K** | logit quality |
| Norms / scales | **F32** | free |

Two levers that decide whether 2-bit is usable:
1. **imatrix calibration** on a code-heavy corpus (it's an agentic coding model),
   so rounding error is weighted by activation importance.
2. **per-expert mixed precision** using router hit-frequency from the imatrix
   run: bump the hottest ~10-20% of experts to 3-bit, leave the cold tail at
   2-bit.

Full policy and the tensor-name -> quant-type mapping live in
[tools/quantize/POLICY.md](tools/quantize/POLICY.md). The source is the FP8
checkpoint (dequant -> BF16 -> imatrix -> requant to GGUF); the FP8 itself is the
reference, not the deployable artifact.

## 4. Memory model

### Weights (asymmetric quant)

397B: routed experts blend to ~2.24 bpw -> ~108GB; non-expert path at ~6 bpw ->
~8GB; **total ~113-120GB**, tunable toward ~110GB. 35B: **~10-12GB**.

### KV cache — only full-attn layers

Per token: `n_full_layers x 2 (K,V) x n_kv_heads x head_dim x bytes`.

- 397B: `15 x 2 x 2 x 256 = 15360 elems/token` -> **30 KB/token** fp16,
  15 KB Q8. Full 262K context = **~8GB fp16 / ~4GB Q8**.
- 35B: `10 x 2 x 2 x 256 = 10240` -> **20 KB/token** fp16. 262K = ~5.4GB fp16.

Linear-attn layers contribute a fixed ~0.1-0.2GB of recurrent state regardless
of context. This is the whole point: long context is cheap.

### Fitting 397B on a 128GB Mac

~110GB weights (aggressive quant) + a few GB of Metal compute/dequant scratch
leaves room for **~128K context at fp16 KV, or the full 262K at Q8 KV**. 128GB
is the floor; 192/256GB is comfortable. ds4-style on-disk KV spill and SSD
weight streaming are the escape valves when memory is tight.

## 5. Backends

`make` builds the CPU/CLI path (introspection + tests; CPU inference is
reference-only). Inference backends, same kernel set each:

- **Metal** (primary): unified-memory expert pool, the four kernel families in
  `ornith_metal.m`.
- **CUDA** (`cuda`, `cuda-spark`): generic + DGX Spark/GB10 (sm_121a, 128GB
  coherent memory — the closest CUDA analog to the Apple-silicon design point).

Kernel families (in dependency order): (1) asymmetric-quant expert GEMM,
(2) GQA full-attention with paged KV, (3) gated delta-net linear attention +
causal conv, (4) top-k MoE router + grouped expert GEMM.

## 6. Validation

Correctness is gated on **logit parity vs the HF reference**:
1. Run HF `transformers` on a fixed set of prompts at BF16, dump per-layer
   hidden states and final logits (small/35B first — it's the bring-up target
   and has an official GGUF).
2. Assert top-1 token agreement and bounded KL on final logits; assert
   per-layer hidden-state closeness to localize bugs (esp. linear-attn
   prefill/decode parity).
3. Ship the reference vectors in `tests/test-vectors/` and check them in CI,
   exactly as ds4 does.

Quant quality is gated separately: perplexity on held-out code + top-1 agreement
vs the BF16 reference, per quant recipe.

## 7. Scope

- **v1 is text-only.** The vision tower (depth-27, hidden-1152) is parsed and
  acknowledged but not run; the image/video token ids are reserved.
- Targets are **397B and 35B** (both `qwen3_5_moe`). The dense 9B is out of
  scope.
- This is **not** a generic GGUF runner. It accepts Ornith GGUFs with the
  expected tensor layout and quant mix.
