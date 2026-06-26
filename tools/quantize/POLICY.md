# Quantization policy

The deployable Ornith-1.0-397B artifact is a GGUF we build ourselves (no
official one exists). The strategy is **asymmetric quantization**, ds4-style:
quantize the cold, dominant routed experts hard; keep the always-on path clean.

## Tensor -> quant mapping

Matched by tensor name substring (GGUF naming). First match wins.

| Match | Quant | Approx bpw | Why |
|---|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` | `IQ2_XXS` | 2.06 | routed experts, cold; error averages over 512 |
| `ffn_down_exps` | `Q2_K` (hot experts: `IQ3_S`) | 2.6 / 3.1 | writes into residual; more sensitive |
| `ffn_*_shexp` (shared expert) | `Q5_K` | 5.5 | runs every token |
| `ffn_gate_inp` (router) | `F16` | 16 | a wrong route is unrecoverable |
| `*_conv*`, linear-attn `in_proj`/`out_proj`, `dt`/`a`/`beta` | `Q8_0` | 8.5 | recurrent; error compounds over 262K steps |
| full-attn `attn_q/k/v/output` | `Q6_K` | 6.56 | every token |
| `token_embd` | `Q5_K` | 5.5 | input side |
| `output.weight` (lm_head) | `Q6_K` | 6.56 | logit quality |
| `*_norm`, scales | `F32` | 32 | negligible size |

Resulting 397B footprint: routed experts blend to ~2.24 bpw (~108GB), the rest
at ~6 bpw (~8GB) -> **~113GB**, tunable toward ~110GB by widening the IQ2_XXS
share and using Q8 KV at runtime.

## The two quality levers

1. **imatrix calibration.** Collect an importance matrix on a code-heavy corpus
   (agentic-coding workload) so the quantizer weights rounding error by
   per-channel activation magnitude. This is the difference between 2-bit that
   works and 2-bit that babbles.
2. **per-expert mixed precision.** The imatrix run also yields router
   hit-frequency. Promote the hottest ~10-20% of experts (down-proj especially)
   to 3-bit; leave the cold long tail at 2-bit. Cheap GBs, real quality.

## Pipeline

```
FP8 checkpoint  (deepreinforce-ai/Ornith-1.0-397B-FP8, ~397GB, reference only)
   |  dequantize -> BF16
   v
BF16 tensors
   |  imatrix collect (code corpus)  ----> imatrix.dat + router hit stats
   v
GGUF quantize (apply mapping above + per-expert promotion)
   v
Ornith-1.0-397B-ornith.gguf  (~110-120GB, deployable)
```

Bring up on **35B first**: an official 35B GGUF exists, so quantize the 35B
ourselves and diff against the official one to validate the pipeline before
spending a 397B-sized conversion.

## Validation

- perplexity on held-out code vs the BF16 reference, per recipe
- top-1 token agreement and bounded KL on a fixed prompt set
- spot-check the hottest experts kept at higher precision actually move the
  needle (ablate: all-2-bit vs mixed)

> This file is the spec. The implementing tool (M1) reads this mapping; until
> then this is the human-facing contract for what the quantizer must do.
