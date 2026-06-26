#!/usr/bin/env bash
# make_gguf.sh — build an Ornith GGUF from the published weights.
#
# This is the answer to "how are the quants created": YOU create them. For ds4,
# antirez pre-quantized DeepSeek V4 and you just downloaded the result. Ornith
# publishes BF16 + FP8 for the 397B but NO 397B GGUF, so we run the conversion
# ourselves. (An official 35B GGUF exists — convert 35B yourself first and diff
# against it to prove the pipeline before spending a 397B-sized run.)
#
# STATUS: orchestration scaffold. It shells out to llama.cpp tooling, which is
# the realistic way to bootstrap quants today; ornithology's own GGUF writer
# (ROADMAP M1) will later replace these steps so the asymmetric per-expert
# policy in POLICY.md is applied natively.
#
# Prerequisites:
#   - llama.cpp checked out + built  (LLAMA_CPP=/path/to/llama.cpp)
#     NOTE: needs a build whose convert + quantize understands qwen3_5_moe.
#     As of now that may require a recent/patched llama.cpp; if convert errors
#     on the arch, that gap is itself an M1 task.
#   - a calibration corpus for the importance matrix (code-heavy)
#
# Usage:
#   LLAMA_CPP=~/src/llama.cpp tools/quantize/make_gguf.sh \
#       models/Ornith-1.0-35B  calib/code.txt  out/ornith-35b
set -euo pipefail

SRC="${1:?source model dir (HF safetensors)}"
CALIB="${2:?calibration corpus text file}"
OUT="${3:?output prefix, e.g. out/ornith-35b}"
: "${LLAMA_CPP:?set LLAMA_CPP to a built llama.cpp checkout}"

mkdir -p "$(dirname "$OUT")"

# 1. Convert HF safetensors -> a full-precision (BF16) GGUF.
#    This is where llama.cpp must understand the qwen3_5_moe tensor layout
#    (hybrid linear/full attention + fine-grained MoE + shared expert).
echo "== [1/3] convert -> BF16 GGUF =="
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" \
    --outfile "${OUT}-bf16.gguf" --outtype bf16

# 2. Collect an importance matrix on the calibration corpus.
#    Drives both better rounding AND per-expert hit-frequency for mixed precision.
echo "== [2/3] imatrix =="
"$LLAMA_CPP/llama-imatrix" \
    -m "${OUT}-bf16.gguf" -f "$CALIB" -o "${OUT}.imatrix" \
    --chunks 512

# 3. Quantize with the asymmetric policy.
#    llama-quantize applies one base type globally and --tensor-type overrides
#    per name pattern. The overrides below encode tools/quantize/POLICY.md:
#    cold routed experts at 2-bit, everything always-on kept high precision.
echo "== [3/3] quantize (asymmetric, see POLICY.md) =="
"$LLAMA_CPP/llama-quantize" \
    --imatrix "${OUT}.imatrix" \
    --tensor-type 'ffn_gate_exps=iq2_xxs' \
    --tensor-type 'ffn_up_exps=iq2_xxs' \
    --tensor-type 'ffn_down_exps=q2_k' \
    --tensor-type 'ffn_gate_shexp=q5_k' \
    --tensor-type 'ffn_up_shexp=q5_k' \
    --tensor-type 'ffn_down_shexp=q5_k' \
    --tensor-type 'ffn_gate_inp=f16' \
    --tensor-type 'attn_q=q6_k' \
    --tensor-type 'attn_k=q6_k' \
    --tensor-type 'attn_v=q6_k' \
    --tensor-type 'attn_output=q6_k' \
    --tensor-type 'token_embd=q5_k' \
    --tensor-type 'output=q6_k' \
    "${OUT}-bf16.gguf" "${OUT}-ornith.gguf" Q2_K

echo
echo "== done: ${OUT}-ornith.gguf =="
echo "verify shape + quant histogram with:"
echo "    ./ornith inspect ${OUT}-ornith.gguf"
echo
echo "NOTE: --tensor-type pattern support and qwen3_5_moe arch support depend on"
echo "your llama.cpp version. Linear-attn weights (conv/delta) and per-expert"
echo "mixed precision (hot experts -> iq3_s) are not yet expressed here; that is"
echo "the native quantizer work in ROADMAP M1."
