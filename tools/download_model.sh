#!/usr/bin/env bash
# Download Ornith-1.0 weights from Hugging Face into ./models.
#
# Usage:
#   tools/download_model.sh 35b            # BF16 safetensors (bring-up target)
#   tools/download_model.sh 35b-gguf       # official 35B GGUF (reference)
#   tools/download_model.sh 397b-fp8       # FP8 (quant source for the 397B GGUF)
#   tools/download_model.sh 397b           # BF16 safetensors (~794GB, careful)
#
# Requires the `huggingface-cli` (pip install huggingface_hub) or `hf`.
set -euo pipefail

variant="${1:-}"
case "$variant" in
  35b)        repo="deepreinforce-ai/Ornith-1.0-35B" ;;
  35b-fp8)    repo="deepreinforce-ai/Ornith-1.0-35B-FP8" ;;
  35b-gguf)   repo="deepreinforce-ai/Ornith-1.0-35B-GGUF" ;;
  397b)       repo="deepreinforce-ai/Ornith-1.0-397B" ;;
  397b-fp8)   repo="deepreinforce-ai/Ornith-1.0-397B-FP8" ;;
  *)
    echo "usage: $0 {35b|35b-fp8|35b-gguf|397b|397b-fp8}" >&2
    exit 1 ;;
esac

dest="models/$(basename "$repo")"
mkdir -p "$dest"
echo ">> downloading $repo -> $dest"

if command -v hf >/dev/null 2>&1; then
  hf download "$repo" --local-dir "$dest"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$repo" --local-dir "$dest"
else
  echo "error: install huggingface_hub (pip install huggingface_hub)" >&2
  exit 1
fi

echo ">> done. inspect a GGUF with:  ./ornith inspect $dest/*.gguf"
echo ">> inspect a safetensors config with:  ./ornith config $dest/config.json"
