# ornithology

A narrow, GGML-free local inference engine for the **Ornith-1.0** model family,
in the spirit of [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4) for
DeepSeek V4. The goal: run **Ornith-1.0-397B** and **Ornith-1.0-35B** on a
single high-memory machine (Apple silicon first, then CUDA), with a ds4-style
asymmetric expert quantization that fits the 397B flagship into ~110-130GB.

> Status: **early scaffold.** The model-introspection tooling
> (`inspect`/`config`/`arch`) works and is tested. The inference forward pass
> (the hybrid linear/full attention + MoE engine) is the next milestone and is
> currently stubbed with honest "not implemented" errors. See
> [ROADMAP.md](ROADMAP.md).

## Why a new engine instead of forking ds4

ds4 is built around DeepSeek's **MLA + compressed KV cache**. Ornith-1.0 is a
different architecture (`model_type: qwen3_5_moe`, post-trained on Gemma 4 +
Qwen 3.5), so the central engine is necessarily different:

| | ds4 / DeepSeek V4 | ornithology / Ornith-1.0 |
|---|---|---|
| Attention | MLA, compressed KV on every layer | **Hybrid**: gated delta-net *linear* attention on 3 of 4 layers, GQA *full* attention on the 4th |
| KV cache | compressed, all layers | only the full-attn layers (¼ of layers); linear layers keep constant-size state |
| MoE | ~160 experts | **512 experts / top-10** (397B), **256 / top-8** (35B) |
| Modality | text | multimodal (vision); v1 here is text-only |

The reusable ideas carry over (asymmetric expert quant, GGUF tooling, server +
agent surface); the attention kernels are new work. See [DESIGN.md](DESIGN.md).

## What the weights situation forces

From the published Ornith-1.0 repos:

- **397B**: BF16 only (122 safetensors shards, ~794GB) + an FP8 variant
  (~397GB, `compressed-tensors`). **There is no official 397B GGUF.**
- **35B**: BF16, FP8, **and an official GGUF**.
- **9B**: BF16 + GGUF (dense `qwen3_5`, out of scope here).

So shipping a runnable 397B means **producing the GGUF ourselves** via the
asymmetric quant pipeline (see [tools/quantize/POLICY.md](tools/quantize/POLICY.md)).
The 35B GGUF already exists, which makes it the natural **bring-up target**:
get the engine correct on 35B first, then quantize and run 397B.

## Build

Default build is the portable CPU/CLI path (no GPU, no GGML). It works anywhere
and is what the test suite runs against:

```sh
make            # -> ./ornith
make test       # build + run the test suite
```

GPU inference backends (kernels still to be written, see ROADMAP):

```sh
make metal      # macOS Metal  (primary inference target)
make cuda       # generic Linux CUDA
make cuda-spark # DGX Spark / GB10
```

## Usage (today)

```sh
./ornith version
./ornith arch 397b                 # print the 397B reference architecture + cost model
./ornith arch 35b                  # same for 35B
./ornith config path/to/config.json   # parse a HF config.json, validate it's qwen3_5_moe
./ornith inspect model.gguf           # parse GGUF header/metadata/tensor index
./ornith inspect --tensors model.gguf # also list every tensor
./ornith run model.gguf               # NOT IMPLEMENTED YET (forward pass is M2/M3)
```

`arch` prints a back-of-envelope memory model: parameter count, weight
footprint at a ds4-style ~2.24 bpw expert blend, and KV-cache cost per token
and at full context. For the 397B that lands at **~113GB of weights** and
**~30 KB/token** of KV (only the full-attn layers cost KV), i.e. ~8GB for the
full 256K context at FP16, half that at Q8.

## Layout

```
src/
  ornith.c          CLI entry (version/arch/config/inspect/run)
  ornith_model.*    qwen3_5_moe architecture + cost model (397B and 35B)
  ornith_config.*   Hugging Face config.json -> ornith_arch
  ornith_gguf.*     GGUF v2/v3 reader (header, metadata, tensor index)
  ornith_json.*     tiny dependency-free JSON parser
  ornith_util.c     error handling
  ornith_metal.m    Metal backend            (stub: kernels are the next milestone)
  ornith_cuda.cu    CUDA backend             (stub)
  ornith_server.c   OpenAI/Anthropic HTTP API (stub)
  ornith_agent.c    integrated coding agent   (stub)
tools/
  quantize/POLICY.md   the asymmetric quantization policy for the 397B GGUF
  download_model.sh    fetch weights from Hugging Face
tests/
  test_gguf.c       JSON/config/GGUF/arch tests (run by `make test`)
```

## License

MIT. The Ornith-1.0 weights are also MIT (deepreinforce-ai). This project reuses
GGUF format conventions and (planned) quant block layouts from the llama.cpp
project, also MIT.
