# ornithology

A narrow, GGML-free local inference engine for the **Ornith-1.0** model family,
in the spirit of [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4) for
DeepSeek V4. The goal: run **Ornith-1.0-397B** and **Ornith-1.0-35B** on a
single high-memory machine (Apple silicon first, then CUDA), with a ds4-style
asymmetric expert quantization that fits the 397B flagship into ~110-130GB.

> Status: **the CPU stack is feature-complete and broadly at ds4 parity.** It
> loads a real Ornith GGUF (dense 9B and MoE 35B) and generates coherent text;
> the forward pass (hybrid linear/full attention + MoE, exact qwen3.5 gating) is
> validated against llama.cpp (logits match to ~0.8%). Shipped on CPU:
>
> - byte-level BPE tokenizer; **mmap** weight loading (so the 397B's ~113GB never
>   needs a single allocation); on-the-fly dequant
> - quant codecs: F32/F16/BF16/Q8_0/Q4_0, k-quants **Q2_K/Q4_K/Q5_K/Q6_K**, and
>   sub-2-bit **IQ2_XXS** — encode + decode; **imatrix** importance-weighted quant
> - quant-aware **Q8 matvec** (the ds4-style integer dot; ~3x faster than f32),
>   **Q8 KV-cache**, **sampling** (temp/top-p/top-k/min-p/repeat-penalty/seed)
> - **OpenAI + Anthropic + OpenAI-Responses** HTTP server (`ornith serve`) with
>   streaming, **tool/function calling** on all three, and a built-in web UI
> - **integrated coding agent** (`ornith agent`, read/write/run tools with a
>   confirm gate) + interactive **REPL** (`ornith repl`)
> - **persistent KV sessions** (bit-exact save/resume), and **bench**/**eval**
>   (tok/s, perplexity) harnesses
>
> Remaining for full parity is the **GPU half**: the Metal backend is a first cut
> (compiles on Apple silicon via `make metal`; kernels mirror the CPU oracle but
> are not yet verified on-device), and CUDA/ROCm, distributed inference, and
> speculative decoding are not started. See [ROADMAP.md](ROADMAP.md).

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

## Quickstart on a Mac (CPU today)

The CPU engine is real and generates coherent text from real Ornith GGUFs. Metal
is the primary *target* but its kernels are still stubs, so **build the default
(CPU) target for now** — `make metal` is not functional yet (see ROADMAP M3).

```sh
make                                   # CPU build -> ./ornith  (NOT `make metal` yet)

# get a model (no weights are committed). The dense 9B is the smallest;
# the 35B is the smallest MoE and fits a 128GB Mac comfortably.
tools/download_model.sh 9b-gguf        # ~5.6GB  (or: 35b-gguf ~21GB)

# one-shot generation (works today)
./ornith run --prompt "The capital of France is" -n 32 \
    models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf

# OpenAI / Anthropic / Responses compatible server (CPU-backed)
./ornith serve --port 8080 models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf
curl -s localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":64}'
# also: POST /v1/responses, POST /v1/messages, GET /v1/models, GET /health
```

Expect CPU speed (a naive reference kernel; faster on a many-core Studio, but not
GPU-fast). The **397B does not run yet** — it needs the sub-2-bit asymmetric quant
*and* the Metal backend. On a 128GB Studio the **35B** is the model to try today.
Weights stay quantized in RAM and are dequantized on the fly, so peak memory is
roughly the GGUF file size, not the f32 model.

## Usage (today)

```sh
./ornith version
./ornith arch 397b                 # print the 397B reference architecture + cost model
./ornith arch 35b                  # same for 35B
./ornith config path/to/config.json   # parse a HF config.json, validate it's qwen3_5_moe
./ornith inspect model.gguf           # parse GGUF header/metadata/tensor index
./ornith inspect --tensors model.gguf # also list every tensor
./ornith run --prompt "..." [-n N] [--temp T --top-p P --top-k K --seed S] \
             [--kv-q8] [--session FILE] model.gguf   # real-weight generation (CPU)
./ornith serve [--host H] [--port P] model.gguf # OpenAI/Anthropic/Responses server + web UI
./ornith repl model.gguf                        # interactive multi-turn chat
./ornith agent [--yolo] [--max-iters N] --task "..." model.gguf  # coding agent (tools)
./ornith imatrix model.gguf corpus.txt -o imatrix.dat   # collect importance matrix
./ornith quantize [--base TYPE] [--imatrix imatrix.dat] in.gguf out.gguf
./ornith bench [--prompt-len P] [--gen N] model.gguf    # prefill/decode tok/s + RSS
./ornith eval model.gguf corpus.txt                     # perplexity
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
  ornith_tokenizer.* byte-level BPE tokenizer (loaded from GGUF metadata)
  ornith_rforward.*  real-weight forward + on-the-fly dequant generation
  ornith_server.*   OpenAI/Anthropic/Responses HTTP server (CPU-backed)
  ornith_metal.m    Metal backend            (stub: kernels are the next milestone)
  ornith_cuda.cu    CUDA backend             (stub)
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
