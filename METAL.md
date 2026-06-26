# Metal (Apple GPU) backend — first cut

This documents the Metal backend's first cut: the per-token quant-aware matvec
hot path, mirrored from the CPU oracle. It is written against the CPU reference
but **has not been compiled or run** — there is no Metal toolchain on the
development box (Linux, no GPU). The intent is for it to be compiled with
`make metal` on the Mac Studio and iterated on from real compiler/runtime
output.

## What's implemented

New, self-contained files (nothing else in the tree is touched; the CPU engine
does not call into this yet):

- `src/ornith_metal.h` — plain-C backend interface: context init/free, weight
  upload to a `MTLBuffer`, and `ornith_metal_matvec{,_q4k,_q6k}` entry points.
- `src/ornith_metal.m` — Objective-C host code + embedded Metal Shading Language
  (MSL) kernels compiled at runtime via `newLibraryWithSource` (ds4 style). All
  of it is gated behind `#if defined(ORNITH_BACKEND_METAL)`, so on the default
  CPU build and on Linux CI the translation unit is empty.
- `Makefile` — `metal:` target compiles `ornith_metal.m` to its own object with
  `-DORNITH_BACKEND_METAL` and links `-framework Metal -framework Foundation
  -framework Accelerate`.

Kernels (all in the embedded MSL string in `ornith_metal.m`):

1. `kernel_quantize_q8_K` — activation row `f32 -> Q8_K` super-blocks. One
   thread per 256-element super-block. A line-for-line transcription of
   `qdot_quantize_row_q8_K` (`amax`/`max` scan, `iscale = -127/max`, clamp to
   `[-128,127]`, `bsums` group sums, `d = 1/iscale`).
2. `kernel_mul_mv_q4_K_q8_K` — Q4_K weight × Q8_K activation integer dot.
3. `kernel_mul_mv_q6_K_q8_K` — Q6_K weight × Q8_K activation integer dot.
4. `kernel_mul_mv_q8_0_q8_K` — Q8_0 weight × Q8_K activation integer dot.

### How the kernels mirror the CPU quant-aware matvec

The CPU oracle (`src/ornith_qdot.c`) does **not** dequantize weights to f32.
It quantizes the *activation* row once to Q8_K, then runs a type-specific
*integer* dot directly against the packed weight bytes, scaling by the per-block
`d` (and, for Q4_K, folding the per-sub-block mins via the activation `bsums`).
The Metal kernels do the same thing, byte-for-byte:

- The MSL `block_q4_K` / `block_q6_K` / `block_q8_0` structs match the GGUF
  on-disk layouts used by `ornith_quant.c` (144 / 210 / 34 bytes). f16 fields
  are read as MSL `half` (IEEE binary16, same as ggml f16 on little-endian
  Apple silicon). `static_assert`s on `sizeof` each block catch any device-side
  padding at shader-compile time.
- The Q8_K activation block is produced *and* consumed entirely on-device, so
  its layout only has to be self-consistent (it matches the C `oq8k_block`:
  `float d; int8 qs[256]; int16 bsums[16]`).
- `q4k_scale_min` (the 6-bit scale/min unpack) is transcribed exactly from the
  CPU `q4k_scale_min` / `get_scale_min_k4`.
- The Q4_K inner loop reproduces the CPU's `summs` (mins × bsums) and `isum`
  (4-bit nibble × int8 activation, scaled by the sub-block scale) and combines
  them as `d*isum + dm*summs`. Q6_K reproduces the `q1..q4` 6-bit reconstruction
  (`-32` bias) and the `sc[is + {0,2,4,6}]` indexing. Q8_0 reproduces the
  "8 Q8_0 blocks per Q8_K super-block" plain int8 dot.

Parallelization: **one threadgroup per output row** (the `lm_head` / projection
output index), 64 threads striding over the row's super-blocks, then a
threadgroup-memory tree reduction to `out[row]`. This is the simplest layout
that keeps the numerics identical to the CPU loop; it is not yet optimized
(see plan below).

## How to compile on the Mac Studio

From the repo root:

```bash
make clean
make metal        # builds ./ornith with the Metal backend compiled in
```

Requirements: Xcode command-line tools (`xcode-select --install`) so `clang`
finds the `Metal`, `Foundation`, and `Accelerate` frameworks. The kernels are
compiled from source at *runtime* (`newLibraryWithSource`), so there is no
offline `.metallib` step.

There is no standalone entry point wired up yet (CPU-engine integration is a
later pass). To smoke-test the backend in isolation, add a tiny harness that
calls `ornith_metal_init` → `ornith_metal_upload_weight` → `ornith_metal_matvec`
on a small Q4_K tensor and diffs the result against `qdot_vec_dot` from
`ornith_qdot.c`. That is the **first thing to run on the Studio**: a numeric
parity check against the CPU oracle on a single matvec.

## Expected issues on first compile / run (be skeptical)

This is unverified code. Likely first-pass problems, roughly in order:

1. **`-std=c11` on the `.m` file.** The `metal:` rule reuses `$(CFLAGS)` which
   includes `-std=c11`. clang accepts Objective-C with a C `-std`, but if it
   complains, compile `ornith_metal.o` without `-std=c11` (drop `$(CSTD)` for
   that one rule).
2. **MSL `static_assert` failures.** If Metal pads any block struct, the
   `static_assert`s fire with a clear message. Fix by reordering / adding
   explicit padding or `packed` attributes until each `sizeof` matches
   144 / 210 / 34 / 292.
3. **Fast-math rounding drift in the activation quant.** Metal enables
   fast-math by default; `rint(iscale*x)` may not match the CPU's `lrintf`
   exactly at ties, producing off-by-one int8 values and small output drift.
   If parity is off, recompile the library with strict math
   (`MTLMathModeSafe`, macOS 15+, or `options.fastMathEnabled = NO`) — wire an
   env toggle like ds4's `DS4_METAL_MATH_SAFE`.
4. **`char` signedness.** MSL `char` is signed (matches int8), but double-check
   the Q8_K `qs`/`bsums` and Q6_K `scales` are read signed; a signedness slip
   shows up as large wrong dot products.
5. **f16 endianness / NaN handling.** Reading GGUF f16 as `half` assumes
   little-endian (true on Apple silicon) and IEEE binary16. Should be fine, but
   it is an assumption.
6. **Threadgroup size vs. pipeline limit.** 64 threads/threadgroup is safe, but
   the tree reduction assumes a power-of-two thread count; keep it ≥ a power of
   two if you retune.
7. **Buffer alignment on the weight cast.** Per-block offsets (144/210/34) are
   even and the `MTLBuffer` base is page-aligned, so the
   `device const uchar* -> device const block_*` casts should satisfy the
   2-byte alignment requirement. If Metal asserts on alignment, switch to
   reading the half scale via `as_type`/byte loads.
8. **MRC memory management.** The file uses manual retain/release (no ARC;
   `-fno-objc-arc` is passed). If you flip to ARC, the `id<>` fields in the
   malloc'd context struct need `__bridge` handling.

## Plan for the remaining kernels

This first cut is deliberately scoped to the matvec hot path. Still to do,
roughly in dependency order (tracked against `ROADMAP.md` and the ds4 graph):

1. **Batched / multi-token matvec** and weights resident across calls (avoid
   re-uploading the activation buffer each call; keep a persistent Q8_K scratch
   buffer; fuse the quantize + matvec into one command buffer with multiple
   encoders — already done — but also batch over tokens).
2. **GQA full-attention layer** — 32 q-heads : 2 kv-heads, head_dim 256, with a
   paged FP16/Q8 KV cache. Crib structure from `ds4-ref/metal/flash_attn.metal`.
3. **Gated delta-net linear-attention layer** — recurrence + causal depthwise
   conv (kernel 4), constant per-layer state. Highest-risk kernel (no DeepSeek
   analogue); reference llama.cpp's qwen3next graph.
4. **MoE dispatch** — top-k router + grouped expert GEMM with expert-parallel
   dispatch over the unified-memory expert pool (sub-2-bit IQ2_XXS / Q2_K
   expert quant). Reference `ds4-ref/metal/moe.metal` and ds4's expert
   streaming/residency strategy.
5. **CPU-engine integration** — have `ornith_forward.c` / `ornith_rforward.c`
   route matvecs through the Metal backend when `ORNITH_BACKEND_METAL` is set
   and `qdot_can(wtype, n)` holds, falling back to CPU otherwise.

## Attribution

The quant block layouts and the integer-dot math are derived from ggml (MIT)
via the ds4 reference clone. The Objective-C / Metal scaffolding follows the
structure of `ds4-ref/ds4_metal.m`. See `LICENSE`.
