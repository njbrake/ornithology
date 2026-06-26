/* ornith_metal.m — Metal backend (PRIMARY inference target).
 *
 * STATUS: stub. This file establishes the backend seam; the kernels are the
 * core engineering work tracked in ROADMAP.md (M2/M3). Metal is the primary
 * target because the design point is a single Apple-silicon machine with
 * 128GB+ unified memory running the 397B model at a ds4-style ~2-bit expert
 * quant.
 *
 * Kernels to implement here, in dependency order:
 *   1. dequant + matmul for the asymmetric expert quant mix (IQ2_XXS / Q2_K),
 *      reusing llama.cpp block layouts.
 *   2. Full-attention layer: GQA (32 q-heads : 2 kv-heads, head_dim 256) with
 *      a paged FP16/Q8 KV cache. Only ~15 of 60 layers need this.
 *   3. Linear-attention layer: gated delta-net recurrence + causal depthwise
 *      conv (kernel 4). Constant per-layer state, no growing KV. This is the
 *      piece with no DeepSeek analogue and the highest-risk kernel.
 *   4. MoE router (top-10 of 512) + grouped expert GEMM with expert-parallel
 *      dispatch over the unified-memory expert pool.
 *
 * Reference to crib from: llama.cpp's qwen3next graph for the linear-attention
 * recurrence, and ds4's Metal expert-streaming for the MoE memory strategy.
 */

#if defined(ORNITH_BACKEND_METAL)
#import <Foundation/Foundation.h>

int ornith_backend_metal_available(void) {
    /* TODO: query MTLCreateSystemDefaultDevice and unified memory size. */
    return 0;
}
#endif
