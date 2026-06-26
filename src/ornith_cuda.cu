/* ornith_cuda.cu — CUDA backend (generic Linux + DGX Spark/GB10).
 *
 * STATUS: stub. Same kernel set as the Metal backend (see ornith_metal.m):
 * asymmetric-quant expert GEMM, GQA full-attention with paged KV, gated
 * delta-net linear attention + causal conv, and top-10/512 MoE routing.
 *
 * The Spark target (sm_121a, 128GB coherent memory) is the closest CUDA analog
 * to the primary Apple-silicon design point, so it gets a dedicated make target.
 */

#if defined(ORNITH_BACKEND_CUDA)
extern "C" int ornith_backend_cuda_available(void) {
    /* TODO: cudaGetDeviceCount + report VRAM / coherent-memory budget. */
    return 0;
}
#endif
