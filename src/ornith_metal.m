/* ornith_metal.m — Metal (Apple GPU) backend, FIRST CUT.
 *
 * STATUS: written against the CPU oracle (ornith_qdot.c) but NOT yet compiled
 * or run — there is no Metal toolchain on the dev box. Compile on the Studio
 * with `make metal`; see METAL.md for the build command, the expected first
 * errors, and the remaining-kernel plan.
 *
 * What this implements (the per-token hot path, incl. lm_head):
 *   1. kernel_quantize_q8_K       — activation row f32 -> Q8_K super-blocks,
 *                                    a transcription of qdot_quantize_row_q8_K.
 *   2. kernel_mul_mv_q4_K_q8_K    — integer vec_dot for Q4_K weights, one
 *                                    threadgroup per output row, transcribed
 *                                    block-for-block from vec_dot_q4_K.
 *   3. kernel_mul_mv_q6_K_q8_K    — same for Q6_K (transcribes vec_dot_q6_K).
 *   4. kernel_mul_mv_q8_0_q8_K    — same for Q8_0 (transcribes vec_dot_q8_0).
 *
 * Design choice: rather than copy ds4's llama.cpp-derived f32-activation Metal
 * matvec, the kernels here mirror ornithology's OWN integer-dot oracle so the
 * GPU result matches the CPU reference numerically (the activation is quantized
 * to Q8_K exactly as on CPU, then an integer dot runs against the packed weight
 * bytes — no f32 weight materialization). The Objective-C scaffolding (device /
 * queue / newLibraryWithSource / pipelines / shared buffers / dispatch) follows
 * ds4-ref/ds4_metal.m.
 *
 * Everything is gated behind ORNITH_BACKEND_METAL: on the default CPU build and
 * on Linux CI this file compiles to nothing.
 *
 * Attribution: quant block layouts + integer-dot math derived from ggml (MIT)
 * via the ds4 reference; see LICENSE.
 */

#if defined(ORNITH_BACKEND_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ornith_metal.h"
#include "ornith_gguf.h"   /* oggml_type: OGGML_Q4_K=12, OGGML_Q6_K=14, OGGML_Q8_0=8 */

#define ORNITH_METAL_QK_K   256
/* Threads per threadgroup for the matvec reduction. Must be a power of two and
 * match the `scratch[]` size declared in the MSL reduction below. */
#define ORNITH_METAL_TG     64

/* ---- embedded Metal Shading Language source -------------------------------
 * Compiled at runtime via newLibraryWithSource (ds4 style). The device-side
 * struct layouts match ornith_quant.c / ornith_qdot.c byte-for-byte:
 *   block_q4_K  144B : d:half dmin:half scales[12] qs[128]
 *   block_q6_K  210B : ql[128] qh[64] scales[16]:int8 d:half
 *   block_q8_0   34B : d:half qs[32]:int8
 *   block_q8_K  292B : d:float qs[256]:int8 bsums[16]:int16  (device-internal)
 * Each physical line is its own string literal so MSL compile errors report a
 * usable line number. */
static const char *ORNITH_METAL_SOURCE =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"#define QK_K 256\n"
"\n"
"typedef struct { half d; half dmin; uchar scales[12]; uchar qs[128]; } block_q4_K;\n"
"typedef struct { uchar ql[128]; uchar qh[64]; char scales[16]; half d; } block_q6_K;\n"
"typedef struct { half d; char qs[32]; } block_q8_0;\n"
"typedef struct { float d; char qs[256]; short bsums[16]; } block_q8_K;\n"
"\n"
"/* Catch any device-side struct padding at shader-compile time: the GGUF block\n"
"   layouts must be byte-identical to ornith_quant.c / ornith_qdot.c. */\n"
"static_assert(sizeof(block_q4_K) == 144, \"block_q4_K must be 144 bytes\");\n"
"static_assert(sizeof(block_q6_K) == 210, \"block_q6_K must be 210 bytes\");\n"
"static_assert(sizeof(block_q8_0) ==  34, \"block_q8_0 must be 34 bytes\");\n"
"static_assert(sizeof(block_q8_K) == 292, \"block_q8_K must be 292 bytes\");\n"
"\n"
"/* Unpack 6-bit scale `d` and 6-bit min `m` for sub-block j (0..7) from a\n"
"   Q4_K scales[12] field. Mirrors get_scale_min_k4 / q4k_scale_min. */\n"
"static inline void q4k_scale_min(int j, device const uchar *q,\n"
"                                 thread uchar &d, thread uchar &m) {\n"
"    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }\n"
"    else {\n"
"        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);\n"
"        m = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);\n"
"    }\n"
"}\n"
"\n"
"/* activation row f32 -> Q8_K, one thread per 256-element super-block.\n"
"   Transcription of qdot_quantize_row_q8_K. */\n"
"kernel void kernel_quantize_q8_K(\n"
"        device const float    *x   [[buffer(0)]],\n"
"        device       block_q8_K *y [[buffer(1)]],\n"
"        constant     uint     &nb  [[buffer(2)]],\n"
"        uint i [[thread_position_in_grid]]) {\n"
"    if (i >= nb) return;\n"
"    device const float *xb = x + (uint)i * QK_K;\n"
"    device block_q8_K  *yb = y + i;\n"
"    float amax = 0.0f, mx = 0.0f;\n"
"    for (int j = 0; j < QK_K; j++) {\n"
"        float ax = fabs(xb[j]);\n"
"        if (ax > amax) { amax = ax; mx = xb[j]; }\n"
"    }\n"
"    if (amax == 0.0f) {\n"
"        yb->d = 0.0f;\n"
"        for (int j = 0; j < QK_K; j++) yb->qs[j] = 0;\n"
"        for (int j = 0; j < QK_K/16; j++) yb->bsums[j] = 0;\n"
"        return;\n"
"    }\n"
"    const float iscale = -127.0f / mx;\n"
"    for (int j = 0; j < QK_K; j++) {\n"
"        int v = (int)rint(iscale * xb[j]);\n"
"        v = min(v, 127); v = max(v, -128);\n"
"        yb->qs[j] = (char)v;\n"
"    }\n"
"    for (int j = 0; j < QK_K/16; j++) {\n"
"        int sum = 0;\n"
"        for (int t = 0; t < 16; t++) sum += (int)yb->qs[j*16 + t];\n"
"        yb->bsums[j] = (short)sum;\n"
"    }\n"
"    yb->d = 1.0f / iscale;\n"
"}\n"
"\n"
"/* Q4_K x Q8_K matvec. One threadgroup per output row; threads stride over the\n"
"   row's super-blocks and reduce. Transcribes vec_dot_q4_K. */\n"
"kernel void kernel_mul_mv_q4_K_q8_K(\n"
"        device const uchar      *wbytes [[buffer(0)]],\n"
"        device const block_q8_K *yq     [[buffer(1)]],\n"
"        device       float      *out    [[buffer(2)]],\n"
"        constant     int        &n      [[buffer(3)]],\n"
"        constant     int        &nrows  [[buffer(4)]],\n"
"        uint  row [[threadgroup_position_in_grid]],\n"
"        uint  tid [[thread_position_in_threadgroup]],\n"
"        uint  tpg [[threads_per_threadgroup]]) {\n"
"    threadgroup float scratch[64];\n"
"    if (row >= (uint)nrows) { return; }\n"
"    const int nb = n / QK_K;\n"
"    device const uchar *wr = wbytes + (size_t)row * (size_t)nb * 144;\n"
"    float partial = 0.0f;\n"
"    for (int i = (int)tid; i < nb; i += (int)tpg) {\n"
"        device const block_q4_K *blk = (device const block_q4_K *)(wr + (size_t)i * 144);\n"
"        float d  =  yq[i].d * (float)blk->d;\n"
"        float dm = -yq[i].d * (float)blk->dmin;\n"
"        device const uchar *sc = blk->scales;\n"
"        device const uchar *qs = blk->qs;\n"
"        device const char  *q8 = yq[i].qs;\n"
"        int summs = 0;\n"
"        for (int j = 0; j < QK_K/32; j++) {\n"
"            uchar scv, mv; q4k_scale_min(j, sc, scv, mv);\n"
"            summs += (int)mv * ((int)yq[i].bsums[j*2] + (int)yq[i].bsums[j*2 + 1]);\n"
"        }\n"
"        int isum = 0;\n"
"        for (int j = 0; j < QK_K/32; j++) {\n"
"            uchar scv, mv; q4k_scale_min(j, sc, scv, mv);\n"
"            const int byte_off = (j >> 1) * 32;\n"
"            const int shift    = (j & 1) * 4;\n"
"            int acc = 0;\n"
"            for (int l = 0; l < 32; l++)\n"
"                acc += (int)((qs[byte_off + l] >> shift) & 0xF) * (int)q8[j*32 + l];\n"
"            isum += acc * (int)scv;\n"
"        }\n"
"        partial += d * (float)isum + dm * (float)summs;\n"
"    }\n"
"    scratch[tid] = partial;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = tpg >> 1; s > 0; s >>= 1) {\n"
"        if (tid < s) scratch[tid] += scratch[tid + s];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out[row] = scratch[0];\n"
"}\n"
"\n"
"/* Q6_K x Q8_K matvec. Transcribes vec_dot_q6_K. */\n"
"kernel void kernel_mul_mv_q6_K_q8_K(\n"
"        device const uchar      *wbytes [[buffer(0)]],\n"
"        device const block_q8_K *yq     [[buffer(1)]],\n"
"        device       float      *out    [[buffer(2)]],\n"
"        constant     int        &n      [[buffer(3)]],\n"
"        constant     int        &nrows  [[buffer(4)]],\n"
"        uint  row [[threadgroup_position_in_grid]],\n"
"        uint  tid [[thread_position_in_threadgroup]],\n"
"        uint  tpg [[threads_per_threadgroup]]) {\n"
"    threadgroup float scratch[64];\n"
"    if (row >= (uint)nrows) { return; }\n"
"    const int nb = n / QK_K;\n"
"    device const uchar *wr = wbytes + (size_t)row * (size_t)nb * 210;\n"
"    float partial = 0.0f;\n"
"    for (int i = (int)tid; i < nb; i += (int)tpg) {\n"
"        device const block_q6_K *blk = (device const block_q6_K *)(wr + (size_t)i * 210);\n"
"        device const uchar *ql = blk->ql;\n"
"        device const uchar *qh = blk->qh;\n"
"        device const char  *sc = blk->scales;\n"
"        float d = yq[i].d * (float)blk->d;\n"
"        device const char  *q8 = yq[i].qs;\n"
"        int isum = 0;\n"
"        for (int n2 = 0; n2 < QK_K; n2 += 128) {\n"
"            for (int l = 0; l < 32; l++) {\n"
"                int is = l / 16;\n"
"                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;\n"
"                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;\n"
"                int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;\n"
"                int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;\n"
"                isum += (int)q8[l +  0] * q1 * (int)sc[is + 0]\n"
"                      + (int)q8[l + 32] * q2 * (int)sc[is + 2]\n"
"                      + (int)q8[l + 64] * q3 * (int)sc[is + 4]\n"
"                      + (int)q8[l + 96] * q4 * (int)sc[is + 6];\n"
"            }\n"
"            q8 += 128; ql += 64; qh += 32; sc += 8;\n"
"        }\n"
"        partial += d * (float)isum;\n"
"    }\n"
"    scratch[tid] = partial;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = tpg >> 1; s > 0; s >>= 1) {\n"
"        if (tid < s) scratch[tid] += scratch[tid + s];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out[row] = scratch[0];\n"
"}\n"
"\n"
"/* Q8_0 x Q8_K matvec. Transcribes vec_dot_q8_0 (8 Q8_0 blocks per Q8_K). */\n"
"kernel void kernel_mul_mv_q8_0_q8_K(\n"
"        device const uchar      *wbytes [[buffer(0)]],\n"
"        device const block_q8_K *yq     [[buffer(1)]],\n"
"        device       float      *out    [[buffer(2)]],\n"
"        constant     int        &n      [[buffer(3)]],\n"
"        constant     int        &nrows  [[buffer(4)]],\n"
"        uint  row [[threadgroup_position_in_grid]],\n"
"        uint  tid [[thread_position_in_threadgroup]],\n"
"        uint  tpg [[threads_per_threadgroup]]) {\n"
"    threadgroup float scratch[64];\n"
"    if (row >= (uint)nrows) { return; }\n"
"    const int nb = n / QK_K;\n"
"    device const uchar *wr = wbytes + (size_t)row * (size_t)nb * 8 * 34;\n"
"    float partial = 0.0f;\n"
"    for (int i = (int)tid; i < nb; i += (int)tpg) {\n"
"        device const uchar *w = wr + (size_t)i * 8 * 34;\n"
"        device const char  *q8 = yq[i].qs;\n"
"        float blk_sum = 0.0f;\n"
"        for (int sb = 0; sb < 8; sb++) {\n"
"            device const block_q8_0 *wb = (device const block_q8_0 *)(w + (size_t)sb * 34);\n"
"            float wd = (float)wb->d;\n"
"            device const char *wq = wb->qs;\n"
"            device const char *yq8 = q8 + sb * 32;\n"
"            int isum = 0;\n"
"            for (int l = 0; l < 32; l++) isum += (int)wq[l] * (int)yq8[l];\n"
"            blk_sum += wd * (float)isum;\n"
"        }\n"
"        partial += yq[i].d * blk_sum;\n"
"    }\n"
"    scratch[tid] = partial;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = tpg >> 1; s > 0; s >>= 1) {\n"
"        if (tid < s) scratch[tid] += scratch[tid + s];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) out[row] = scratch[0];\n"
"}\n";

/* ---- backend context ------------------------------------------------------ */

struct ornith_metal_ctx {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> pso_quant_q8_K;
    id<MTLComputePipelineState> pso_q4_K;
    id<MTLComputePipelineState> pso_q6_K;
    id<MTLComputePipelineState> pso_q8_0;
};

struct ornith_metal_tensor {
    id<MTLBuffer> buffer;   /* packed GGUF block bytes, resident on device */
    uint32_t      wtype;    /* oggml_type */
    int64_t       n_cols;   /* elements per row */
    int64_t       n_rows;   /* number of weight rows / matvec outputs */
};

/* Per-type packed bytes for one weight row of `n_cols` elements. 0 if the type
 * is not a supported integer-dot weight type or `n_cols` is not 256-aligned. */
static size_t metal_row_bytes(uint32_t wtype, int64_t n_cols) {
    if (n_cols <= 0 || (n_cols % ORNITH_METAL_QK_K) != 0) return 0;
    const int64_t nb = n_cols / ORNITH_METAL_QK_K;   /* 256-superblocks */
    switch (wtype) {
    case OGGML_Q4_K: return (size_t)nb * 144;
    case OGGML_Q6_K: return (size_t)nb * 210;
    case OGGML_Q8_0: return (size_t)nb * 8 * 34;     /* 8 Q8_0 blocks / 256 */
    default:         return 0;
    }
}

/* ---- pipeline build helper ------------------------------------------------ */

static id<MTLComputePipelineState>
metal_make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, const char *name) {
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        fprintf(stderr, "ornith metal: function '%s' not found in library\n", name);
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    if (!pso) {
        fprintf(stderr, "ornith metal: pipeline '%s' failed: %s\n", name,
                error ? [[error localizedDescription] UTF8String] : "(unknown)");
        return nil;
    }
    return pso;
}

/* ---- public API ----------------------------------------------------------- */

int ornith_backend_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        int ok = (dev != nil);
        [dev release];
        return ok;
    }
}

ornith_metal_ctx *ornith_metal_init(void) {
    @autoreleasepool {
        ornith_metal_ctx *ctx = (ornith_metal_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) return NULL;

        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "ornith metal: no Metal device available\n");
            free(ctx);
            return NULL;
        }
        fprintf(stderr, "ornith metal: device '%s', unified=%d\n",
                [[ctx->device name] UTF8String], (int)[ctx->device hasUnifiedMemory]);

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "ornith metal: failed to create command queue\n");
            ornith_metal_free(ctx);
            return NULL;
        }

        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:ORNITH_METAL_SOURCE];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
        [opts release];
        if (!ctx->library) {
            fprintf(stderr, "ornith metal: MSL compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            ornith_metal_free(ctx);
            return NULL;
        }

        ctx->pso_quant_q8_K = metal_make_pipeline(ctx->device, ctx->library, "kernel_quantize_q8_K");
        ctx->pso_q4_K = metal_make_pipeline(ctx->device, ctx->library, "kernel_mul_mv_q4_K_q8_K");
        ctx->pso_q6_K = metal_make_pipeline(ctx->device, ctx->library, "kernel_mul_mv_q6_K_q8_K");
        ctx->pso_q8_0 = metal_make_pipeline(ctx->device, ctx->library, "kernel_mul_mv_q8_0_q8_K");
        if (!ctx->pso_quant_q8_K || !ctx->pso_q4_K || !ctx->pso_q6_K || !ctx->pso_q8_0) {
            ornith_metal_free(ctx);
            return NULL;
        }
        return ctx;
    }
}

void ornith_metal_free(ornith_metal_ctx *ctx) {
    if (!ctx) return;
    [ctx->pso_quant_q8_K release];
    [ctx->pso_q4_K release];
    [ctx->pso_q6_K release];
    [ctx->pso_q8_0 release];
    [ctx->library release];
    [ctx->queue release];
    [ctx->device release];
    free(ctx);
}

ornith_metal_tensor *ornith_metal_upload_weight(ornith_metal_ctx *ctx,
                                                uint32_t wtype,
                                                const void *data, size_t nbytes,
                                                int64_t n_cols, int64_t n_rows) {
    if (!ctx || !data || n_rows <= 0) return NULL;
    const size_t row_bytes = metal_row_bytes(wtype, n_cols);
    if (row_bytes == 0) {
        fprintf(stderr, "ornith metal: unsupported wtype %u or bad n_cols %lld\n",
                wtype, (long long)n_cols);
        return NULL;
    }
    const size_t expect = row_bytes * (size_t)n_rows;
    if (nbytes < expect) {
        fprintf(stderr, "ornith metal: weight buffer too small (%zu < %zu)\n", nbytes, expect);
        return NULL;
    }
    @autoreleasepool {
        ornith_metal_tensor *t = (ornith_metal_tensor *)calloc(1, sizeof(*t));
        if (!t) return NULL;
        t->buffer = [ctx->device newBufferWithBytes:data
                                             length:expect
                                            options:MTLResourceStorageModeShared];
        if (!t->buffer) {
            fprintf(stderr, "ornith metal: weight buffer alloc failed (%zu bytes)\n", expect);
            free(t);
            return NULL;
        }
        t->wtype = wtype;
        t->n_cols = n_cols;
        t->n_rows = n_rows;
        return t;
    }
}

void ornith_metal_free_tensor(ornith_metal_tensor *t) {
    if (!t) return;
    [t->buffer release];
    free(t);
}

/* Pick the matvec pipeline for a weight type. */
static id<MTLComputePipelineState>
metal_matvec_pso(ornith_metal_ctx *ctx, uint32_t wtype) {
    switch (wtype) {
    case OGGML_Q4_K: return ctx->pso_q4_K;
    case OGGML_Q6_K: return ctx->pso_q6_K;
    case OGGML_Q8_0: return ctx->pso_q8_0;
    default:         return nil;
    }
}

int ornith_metal_matvec(ornith_metal_ctx *ctx, const ornith_metal_tensor *w,
                        const float *x, float *out) {
    if (!ctx || !w || !x || !out) return 1;
    id<MTLComputePipelineState> pso = metal_matvec_pso(ctx, w->wtype);
    if (!pso) {
        fprintf(stderr, "ornith metal: no matvec pipeline for wtype %u\n", w->wtype);
        return 1;
    }
    const int   n     = (int)w->n_cols;
    const int   nrows = (int)w->n_rows;
    const uint  nb    = (uint)(w->n_cols / ORNITH_METAL_QK_K);

    @autoreleasepool {
        /* Transient shared buffers (unified memory): activation, Q8_K, output. */
        id<MTLBuffer> bx = [ctx->device newBufferWithBytes:x
                                                    length:(size_t)n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq = [ctx->device newBufferWithLength:(size_t)nb * 292
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [ctx->device newBufferWithLength:(size_t)nrows * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        if (!bx || !bq || !bo) {
            fprintf(stderr, "ornith metal: matvec scratch alloc failed\n");
            [bx release]; [bq release]; [bo release];
            return 1;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];

        /* 1) quantize activation row to Q8_K: one thread per super-block. */
        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        [e1 setComputePipelineState:ctx->pso_quant_q8_K];
        [e1 setBuffer:bx offset:0 atIndex:0];
        [e1 setBuffer:bq offset:0 atIndex:1];
        [e1 setBytes:&nb length:sizeof(nb) atIndex:2];
        [e1 dispatchThreadgroups:MTLSizeMake((nb + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e1 endEncoding];

        /* 2) integer-dot matvec: one threadgroup per output row. */
        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:pso];
        [e2 setBuffer:w->buffer offset:0 atIndex:0];
        [e2 setBuffer:bq offset:0 atIndex:1];
        [e2 setBuffer:bo offset:0 atIndex:2];
        [e2 setBytes:&n length:sizeof(n) atIndex:3];
        [e2 setBytes:&nrows length:sizeof(nrows) atIndex:4];
        [e2 dispatchThreadgroups:MTLSizeMake((NSUInteger)nrows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(ORNITH_METAL_TG, 1, 1)];
        [e2 endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "ornith metal: matvec command buffer error: %s\n",
                    cb.error ? [[cb.error localizedDescription] UTF8String] : "(unknown)");
            [bx release]; [bq release]; [bo release];
            return 1;
        }

        memcpy(out, [bo contents], (size_t)nrows * sizeof(float));
        [bx release]; [bq release]; [bo release];
        return 0;
    }
}

int ornith_metal_matvec_q4k(ornith_metal_ctx *ctx, const ornith_metal_tensor *w,
                            const float *x, float *out) {
    if (!w || w->wtype != OGGML_Q4_K) return 1;
    return ornith_metal_matvec(ctx, w, x, out);
}

int ornith_metal_matvec_q6k(ornith_metal_ctx *ctx, const ornith_metal_tensor *w,
                            const float *x, float *out) {
    if (!w || w->wtype != OGGML_Q6_K) return 1;
    return ornith_metal_matvec(ctx, w, x, out);
}

#endif /* ORNITH_BACKEND_METAL */
