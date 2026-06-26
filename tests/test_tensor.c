/* test_tensor.c — unit tests for the core f32 kernels, each against a
 * hand-computed or independently-derived value. Run via `make test`. */
#include "ornith_tensor.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

static int close(float a, float b, float tol) { return fabsf(a-b) <= tol; }

int main(void) {
    printf("== tensor: linear/matmul ==\n");
    {
        /* y = W x, W = [[1,2,3],[4,5,6]], x=[1,0,-1] -> [-2,-2] */
        float W[6] = {1,2,3, 4,5,6};
        float x[3] = {1,0,-1};
        float y[2];
        ot_linear(W, x, y, 2, 3);
        CHECK(close(y[0], -2, 1e-6f) && close(y[1], -2, 1e-6f), "ot_linear");

        float bias[2] = {10, 20};
        ot_linear_bias(W, bias, x, y, 2, 3);
        CHECK(close(y[0], 8, 1e-6f) && close(y[1], 18, 1e-6f), "ot_linear_bias");

        /* matmul: A[2x3] * B[3x2] */
        float A[6] = {1,2,3, 4,5,6};
        float B[6] = {7,8, 9,10, 11,12};
        float C[4];
        ot_matmul(A, B, C, 2, 3, 2);
        /* C00=1*7+2*9+3*11=58, C01=64, C10=139, C11=154 */
        CHECK(close(C[0],58,1e-4f)&&close(C[1],64,1e-4f)&&
              close(C[2],139,1e-4f)&&close(C[3],154,1e-4f), "ot_matmul");
    }

    printf("== tensor: rmsnorm/softmax ==\n");
    {
        float x[4] = {1,2,3,4};
        float out[4];
        ot_rmsnorm(x, NULL, out, 4, 0.0f);
        /* rms = sqrt((1+4+9+16)/4)=sqrt(7.5)=2.7386 */
        float rms = sqrtf(7.5f);
        CHECK(close(out[0], 1.0f/rms, 1e-5f) && close(out[3], 4.0f/rms, 1e-5f),
              "ot_rmsnorm");

        float s[3] = {1, 2, 3};
        ot_softmax(s, 3);
        float sum = s[0]+s[1]+s[2];
        CHECK(close(sum, 1.0f, 1e-6f), "softmax sums to 1");
        CHECK(s[2] > s[1] && s[1] > s[0], "softmax monotone");
        /* known value: exp ratios */
        float e0=expf(-2), e1=expf(-1), e2=1, z=e0+e1+e2;
        CHECK(close(s[0], e0/z, 1e-5f), "softmax exact");
    }

    printf("== tensor: activations ==\n");
    {
        CHECK(close(ot_sigmoid(0), 0.5f, 1e-6f), "sigmoid(0)=0.5");
        CHECK(close(ot_silu1(0), 0.0f, 1e-6f), "silu(0)=0");
        /* silu(1)=1*sigmoid(1)=0.731059 */
        CHECK(close(ot_silu1(1), 0.7310586f, 1e-5f), "silu(1)");
        float g[2]={1,2}, u[2]={3,4}, o[2];
        ot_swiglu(g,u,o,2);
        CHECK(close(o[0], ot_silu1(1)*3, 1e-5f) &&
              close(o[1], ot_silu1(2)*4, 1e-5f), "swiglu");
    }

    printf("== tensor: rope ==\n");
    {
        /* RoPE preserves the norm of each rotated pair. */
        float v[4] = {0.3f, -0.7f, 1.1f, 0.5f};
        float n0 = sqrtf(ot_dot(v,v,4));
        ot_rope_inplace(v, 4, 5, 10000.0f);
        float n1 = sqrtf(ot_dot(v,v,4));
        CHECK(close(n0, n1, 1e-4f), "rope preserves norm");
        /* pos 0 is identity */
        float w[4] = {1,2,3,4}, w0[4]={1,2,3,4};
        ot_rope_inplace(w, 4, 0, 10000.0f);
        CHECK(close(w[0],w0[0],1e-6f)&&close(w[2],w0[2],1e-6f),
              "rope pos 0 = identity");
    }

    printf("== tensor: softplus ==\n");
    {
        /* softplus(0)=ln2; large x -> ~x; stable (no overflow) at x=100 */
        CHECK(close(ot_softplus(0.0f), 0.6931472f, 1e-5f), "softplus(0)=ln2");
        CHECK(close(ot_softplus(100.0f), 100.0f, 1e-3f), "softplus(100)~100 (stable)");
        CHECK(ot_softplus(-100.0f) >= 0.0f && ot_softplus(-100.0f) < 1e-3f,
              "softplus(-100)~0");
    }

    printf("== tensor: l2norm_eps ==\n");
    {
        float x[4] = {3,4,0,0}, y[4];
        ot_l2norm_eps(x, y, 4, 0.0f);   /* ||x||=5 -> unit */
        CHECK(close(y[0],0.6f,1e-5f) && close(y[1],0.8f,1e-5f),
              "l2norm_eps normalizes [3,4]->[0.6,0.8]");
        CHECK(close(ot_dot(y,y,4), 1.0f, 1e-5f), "l2norm_eps yields unit vector");
        /* eps regularizes a near-zero vector toward 0 rather than exploding */
        float z[2] = {1e-4f, 0.0f}, zo[2];
        ot_l2norm_eps(z, zo, 2, 1e-6f);
        CHECK(zo[0] < 1.0f, "l2norm_eps with eps damps a tiny vector");
    }

    printf("== tensor: rng determinism ==\n");
    {
        ot_rng a = ot_rng_seed(123), b = ot_rng_seed(123);
        float fa[8], fb[8];
        ot_rng_fill(&a, fa, 8, -1, 1);
        ot_rng_fill(&b, fb, 8, -1, 1);
        int same = 1;
        for (int i=0;i<8;i++) if (fa[i]!=fb[i]) same=0;
        CHECK(same, "same seed -> same stream");
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "TENSOR TESTS FAILED" : "ALL TENSOR TESTS PASSED",
           failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
