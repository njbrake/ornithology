/* test_moe.c — MoE router selection and that the batched combine equals a
 * naive per-token loop over the selected experts + shared expert. */
#include "ornith_moe.h"
#include "ornith_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else         { printf("  ok  : %s\n", msg); } \
} while (0)

static int close(float a, float b, float tol){ return fabsf(a-b)<=tol; }

int main(void) {
    printf("== moe: router top-k selection ==\n");
    {
        float logits[6] = {0.1f, 2.0f, -1.0f, 1.5f, 0.5f, 1.9f};
        int idx[3]; float w[3];
        ornith_moe_route(logits, 6, 3, idx, w);
        /* top-3 are experts 1(2.0),5(1.9),3(1.5) */
        CHECK(idx[0]==1 && idx[1]==5 && idx[2]==3, "top-3 indices");
        float sum = w[0]+w[1]+w[2];
        CHECK(close(sum,1.0f,1e-6f), "weights sum to 1");
        CHECK(w[0]>w[1] && w[1]>w[2], "weights descending");
        /* exact softmax over the three selected logits */
        float e0=expf(2.0f-2.0f), e1=expf(1.9f-2.0f), e2=expf(1.5f-2.0f);
        float z=e0+e1+e2;
        CHECK(close(w[0],e0/z,1e-5f), "router softmax exact");
    }

    printf("== moe: forward == naive per-token combine ==\n");
    {
        int H=8, I=6, ne=5, K=2, sI=4;
        ot_rng r = ot_rng_seed(99);
        float *router=malloc((size_t)ne*H*sizeof(float));
        float *eg=malloc((size_t)ne*I*H*sizeof(float));
        float *eu=malloc((size_t)ne*I*H*sizeof(float));
        float *ed=malloc((size_t)ne*H*I*sizeof(float));
        float *sg=malloc((size_t)sI*H*sizeof(float));
        float *su=malloc((size_t)sI*H*sizeof(float));
        float *sd=malloc((size_t)H*sI*sizeof(float));
        float *x=malloc((size_t)H*sizeof(float));
        ot_rng_fill(&r,router,ne*H,-0.5f,0.5f);
        ot_rng_fill(&r,eg,ne*I*H,-0.3f,0.3f);
        ot_rng_fill(&r,eu,ne*I*H,-0.3f,0.3f);
        ot_rng_fill(&r,ed,ne*H*I,-0.3f,0.3f);
        ot_rng_fill(&r,sg,sI*H,-0.3f,0.3f);
        ot_rng_fill(&r,su,sI*H,-0.3f,0.3f);
        ot_rng_fill(&r,sd,H*sI,-0.3f,0.3f);
        ot_rng_fill(&r,x,H,-1,1);

        ornith_moe m={0};
        m.hidden=H; m.inter=I; m.n_experts=ne; m.top_k=K; m.shared_inter=sI;
        m.w_router=router; m.w_gate=eg; m.w_up=eu; m.w_down=ed;
        m.sw_gate=sg; m.sw_up=su; m.sw_down=sd;

        float *out=malloc((size_t)H*sizeof(float));
        float *scr=malloc((size_t)(H+2*I)*sizeof(float));
        ornith_moe_forward(&m, x, out, scr);

        /* independent reference */
        float logits[5]; ot_linear(router,x,logits,ne,H);
        int idx[2]; float w[2]; ornith_moe_route(logits,ne,K,idx,w);
        float ref[8]={0};
        float fsc[64];
        for (int s=0;s<K;s++){
            int e=idx[s];
            float ep[8];
            ornith_ffn_swiglu(eg+(size_t)e*I*H, eu+(size_t)e*I*H, ed+(size_t)e*H*I,
                              x, ep, H, I, fsc);
            for(int i=0;i<H;i++) ref[i]+=w[s]*ep[i];
        }
        float sp[8];
        ornith_ffn_swiglu(sg,su,sd,x,sp,H,sI,fsc);
        for(int i=0;i<H;i++) ref[i]+=sp[i];

        float md=0; for(int i=0;i<H;i++){float d=fabsf(ref[i]-out[i]);if(d>md)md=d;}
        CHECK(md < 1e-5f, "moe_forward == naive combine + shared");

        free(router);free(eg);free(eu);free(ed);free(sg);free(su);free(sd);
        free(x);free(out);free(scr);
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "MOE TESTS FAILED" : "ALL MOE TESTS PASSED",
           failures, failures==1?"":"s");
    return failures ? 1 : 0;
}
