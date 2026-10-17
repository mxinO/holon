#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);

// Holon: the MoE router in one launch (topk-moe.cu), for up to HOLON_ROUTER_MAX_ROWS rows: weights and ids of the top
// `n_expert_used` of the softmax over x · Wᵀ in f32 (x [n_rows, hidden] bf16 or f32, W [n_experts, hidden] f32;
// hidden a multiple of 4 · WARP_SIZE · HOLON_ROUTER_WARPS), divided by their sum clamped at `clamp_val` (-INFINITY: not
// renormalised); `logits` [n_rows, n_experts] f32 its scratch, `done` one zeroed counter per HOLON_ROUTER_ROWS rows,
// left zeroed.
#define HOLON_ROUTER_WARPS 4
#define HOLON_ROUTER_ROWS 4
#define HOLON_ROUTER_MAX_ROWS 16

void ggml_cuda_holon_moe_router(const void * x, bool x_bf16, const float * w, int n_rows, int hidden, int n_experts,
                                int n_expert_used, float clamp_val, float * logits, unsigned int * done,
                                float * weights, int32_t * ids, cudaStream_t stream);
