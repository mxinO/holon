#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_expert_used * row; // Holon: ids at row stride k (a contiguous [T, k] tensor)

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) weights->ne[1]); // Holon: ids at row stride k

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}

// Holon: the MoE router in one launch (docs/v1-offload.md §14, step 3c), for decode and verify steps: logits =
// x · Wᵀ in f32, x [n_rows, hidden] (bf16 or f32) read as its exact f32 values and W [n_experts, hidden] f32, then
// topk_moe_cuda's selection on them (holon_topk_select). Block (e, t) computes expert e for the t-th tile of HOLON_ROUTER_ROWS rows: warp
// q of HOLON_ROUTER_WARPS sums the q-th of equal slices of hidden — lane l the products x[r, h] * W[e, h] over h =
// 4 l + 4 * WARP_SIZE * i + j of the slice, i ascending, j = 0..3, one fma each, then warp_reduce_sum's xor tree —
// and the slices' sums are added in slice order: one order per hidden, the same for a row at any row count. The
// tile's logits go to `logits` [n_rows, n_experts]; the last of the tile's blocks to finish (a counter per tile in
// `done`, zero before the launch and reset by that block) selects the tile's rows, one warp per row, as
// topk_moe_cuda does from those logits.
static __device__ __forceinline__ float4 holon_router_load4(const float * x) { return *(const float4 *) x; }

static __device__ __forceinline__ float4 holon_router_load4(const nv_bfloat16 * x) {
    const uint2 v = *(const uint2 *) x;
    const float2 a = __bfloat1622float2(*(const nv_bfloat162 *) &v.x);
    const float2 b = __bfloat1622float2(*(const nv_bfloat162 *) &v.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

// The selection of topk_moe_cuda (softmax, no bias, no delayed softmax, scale 1), the same arithmetic in the same
// order, for n_expert_used <= WARP_SIZE: its per-lane arrays indexed by unrolled constants (registers) where the
// vendored loop indexes them by the round (local memory). Reads the row's logits from L2 (other blocks wrote them).
template <int n_experts>
static __device__ __forceinline__ void holon_topk_select(const float * logits, float * weights, int32_t * ids,
                                                         const int n_expert_used, const float clamp_val) {
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    const int lane = threadIdx.x;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = i * WARP_SIZE + lane;
        wt[i] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? __ldcg(logits + expert) : -INFINITY;
    }
    softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }
    float wt_sum = 0.f, output_weight = 0.f;
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if (max_expert == lane + i * WARP_SIZE) {
                wt[i] = -INFINITY;
            }
        }
        if (k == lane) {
            output_weight = max_val;
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            if (clamp_val != -INFINITY) {
                wt_sum += max_val;
            }
        }
    }
    if (clamp_val != -INFINITY) {                     // with_norm
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;
        output_weight *= inv_sum;
    }
    if (lane < n_expert_used) {
        weights[lane] = output_weight;                // times the scale, 1
    }
}

template <int n_experts, typename src_t>
__launch_bounds__(HOLON_ROUTER_WARPS * WARP_SIZE)
__global__ void holon_moe_router(const src_t *         x,
                                 const float *         w,
                                 float *               logits,
                                 unsigned int *        done,
                                 float *               weights,
                                 int32_t *             ids,
                                 const int             n_rows,
                                 const int             hidden,
                                 const int             n_expert_used,
                                 const float           clamp_val) {
    constexpr int R = HOLON_ROUTER_ROWS;
    const int lane = threadIdx.x, q = threadIdx.y, expert = blockIdx.x;
    const int row0 = blockIdx.y * R, rows = min(R, n_rows - row0), slice = hidden / HOLON_ROUTER_WARPS;
    const float * wq = w + (int64_t) expert * hidden + q * slice;
    const src_t * xq = x + (int64_t) row0 * hidden + q * slice;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }
    ggml_cuda_pdl_sync();
#pragma unroll 4
    for (int h = 4 * lane; h < slice; h += 4 * WARP_SIZE) {
        const float4 wv = __ldg((const float4 *) (wq + h));      // W is never written: read-only
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (r < rows) {
                const float4 xv = holon_router_load4(xq + (int64_t) r * hidden + h);
                acc[r] = fmaf(xv.x, wv.x, acc[r]);
                acc[r] = fmaf(xv.y, wv.y, acc[r]);
                acc[r] = fmaf(xv.z, wv.z, acc[r]);
                acc[r] = fmaf(xv.w, wv.w, acc[r]);
            }
        }
    }
    __shared__ float part[HOLON_ROUTER_WARPS][R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r] = warp_reduce_sum(acc[r]);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            part[q][r] = acc[r];
        }
    }
    __syncthreads();
    if (q == 0 && lane < rows) {                                  // lane r adds row r's slices in order
        float sum = part[0][lane];
#pragma unroll
        for (int i = 1; i < HOLON_ROUTER_WARPS; i++) {
            sum += part[i][lane];
        }
        logits[(int64_t) (row0 + lane) * n_experts + expert] = sum;
    }

    // the tile's last block selects: every block's logits are visible to it once it has seen their counts
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (lane == 0 && q == 0) {
        last = atomicAdd(done + blockIdx.y, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    for (int r = q; r < rows; r += HOLON_ROUTER_WARPS) {
        holon_topk_select<n_experts>(logits + (int64_t) (row0 + r) * n_experts,
                                     weights + (int64_t) (row0 + r) * n_expert_used,
                                     ids + (int64_t) (row0 + r) * n_expert_used, n_expert_used, clamp_val);
    }
    if (lane == 0 && q == 0) {
        done[blockIdx.y] = 0;
    }
}

template <typename src_t>
static void holon_moe_router_cuda(const src_t * x, const float * w, const int n_rows, const int hidden,
                                  const int n_experts, const int n_expert_used, const float clamp_val, float * logits,
                                  unsigned int * done, float * weights, int32_t * ids, cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params params(dim3(n_experts, (n_rows + HOLON_ROUTER_ROWS - 1) / HOLON_ROUTER_ROWS, 1),
                                                dim3(WARP_SIZE, HOLON_ROUTER_WARPS, 1), 0, stream);
    switch (n_experts) {
#define HOLON_ROUTER_CASE(E) case E: ggml_cuda_kernel_launch(holon_moe_router<E, src_t>, params, x, w, logits, done, \
                                     weights, ids, n_rows, hidden, n_expert_used, clamp_val); break;
        HOLON_ROUTER_CASE(1) HOLON_ROUTER_CASE(2) HOLON_ROUTER_CASE(4) HOLON_ROUTER_CASE(8) HOLON_ROUTER_CASE(16)
        HOLON_ROUTER_CASE(32) HOLON_ROUTER_CASE(64) HOLON_ROUTER_CASE(128) HOLON_ROUTER_CASE(256)
        HOLON_ROUTER_CASE(512)
#undef HOLON_ROUTER_CASE
        default: GGML_ABORT("holon_moe_router: %d experts (a power of 2 up to 512)", n_experts);
    }
}

void ggml_cuda_holon_moe_router(const void * x, const bool x_bf16, const float * w, const int n_rows,
                                const int hidden, const int n_experts, const int n_expert_used,
                                const float clamp_val, float * logits, unsigned int * done, float * weights,
                                int32_t * ids, cudaStream_t stream) {
    GGML_ASSERT(hidden % (4 * WARP_SIZE * HOLON_ROUTER_WARPS) == 0 && n_expert_used <= n_experts &&
                n_expert_used <= WARP_SIZE && n_rows <= HOLON_ROUTER_MAX_ROWS);
    if (x_bf16) {
        holon_moe_router_cuda((const nv_bfloat16 *) x, w, n_rows, hidden, n_experts, n_expert_used, clamp_val,
                              logits, done, weights, ids, stream);
    } else {
        holon_moe_router_cuda((const float *) x, w, n_rows, hidden, n_experts, n_expert_used, clamp_val, logits,
                              done, weights, ids, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
