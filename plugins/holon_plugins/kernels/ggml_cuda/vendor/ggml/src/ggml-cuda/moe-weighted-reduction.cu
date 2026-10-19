// Holon (docs/v1-moe-prefill.md D8): every semantic hunk carries a "Holon:" marker; no signature plumbing beyond
// the marked hunks.

#include "moe-weighted-reduction.cuh"

// Holon: a + b in f32 without flushing subnormals (fast math would), as torch's bf16 residual add computes it
static __device__ __forceinline__ float holon_add_rn(float a, const float b) {
    asm("add.rn.f32 %0, %0, %1;" : "+f"(a) : "f"(b));
    return a;
}

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  nv_bfloat16 * __restrict__ dst, // Holon: bf16 out
                                                  const nv_bfloat16 * __restrict__ residual, // Holon: or NULL
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    // Holon: plus the residual stream, as `residual + bf16(sum)`: the sum rounded first, then the add
    dst[token * n_embd + col] = residual == nullptr ? __float2bfloat16(sum) :
        __float2bfloat16(holon_add_rn(__bfloat162float(residual[token * n_embd + col]),
                                      __bfloat162float(__float2bfloat16(sum))));
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          nv_bfloat16 * dst,
                                          const nv_bfloat16 * residual,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, residual, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         residual) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_BF16); // Holon: the caller's bf16 activations
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(residual == nullptr || (residual->type == GGML_TYPE_BF16 && ggml_is_contiguous(residual) &&
                                        ggml_are_same_shape(residual, dst))); // Holon: dst = residual + the sum

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (nv_bfloat16 *) dst->data, // Holon: bf16 out, the residual add
                                  residual ? (const nv_bfloat16 *) residual->data : nullptr,
                                  n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
