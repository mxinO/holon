// plug-serve S2: our Torch wrapper over the vendored ggml-derived GGUF kernels (third_party/sglang_gguf).
// Every op writes into caller-allocated tensors (outputs and the q8_1 activation workspace), never allocates,
// and launches on the current stream, so it can be captured in a CUDA graph and fed from Ledger buffers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "compat.h"
// clang-format off
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
// clang-format on

// ggml type id -> kernel name suffix. Vector kernels exist for every type below; batched mmq / MoE only for the
// legacy and k-quant types (the vendored set predates llama.cpp's i-quant mmq).
#define MMQ_TYPES(X)                                                                                    \
  X(2, q4_0, Q4_0) X(3, q4_1, Q4_1) X(6, q5_0, Q5_0) X(7, q5_1, Q5_1) X(8, q8_0, Q8_0) X(10, q2_K, Q2_K) \
  X(11, q3_K, Q3_K) X(12, q4_K, Q4_K) X(13, q5_K, Q5_K) X(14, q6_K, Q6_K)
#define VEC_TYPES(X)                                                                                        \
  MMQ_TYPES(X) X(16, iq2_xxs, IQ2_XXS) X(17, iq2_xs, IQ2_XS) X(18, iq3_xxs, IQ3_XXS) X(19, iq1_s, IQ1_S)     \
  X(20, iq4_nl, IQ4_NL) X(21, iq3_s, IQ3_S) X(22, iq2_s, IQ2_S) X(23, iq4_xs, IQ4_XS) X(29, iq1_m, IQ1_M)

#define DISPATCH_ACT_TYPES(TYPE, NAME, ...)                     \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                \
                     AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
                     AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

#define DISPATCH_OUT_TYPES(TYPE, NAME, ...)                      \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                 \
                     AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
                     AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
                     AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

static int64_t padded_cols(int64_t k) { return (k + 511) / 512 * 512; }

static void check_q8_1_workspace(const torch::Tensor& x, const torch::Tensor& xq) {
  TORCH_CHECK(xq.scalar_type() == torch::kInt32 && xq.is_contiguous(), "xq must be contiguous int32");
  TORCH_CHECK(xq.numel() >= x.size(0) * (padded_cols(x.size(1)) / 32 * 9), "xq workspace too small");
}

static void check_io(const torch::Tensor& w, const torch::Tensor& x, const torch::Tensor& y) {
  TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda(), "tensors must be on CUDA");
  TORCH_CHECK(w.scalar_type() == torch::kUInt8 && w.is_contiguous(), "w must be contiguous uint8 ggml blocks");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && y.is_contiguous(), "x, y must be contiguous");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y dtype differ");
}

template <typename scalar_t>
static __global__ void quantize_q8_1_kernel(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx,
                                            const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) return;
  const auto iy = blockIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1* y = (block_q8_1*)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;
  const float xi = ix < kx ? static_cast<float>(x[(int64_t)iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
    sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
  }
  const float d = amax / 127;
  y[ib].qs[iqs] = amax == 0.0f ? 0 : roundf(xi / d);
  if (iqs == 0) {
    y[ib].ds.x = __float2half(d);
    y[ib].ds.y = __float2half(sum);
  }
}

template <typename scalar_t>
static void quantize_q8_1_launch(const scalar_t* x, void* vy, int kx, int ky, cudaStream_t stream) {
  const int kx_padded = padded_cols(kx);
  const int nbx = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  for (int off = 0; off < ky; off += 65535) {
    const dim3 grid(nbx, std::min(ky - off, 65535), 1);
    quantize_q8_1_kernel<<<grid, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(
        x + (int64_t)off * kx, (int32_t*)vy + (int64_t)off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

void quantize_q8_1(torch::Tensor x, torch::Tensor xq) {
  check_q8_1_workspace(x, xq);
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ACT_TYPES(x.scalar_type(), "quantize_q8_1", [&] {
    quantize_q8_1_launch<scalar_t>(x.data_ptr<scalar_t>(), xq.data_ptr(), x.size(1), x.size(0), stream);
  });
}

// w: ggml blocks for an [m, n] matrix; out: preallocated [m, n] half/bf16/float.
void dequantize(torch::Tensor w, int64_t type, torch::Tensor out) {
  TORCH_CHECK(w.scalar_type() == torch::kUInt8 && w.is_contiguous() && out.is_contiguous());
  const at::cuda::OptionalCUDAGuard guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_OUT_TYPES(out.scalar_type(), "dequantize", [&] {
    auto fn = ggml_get_to_cuda<scalar_t>(type);
    TORCH_CHECK(fn != nullptr, "dequantize: unsupported ggml type ", type);
    fn(w.data_ptr(), out.data_ptr<scalar_t>(), out.numel(), stream);
  });
}

// Type-dispatched launchers (file scope: the switch bodies are generated from the type lists above).
template <typename scalar_t>
static bool launch_vec(int64_t type, const void* w, const void* xq, scalar_t* y, int cols, int rows, int vecs,
                       cudaStream_t s) {
  switch (type) {
#define CASE(id, name, NAME) \
  case id: mul_mat_vec_##name##_q8_1_cuda<scalar_t>(w, xq, y, cols, rows, vecs, s); return true;
    VEC_TYPES(CASE)
#undef CASE
  }
  return false;
}

template <typename scalar_t>
static bool launch_mmq(int64_t type, const void* w, const void* xq, scalar_t* y, int cols, int rows, int batch,
                       cudaStream_t s) {
  const int padded = padded_cols(cols);
  switch (type) {
#define CASE(id, name, NAME) \
  case id: ggml_mul_mat_##name##_q8_1_cuda(w, xq, y, cols, rows, batch, padded, rows, s); return true;
    MMQ_TYPES(CASE)
#undef CASE
  }
  return false;
}

template <typename scalar_t>
static bool launch_moe(int64_t type, const void* xq, const void* w, scalar_t* y, const int* sorted, const int* eids,
                       const int* npad, int exp_stride, int cols, int rows, int tokens, int top_k, int sorted_len,
                       cudaStream_t s) {
  const int padded = padded_cols(cols);
  switch (type) {
#define CASE(id, name, NAME)                                                                                   \
  case id:                                                                                                     \
    ggml_moe_##name##_q8_1_cuda(xq, w, y, sorted, eids, npad, exp_stride, cols, rows, tokens, padded, rows, top_k, \
                                sorted_len, s);                                                                \
    return true;
    MMQ_TYPES(CASE)
#undef CASE
  }
  return false;
}

template <typename scalar_t>
static bool launch_moe_vec(int64_t type, const void* w, const void* xq, scalar_t* y, const int* topk_ids, int top_k,
                           int tokens, int cols, int rows, cudaStream_t s) {
  const int token_stride = padded_cols(cols) / 32 * 9;
  switch (type) {
#define CASE(id, name, NAME)                                                                                   \
  case id:                                                                                                     \
    moe_vec_##name##_q8_1_cuda<scalar_t>(w, xq, y, topk_ids, top_k, tokens, cols, rows, token_stride, s);     \
    return true;
    VEC_TYPES(CASE)
#undef CASE
  }
  return false;
}

// y[v, :] = W @ x[v, :] for each of the x.size(0) vectors; W is [rows, x.size(1)].
void mul_mat_vec(torch::Tensor w, torch::Tensor x, int64_t type, torch::Tensor y, torch::Tensor xq) {
  check_io(w, x, y);
  check_q8_1_workspace(x, xq);
  const int cols = x.size(1), vecs = x.size(0), rows = y.size(1);
  TORCH_CHECK(y.size(0) == vecs, "y rows != x rows");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ACT_TYPES(x.scalar_type(), "mul_mat_vec", [&] {
    quantize_q8_1_launch<scalar_t>(x.data_ptr<scalar_t>(), xq.data_ptr(), cols, vecs, stream);
    TORCH_CHECK(launch_vec<scalar_t>(type, w.data_ptr(), xq.data_ptr(), y.data_ptr<scalar_t>(), cols, rows, vecs,
                                     stream), "mul_mat_vec: unsupported ggml type ", type);
  });
}

// Batched (tiled) y = x @ W^T, legacy and k-quant types only.
void mul_mat(torch::Tensor w, torch::Tensor x, int64_t type, torch::Tensor y, torch::Tensor xq) {
  check_io(w, x, y);
  check_q8_1_workspace(x, xq);
  const int cols = x.size(1), batch = x.size(0), rows = y.size(1);
  TORCH_CHECK(y.size(0) == batch, "y rows != x rows");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ACT_TYPES(x.scalar_type(), "mul_mat", [&] {
    quantize_q8_1_launch<scalar_t>(x.data_ptr<scalar_t>(), xq.data_ptr(), cols, batch, stream);
    TORCH_CHECK(launch_mmq<scalar_t>(type, w.data_ptr(), xq.data_ptr(), y.data_ptr<scalar_t>(), cols, rows, batch,
                                     stream), "mul_mat: no batched kernel for ggml type ", type);
  });
}

// Batched MoE over moe_align_block_size-style routing (sorted_token_ids padded to moe_block_size(type), padding
// entries = tokens * top_k). w: [experts, rows, cols] ggml blocks; y: [tokens * top_k, rows].
void moe(torch::Tensor x, torch::Tensor w, torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
         torch::Tensor num_tokens_post_padded, int64_t type, int64_t top_k, torch::Tensor y, torch::Tensor xq) {
  check_io(w, x, y);
  check_q8_1_workspace(x, xq);
  const int cols = x.size(1), tokens = x.size(0), rows = y.size(1);
  TORCH_CHECK(y.size(0) == tokens * top_k, "y must be [tokens*top_k, rows]");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ACT_TYPES(x.scalar_type(), "moe", [&] {
    quantize_q8_1_launch<scalar_t>(x.data_ptr<scalar_t>(), xq.data_ptr(), cols, tokens, stream);
    TORCH_CHECK(launch_moe<scalar_t>(type, xq.data_ptr(), w.data_ptr(), y.data_ptr<scalar_t>(),
                                     sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
                                     num_tokens_post_padded.data_ptr<int>(), w.stride(0), cols, rows, tokens, top_k,
                                     sorted_token_ids.size(0), stream),
                "moe: no batched MoE kernel for ggml type ", type);
  });
}

// Per-(token, k) vector MoE: y[t*top_k + k] = W[topk_ids[t, k]] @ x[t].
void moe_vec(torch::Tensor x, torch::Tensor w, torch::Tensor topk_ids, int64_t type, torch::Tensor y,
             torch::Tensor xq) {
  check_io(w, x, y);
  check_q8_1_workspace(x, xq);
  const int cols = x.size(1), tokens = x.size(0), rows = y.size(1), top_k = topk_ids.size(1);
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.is_contiguous(), "topk_ids must be int32");
  TORCH_CHECK(y.size(0) == tokens * top_k, "y must be [tokens*top_k, rows]");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ACT_TYPES(x.scalar_type(), "moe_vec", [&] {
    quantize_q8_1_launch<scalar_t>(x.data_ptr<scalar_t>(), xq.data_ptr(), cols, tokens, stream);
    TORCH_CHECK(launch_moe_vec<scalar_t>(type, w.data_ptr(), xq.data_ptr(), y.data_ptr<scalar_t>(),
                                         topk_ids.data_ptr<int>(), top_k, tokens, cols, rows, stream),
                "moe_vec: unsupported ggml type ", type);
  });
}

int64_t moe_block_size(int64_t type) {
  switch (type) {
#define CASE(id, name, NAME) \
  case id: return MOE_X_##NAME;
    MMQ_TYPES(CASE)
#undef CASE
  }
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_q8_1", &quantize_q8_1);
  m.def("dequantize", &dequantize);
  m.def("mul_mat_vec", &mul_mat_vec);
  m.def("mul_mat", &mul_mat);
  m.def("moe", &moe);
  m.def("moe_vec", &moe_vec);
  m.def("moe_block_size", &moe_block_size);
}
