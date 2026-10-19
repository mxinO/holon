// std.state_tier's `state_copy@1` (std.state_tier.kernels): row copies between device pools and mapped host pools from
// host row lists, on torch's current stream (the tier's copy stream). Not graph safe: the lists change per call.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

// state_copy@1 (std.state_tier.kernels): dst row dst_rows[i] = src row src_rows[i], rows of C bytes, on the current stream.
// Two paths. "sm": zero-copy SM copies (K step 0's K-M2: 8 CTAs x 256 threads of 16-byte loads, 2 deep, reach 0.97-0.99x of
// PCIe on fragmented 34,816 B rows and take no copy engine, so a load never queues an engine step's own copies behind
// it); the lists go to the device in one copy through torch's pinned (caching) host allocator. "dma": one
// cudaMemcpyBatchAsync of the lists, runs of rows consecutive in both views merged (copy engines; K-M2: 0.89-1.03x).
constexpr int ROW_CTAS = 8, ROW_THREADS = 256, ROW_UNROLL = 2;
constexpr int64_t ROW_TILE = 64 << 10;                 // rows split into tiles of at most 64 KiB, so a few big rows
                                                       // (a checkpoint's 3 MiB segments) still spread over the CTAs

template <int UNROLL>
__global__ void k_rows(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const int64_t * __restrict__ rows,
                       int64_t n, int64_t chunk, int64_t tiles) {
    const int64_t tile_words = ROW_TILE / 16, words = chunk / 16;
    for (int64_t w = blockIdx.x; w < n * tiles; w += gridDim.x) {
        const int64_t i = w / tiles, first = (w % tiles) * tile_words, count = min(tile_words, words - first);
        const uint4 * __restrict__ s = reinterpret_cast<const uint4 *>(src + rows[i] * chunk) + first;
        uint4 * __restrict__ d = reinterpret_cast<uint4 *>(dst + rows[n + i] * chunk) + first;
        for (int64_t base = threadIdx.x; base < count; base += int64_t(blockDim.x) * UNROLL) {
            uint4 v[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int64_t j = base + int64_t(u) * blockDim.x;
                if (j < count) {
                    v[u] = s[j];
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int64_t j = base + int64_t(u) * blockDim.x;
                if (j < count) {
                    d[j] = v[u];
                }
            }
        }
    }
}

int64_t check_rows(const at::Tensor & src, const at::Tensor & src_rows, const at::Tensor & dst, const at::Tensor & dst_rows) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device() && src.dim() == 2 && dst.dim() == 2 &&
                src.scalar_type() == at::kByte && dst.scalar_type() == at::kByte && src.is_contiguous() &&
                dst.is_contiguous() && src.size(1) == dst.size(1) && src.size(1) % 16 == 0,
                "state_copy@1: src [Rs, C] and dst [Rd, C] contiguous uint8 views on one CUDA device, C a multiple of 16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0,
                "state_copy@1: src and dst 16-byte aligned");
    TORCH_CHECK(!src_rows.is_cuda() && !dst_rows.is_cuda() && src_rows.scalar_type() == at::kLong &&
                dst_rows.scalar_type() == at::kLong && src_rows.dim() == 1 && dst_rows.dim() == 1 &&
                src_rows.numel() == dst_rows.numel(), "state_copy@1: src_rows and dst_rows [n] int64 on the host");
    return src_rows.numel();
}

// The rows of one side as host values, bounds checked.
const int64_t * host_rows(const at::Tensor & rows, int64_t limit, const char * name) {
    TORCH_CHECK(rows.is_contiguous(), "state_copy@1: ", name, " contiguous");
    const int64_t * r = rows.data_ptr<int64_t>();
    for (int64_t i = 0; i < rows.numel(); ++i) {
        TORCH_CHECK(r[i] >= 0 && r[i] < limit, "state_copy@1: ", name, "[", i, "] = ", r[i], " outside [0, ", limit, ")");
    }
    return r;
}

void state_copy_sm(const at::Tensor & src, const at::Tensor & src_rows, at::Tensor & dst, const at::Tensor & dst_rows) {
    const int64_t n = check_rows(src, src_rows, dst, dst_rows);
    if (n == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(dst.device());
    auto lists = at::empty({2, n}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    int64_t * l = lists.data_ptr<int64_t>();
    const int64_t * s = host_rows(src_rows.contiguous(), src.size(0), "src_rows");
    const int64_t * d = host_rows(dst_rows.contiguous(), dst.size(0), "dst_rows");
    std::copy(s, s + n, l);
    std::copy(d, d + n, l + n);
    // the caching host allocator keeps the block until this copy is done on the current stream
    const auto rows = lists.to(at::TensorOptions().device(dst.device()), /*non_blocking=*/true);
    const int64_t chunk = src.size(1), tiles = (chunk + ROW_TILE - 1) / ROW_TILE;
    k_rows<ROW_UNROLL><<<ROW_CTAS, ROW_THREADS, 0, stream()>>>(static_cast<const uint8_t *>(src.data_ptr()),
                                                               static_cast<uint8_t *>(dst.data_ptr()),
                                                               rows.data_ptr<int64_t>(), n, chunk, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void state_copy_dma(const at::Tensor & src, const at::Tensor & src_rows, at::Tensor & dst, const at::Tensor & dst_rows) {
    const int64_t n = check_rows(src, src_rows, dst, dst_rows);
    if (n == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(dst.device());
    const auto sr = src_rows.contiguous(), dr = dst_rows.contiguous();
    const int64_t * s = host_rows(sr, src.size(0), "src_rows");
    const int64_t * d = host_rows(dr, dst.size(0), "dst_rows");
    const int64_t chunk = src.size(1);
    const auto * from = static_cast<const uint8_t *>(src.data_ptr());
    auto * to = static_cast<uint8_t *>(dst.data_ptr());
    std::vector<void *> dsts;
    std::vector<const void *> srcs;
    std::vector<size_t> sizes;
    for (int64_t i = 0; i < n;) {
        int64_t run = 1;
        while (i + run < n && s[i + run] == s[i] + run && d[i + run] == d[i] + run) {
            ++run;
        }
        srcs.push_back(from + s[i] * chunk);
        dsts.push_back(to + d[i] * chunk);
        sizes.push_back(size_t(run * chunk));
        i += run;
    }
    const cudaStream_t st = stream();
#if CUDART_VERSION >= 12080
    if (st != nullptr && st != cudaStreamLegacy && st != cudaStreamPerThread) {   // the batch refuses the default streams
        cudaMemcpyAttributes attrs = {};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        size_t index = 0;
#if CUDART_VERSION >= 13000
        C10_CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attrs, &index, 1, st));
#else
        size_t failed = 0;
        C10_CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), const_cast<void **>(srcs.data()), sizes.data(), dsts.size(),
                                            &attrs, &index, 1, &failed, st));
#endif
        return;
    }
#endif
    for (size_t i = 0; i < dsts.size(); ++i) {
        C10_CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, st));
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("state_copy_sm", &state_copy_sm, "state_copy@1 by SM copies: dst[dst_rows] = src[src_rows] (host lists)");
    m.def("state_copy_dma", &state_copy_dma, "state_copy@1 by one cudaMemcpyBatchAsync: dst[dst_rows] = src[src_rows]");
}
