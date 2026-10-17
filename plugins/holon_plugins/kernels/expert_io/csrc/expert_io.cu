// The offload store's in-graph kernels (std.moe.kernels: `moe_resolve@1`, `expert_fetch@1`). Every call runs on the
// tensors' device and torch's current stream, reads its sizes from the device only, and is CUDA-graph capturable.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int PADDING = 0, DECODE = 1;                 // row kinds (std.moe.kernels)
constexpr int RANGE = 1;                               // error bit: a resident slot outside the arena's cache slots
constexpr int RESOLVE_THREADS = 256;                   // 8 warps: one per role for the scans
constexpr int FETCH_CTAS = 8, FETCH_THREADS = 256, FETCH_UNROLL = 8;   // step 0's M1: saturates PCIe

// One CTA per call. need[r][e]: role r of expert e is routed and missing; a warp per role ranks the missing experts
// by id (ballot scans), so staging slots and fetch entries are deterministic; then every entry gets its slot.
__global__ void k_resolve(const int32_t * __restrict__ ids, const uint8_t * __restrict__ kinds,
                          const int32_t * __restrict__ residency, const int32_t * __restrict__ host,
                          const int32_t * __restrict__ limits, int32_t * __restrict__ counts,
                          int32_t * __restrict__ misses, int32_t * __restrict__ errors, int32_t * __restrict__ slots,
                          int32_t * __restrict__ fetch, int32_t * __restrict__ n_fetch, int tokens, int k, int experts,
                          int roles, int staging) {
    extern __shared__ int32_t shared[];
    int32_t * need = shared;                           // [roles, experts]: 0/1, then the miss's rank
    const int entries = tokens * k;
    for (int i = threadIdx.x; i < roles * experts; i += blockDim.x) {
        need[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < entries; i += blockDim.x) {
        const int e = ids[i], kind = kinds[i / k];
        if (kind == PADDING || e < 0) {
            continue;
        }
        if (kind == DECODE) {
            atomicAdd(&counts[e], 1);
        }
        for (int r = 0; r < roles; ++r) {
            if (residency[e * roles + r] < 0) {
                need[r * experts + e] = 1;
            }
        }
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp < roles) {
        int32_t * row = need + warp * experts;
        int seen = 0;
        for (int base = 0; base < experts; base += 32) {
            const int e = base + lane;
            const bool miss = e < experts && row[e];
            const unsigned mask = __ballot_sync(0xffffffffu, miss);
            const int rank = seen + __popc(mask & ((1u << lane) - 1));
            if (miss) {
                row[e] = rank;
                if (rank < staging) {
                    fetch[warp * staging + rank] = host[e * roles + warp];
                }
            }
            seen += __popc(mask);
        }
        if (lane == 0) {
            n_fetch[warp] = seen;
            atomicMax(misses, seen);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < entries; i += blockDim.x) {
        const int e = ids[i], j = i % k;
        const bool live = kinds[i / k] != PADDING && e >= 0;
        for (int r = 0; r < roles; ++r) {
            int32_t at = live ? residency[e * roles + r] : j;
            if (live && at < 0) {
                at = k + min(need[r * experts + e], staging - 1);
            } else if (live && (at < k + staging || at >= limits[r])) {     // corrupt: never read past the arena
                at = j;
                atomicOr(errors, RANGE);
            }
            slots[r * entries + i] = at;
        }
    }
}

// Every CTA walks every fetched unit; its threads move FETCH_UNROLL 16-byte words each per trip, all loads before all
// stores, so a CTA keeps blockDim * FETCH_UNROLL * 16 bytes of reads in flight (zero-copy reads of host memory).
__global__ void k_fetch(const int32_t * __restrict__ fetch, const int32_t * __restrict__ n, const uint4 * __restrict__ src,
                        uint4 * __restrict__ dst, int64_t words, int first, int staging) {
    const int count = min(*n, staging);
    const int64_t stride = int64_t(gridDim.x) * blockDim.x * FETCH_UNROLL;
    for (int b = 0; b < count; ++b) {
        const uint4 * __restrict__ s = src + fetch[b] * words;
        uint4 * __restrict__ d = dst + (first + b) * words;
        for (int64_t base = int64_t(blockIdx.x) * blockDim.x * FETCH_UNROLL + threadIdx.x; base < words;
             base += stride) {
            uint4 v[FETCH_UNROLL];
#pragma unroll
            for (int u = 0; u < FETCH_UNROLL; ++u) {
                const int64_t i = base + int64_t(u) * blockDim.x;
                if (i < words) {
                    v[u] = s[i];
                }
            }
#pragma unroll
            for (int u = 0; u < FETCH_UNROLL; ++u) {
                const int64_t i = base + int64_t(u) * blockDim.x;
                if (i < words) {
                    d[i] = v[u];
                }
            }
        }
    }
}

void check(const at::Tensor & t, const at::Tensor & like, at::ScalarType dtype, at::IntArrayRef shape, const char * name) {
    TORCH_CHECK(t.scalar_type() == dtype && t.device() == like.device() && t.is_contiguous() && t.sizes() == shape,
                name, ": contiguous ", shape, " of ", dtype, " on ", like.device(), ", got ", t.sizes(), " of ",
                t.scalar_type(), " on ", t.device());
}

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void resolve(const at::Tensor & ids, const at::Tensor & kinds, const at::Tensor & residency, const at::Tensor & host,
             const at::Tensor & limits, at::Tensor & counts, at::Tensor & misses, at::Tensor & errors,
             at::Tensor & slots, at::Tensor & fetch, at::Tensor & n_fetch) {
    const c10::cuda::CUDAGuard guard(ids.device());
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && residency.dim() == 2 && fetch.dim() == 2, "ids [T, k] on CUDA");
    const int64_t tokens = ids.size(0), k = ids.size(1), experts = residency.size(0), roles = residency.size(1);
    const int64_t staging = fetch.size(1);
    TORCH_CHECK(roles <= RESOLVE_THREADS / 32 && staging > 0, "at most ", RESOLVE_THREADS / 32, " roles, staging > 0");
    check(ids, ids, at::kInt, {tokens, k}, "ids");
    check(kinds, ids, at::kByte, {tokens}, "kinds");
    check(residency, ids, at::kInt, {experts, roles}, "residency");
    check(host, ids, at::kInt, {experts, roles}, "host");
    check(limits, ids, at::kInt, {roles}, "limits");
    check(counts, ids, at::kInt, {experts}, "counts");
    check(misses, ids, at::kInt, {1}, "misses");
    check(errors, ids, at::kInt, {1}, "errors");
    check(slots, ids, at::kInt, {roles, tokens, k}, "slots");
    check(fetch, ids, at::kInt, {roles, staging}, "fetch");
    check(n_fetch, ids, at::kInt, {roles}, "n_fetch");
    const size_t smem = roles * experts * sizeof(int32_t);
    k_resolve<<<1, RESOLVE_THREADS, smem, stream()>>>(
        ids.data_ptr<int32_t>(), kinds.data_ptr<uint8_t>(), residency.data_ptr<int32_t>(), host.data_ptr<int32_t>(),
        limits.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), misses.data_ptr<int32_t>(), errors.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(), fetch.data_ptr<int32_t>(), n_fetch.data_ptr<int32_t>(), tokens, k, experts, roles,
        staging);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather(const at::Tensor & fetch, const at::Tensor & n, const at::Tensor & src, at::Tensor & dst, int64_t first) {
    const c10::cuda::CUDAGuard guard(dst.device());
    TORCH_CHECK(dst.is_cuda() && dst.dim() == 2 && src.dim() == 2 && dst.size(1) == src.size(1) && dst.size(1) % 16 == 0,
                "src [U, unit], dst [S, unit] with unit a multiple of 16 bytes");
    check(fetch, dst, at::kInt, {fetch.size(0)}, "fetch");
    check(n, dst, at::kInt, {1}, "n");
    check(src, dst, at::kByte, {src.size(0), src.size(1)}, "src");
    check(dst, dst, at::kByte, {dst.size(0), dst.size(1)}, "dst");
    TORCH_CHECK(first >= 0 && first + fetch.size(0) <= dst.size(0), "dst holds no slots ", first, " + ", fetch.size(0));
    TORCH_CHECK(reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0,
                "src and dst 16-byte aligned");
    k_fetch<<<FETCH_CTAS, FETCH_THREADS, 0, stream()>>>(
        fetch.data_ptr<int32_t>(), n.data_ptr<int32_t>(), reinterpret_cast<const uint4 *>(src.data_ptr()),
        reinterpret_cast<uint4 *>(dst.data_ptr()), dst.size(1) / 16, int(first), int(fetch.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resolve", &resolve, "moe_resolve@1 into slots, fetch, n_fetch (and counts, misses, errors in place)");
    m.def("fetch", &gather, "expert_fetch@1: dst[first + i] = src[fetch[i]] for i < min(n, len(fetch))");
}
