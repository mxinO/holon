// GPU side of K's step 0: the SM chunk-list copy (gather H2D from mapped host memory, scatter D2H into it, or
// device to device) and a busy kernel that keeps the stream behind while the host runs ahead.
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels.h"

namespace {

// Work items are (src, dst, 16-byte words), each at most a tile (the host splits larger chunks), and CTA b takes items
// b, b + gridDim.x, ...: its threads move U words each per trip, all loads before all stores, so a CTA keeps
// blockDim * U * 16 bytes in flight. Unlike x0's blob gather (every CTA walks every blob), a list of tens of thousands
// of 34 KB chunks costs no idle trips.
template <int U>
__global__ void k_copy(const k0::Item* __restrict__ items, int64_t n) {
  for (int64_t b = blockIdx.x; b < n; b += gridDim.x) {
    const uint4* __restrict__ s = items[b].src;
    uint4* __restrict__ d = items[b].dst;
    const int64_t w = items[b].words;
    for (int64_t base = threadIdx.x; base < w; base += int64_t(blockDim.x) * U) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int64_t i = base + int64_t(u) * blockDim.x;
        if (i < w) v[u] = s[i];
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int64_t i = base + int64_t(u) * blockDim.x;
        if (i < w) d[i] = v[u];
      }
    }
  }
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void k_busy(uint64_t ns) {
  const uint64_t t0 = globaltimer();
  while (globaltimer() - t0 < ns) {
  }
}

}  // namespace

namespace k0 {

void launch_copy(cudaStream_t s, const Item* items, int64_t n, int ctas, int threads, int unroll) {
  switch (unroll) {
    case 1: k_copy<1><<<ctas, threads, 0, s>>>(items, n); break;
    case 2: k_copy<2><<<ctas, threads, 0, s>>>(items, n); break;
    case 4: k_copy<4><<<ctas, threads, 0, s>>>(items, n); break;
    case 8: k_copy<8><<<ctas, threads, 0, s>>>(items, n); break;
    default: k_copy<4><<<ctas, threads, 0, s>>>(items, n);
  }
}

void launch_busy(cudaStream_t s, uint64_t ns) { k_busy<<<1, 32, 0, s>>>(ns); }

}  // namespace k0
