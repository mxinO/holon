// GPU side of step 0: the zero-copy SM gather (M1, M2), S6's flag pair and busy kernel, and M5's per-layer
// producer / consumer kernels around the CPU hand-off.
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels.h"

namespace {

using SysRef = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

// Every CTA walks every blob; its threads move U 16-byte words each per trip, all loads before all stores, so a
// CTA keeps blockDim * U * 16 bytes of PCIe reads in flight.
template <int U>
__global__ void k_gather(const x0::Blob* __restrict__ blobs, int n) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x * U;
  for (int b = 0; b < n; ++b) {
    const uint4* __restrict__ s = blobs[b].src;
    uint4* __restrict__ d = blobs[b].dst;
    const uint64_t w = blobs[b].words;
    for (uint64_t base = uint64_t(blockIdx.x) * blockDim.x * U + threadIdx.x; base < w; base += stride) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint64_t i = base + uint64_t(u) * blockDim.x;
        if (i < w) v[u] = s[i];
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint64_t i = base + uint64_t(u) * blockDim.x;
        if (i < w) d[i] = v[u];
      }
    }
  }
}

__global__ void k_signal(uint32_t* flag, uint32_t value) {
  SysRef(*flag).store(value, cuda::memory_order_release);
}

__global__ void k_wait(uint32_t* flag, uint32_t value) {
  SysRef f(*flag);
  while (f.load(cuda::memory_order_acquire) < value) __nanosleep(64);
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

// Device-planned layer: the job list (k distinct experts, all `rows` tokens each) and the activations, written
// straight into mapped memory.
__global__ void k_plan(x0::Desc* d, float* x, const int64_t* step, int layer, int k, int rows, int experts,
                       int hidden) {
  const int s = int(*step);
  if (threadIdx.x == 0) {
    d->njobs = k;
    d->rows = rows;
    for (int j = 0; j < k; ++j) d->ids[j] = (s * 7 + layer * 13 + j * 37) % experts;
  }
  for (int i = threadIdx.x; i < rows * hidden; i += blockDim.x) x[i] = __sinf(0.001f * i + 0.37f * s + layer);
}

// Host-planned layer: the routes (rows x top_k ids) and the activations; the CPU builds the job list.
__global__ void k_route(int32_t* route, float* x, const int64_t* step, int layer, int rows, int top_k, int experts,
                        int hidden) {
  const int s = int(*step);
  for (int i = threadIdx.x; i < rows * top_k; i += blockDim.x)
    route[i] = (s * 7 + layer * 13 + (i / top_k) * 3 + (i % top_k) * 37) % experts;
  for (int i = threadIdx.x; i < rows * hidden; i += blockDim.x) x[i] = __sinf(0.001f * i + 0.37f * s + layer);
}

// The patch: read every CPU row back over PCIe and reduce over jobs.
__global__ void k_consume(const float* y, float* out, int k, int rows, int max_rows, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * hidden) return;
  const int r = i / hidden, h = i % hidden;
  float acc = 0.f;
  for (int j = 0; j < k; ++j) acc += y[(size_t(j) * max_rows + r) * hidden + h];
  out[i] = acc;
}

}  // namespace

namespace x0 {

void launch_gather(cudaStream_t s, const Blob* blobs, int n, int ctas, int threads, int unroll) {
  switch (unroll) {
    case 1: k_gather<1><<<ctas, threads, 0, s>>>(blobs, n); break;
    case 2: k_gather<2><<<ctas, threads, 0, s>>>(blobs, n); break;
    case 4: k_gather<4><<<ctas, threads, 0, s>>>(blobs, n); break;
    case 8: k_gather<8><<<ctas, threads, 0, s>>>(blobs, n); break;
    default: k_gather<4><<<ctas, threads, 0, s>>>(blobs, n);
  }
}
void launch_signal(cudaStream_t s, uint32_t* flag, uint32_t value) { k_signal<<<1, 1, 0, s>>>(flag, value); }
void launch_wait(cudaStream_t s, uint32_t* flag, uint32_t value) { k_wait<<<1, 1, 0, s>>>(flag, value); }
void launch_busy(cudaStream_t s, uint64_t ns) { k_busy<<<1, 32, 0, s>>>(ns); }
void launch_plan(cudaStream_t s, Desc* d, float* x, const int64_t* step, int layer, int k, int rows, int experts,
                 int hidden) {
  k_plan<<<1, 256, 0, s>>>(d, x, step, layer, k, rows, experts, hidden);
}
void launch_route(cudaStream_t s, int32_t* route, float* x, const int64_t* step, int layer, int rows, int top_k,
                  int experts, int hidden) {
  k_route<<<1, 256, 0, s>>>(route, x, step, layer, rows, top_k, experts, hidden);
}
void launch_consume(cudaStream_t s, const float* y, float* out, int k, int rows, int max_rows, int hidden) {
  const int n = rows * hidden;
  k_consume<<<(n + 255) / 256, 256, 0, s>>>(y, out, k, rows, max_rows, hidden);
}

}  // namespace x0
