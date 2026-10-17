// Shared between kernels.cu and x0.cpp.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace x0 {

struct Blob {
  const uint4* src;  // mapped pinned host (device alias) or device memory
  uint4* dst;
  uint64_t words;    // 16-byte words
};

constexpr int kMaxJobs = 30;
struct Desc {  // one layer's CPU job list in mapped memory, 128 bytes
  int32_t njobs, rows;
  int32_t ids[kMaxJobs];
};

void launch_gather(cudaStream_t s, const Blob* blobs, int n, int ctas, int threads, int unroll);
void launch_signal(cudaStream_t s, uint32_t* flag, uint32_t value);
void launch_wait(cudaStream_t s, uint32_t* flag, uint32_t value);
void launch_busy(cudaStream_t s, uint64_t ns);
void launch_plan(cudaStream_t s, Desc* d, float* x, const int64_t* step, int layer, int k, int rows, int experts,
                 int hidden);
void launch_route(cudaStream_t s, int32_t* route, float* x, const int64_t* step, int layer, int rows, int top_k,
                  int experts, int hidden);
void launch_consume(cudaStream_t s, const float* y, float* out, int k, int rows, int max_rows, int hidden);

}  // namespace x0
