// Shared between kernels.cu and k0.cpp.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace k0 {

struct Item {        // one work item of the SM copy: 16-byte words from src to dst (device or mapped host addresses)
  const uint4* src;
  uint4* dst;
  int64_t words;
};

void launch_copy(cudaStream_t s, const Item* items, int64_t n, int ctas, int threads, int unroll);
void launch_busy(cudaStream_t s, uint64_t ns);

}  // namespace k0
