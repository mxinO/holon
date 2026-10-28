// GPU side of the S6 handshake variants that need a kernel: a flag write/spin-wait pair
// on host-mapped memory (the alternative to stream memops) and a fixed-duration busy
// kernel that stands in for independent GPU work in the overlap test.
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

using SysRef = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

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

}  // namespace

void launch_signal(cudaStream_t s, uint32_t* flag, uint32_t value) { k_signal<<<1, 1, 0, s>>>(flag, value); }
void launch_wait(cudaStream_t s, uint32_t* flag, uint32_t value) { k_wait<<<1, 1, 0, s>>>(flag, value); }
void launch_busy(cudaStream_t s, uint64_t ns) { k_busy<<<1, 32, 0, s>>>(ns); }
