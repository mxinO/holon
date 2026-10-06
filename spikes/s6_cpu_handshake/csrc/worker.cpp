// S6: a persistent native CPU worker served from inside CUDA graphs.
//
// Protocol: one 32-bit word per slot (one slot per MoE layer), each on its own 64-byte line
// of mapped pinned memory.
//   GPU (in the graph):  write word = REQ  ...  wait until word >= DONE
//   CPU (coordinator):   sees REQ -> runs the slot's task on the pool -> stores DONE
// The GPU writes REQ only after it has observed DONE for that slot (stream order across
// replays) and the CPU writes DONE only after observing REQ, so a single word needs no
// separate reset write and both immediates are constants that can be baked into a graph.
// Inputs/outputs travel through per-slot pinned buffers copied in-stream around the pair.
//
// The GPU side comes in three flavours so they can be compared under identical graphs:
//   memop    cuStreamWriteValue32 / cuStreamWaitValue32 (front-end ops, no SM occupied)
//   kernel   1-thread kernels doing a system-scope release store / acquire spin
//   hostfunc cudaLaunchHostFunc running the task inline on the driver's callback thread
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <pthread.h>
#include <torch/extension.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

void launch_signal(cudaStream_t s, uint32_t* flag, uint32_t value);
void launch_wait(cudaStream_t s, uint32_t* flag, uint32_t value);
void launch_busy(cudaStream_t s, uint64_t ns);

namespace {

constexpr uint32_t kReq = 1;
constexpr uint32_t kDone = 2;
constexpr int64_t kStride = 16;  // uint32 words per 64-byte line

using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}
void cu_check(CUresult r, const char* what) { TORCH_CHECK(r == CUDA_SUCCESS, what, " failed: CUresult=", int(r)); }

// Driver entry points resolved through the runtime, so the extension does not link libcuda.
struct Driver {
  decltype(&cuStreamWriteValue32_v2) write32 = nullptr;
  decltype(&cuStreamWaitValue32_v2) wait32 = nullptr;
  decltype(&cuDeviceGetAttribute) attr = nullptr;
  decltype(&cuGraphGetNodes) graph_nodes = nullptr;
  decltype(&cuGraphNodeGetType) node_type = nullptr;
};

const Driver& driver() {
  static const Driver d = [] {
    Driver d;
    auto get = [](const char* name, void** fn) {
      cudaDriverEntryPointQueryResult q;
      cuda_check(cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q), name);
      TORCH_CHECK(q == cudaDriverEntryPointSuccess && *fn != nullptr, "driver symbol not found: ", name);
    };
    get("cuStreamWriteValue32", reinterpret_cast<void**>(&d.write32));
    get("cuStreamWaitValue32", reinterpret_cast<void**>(&d.wait32));
    get("cuDeviceGetAttribute", reinterpret_cast<void**>(&d.attr));
    get("cuGraphGetNodes", reinterpret_cast<void**>(&d.graph_nodes));
    get("cuGraphNodeGetType", reinterpret_cast<void**>(&d.node_type));
    return d;
  }();
  return d;
}

void pin_to(int core) {
  if (core < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// Hot spin while work flows (wakeup = one poll), doze once idle for kHot so an idle engine
// does not burn cores. The clock is read only every 1024 polls to keep the hot loop cheap.
class Backoff {
 public:
  void reset() {
    last_ = Clock::now();
    polls_ = 0;
    dozing_ = false;
  }
  void wait() {
    if (!dozing_) {
      if ((++polls_ & 1023u) != 0 || Clock::now() - last_ < kHot) {
        _mm_pause();
        return;
      }
      dozing_ = true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

 private:
  static constexpr auto kHot = std::chrono::milliseconds(100);
  Clock::time_point last_ = Clock::now();
  uint32_t polls_ = 0;
  bool dozing_ = false;
};

// --- simulated CPU expert: y = W x, W bf16 [rows, cols], x/y fp32 ---------------------------

inline __m256 bf16x8(const uint16_t* p) {
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

inline float bf16(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Gemv {
  const uint16_t* w = nullptr;
  const float* x = nullptr;
  float* y = nullptr;
  int64_t rows = 0, cols = 0;

  void run(int64_t r0, int64_t r1) const {
    for (int64_t r = r0; r < r1; ++r) {
      const uint16_t* wr = w + r * cols;
      __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
      int64_t k = 0;
      for (; k + 32 <= cols; k += 32) {
        a0 = _mm256_fmadd_ps(bf16x8(wr + k), _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(bf16x8(wr + k + 8), _mm256_loadu_ps(x + k + 8), a1);
        a2 = _mm256_fmadd_ps(bf16x8(wr + k + 16), _mm256_loadu_ps(x + k + 16), a2);
        a3 = _mm256_fmadd_ps(bf16x8(wr + k + 24), _mm256_loadu_ps(x + k + 24), a3);
      }
      const __m256 a = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
      const __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
      float lanes[4];
      _mm_storeu_ps(lanes, h);
      float s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
      for (; k < cols; ++k) s += bf16(wr[k]) * x[k];
      y[r] = s;
    }
  }
};

// Persistent spinning pool; the calling thread computes chunk 0 and waits for the rest.
class Pool {
 public:
  explicit Pool(const std::vector<int>& cores) : n_(int(cores.size())) {
    for (int i = 1; i < n_; ++i)
      threads_.emplace_back([this, i, c = cores[i]] {
        pin_to(c);
        loop(i);
      });
  }
  ~Pool() {
    stop_.store(true, std::memory_order_relaxed);
    for (auto& t : threads_) t.join();
  }
  void run(const Gemv& g) {
    if (g.rows == 0) return;
    job_ = g;
    pending_.store(n_ - 1, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    chunk(0);
    while (pending_.load(std::memory_order_acquire) != 0) _mm_pause();
  }

 private:
  void chunk(int i) const {
    const int64_t per = (job_.rows + n_ - 1) / n_;
    job_.run(std::min(job_.rows, i * per), std::min(job_.rows, (i + 1) * per));
  }
  void loop(int i) {
    uint64_t seen = 0;
    Backoff b;
    b.reset();
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint64_t g = gen_.load(std::memory_order_acquire);
      if (g == seen) {
        b.wait();
        continue;
      }
      seen = g;
      chunk(i);
      pending_.fetch_sub(1, std::memory_order_release);
      b.reset();
    }
  }

  const int n_;
  Gemv job_;
  std::vector<std::thread> threads_;
  alignas(64) std::atomic<uint64_t> gen_{0};
  alignas(64) std::atomic<int> pending_{0};
  alignas(64) std::atomic<bool> stop_{false};
};

class Worker {
 public:
  Worker(torch::Tensor w, torch::Tensor x, torch::Tensor y, std::vector<int64_t> cores)
      : w_(w), x_(x), y_(y), slots_(w.size(0)), cores_(cores.begin(), cores.end()) {
    TORCH_CHECK(w.device().is_cpu() && w.scalar_type() == torch::kBFloat16 && w.dim() == 3 && w.is_contiguous());
    TORCH_CHECK(x.is_pinned() && x.scalar_type() == torch::kFloat && x.size(0) == slots_ && x.size(1) == w.size(2));
    TORCH_CHECK(y.is_pinned() && y.scalar_type() == torch::kFloat && y.size(0) == slots_ && y.size(1) == w.size(1));
    TORCH_CHECK(!cores_.empty(), "need at least one core (the coordinator)");
    const size_t bytes = size_t(slots_ * kStride) * sizeof(uint32_t);
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&flags_), bytes, cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(flags)");
    std::memset(flags_, 0, bytes);
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dflags_), flags_, 0), "cudaHostGetDevicePointer");
    for (int64_t s = 0; s < slots_; ++s) host_ctx_.push_back({this, s});
  }

  ~Worker() {
    stop();
    cudaFreeHost(flags_);
  }

  void start() {
    TORCH_CHECK(!coord_.joinable(), "worker already running");
    stop_.store(false);
    abort_.store(false);
    pool_ = std::make_unique<Pool>(cores_);
    coord_ = std::thread([this] {
      pin_to(cores_[0]);
      serve_loop();
    });
  }

  // Joins the threads. Must run after in-flight replays drained (quiesce); returns the number
  // of slots left in REQ, i.e. graph waits that would now never complete (0 when clean).
  int64_t stop() {
    if (coord_.joinable()) {
      stop_.store(true, std::memory_order_relaxed);
      coord_.join();
      pool_.reset();
    }
    int64_t stranded = 0;
    for (int64_t s = 0; s < slots_; ++s) stranded += load(s) == kReq;
    return stranded;
  }

  // Fail-open: answer every request without computing, so in-flight replays drain (their
  // outputs are garbage and must be discarded by the caller). Used for abort/teardown.
  void abort() { abort_.store(true, std::memory_order_release); }

  void set_compute(bool on) { compute_.store(on); }
  void set_spin(int64_t us) { spin_us_.store(us); }
  // Sleep `us` before every `every`-th task, counting from the next one (every=huge: stall once).
  void set_delay(int64_t us, int64_t every) {
    delay_base_.store(tasks_.load());
    delay_every_.store(every);
    delay_us_.store(us);
  }
  int64_t tasks() const { return tasks_.load(); }
  int64_t aborted() const { return aborted_.load(); }
  int64_t compute_ns() const { return compute_ns_.load(); }

  // ---- GPU side, enqueued on the current (possibly capturing) stream ----
  void signal_memop(int64_t s) {
    cu_check(driver().write32(cu_stream(), dflag(s), kReq, CU_STREAM_WRITE_VALUE_DEFAULT), "cuStreamWriteValue32");
  }
  void wait_memop(int64_t s) {
    cu_check(driver().wait32(cu_stream(), dflag(s), kDone, CU_STREAM_WAIT_VALUE_GEQ), "cuStreamWaitValue32");
  }
  void signal_kernel(int64_t s) {
    launch_signal(stream(), reinterpret_cast<uint32_t*>(dflag(s)), kReq);
    cuda_check(cudaGetLastError(), "launch_signal");
  }
  void wait_kernel(int64_t s) {
    launch_wait(stream(), reinterpret_cast<uint32_t*>(dflag(s)), kDone);
    cuda_check(cudaGetLastError(), "launch_wait");
  }
  void hostfunc(int64_t s) {
    TORCH_CHECK(s >= 0 && s < slots_);
    cuda_check(cudaLaunchHostFunc(stream(), &Worker::host_cb, &host_ctx_[s]), "cudaLaunchHostFunc");
  }

  // Mean wall time of one task on the pool (caller thread = chunk 0); only while idle.
  double task_us(int64_t s, int64_t iters) {
    TORCH_CHECK(pool_, "worker not started");
    const auto t0 = Clock::now();
    for (int64_t i = 0; i < iters; ++i) pool_->run(gemv(s));
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / double(iters);
  }

 private:
  struct HostCtx {
    Worker* w;
    int64_t slot;
  };
  static void CUDART_CB host_cb(void* p) {
    auto* c = static_cast<HostCtx*>(p);
    c->w->serve(c->slot);
  }

  static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }
  static CUstream cu_stream() { return reinterpret_cast<CUstream>(stream()); }
  CUdeviceptr dflag(int64_t s) const {
    TORCH_CHECK(s >= 0 && s < slots_);
    return dflags_ + CUdeviceptr(s * kStride * sizeof(uint32_t));
  }
  uint32_t load(int64_t s) const { return __atomic_load_n(flags_ + s * kStride, __ATOMIC_ACQUIRE); }

  Gemv gemv(int64_t s) const {
    const int64_t rows = w_.size(1), cols = w_.size(2);
    return {reinterpret_cast<const uint16_t*>(w_.data_ptr<at::BFloat16>()) + s * rows * cols,
            x_.data_ptr<float>() + s * cols, y_.data_ptr<float>() + s * rows, rows, cols};
  }

  void serve(int64_t s) {
    const int64_t n = tasks_.fetch_add(1, std::memory_order_relaxed);
    if (abort_.load(std::memory_order_acquire)) {
      aborted_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const int64_t d = delay_us_.load(std::memory_order_relaxed), every = delay_every_.load(std::memory_order_relaxed);
    if (d > 0 && every > 0 && (n - delay_base_.load(std::memory_order_relaxed)) % every == 0)
      std::this_thread::sleep_for(std::chrono::microseconds(d));
    const bool compute = compute_.load(std::memory_order_relaxed);
    const auto spin = std::chrono::microseconds(spin_us_.load(std::memory_order_relaxed));
    if (!compute && spin.count() == 0) return;
    const auto t0 = Clock::now();
    while (Clock::now() - t0 < spin) _mm_pause();  // fixed-length stand-in for CPU work
    if (compute) pool_->run(gemv(s));
    compute_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count(),
                          std::memory_order_relaxed);
  }

  void serve_loop() {
    Backoff b;
    b.reset();
    while (!stop_.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int64_t s = 0; s < slots_; ++s) {
        if (load(s) != kReq) continue;  // acquire: the in-stream D2H of x is visible
        serve(s);
        __atomic_store_n(flags_ + s * kStride, kDone, __ATOMIC_RELEASE);  // release: y visible first
        any = true;
      }
      if (any)
        b.reset();
      else
        b.wait();
    }
  }

  torch::Tensor w_, x_, y_;
  const int64_t slots_;
  const std::vector<int> cores_;
  uint32_t* flags_ = nullptr;
  CUdeviceptr dflags_ = 0;
  std::vector<HostCtx> host_ctx_;
  std::unique_ptr<Pool> pool_;
  std::thread coord_;
  std::atomic<bool> stop_{false}, abort_{false}, compute_{true};
  std::atomic<int64_t> delay_us_{0}, delay_every_{0}, delay_base_{0}, spin_us_{0}, tasks_{0}, aborted_{0}, compute_ns_{0};
};

// CUDA tensor aliasing a pinned host tensor (zero-copy: kernels read/write it over PCIe).
torch::Tensor device_view(torch::Tensor host) {
  TORCH_CHECK(host.is_pinned(), "device_view needs pinned memory");
  void* dptr = nullptr;
  cuda_check(cudaHostGetDevicePointer(&dptr, host.data_ptr(), 0), "cudaHostGetDevicePointer");
  int dev = 0;
  cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
  return torch::from_blob(dptr, host.sizes(), host.strides(), [host](void*) {},
                          host.options().device(torch::kCUDA, dev).pinned_memory(false));
}

void busy(int64_t ns) {
  launch_busy(at::cuda::getCurrentCUDAStream().stream(), uint64_t(ns));
  cuda_check(cudaGetLastError(), "launch_busy");
}

std::map<std::string, int64_t> memop_caps() {
  int dev = 0;
  cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
  auto get = [&](CUdevice_attribute a) {
    int v = -1;
    driver().attr(&v, a, dev);
    return int64_t(v);
  };
  return {{"can_use_64_bit_stream_mem_ops", get(CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS)},
          {"can_use_stream_wait_value_nor", get(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR)},
          {"can_flush_remote_writes", get(CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES)},
          {"can_use_host_pointer_for_registered_mem", get(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM)},
          {"host_native_atomic_supported", get(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)},
          {"pageable_memory_access", get(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)}};
}

// Node types of a captured graph (torch.cuda.CUDAGraph(keep_graph=True).raw_cuda_graph()). Driver API:
// the runtime's cudaGraphNodeGetType fails with "unknown error" on batch-memop nodes (CUDA 13.0).
std::vector<std::string> graph_node_types(int64_t graph) {
  static const char* names[] = {"kernel", "memcpy", "memset", "host", "child_graph", "empty", "wait_event",
                                "event_record", "ext_sema_signal", "ext_sema_wait", "mem_alloc", "mem_free",
                                "batch_mem_op", "conditional"};
  auto g = reinterpret_cast<CUgraph>(graph);
  size_t n = 0;
  cu_check(driver().graph_nodes(g, nullptr, &n), "cuGraphGetNodes");
  std::vector<CUgraphNode> nodes(n);
  cu_check(driver().graph_nodes(g, nodes.data(), &n), "cuGraphGetNodes");
  std::vector<std::string> out;
  for (auto node : nodes) {
    CUgraphNodeType t;
    cu_check(driver().node_type(node, &t), "cuGraphNodeGetType");
    const int i = int(t);
    out.push_back(i >= 0 && i < int(std::size(names)) ? names[i] : "type_" + std::to_string(i));
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Worker>(m, "Worker")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>>())
      .def("start", &Worker::start)
      .def("stop", &Worker::stop, py::call_guard<py::gil_scoped_release>())
      .def("abort", &Worker::abort)
      .def("set_compute", &Worker::set_compute)
      .def("set_delay", &Worker::set_delay)
      .def("set_spin", &Worker::set_spin)
      .def("tasks", &Worker::tasks)
      .def("aborted", &Worker::aborted)
      .def("compute_ns", &Worker::compute_ns)
      .def("signal_memop", &Worker::signal_memop)
      .def("wait_memop", &Worker::wait_memop)
      .def("signal_kernel", &Worker::signal_kernel)
      .def("wait_kernel", &Worker::wait_kernel)
      .def("hostfunc", &Worker::hostfunc)
      .def("task_us", &Worker::task_us, py::call_guard<py::gil_scoped_release>());
  m.def("busy", &busy);
  m.def("device_view", &device_view);
  m.def("memop_caps", &memop_caps);
  m.def("graph_node_types", &graph_node_types);
}
