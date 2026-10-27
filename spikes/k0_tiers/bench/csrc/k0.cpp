// Step 0 of milestone K (docs/dev/v1-kv-tiers.md §8): K-M2's copy paths over chunk address lists (one
// cudaMemcpyBatchAsync, the SM copy kernel), K-M3's copy feeder beside a decoding engine, and K-M4's C reference for
// preadv / pwritev. One torch extension; bench.py drives it.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <torch/extension.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <thread>
#include <vector>

#include "kernels.h"

namespace {

using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}
cudaStream_t current() { return at::cuda::getCurrentCUDAStream().stream(); }
cudaStream_t as_stream(int64_t handle) { return handle ? reinterpret_cast<cudaStream_t>(handle) : current(); }

const int64_t* host_i64(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == torch::kInt64 && t.dim() == 1 && t.is_contiguous(), name,
              ": a contiguous 1-d int64 CPU tensor");
  return t.data_ptr<int64_t>();
}

// ------------------------------------------------------------------------------------------------ copies

// One cudaMemcpyBatchAsync over the (src, dst, bytes) lists, with expert_io's attributes (source read in stream
// order; prefer overlap with compute). `per_call` > 0 splits the list into calls of at most that many copies.
struct BatchLists {
  std::vector<void*> dst;
  std::vector<const void*> src;
  std::vector<size_t> sizes;
};

void batch_on(cudaStream_t s, const BatchLists& l, int64_t per_call, unsigned flags = cudaMemcpyFlagPreferOverlapWithCompute,
              const std::vector<cudaEvent_t>* paced = nullptr) {
  cudaMemcpyAttributes a{};
  a.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  a.flags = flags;
  size_t idx = 0;
  const size_t n = l.dst.size(), step = per_call > 0 ? size_t(per_call) : n;
  for (size_t i = 0, k = 0; i < n; i += step, ++k) {
    const size_t m = std::min(step, n - i);
    // paced: at most paced->size() pieces in flight; a piece is issued once the one `depth` before it has landed, so
    // a copy another stream queues behind them on the copy engine waits for at most that many pieces
    if (paced && k >= paced->size()) cuda_check(cudaEventSynchronize((*paced)[k % paced->size()]), "pace");
    cuda_check(cudaMemcpyBatchAsync(const_cast<void**>(l.dst.data() + i), const_cast<const void**>(l.src.data() + i),
                                    const_cast<size_t*>(l.sizes.data() + i), m, &a, &idx, 1, s),
               "cudaMemcpyBatchAsync");
    if (paced) cudaEventRecord((*paced)[k % paced->size()], s);
  }
}

void each_on(cudaStream_t s, const BatchLists& l) {
  for (size_t i = 0; i < l.dst.size(); ++i)
    cuda_check(cudaMemcpyAsync(l.dst[i], l.src[i], l.sizes[i], cudaMemcpyDefault, s), "cudaMemcpyAsync");
}

BatchLists lists(const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& sizes) {
  const int64_t n = src.numel();
  TORCH_CHECK(dst.numel() == n && sizes.numel() == n, "src, dst and sizes: one entry per copy");
  const int64_t *s = host_i64(src, "src"), *d = host_i64(dst, "dst"), *b = host_i64(sizes, "sizes");
  BatchLists l;
  l.dst.resize(n);
  l.src.resize(n);
  l.sizes.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    l.src[i] = reinterpret_cast<const void*>(s[i]);
    l.dst[i] = reinterpret_cast<void*>(d[i]);
    l.sizes[i] = size_t(b[i]);
  }
  return l;
}

void memcpy_batch(const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& sizes, int64_t per_call,
                  int64_t stream) {
  if (src.numel()) batch_on(as_stream(stream), lists(src, dst, sizes), per_call);
}

const k0::Item* items(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt64 && t.dim() == 2 && t.size(1) == 3 && t.is_contiguous(),
              "items: device int64 [n, 3] (src, dst, 16-byte words)");
  return reinterpret_cast<const k0::Item*>(t.data_ptr<int64_t>());
}

void sm_copy(const torch::Tensor& table, int64_t ctas, int64_t threads, int64_t unroll, int64_t stream) {
  if (!table.size(0)) return;
  k0::launch_copy(as_stream(stream), items(table), table.size(0), int(ctas), int(threads), int(unroll));
  cuda_check(cudaGetLastError(), "sm_copy");
}

// ------------------------------------------------------------------------------------------------ K-M3 feeder

// Keeps copies in flight on its own non-blocking stream from a C++ thread (no GIL), one copy list at a time back to
// back: "dma" (cudaMemcpyBatchAsync; `per_call` > 0 splits it into calls of that many copies, with at most `depth`
// of them in flight when `depth` > 0; `flags` the batch attribute flags), "each" (one cudaMemcpyAsync per chunk) or "sm" (the SM
// copy kernel); `loops` > 0 stops after that many copies. The waits sleep (blocking-sync events), so the feeder does not spin a core beside the engine's loop.
class Feeder {
 public:
  Feeder(const std::string& kind, torch::Tensor src, torch::Tensor dst, torch::Tensor sizes, torch::Tensor table,
         int64_t ctas, int64_t unroll, int64_t bytes, int64_t per_call, int64_t depth, int64_t flags, int64_t loops)
      : kind_(kind), table_(table), ctas_(int(ctas)), unroll_(int(unroll)), bytes_(bytes), per_call_(per_call),
        depth_(depth), loops_(loops), flags_(unsigned(flags)) {
    TORCH_CHECK(kind == "dma" || kind == "each" || kind == "sm", "kind: dma, each or sm");
    if (kind != "sm") lists_ = lists(src, dst, sizes);
    cuda_check(cudaGetDevice(&device_), "cudaGetDevice");
  }
  ~Feeder() {
    if (thread_.joinable()) stop();
  }

  void start() {
    TORCH_CHECK(!thread_.joinable(), "feeder already running");
    stop_ = false;
    copies_ = 0;
    t0_ = Clock::now();
    thread_ = std::thread([this] {
      cudaSetDevice(device_);
      cudaStream_t s;
      cudaEvent_t e;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      cudaEventCreateWithFlags(&e, cudaEventBlockingSync | cudaEventDisableTiming);
      std::vector<cudaEvent_t> paced(size_t(std::max<int64_t>(depth_, 0)));
      for (auto& p : paced) cudaEventCreateWithFlags(&p, cudaEventBlockingSync | cudaEventDisableTiming);
      while (!stop_.load() && (loops_ == 0 || copies_.load() < loops_)) {
        if (kind_ == "dma") batch_on(s, lists_, per_call_, flags_, depth_ > 0 ? &paced : nullptr);
        else if (kind_ == "each") each_on(s, lists_);
        else k0::launch_copy(s, items(table_), table_.size(0), ctas_, 256, unroll_);
        cudaEventRecord(e, s);
        cudaEventSynchronize(e);
        copies_++;
      }
      for (auto& p : paced) cudaEventDestroy(p);
      cudaEventDestroy(e);
      cudaStreamDestroy(s);
    });
  }

  bool busy() const { return thread_.joinable() && (loops_ == 0 || copies_.load() < loops_); }

  std::map<std::string, double> stop() {
    stop_ = true;
    thread_.join();
    const double secs = std::chrono::duration<double>(Clock::now() - t0_).count();
    return {{"copies", double(copies_.load())}, {"seconds", secs},
            {"gbps", double(copies_.load()) * double(bytes_) / secs / 1e9}};
  }

 private:
  std::string kind_;
  BatchLists lists_;
  torch::Tensor table_;
  int ctas_, unroll_, device_ = 0;
  int64_t bytes_, per_call_, depth_, loops_;
  unsigned flags_;
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> copies_{0};
  Clock::time_point t0_;
  std::thread thread_;
};

// ------------------------------------------------------------------------------------------------ K-M4 C reference

// `threads` threads each issue blocking preadv (or pwritev) of one slot at a time, at random slots of the file, for
// `seconds`. A slot is `iov` iovecs of the given sizes, each in its own 4 KiB-aligned buffer (as an S2 pool's segments
// sit apart). Returns GB/s over the window (a write window ends with fdatasync).
std::map<std::string, double> io_run(const std::string& path, bool write, bool direct, int64_t slot_bytes,
                                     std::vector<int64_t> iov, int64_t slots, int64_t threads, double seconds) {
  const int fd = open(path.c_str(), (write ? O_WRONLY : O_RDONLY) | (direct ? O_DIRECT : 0));
  TORCH_CHECK(fd >= 0, "open failed: ", strerror(errno));
  std::atomic<int64_t> bytes{0}, ops{0};
  std::atomic<bool> fail{false};
  const auto t0 = Clock::now();
  const auto until = t0 + std::chrono::microseconds(int64_t(seconds * 1e6));
  std::vector<std::thread> ts;
  for (int64_t w = 0; w < threads; ++w)
    ts.emplace_back([&, w] {
      std::mt19937_64 g(1000 + w);
      std::vector<void*> bufs;
      std::vector<iovec> v;
      for (int64_t n : iov) {
        void* p = nullptr;
        if (posix_memalign(&p, 4096, size_t(n))) {
          fail = true;
          return;
        }
        std::memset(p, int(w + 1), size_t(n));
        bufs.push_back(p);
        v.push_back({p, size_t(n)});
      }
      int64_t total = 0;
      for (int64_t n : iov) total += n;
      while (Clock::now() < until && !fail) {
        const off_t off = off_t(int64_t(g() % uint64_t(slots)) * slot_bytes);
        const ssize_t r = write ? pwritev(fd, v.data(), int(v.size()), off) : preadv(fd, v.data(), int(v.size()), off);
        if (r != ssize_t(total)) {
          fail = true;
          break;
        }
        bytes += total;
        ops++;
      }
      for (void* p : bufs) free(p);
    });
  for (auto& t : ts) t.join();
  if (write) fdatasync(fd);
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  close(fd);
  TORCH_CHECK(!fail, "io_run: an I/O failed (", strerror(errno), ")");
  return {{"gbps", double(bytes.load()) / secs / 1e9}, {"ops", double(ops.load())}, {"seconds", secs}};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("busy", [](int64_t ns, int64_t stream) {
    k0::launch_busy(as_stream(stream), uint64_t(ns));
    cuda_check(cudaGetLastError(), "busy");
  }, py::arg("ns"), py::arg("stream") = 0);
  m.def("memcpy_batch", &memcpy_batch, py::arg("src"), py::arg("dst"), py::arg("sizes"), py::arg("per_call") = 0,
        py::arg("stream") = 0);
  m.def("sm_copy", &sm_copy, py::arg("table"), py::arg("ctas"), py::arg("threads") = 256, py::arg("unroll") = 4,
        py::arg("stream") = 0);
  py::class_<Feeder>(m, "Feeder")
      .def(py::init<const std::string&, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t,
                    int64_t, int64_t, int64_t, int64_t, int64_t>(),
           py::arg("kind"), py::arg("src"), py::arg("dst"), py::arg("sizes"), py::arg("table"), py::arg("ctas"),
           py::arg("unroll"), py::arg("bytes"), py::arg("per_call") = 0, py::arg("depth") = 0,
           py::arg("flags") = int64_t(cudaMemcpyFlagPreferOverlapWithCompute), py::arg("loops") = 0)
      .def("start", &Feeder::start)
      .def("busy", &Feeder::busy)
      .def("stop", &Feeder::stop, py::call_guard<py::gil_scoped_release>());
  m.def("io_run", &io_run, py::call_guard<py::gil_scoped_release>());
  // a wait that keeps the core busy without the GIL, as an engine step's CUDA sync does (spin, GIL released)
  m.def("spin", [](double seconds) {
    const auto until = Clock::now() + std::chrono::nanoseconds(int64_t(seconds * 1e9));
    while (Clock::now() < until) {
    }
  }, py::call_guard<py::gil_scoped_release>());
}
