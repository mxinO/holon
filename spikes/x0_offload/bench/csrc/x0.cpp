// Step 0 of milestone X (docs/dev/v1-offload.md §8): M1 host-to-device rates, M2 CPU/PCIe contention, M3 CPU expert
// throughput (ggml-cpu), M5 the S6 handshake with a real expert payload, M7 O_DIRECT reads. One torch extension;
// bench.py drives it. The handshake protocol and worker follow spikes/s6_cpu_handshake (same single-word flags).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <torch/extension.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <map>
#include <memory>
#include <optional>
#include <random>

#include "cpu_experts.h"
#include "kernels.h"

namespace {

using x0::Clock;

void cuda_check(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }
double us_since(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }
std::vector<int> ints(const std::vector<int64_t>& v) { return {v.begin(), v.end()}; }

// ------------------------------------------------------------------------------------------------ host memory

// Exactly `bytes` of mapped pinned memory (cudaHostAlloc, no power-of-two rounding: D10).
class HostArena {
 public:
  explicit HostArena(int64_t bytes) : bytes_(bytes) {
    cuda_check(cudaHostAlloc(&p_, size_t(bytes), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    cuda_check(cudaHostGetDevicePointer(&d_, p_, 0), "cudaHostGetDevicePointer");
  }
  ~HostArena() { cudaFreeHost(p_); }
  torch::Tensor tensor() { return torch::from_blob(p_, {bytes_}, [](void*) {}, torch::kUInt8); }
  int64_t host_ptr() const { return int64_t(p_); }
  int64_t device_ptr() const { return int64_t(d_); }
  int64_t bytes() const { return bytes_; }

 private:
  int64_t bytes_;
  void* p_ = nullptr;
  void* d_ = nullptr;
};

// ------------------------------------------------------------------------------------------------ M1 copies

const x0::Blob* blob_table(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt64 && t.dim() == 2 && t.size(1) == 3 && t.is_contiguous(),
              "blobs: device int64 [n, 3] (src, dst, 16-byte words)");
  return reinterpret_cast<const x0::Blob*>(t.data_ptr<int64_t>());
}

void gather_on(cudaStream_t s, const x0::Blob* blobs, int n, int ctas, int threads, int unroll) {
  x0::launch_gather(s, blobs, n, ctas, threads, unroll);
  cuda_check(cudaGetLastError(), "gather");
}

void memcpy_each_on(cudaStream_t s, const int64_t* t, int64_t n) {  // host table [n, 3]
  for (int64_t i = 0; i < n; ++i)
    cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(t[3 * i + 1]), reinterpret_cast<void*>(t[3 * i]),
                               size_t(t[3 * i + 2]) * 16, cudaMemcpyDefault, s),
               "cudaMemcpyAsync");
}

void memcpy_batch_on(cudaStream_t s, const int64_t* t, int64_t n) {
  std::vector<void*> dst(n);
  std::vector<const void*> src(n);
  std::vector<size_t> sizes(n);
  for (int64_t i = 0; i < n; ++i) {
    src[i] = reinterpret_cast<const void*>(t[3 * i]);
    dst[i] = reinterpret_cast<void*>(t[3 * i + 1]);
    sizes[i] = size_t(t[3 * i + 2]) * 16;
  }
  cudaMemcpyAttributes a{};
  a.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t idx = 0;
  cuda_check(cudaMemcpyBatchAsync(dst.data(), src.data(), sizes.data(), size_t(n), &a, &idx, 1, s),
             "cudaMemcpyBatchAsync");
}

const int64_t* host_table(const torch::Tensor& t) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == torch::kInt64 && t.dim() == 2 && t.size(1) == 3 &&
                  t.is_contiguous(),
              "blobs: host int64 [n, 3]");
  return t.data_ptr<int64_t>();
}

// ------------------------------------------------------------------------------------------------ M3 CPU experts

class Cpu {
 public:
  explicit Cpu(const std::vector<int64_t>& cores) : cores_(ints(cores)), pool_(cores_) {}
  int threads() const { return pool_.size(); }

  // Per-batch wall times (us) of `iters` batches of `njobs` experts x `tokens` tokens; batch i takes experts
  // first + i*njobs + j (mod experts), so a sweep over more experts than the L3 reads DRAM.
  std::vector<double> time(const x0::ExpertClass& c, int64_t arena, int64_t experts, int njobs, int tokens,
                           int iters, int64_t first) {
    x0::PinScope pin(cores_[0]);
    prepare(c, njobs, tokens);
    std::vector<double> out;
    out.reserve(iters);
    int64_t e = first;
    for (int i = 0; i < iters; ++i) {
      for (int j = 0; j < njobs; ++j, ++e) jobs_[j].w = reinterpret_cast<const uint8_t*>(arena) + (e % experts) * c.bytes;
      const auto t0 = Clock::now();
      batch_.run(pool_, c, jobs_.data(), njobs, tokens);
      out.push_back(us_since(t0));
    }
    return out;
  }

  // y[j] = expert ids[j] applied to x[j] ([njobs, tokens, hidden] f32 each), for the bitwise checks.
  void run(const x0::ExpertClass& c, int64_t arena, const std::vector<int64_t>& ids, torch::Tensor x,
           torch::Tensor y) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat && x.is_contiguous() && x.dim() == 3 && x.size(2) == c.hidden);
    TORCH_CHECK(y.sizes() == x.sizes() && y.scalar_type() == torch::kFloat && y.is_contiguous());
    TORCH_CHECK(int64_t(ids.size()) == x.size(0));
    x0::PinScope pin(cores_[0]);
    const int njobs = int(ids.size()), tokens = int(x.size(1));
    std::vector<x0::Job> jobs(njobs);
    for (int j = 0; j < njobs; ++j)
      jobs[j] = {reinterpret_cast<const uint8_t*>(arena) + ids[j] * c.bytes,
                 x.data_ptr<float>() + size_t(j) * tokens * c.hidden, y.data_ptr<float>() + size_t(j) * tokens * c.hidden};
    batch_.run(pool_, c, jobs.data(), njobs, tokens);
  }

  // One batch with no pinning (the caller is already on cores[0]): M2's loops.
  void run_jobs(const x0::ExpertClass& c, const uint8_t* const* w, int njobs, int tokens) {
    prepare(c, njobs, tokens);
    for (int j = 0; j < njobs; ++j) jobs_[j].w = w[j];
    batch_.run(pool_, c, jobs_.data(), njobs, tokens);
  }
  int core0() const { return cores_[0]; }

 private:
  void prepare(const x0::ExpertClass& c, int njobs, int tokens) {
    const size_t n = size_t(njobs) * tokens * c.hidden;
    if (x_.size() < n) {
      std::mt19937 g(1);
      std::normal_distribution<float> d(0.f, 1.f);
      x_.resize(n);
      for (auto& v : x_) v = d(g);
      y_.assign(n, 0.f);
    }
    jobs_.resize(njobs);
    for (int j = 0; j < njobs; ++j)
      jobs_[j] = {nullptr, x_.data() + size_t(j) * tokens * c.hidden, y_.data() + size_t(j) * tokens * c.hidden};
  }

  std::vector<int> cores_;
  x0::Pool pool_;
  x0::Batch batch_;
  std::vector<float> x_, y_;
  std::vector<x0::Job> jobs_;
};

// ------------------------------------------------------------------------------------------------ M2 contention

// Background GPU traffic (DMA, batched DMA or the SM gather of `blobs`, round after round on its own stream) while
// the CPU computes one-expert jobs on `cpu` (if any). Phase 1 (duration_ms): the CPU rate under GPU traffic.
// Phase 2 (duration_ms, the CPU still running): the GPU rate under CPU load.
std::map<std::string, double> contend(Cpu* cpu, const x0::ExpertClass& c, int64_t arena, int64_t experts, int tokens, std::string mode,
                 torch::Tensor blobs_host, torch::Tensor blobs_dev, int ctas, int threads, int unroll, int feeder_core,
                 double duration_ms) {
  const bool gpu = mode != "none";
  const int64_t* th = host_table(blobs_host);
  const x0::Blob* td = blob_table(blobs_dev);
  const int nblobs = int(blobs_host.size(0));
  double batch_bytes = 0;
  for (int i = 0; i < nblobs; ++i) batch_bytes += double(th[3 * i + 2]) * 16;
  const int dev = blobs_dev.device().index();
  const auto dur = std::chrono::microseconds(int64_t(duration_ms * 1000));

  std::atomic<bool> cpu_started{false}, stop{false}, feeder_ready{!gpu};
  double gpu_gbps = 0;
  int64_t gpu_rounds = 0;
  std::thread feeder;
  if (gpu)
    feeder = std::thread([&] {
      x0::pin_to(feeder_core);
      cuda_check(cudaSetDevice(dev), "cudaSetDevice");
      cudaStream_t s;
      cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
      std::vector<cudaEvent_t> ev;
      int64_t a = -1, b = -1;
      Clock::time_point t_cpu;
      bool seen = false;
      while (true) {
        if (mode == "gather")
          gather_on(s, td, nblobs, ctas, threads, unroll);
        else if (mode == "batch")
          memcpy_batch_on(s, th, nblobs);
        else
          memcpy_each_on(s, th, nblobs);
        cudaEvent_t e;
        cuda_check(cudaEventCreate(&e), "cudaEventCreate");
        cuda_check(cudaEventRecord(e, s), "cudaEventRecord");
        ev.push_back(e);
        if (ev.size() > 4) cudaEventSynchronize(ev[ev.size() - 5]);
        if (ev.size() == 8) feeder_ready.store(true);
        if (!seen && cpu_started.load()) {
          seen = true;
          t_cpu = Clock::now();
        }
        if (seen && a < 0 && Clock::now() - t_cpu >= dur) a = int64_t(ev.size()) - 1;
        if (a >= 0 && Clock::now() - t_cpu >= 2 * dur) {
          b = int64_t(ev.size()) - 1;
          break;
        }
      }
      stop.store(true);
      cudaStreamSynchronize(s);
      float ms = 0;
      cudaEventElapsedTime(&ms, ev[a], ev[b]);
      gpu_rounds = b - a;
      gpu_gbps = double(gpu_rounds) * batch_bytes / (double(ms) * 1e6);
      for (auto e : ev) cudaEventDestroy(e);
      cudaStreamDestroy(s);
    });

  double cpu_us = 0;
  int64_t cpu_n = 0;
  {
    x0::PinScope pin(cpu ? cpu->core0() : feeder_core);
    while (!feeder_ready.load()) _mm_pause();
    cpu_started.store(true);
    const auto t0 = Clock::now();
    bool counted = false;
    int64_t e = 0, n = 0;
    std::vector<const uint8_t*> w(1);
    while (true) {
      const bool phase1 = Clock::now() - t0 < dur;
      if (!phase1 && !counted) {
        cpu_us = us_since(t0) / double(std::max<int64_t>(n, 1));
        cpu_n = n;
        counted = true;
        if (!gpu) break;
      }
      if (counted && stop.load()) break;
      if (cpu) {
        w[0] = reinterpret_cast<const uint8_t*>(arena) + (e++ % experts) * c.bytes;
        cpu->run_jobs(c, w.data(), 1, tokens);
        ++n;
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
  }
  if (feeder.joinable()) feeder.join();
  const double nan = std::nan("");  // results are built without the GIL: no Python objects here
  return {{"cpu_us_per_expert", cpu ? cpu_us : nan},
          {"cpu_experts", double(cpu_n)},
          {"gpu_gbps", gpu ? gpu_gbps : nan},
          {"gpu_rounds", double(gpu_rounds)}};
}

// One miss split: the GPU path moves blobs [first, first+m) of `blobs` while the CPU computes `cpu_ids`, started
// together. Returns (cpu_us, gpu_us, wall_us): each path's own time and launch -> both done.
std::vector<double> split_once(Cpu& cpu, const x0::ExpertClass& c, int64_t arena, const std::vector<int64_t>& cpu_ids,
                               std::string mode, torch::Tensor blobs_host, torch::Tensor blobs_dev, int64_t first,
                               int64_t m, int ctas, int threads, int unroll) {
  x0::PinScope pin(cpu.core0());
  cudaStream_t s = stream();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<const uint8_t*> w;
  for (auto id : cpu_ids) w.push_back(reinterpret_cast<const uint8_t*>(arena) + id * c.bytes);
  const auto t0 = Clock::now();
  cudaEventRecord(e0, s);
  if (m > 0) {
    if (mode == "gather")
      gather_on(s, blob_table(blobs_dev) + first, int(m), ctas, threads, unroll);
    else if (mode == "batch")
      memcpy_batch_on(s, host_table(blobs_host) + 3 * first, m);
    else
      memcpy_each_on(s, host_table(blobs_host) + 3 * first, m);
  }
  cudaEventRecord(e1, s);
  double cpu_us = 0;
  if (!w.empty()) {
    const auto c0 = Clock::now();
    cpu.run_jobs(c, w.data(), int(w.size()), 1);
    cpu_us = us_since(c0);
  }
  cudaEventSynchronize(e1);
  const double wall = us_since(t0);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return {cpu_us, m > 0 ? double(ms) * 1000 : 0.0, wall};
}

// ------------------------------------------------------------------------------------------------ M5 handshake

constexpr uint32_t kReq = 1, kPlanned = 2, kDone = 3;
constexpr int64_t kStride = 16;  // uint32 words per 64-byte line
constexpr int kTopK = 8;

// S6's persistent worker, with a real payload: per layer slot a job list (device- or host-planned), activations and
// one output row block per job, all in mapped pinned memory; the coordinator (pool thread 0) computes the jobs.
class Handshake {
 public:
  Handshake(const x0::ExpertClass& c, int64_t arena, int64_t experts, int64_t slots, int64_t max_rows,
            const std::vector<int64_t>& cores)
      : c_(c), arena_(reinterpret_cast<const uint8_t*>(arena)), experts_(experts), slots_(slots),
        max_rows_(max_rows), cores_(ints(cores)),
        mem_(slots * (kStride * 4 + sizeof(x0::Desc) + max_rows * kTopK * 4 +
                      max_rows * c.hidden * 4 * (1 + x0::kMaxJobs))) {
    std::memset(reinterpret_cast<void*>(mem_.host_ptr()), 0, size_t(mem_.bytes()));
    off_desc_ = slots * kStride * 4;
    off_route_ = off_desc_ + slots * int64_t(sizeof(x0::Desc));
    off_x_ = off_route_ + slots * max_rows * kTopK * 4;
    off_y_ = off_x_ + slots * max_rows * c.hidden * 4;
  }
  ~Handshake() { stop(); }

  void start() {
    TORCH_CHECK(!coord_.joinable(), "running");
    stop_.store(false);
    coord_ = std::thread([this] {
      x0::pin_to(cores_[0]);
      x0::Pool pool(cores_);
      x0::Batch batch;
      serve_loop(pool, batch);
    });
  }
  int64_t stop() {
    if (coord_.joinable()) {
      stop_.store(true);
      coord_.join();
    }
    int64_t stranded = 0;
    for (int64_t s = 0; s < slots_; ++s) stranded += word(s) == kReq || word(s) == kPlanned;
    return stranded;
  }
  void set_host_plan(bool on, int k, int rows) {
    host_plan_.store(on);
    host_k_.store(k);
    host_rows_.store(rows);
  }
  int64_t tasks() const { return tasks_.load(); }
  int64_t compute_ns() const { return compute_ns_.load(); }

  // GPU side, enqueued on the current (capturing) stream.
  void plan(int64_t s, torch::Tensor step, int k, int rows) {
    TORCH_CHECK(k <= x0::kMaxJobs && rows <= max_rows_);
    x0::launch_plan(stream(), dptr<x0::Desc>(off_desc_, s, sizeof(x0::Desc)), dptr<float>(off_x_, s, max_rows_ * c_.hidden * 4),
                    step.data_ptr<int64_t>(), int(s), k, rows, int(experts_), c_.hidden);
  }
  void route(int64_t s, torch::Tensor step, int rows) {
    x0::launch_route(stream(), dptr<int32_t>(off_route_, s, max_rows_ * kTopK * 4),
                     dptr<float>(off_x_, s, max_rows_ * c_.hidden * 4), step.data_ptr<int64_t>(), int(s), rows, kTopK,
                     int(experts_), c_.hidden);
  }
  void signal(int64_t s) { x0::launch_signal(stream(), dptr<uint32_t>(0, s, kStride * 4), kReq); }
  void wait(int64_t s, int64_t value) { x0::launch_wait(stream(), dptr<uint32_t>(0, s, kStride * 4), uint32_t(value)); }
  void consume(int64_t s, torch::Tensor out, int k, int rows) {
    x0::launch_consume(stream(), dptr<float>(off_y_, s, max_rows_ * c_.hidden * 4 * x0::kMaxJobs), out.data_ptr<float>(),
                       k, rows, int(max_rows_), c_.hidden);
  }
  // The output of slot s, job j, token r (host view), for a correctness spot check.
  torch::Tensor y(int64_t s) {
    return torch::from_blob(hptr<float>(off_y_, s, max_rows_ * c_.hidden * 4 * x0::kMaxJobs),
                            {x0::kMaxJobs, max_rows_, c_.hidden}, torch::kFloat);
  }
  torch::Tensor x(int64_t s) {
    return torch::from_blob(hptr<float>(off_x_, s, max_rows_ * c_.hidden * 4), {max_rows_, c_.hidden}, torch::kFloat);
  }
  std::vector<int64_t> ids(int64_t s) {
    const auto* d = hptr<x0::Desc>(off_desc_, s, sizeof(x0::Desc));
    return {d->ids, d->ids + d->njobs};
  }

 private:
  template <class T>
  T* hptr(int64_t off, int64_t s, int64_t stride) const {
    return reinterpret_cast<T*>(mem_.host_ptr() + off + s * stride);
  }
  template <class T>
  T* dptr(int64_t off, int64_t s, int64_t stride) const {
    TORCH_CHECK(s >= 0 && s < slots_);
    return reinterpret_cast<T*>(mem_.device_ptr() + off + s * stride);
  }
  uint32_t word(int64_t s) const { return __atomic_load_n(hptr<uint32_t>(0, s, kStride * 4), __ATOMIC_ACQUIRE); }
  void set(int64_t s, uint32_t v) { __atomic_store_n(hptr<uint32_t>(0, s, kStride * 4), v, __ATOMIC_RELEASE); }

  void serve(x0::Pool& pool, x0::Batch& batch, int64_t s) {
    auto* d = hptr<x0::Desc>(off_desc_, s, sizeof(x0::Desc));
    const auto t0 = Clock::now();
    if (host_plan_.load(std::memory_order_relaxed)) {  // build the job list from the routes: first k distinct
      const int rows = host_rows_.load(), k = host_k_.load();
      const int32_t* r = hptr<int32_t>(off_route_, s, max_rows_ * kTopK * 4);
      int n = 0;
      for (int i = 0; i < rows * kTopK && n < k; ++i) {
        bool dup = false;
        for (int j = 0; j < n; ++j) dup |= d->ids[j] == r[i];
        if (!dup) d->ids[n++] = r[i];
      }
      d->njobs = n;
      d->rows = rows;
      set(s, kPlanned);
    }
    const int njobs = d->njobs, rows = d->rows;
    x0::Job jobs[x0::kMaxJobs];
    float* x = hptr<float>(off_x_, s, max_rows_ * c_.hidden * 4);
    float* y = hptr<float>(off_y_, s, max_rows_ * c_.hidden * 4 * x0::kMaxJobs);
    for (int j = 0; j < njobs; ++j)
      jobs[j] = {arena_ + int64_t(d->ids[j]) * c_.bytes, x, y + size_t(j) * max_rows_ * c_.hidden};
    batch.run(pool, c_, jobs, njobs, rows);
    compute_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count(),
                          std::memory_order_relaxed);
    tasks_.fetch_add(1, std::memory_order_relaxed);
  }

  void serve_loop(x0::Pool& pool, x0::Batch& batch) {
    x0::Backoff b;
    b.reset();
    while (!stop_.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int64_t s = 0; s < slots_; ++s) {
        if (word(s) != kReq) continue;
        serve(pool, batch, s);
        set(s, kDone);
        any = true;
      }
      if (any)
        b.reset();
      else
        b.wait();
    }
  }

  const x0::ExpertClass c_;
  const uint8_t* arena_;
  const int64_t experts_, slots_, max_rows_;
  const std::vector<int> cores_;
  HostArena mem_;
  int64_t off_desc_ = 0, off_route_ = 0, off_x_ = 0, off_y_ = 0;
  std::thread coord_;
  std::atomic<bool> stop_{false}, host_plan_{false};
  std::atomic<int> host_k_{0}, host_rows_{1};
  std::atomic<int64_t> tasks_{0}, compute_ns_{0};
};

// ------------------------------------------------------------------------------------------------ M7 O_DIRECT

long io_setup(unsigned n, aio_context_t* ctx) { return syscall(__NR_io_setup, n, ctx); }
long io_destroy(aio_context_t ctx) { return syscall(__NR_io_destroy, ctx); }
long io_submit(aio_context_t ctx, long n, iocb** p) { return syscall(__NR_io_submit, ctx, n, p); }
long io_getevents(aio_context_t ctx, long mn, long mx, io_event* ev, timespec* t) {
  return syscall(__NR_io_getevents, ctx, mn, mx, ev, t);
}

constexpr int64_t kAlign = 4096;

struct Aligned {
  explicit Aligned(size_t n) {
    TORCH_CHECK(posix_memalign(&p, kAlign, n) == 0, "posix_memalign");
    std::memset(p, 0, n);
  }
  ~Aligned() { free(p); }
  void* p = nullptr;
};

// Writes `bytes` of pseudo-random data with O_DIRECT (so the SSD stores real data, not trimmed blocks), fsyncs.
double nvme_fill(const std::string& path, int64_t bytes) {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
  TORCH_CHECK(fd >= 0, "open for write failed: ", strerror(errno));
  const size_t chunk = 64 << 20;
  Aligned buf(chunk);
  std::mt19937_64 g(7);
  auto* w = static_cast<uint64_t*>(buf.p);
  const auto t0 = Clock::now();
  for (int64_t done = 0; done < bytes; done += chunk) {
    for (size_t i = 0; i < chunk / 8; i += 512) w[i] = g();  // a fresh word per 4 KiB page: no two pages equal
    for (size_t i = 0; i < chunk / 8; ++i)
      if (i % 512) w[i] = w[i - 1] * 6364136223846793005ULL + 1442695040888963407ULL;
    TORCH_CHECK(write(fd, buf.p, chunk) == ssize_t(chunk), "write failed: ", strerror(errno));
  }
  fsync(fd);
  close(fd);
  return us_since(t0) / 1e6;
}

// Random expert-sized reads (`read_bytes` from an unaligned offset, read as the 4 KiB-aligned superset) for
// `seconds`. aio: each of `workers` threads keeps qd/workers reads in flight with Linux AIO; pread: each worker
// issues one blocking pread at a time (qd = workers).
std::map<std::string, double> nvme_read(const std::string& path, int64_t file_bytes, int64_t read_bytes, int qd, int workers,
                   const std::string& mode, double seconds, const std::vector<int64_t>& cores) {
  const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
  TORCH_CHECK(fd >= 0, "open failed: ", strerror(errno));
  const int per = mode == "pread" ? 1 : std::max(1, qd / workers);
  const int64_t span = (read_bytes + 2 * kAlign - 1) / kAlign * kAlign;  // covers any 16-byte-aligned start
  std::atomic<int64_t> useful{0}, device{0}, reads{0};
  std::vector<std::vector<float>> lat(workers);
  std::atomic<bool> fail{false};
  const auto t0 = Clock::now();
  const auto until = t0 + std::chrono::microseconds(int64_t(seconds * 1e6));
  auto pick = [&](std::mt19937_64& g, int64_t& off, int64_t& len) {
    const int64_t start = int64_t(g() % uint64_t(file_bytes - read_bytes - kAlign)) / 16 * 16;
    off = start / kAlign * kAlign;
    len = (start + read_bytes + kAlign - 1) / kAlign * kAlign - off;
  };
  std::vector<std::thread> ts;
  for (int w = 0; w < workers; ++w)
    ts.emplace_back([&, w] {
      if (!cores.empty()) x0::pin_to(int(cores[w % cores.size()]));
      std::mt19937_64 g(100 + w);
      std::vector<std::unique_ptr<Aligned>> bufs;
      for (int i = 0; i < per; ++i) bufs.push_back(std::make_unique<Aligned>(span));
      if (mode == "pread") {
        while (Clock::now() < until) {
          int64_t off, len;
          pick(g, off, len);
          const auto r0 = Clock::now();
          if (pread(fd, bufs[0]->p, len, off) != len) {
            fail = true;
            return;
          }
          lat[w].push_back(float(us_since(r0)));
          useful += read_bytes;
          device += len;
          ++reads;
        }
        return;
      }
      aio_context_t ctx = 0;
      if (io_setup(per, &ctx) < 0) {
        fail = true;
        return;
      }
      std::vector<iocb> cbs(per);
      std::vector<Clock::time_point> sent(per);
      auto submit = [&](int i) {
        int64_t off, len;
        pick(g, off, len);
        std::memset(&cbs[i], 0, sizeof(iocb));
        cbs[i].aio_fildes = fd;
        cbs[i].aio_lio_opcode = IOCB_CMD_PREAD;
        cbs[i].aio_buf = uint64_t(bufs[i]->p);
        cbs[i].aio_nbytes = len;
        cbs[i].aio_offset = off;
        cbs[i].aio_data = uint64_t(i);
        iocb* p = &cbs[i];
        sent[i] = Clock::now();
        return io_submit(ctx, 1, &p) == 1;
      };
      for (int i = 0; i < per; ++i)
        if (!submit(i)) fail = true;
      int inflight = per;
      std::vector<io_event> ev(per);
      while (inflight > 0 && !fail) {
        const long n = io_getevents(ctx, 1, per, ev.data(), nullptr);
        if (n < 0) {
          fail = true;
          break;
        }
        for (long e = 0; e < n; ++e) {
          const int i = int(ev[e].data);
          inflight--;
          if (ev[e].res != int64_t(cbs[i].aio_nbytes)) fail = true;
          lat[w].push_back(float(us_since(sent[i])));
          useful += read_bytes;
          device += int64_t(cbs[i].aio_nbytes);
          ++reads;
          if (Clock::now() < until && !fail) {
            if (submit(i)) inflight++;
            else fail = true;
          }
        }
      }
      io_destroy(ctx);
    });
  for (auto& t : ts) t.join();
  const double secs = us_since(t0) / 1e6;
  close(fd);
  TORCH_CHECK(!fail, "nvme_read: an I/O failed (", strerror(errno), ")");
  std::vector<float> all;
  for (auto& l : lat) all.insert(all.end(), l.begin(), l.end());
  std::sort(all.begin(), all.end());
  auto pct = [&](double p) { return all.empty() ? 0.0 : double(all[size_t(p * (all.size() - 1))]); };
  return {{"gbps", double(useful.load()) / secs / 1e9},
          {"device_gbps", double(device.load()) / secs / 1e9},
          {"reads", double(reads.load())},
          {"lat_p50_us", pct(0.5)},
          {"lat_p99_us", pct(0.99)},
          {"us_per_read", secs * 1e6 / double(std::max<int64_t>(reads.load(), 1))}};
}

// ------------------------------------------------------------------------------------------------ ggml helpers

torch::Tensor quantize(int64_t type, torch::Tensor src, std::optional<torch::Tensor> imatrix, int64_t threads) {
  TORCH_CHECK(src.scalar_type() == torch::kFloat && src.dim() == 2 && src.is_contiguous());
  const int64_t rows = src.size(0), cols = src.size(1);
  const auto t = ggml_type(type);
  const size_t row = ggml_row_size(t, cols);
  auto out = torch::empty({rows, int64_t(row)}, torch::kUInt8);
  const float* im = imatrix ? imatrix->data_ptr<float>() : nullptr;
  ggml_quantize_init(t);
  std::vector<std::thread> ts;
  for (int64_t i = 0; i < threads; ++i)
    ts.emplace_back([&, i] {
      const int64_t a = rows * i / threads, b = rows * (i + 1) / threads;
      if (b > a)
        ggml_quantize_chunk(t, src.data_ptr<float>() + a * cols, out.data_ptr<uint8_t>() + a * row, 0, b - a, cols, im);
    });
  for (auto& th : ts) th.join();
  return out;
}

torch::Tensor dequantize(int64_t type, torch::Tensor src, int64_t rows, int64_t cols) {
  const auto t = ggml_type(type);
  TORCH_CHECK(src.numel() == int64_t(ggml_row_size(t, cols)) * rows && src.is_contiguous());
  auto out = torch::empty({rows, cols}, torch::kFloat);
  ggml_get_type_traits(t)->to_float(src.data_ptr<uint8_t>(), out.data_ptr<float>(), rows * cols);
  return out;
}

std::map<std::string, int> cpu_features() {
  return {{"avx2", ggml_cpu_has_avx2()},           {"fma", ggml_cpu_has_fma()},
          {"f16c", ggml_cpu_has_f16c()},           {"avx512", ggml_cpu_has_avx512()},
          {"avx512_vbmi", ggml_cpu_has_avx512_vbmi()}, {"avx512_vnni", ggml_cpu_has_avx512_vnni()},
          {"avx512_bf16", ggml_cpu_has_avx512_bf16()}, {"avx_vnni", ggml_cpu_has_avx_vnni()}};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ggml_cpu_init();
  py::class_<HostArena>(m, "HostArena")
      .def(py::init<int64_t>())
      .def("tensor", &HostArena::tensor, py::keep_alive<0, 1>())
      .def("host_ptr", &HostArena::host_ptr)
      .def("device_ptr", &HostArena::device_ptr)
      .def("bytes", &HostArena::bytes);
  py::class_<x0::ExpertClass>(m, "ExpertClass")
      .def(py::init<int, int, int, int>())
      .def_readonly("bytes", &x0::ExpertClass::bytes)
      .def_readonly("down_off", &x0::ExpertClass::down_off)
      .def_readonly("gu_row", &x0::ExpertClass::gu_row)
      .def_readonly("down_row", &x0::ExpertClass::down_row);
  py::class_<Cpu>(m, "Cpu")
      .def(py::init<std::vector<int64_t>>())
      .def("threads", &Cpu::threads)
      .def("time", &Cpu::time, py::call_guard<py::gil_scoped_release>())
      .def("run", &Cpu::run, py::call_guard<py::gil_scoped_release>());
  py::class_<Handshake>(m, "Handshake")
      .def(py::init<const x0::ExpertClass&, int64_t, int64_t, int64_t, int64_t, std::vector<int64_t>>())
      .def("start", &Handshake::start)
      .def("stop", &Handshake::stop, py::call_guard<py::gil_scoped_release>())
      .def("set_host_plan", &Handshake::set_host_plan)
      .def("tasks", &Handshake::tasks)
      .def("compute_ns", &Handshake::compute_ns)
      .def("plan", &Handshake::plan)
      .def("route", &Handshake::route)
      .def("signal", &Handshake::signal)
      .def("wait", &Handshake::wait)
      .def("consume", &Handshake::consume)
      .def("y", &Handshake::y)
      .def("x", &Handshake::x)
      .def("ids", &Handshake::ids);
  m.def("gather", [](torch::Tensor blobs, int64_t n, int ctas, int threads, int unroll) {
    gather_on(stream(), blob_table(blobs), int(n), ctas, threads, unroll);
  });
  m.def("memcpy_each", [](torch::Tensor t) { memcpy_each_on(stream(), host_table(t), t.size(0)); });
  m.def("memcpy_batch", [](torch::Tensor t) { memcpy_batch_on(stream(), host_table(t), t.size(0)); });
  m.def("busy", [](int64_t ns) {
    x0::launch_busy(stream(), uint64_t(ns));
    cuda_check(cudaGetLastError(), "busy");
  });
  m.def("contend", &contend, py::call_guard<py::gil_scoped_release>());
  m.def("split_once", &split_once, py::call_guard<py::gil_scoped_release>());
  m.def("nvme_fill", &nvme_fill, py::call_guard<py::gil_scoped_release>());
  m.def("nvme_read", &nvme_read, py::call_guard<py::gil_scoped_release>());
  m.def("quantize", &quantize, py::call_guard<py::gil_scoped_release>());
  m.def("dequantize", &dequantize);
  m.def("cpu_features", &cpu_features);
  m.def("kReq", [] { return kReq; });
  m.def("kPlanned", [] { return kPlanned; });
  m.def("kDone", [] { return kDone; });
}
