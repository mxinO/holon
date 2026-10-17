// `cpu_task@1` (docs/v1-offload.md D6, spike S6): a persistent native worker that CUDA graphs hand CPU work to, one
// 32-bit word per slot on its own 64-byte line of mapped pinned memory:
//
//   GPU (in the graph):  word := REQ          … independent GPU work …          wait until word >= DONE
//   CPU (coordinator):   sees REQ (acquire) → runs the slot's task on the pool → word := DONE or FAILED (release)
//
// The GPU writes REQ only after it saw the slot's answer to the previous request (stream order), the CPU answers only
// after it saw REQ, so one word needs no reset and both immediates can be baked into a graph. Two GPU flavours: `spin`
// (1-thread kernels: a system-scope release store, an acquire spin; S6's default) and `memop` (cuStreamWriteValue32 /
// cuStreamWaitValue32: no SM held, for display GPUs).
//
// Fail-open (D6): a task error answers FAILED; the spin wait gives up after `timeout_ns`, which covers a dead or hung
// worker; the memop wait cannot time out, so a watchdog thread answers a request left unanswered that long. Either
// sets a sticky flag word (TASK, TIMEOUT), and once it is set every later wait returns at once and the coordinator
// answers every request FAILED without running it: a graph never waits forever and a drain always ends. Each wait ORs
// the flag (and its slot's failure) into the caller's `errors` word, which the caller checks at the step's commit.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda/atomic>
#include <pthread.h>
#include <torch/extension.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

#include "holon_cpu_task.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

constexpr uint32_t REQ = 1, DONE = 2, FAILED = 3, ABANDONED = 4;   // a slot's word (0: never requested); ABANDONED:
                                                                  // answered by the watchdog or at a stop
constexpr uint32_t TASK = 1, TIMEOUT = 2;               // the flag's and `errors`' bits
constexpr int STRIDE = 16;                              // uint32 words per 64-byte line

using Clock = std::chrono::steady_clock;
using SysRef = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void k_signal(uint32_t * word) { SysRef(*word).store(REQ, cuda::memory_order_release); }

// `spin`: wait for the answer (or the flag, or the time bound); else (after a memop wait) only report.
__global__ void k_wait(uint32_t * word, uint32_t * flag, int32_t * errors, uint64_t timeout_ns, int spin) {
    SysRef w(*word), f(*flag);
    uint32_t bits = f.load(cuda::memory_order_acquire);
    if (spin && !bits) {
        const uint64_t t0 = globaltimer();
        while (w.load(cuda::memory_order_acquire) < DONE) {
            if ((bits = f.load(cuda::memory_order_relaxed)) != 0) {
                break;
            }
            if (globaltimer() - t0 > timeout_ns) {
                bits = TIMEOUT;
                f.fetch_or(TIMEOUT, cuda::memory_order_relaxed);
                break;
            }
            __nanosleep(64);
        }
    }
    if (w.load(cuda::memory_order_acquire) == FAILED) {
        bits |= TASK;
    }
    if (bits) {
        atomicOr(errors, int(bits));
    }
}

inline void relax() {
#if defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

void pin_to(int core) {
    if (core < 0) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// S6's backoff: hot polling while requests flow, 50 µs naps after 100 ms without one.
class Backoff {
  public:
    void reset() {
        last_ = Clock::now();
        polls_ = 0;
        dozing_ = false;
    }
    void wait() {
        if (!dozing_) {
            if ((++polls_ & 1023u) != 0 || Clock::now() - last_ < std::chrono::milliseconds(100)) {
                relax();
                return;
            }
            dozing_ = true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

  private:
    Clock::time_point last_ = Clock::now();
    uint32_t polls_ = 0;
    bool dozing_ = false;
};

// Persistent spinning threads; the caller of `run` is thread 0. A task runs on every thread; they meet at `barrier`.
class Pool {
  public:
    explicit Pool(const std::vector<int> & cores) : n_(int(cores.size())) {
        for (int i = 1; i < n_; ++i) {
            threads_.emplace_back([this, i, c = cores[i]] {
                pin_to(c);
                loop(i);
            });
        }
    }
    ~Pool() {
        stop_.store(true, std::memory_order_relaxed);
        for (auto & t : threads_) {
            t.join();
        }
    }
    int size() const { return n_; }

    int run(holon_cpu_task_fn fn, void * arg) {
        fn_ = fn;
        arg_ = arg;
        result_.store(0, std::memory_order_relaxed);
        pending_.store(n_ - 1, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        call(0);
        while (pending_.load(std::memory_order_acquire) != 0) {
            relax();
        }
        return result_.load(std::memory_order_acquire);
    }

    static void barrier(const holon_cpu_env * env) {
        auto * self = static_cast<Pool *>(env->pool);
        if (self->n_ == 1) {
            return;
        }
        const uint32_t g = self->bar_gen_.load(std::memory_order_acquire);
        if (self->bar_count_.fetch_add(1, std::memory_order_acq_rel) == uint32_t(self->n_ - 1)) {
            self->bar_count_.store(0, std::memory_order_relaxed);
            self->bar_gen_.fetch_add(1, std::memory_order_release);
        } else {
            while (self->bar_gen_.load(std::memory_order_acquire) == g) {
                relax();
            }
        }
    }

  private:
    void call(int i) {
        const holon_cpu_env env{i, n_, &Pool::barrier, this};
        int r = 1;
        try {
            r = fn_(arg_, &env);
        } catch (...) {                                // a task never takes the worker down
        }
        if (r) {
            result_.store(1, std::memory_order_relaxed);
        }
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
            call(i);
            pending_.fetch_sub(1, std::memory_order_release);
            b.reset();
        }
    }

    const int n_;
    holon_cpu_task_fn fn_ = nullptr;
    void * arg_ = nullptr;
    std::vector<std::thread> threads_;
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> pending_{0};
    alignas(64) std::atomic<int> result_{0};
    alignas(64) std::atomic<uint32_t> bar_count_{0};
    alignas(64) std::atomic<uint32_t> bar_gen_{0};
    alignas(64) std::atomic<bool> stop_{false};
};

using Task = std::tuple<int64_t, int64_t, int64_t>;     // (ABI version, function, argument): the task ABI's handle

// Driver entry points through the runtime, so the module does not link libcuda.
struct Driver {
    decltype(&cuStreamWriteValue32_v2) write32 = nullptr;
    decltype(&cuStreamWaitValue32_v2) wait32 = nullptr;
};

void cu_check(CUresult r, const char * what) { TORCH_CHECK(r == CUDA_SUCCESS, "cpu_task: ", what, ": CUresult ", int(r)); }

const Driver & driver() {
    static const Driver d = [] {
        Driver d;
        auto get = [](const char * name, void ** fn) {
            cudaDriverEntryPointQueryResult q;
            C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q));
            TORCH_CHECK(q == cudaDriverEntryPointSuccess && *fn != nullptr, "cpu_task: no driver symbol ", name);
        };
        get("cuStreamWriteValue32", reinterpret_cast<void **>(&d.write32));
        get("cuStreamWaitValue32", reinterpret_cast<void **>(&d.wait32));
        return d;
    }();
    return d;
}

class Worker {
  public:
    // `host` and `device`: the same mapped pinned memory, (slots + 1) lines: a word per slot, then the flag.
    Worker(int64_t host, int64_t device, std::vector<Task> tasks, std::vector<int64_t> cores, int64_t timeout_ns,
           bool memop)
        : words_(reinterpret_cast<uint32_t *>(host)), dev_(reinterpret_cast<uint32_t *>(device)),
          timeout_ns_(timeout_ns), memop_(memop) {
        TORCH_CHECK(!cores.empty(), "cpu_task: a worker needs at least one core");
        for (const auto & [abi, fn, arg] : tasks) {
            TORCH_CHECK(abi == HOLON_CPU_TASK_ABI, "cpu_task: a task of ABI ", abi, ", the worker's is ",
                        HOLON_CPU_TASK_ABI);
            tasks_.push_back({reinterpret_cast<holon_cpu_task_fn>(fn), reinterpret_cast<void *>(arg)});
        }
        for (size_t s = 0; s <= tasks_.size(); ++s) {
            word(s).store(0, __ATOMIC_RELAXED);
        }
        if (memop) {
            driver();
        }
        cores_.assign(cores.begin(), cores.end());
        coordinator_ = std::thread([this] { serve(); });
        if (memop) {
            watchdog_ = std::thread([this] { watch(); });
        }
    }
    ~Worker() { stop(); }

    void stop() {
        if (!running_.exchange(false)) {
            return;
        }
        coordinator_.join();
        if (watchdog_.joinable()) {
            watchdog_.join();
        }
        for (size_t s = 0; s < tasks_.size(); ++s) {           // nothing is in flight after a drain; never strand one
            uint32_t req = REQ;
            word(s).compare_exchange_strong(req, ABANDONED, __ATOMIC_RELEASE);
        }
    }

    // Tests: the coordinator stops answering (a dead worker), the watchdog lives on.
    void kill() {
        dead_.store(true, std::memory_order_release);
    }

    int64_t flag() const { return word(tasks_.size()).load(__ATOMIC_ACQUIRE); }
    int64_t served() const { return served_.load(std::memory_order_relaxed); }

    void signal(int64_t slot) {
        check(slot);
        const cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
        if (memop_) {
            cu_check(driver().write32(s, reinterpret_cast<CUdeviceptr>(dev_ + slot * STRIDE), REQ,
                                      CU_STREAM_WRITE_VALUE_DEFAULT), "cuStreamWriteValue32");
        } else {
            k_signal<<<1, 1, 0, s>>>(dev_ + slot * STRIDE);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    void wait(int64_t slot, at::Tensor & errors) {
        check(slot);
        TORCH_CHECK(errors.is_cuda() && errors.scalar_type() == at::kInt && errors.numel() == 1, "errors: int32 [1]");
        const c10::cuda::CUDAGuard guard(errors.device());
        const cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
        uint32_t * w = dev_ + slot * STRIDE;
        if (memop_) {
            cu_check(driver().wait32(s, reinterpret_cast<CUdeviceptr>(w), DONE, CU_STREAM_WAIT_VALUE_GEQ),
                     "cuStreamWaitValue32");
        }
        k_wait<<<1, 1, 0, s>>>(w, dev_ + tasks_.size() * STRIDE, errors.data_ptr<int32_t>(), uint64_t(timeout_ns_),
                               !memop_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

  private:
    struct Bound {
        holon_cpu_task_fn fn;
        void * arg;
    };

    // The words are shared with the GPU (mapped memory): GCC's atomic builtins on them, as std::atomic_ref would.
    struct Word {
        uint32_t * p;
        uint32_t load(int order) const { return __atomic_load_n(p, order); }
        void store(uint32_t v, int order) const { __atomic_store_n(p, v, order); }
        void fetch_or(uint32_t v, int order) const { __atomic_fetch_or(p, v, order); }
        bool compare_exchange_strong(uint32_t & want, uint32_t v, int order) const {
            return __atomic_compare_exchange_n(p, &want, v, false, order, __ATOMIC_RELAXED);
        }
    };
    Word word(size_t s) const { return Word{words_ + s * STRIDE}; }

    void check(int64_t slot) const {
        TORCH_CHECK(slot >= 0 && slot < int64_t(tasks_.size()), "cpu_task: slot ", slot, " of ", tasks_.size());
    }

    // The coordinator: pool thread 0, answering every request (failed at once while the flag is set).
    void serve() {
        pin_to(cores_[0]);
        Pool pool(cores_);
        Backoff b;
        b.reset();
        while (running_.load(std::memory_order_relaxed)) {
            bool any = false;
            for (size_t s = 0; s < tasks_.size() && !dead_.load(std::memory_order_relaxed); ++s) {
                if (word(s).load(__ATOMIC_ACQUIRE) != REQ) {
                    continue;
                }
                any = true;
                const bool failed = flag() != 0 || pool.run(tasks_[s].fn, tasks_[s].arg) != 0;
                if (failed) {
                    word(tasks_.size()).fetch_or(TASK, __ATOMIC_RELAXED);
                }
                uint32_t req = REQ;                        // not answered meanwhile by the watchdog
                word(s).compare_exchange_strong(req, failed ? FAILED : DONE, __ATOMIC_RELEASE);
                served_.fetch_add(1, std::memory_order_relaxed);
            }
            if (any) {
                b.reset();
            } else {
                b.wait();
            }
        }
    }

    // memop only: abandons a request older than the time bound (or any once the flag is set).
    void watch() {
        std::vector<Clock::time_point> since(tasks_.size());
        std::vector<bool> pending(tasks_.size());
        while (running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            const auto now = Clock::now();
            for (size_t s = 0; s < tasks_.size(); ++s) {
                if (word(s).load(__ATOMIC_ACQUIRE) != REQ) {
                    pending[s] = false;
                    continue;
                }
                if (!pending[s]) {
                    pending[s] = true;
                    since[s] = now;
                }
                if (flag() || now - since[s] > std::chrono::nanoseconds(timeout_ns_)) {
                    word(tasks_.size()).fetch_or(TIMEOUT, __ATOMIC_RELAXED);
                    uint32_t req = REQ;
                    word(s).compare_exchange_strong(req, ABANDONED, __ATOMIC_RELEASE);
                }
            }
        }
    }

    uint32_t * words_;
    uint32_t * dev_;
    const int64_t timeout_ns_;
    const bool memop_;
    std::vector<Bound> tasks_;
    std::vector<int> cores_;
    std::atomic<bool> running_{true}, dead_{false};
    std::atomic<int64_t> served_{0};
    std::thread coordinator_, watchdog_;
};

// A task for the protocol's tests: thread 0 sleeps `ns`, every thread counts itself before a barrier and thread 0
// checks the count after it (into `out[1]`), `out[0]` counts the runs; it fails when asked.
struct Probe {
    int64_t ns = 0;
    int fail = 0;
    std::atomic<int64_t> arrived{0};
    int64_t * out = nullptr;

    static int run(void * arg, const holon_cpu_env * env) {
        auto * p = static_cast<Probe *>(arg);
        if (env->thread == 0) {
            p->arrived.store(0);
        }
        env->barrier(env);
        p->arrived.fetch_add(1);
        env->barrier(env);
        if (env->thread == 0) {
            if (p->ns) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(p->ns));
            }
            p->out[0] += 1;
            p->out[1] = p->arrived.load();
        }
        return p->fail;
    }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Worker>(m, "Worker")
        .def(pybind11::init<int64_t, int64_t, std::vector<Task>, std::vector<int64_t>, int64_t, bool>(),
             "host and device addresses of (slots + 1) 64-byte lines of mapped pinned memory, a task per slot (ABI, "
             "function, argument), the cores (the first runs the coordinator), the wait's time bound, memop or spin")
        .def("signal", &Worker::signal, "on the current stream: the slot's task runs once the stream reaches here")
        .def("wait", &Worker::wait, "on the current stream: later work waits for the slot's answer (fail-open)")
        .def("flag", &Worker::flag, "the sticky error bits: 1 a task failed, 2 a wait timed out")
        .def("served", &Worker::served, "requests answered so far")
        .def("stop", &Worker::stop, "join the threads (after a drain); a request still open is answered failed")
        .def("kill", &Worker::kill, "tests: the coordinator stops answering");
    pybind11::class_<Probe>(m, "Probe")
        .def(pybind11::init([](int64_t ns, int fail, at::Tensor out) {
                 TORCH_CHECK(out.scalar_type() == at::kLong && out.numel() == 2 && !out.is_cuda(), "out: int64 [2]");
                 auto * p = new Probe();
                 p->ns = ns;
                 p->fail = fail;
                 p->out = out.data_ptr<int64_t>();
                 return p;
             }),
             pybind11::keep_alive<1, 4>())
        .def("handle", [](Probe & p) {
            return Task{HOLON_CPU_TASK_ABI, reinterpret_cast<int64_t>(&Probe::run), reinterpret_cast<int64_t>(&p)};
        });
}
