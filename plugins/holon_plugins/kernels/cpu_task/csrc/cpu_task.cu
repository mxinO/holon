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
// sets a sticky flag (TASK, TIMEOUT), and once one is set every later wait returns at once and the coordinator
// answers every request FAILED without running it: a graph never waits forever and a drain always ends. Each wait ORs
// the flag (and its slot's failure) into the caller's `errors` word, which the caller checks at the step's commit.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda/atomic>
#include <pthread.h>
#include <unistd.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
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

// `spin`: wait for the answer (or a flag, or the time bound); else (after a memop wait) only report. `flags` is the
// flags' line: the CPU's word (TASK, the watchdog's TIMEOUT), then the GPU's (a spin wait's TIMEOUT): one writer each.
__global__ void k_wait(uint32_t * word, uint32_t * flags, int32_t * errors, uint64_t timeout_ns, int spin) {
    SysRef w(*word), f(flags[0]), g(flags[1]);
    uint32_t bits = f.load(cuda::memory_order_acquire) | g.load(cuda::memory_order_acquire);
    if (spin && !bits) {
        const uint64_t t0 = globaltimer();
        while (w.load(cuda::memory_order_acquire) < DONE) {
            if ((bits = f.load(cuda::memory_order_relaxed) | g.load(cuda::memory_order_relaxed)) != 0) {
                break;
            }
            if (globaltimer() - t0 > timeout_ns) {
                bits = TIMEOUT;
                g.store(TIMEOUT, cuda::memory_order_relaxed);
                break;
            }
            __nanosleep(64);
        }
    }
    const uint32_t answer = w.load(cuda::memory_order_acquire);
    if (answer > DONE) {                               // failed, or abandoned: its rows are not the task's
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

// What the worker's threads use: shared with them, so a coordinator stuck in a task can be left behind at a stop.
struct State {
    uint32_t * words;                                  // (slots + 1) lines; the last: the CPU's flag, the GPU's
    std::vector<std::pair<holon_cpu_task_fn, void *>> tasks;
    std::vector<int> cores;
    int64_t timeout_ns;
    std::atomic<bool> running{true}, dead{false}, exited{false};
    std::atomic<int64_t> served{0};

    Word word(size_t s) const { return Word{words + s * STRIDE}; }
    Word cpu_flag() const { return word(tasks.size()); }
    Word gpu_flag() const { return Word{words + tasks.size() * STRIDE + 1}; }
    uint32_t flag() const { return cpu_flag().load(__ATOMIC_ACQUIRE) | gpu_flag().load(__ATOMIC_ACQUIRE); }
};

// The coordinator: pool thread 0, answering every request (failed at once while a flag is set).
void serve(std::shared_ptr<State> st) {
    pin_to(st->cores[0]);
    {
        Pool pool(st->cores);
        Backoff b;
        b.reset();
        while (st->running.load(std::memory_order_relaxed)) {
            bool any = false;
            for (size_t s = 0; s < st->tasks.size() && !st->dead.load(std::memory_order_relaxed); ++s) {
                if (st->word(s).load(__ATOMIC_ACQUIRE) != REQ) {
                    continue;
                }
                any = true;
                const auto [fn, arg] = st->tasks[s];
                const bool failed = st->flag() != 0 || pool.run(fn, arg) != 0;
                if (failed) {
                    st->cpu_flag().fetch_or(TASK, __ATOMIC_RELAXED);
                }
                uint32_t req = REQ;                        // not answered meanwhile by the watchdog
                st->word(s).compare_exchange_strong(req, failed ? FAILED : DONE, __ATOMIC_RELEASE);
                st->served.fetch_add(1, std::memory_order_relaxed);
            }
            if (any) {
                b.reset();
            } else {
                b.wait();
            }
        }
    }
    st->exited.store(true, std::memory_order_release);
}

// memop only: abandons a request older than the time bound (or any once a flag is set).
void watch(std::shared_ptr<State> st) {
    std::vector<Clock::time_point> since(st->tasks.size());
    std::vector<bool> pending(st->tasks.size());
    while (st->running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto now = Clock::now();
        for (size_t s = 0; s < st->tasks.size(); ++s) {
            if (st->word(s).load(__ATOMIC_ACQUIRE) != REQ) {
                pending[s] = false;
                continue;
            }
            if (!pending[s]) {
                pending[s] = true;
                since[s] = now;
            }
            if (st->flag() || now - since[s] > std::chrono::nanoseconds(st->timeout_ns)) {
                st->cpu_flag().fetch_or(TIMEOUT, __ATOMIC_RELAXED);
                uint32_t req = REQ;
                st->word(s).compare_exchange_strong(req, ABANDONED, __ATOMIC_RELEASE);
            }
        }
    }
}

class Worker {
  public:
    // `host` and `device`: the same mapped pinned memory, (slots + 1) lines: a word per slot, then the flags.
    Worker(int64_t host, int64_t device, std::vector<Task> tasks, std::vector<int64_t> cores, int64_t timeout_ns,
           bool memop)
        : st_(std::make_shared<State>()), dev_(reinterpret_cast<uint32_t *>(device)), memop_(memop) {
        TORCH_CHECK(!cores.empty(), "cpu_task: a worker needs at least one core");
        st_->words = reinterpret_cast<uint32_t *>(host);
        st_->timeout_ns = timeout_ns;
        for (const auto & [abi, fn, arg] : tasks) {
            TORCH_CHECK(abi == HOLON_CPU_TASK_ABI, "cpu_task: a task of ABI ", abi, ", the worker's is ",
                        HOLON_CPU_TASK_ABI);
            st_->tasks.emplace_back(reinterpret_cast<holon_cpu_task_fn>(fn), reinterpret_cast<void *>(arg));
        }
        for (size_t s = 0; s <= st_->tasks.size(); ++s) {
            st_->word(s).store(0, __ATOMIC_RELAXED);
        }
        st_->gpu_flag().store(0, __ATOMIC_RELAXED);
        if (memop) {
            driver();
        }
        st_->cores.assign(cores.begin(), cores.end());
        coordinator_ = std::thread(serve, st_);
        if (memop) {
            watchdog_ = std::thread(watch, st_);
        }
    }
    ~Worker() { stop(); }      // (a coordinator left behind must find its words and tasks: the owner keeps them)

    // After a drain. A coordinator still in a task after the time bound (a task that never returns) is left behind
    // with its state, so a close always ends; a request still open is abandoned (a wait reads it as an error).
    bool stop() {
        if (!st_->running.exchange(false)) {
            return false;
        }
        const auto until = Clock::now() + std::chrono::nanoseconds(st_->timeout_ns);
        while (!st_->exited.load(std::memory_order_acquire) && Clock::now() < until) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        const bool left = !st_->exited.load(std::memory_order_acquire);
        if (left) {
            coordinator_.detach();
        } else {
            coordinator_.join();
        }
        if (watchdog_.joinable()) {
            watchdog_.join();
        }
        for (size_t s = 0; s < st_->tasks.size(); ++s) {
            uint32_t req = REQ;
            st_->word(s).compare_exchange_strong(req, ABANDONED, __ATOMIC_RELEASE);
        }
        return left;
    }

    // Tests: the coordinator stops answering (a dead worker), the watchdog lives on.
    void kill() { st_->dead.store(true, std::memory_order_release); }

    int64_t flag() const { return st_->flag(); }
    int64_t served() const { return st_->served.load(std::memory_order_relaxed); }

    void signal(int64_t slot, const at::Tensor & like) {
        check(slot);
        const c10::cuda::CUDAGuard guard(like.device());
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
        k_wait<<<1, 1, 0, s>>>(w, dev_ + st_->tasks.size() * STRIDE, errors.data_ptr<int32_t>(),
                               uint64_t(st_->timeout_ns), !memop_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

  private:
    void check(int64_t slot) const {
        TORCH_CHECK(slot >= 0 && slot < int64_t(st_->tasks.size()), "cpu_task: slot ", slot, " of ",
                    st_->tasks.size());
    }

    std::shared_ptr<State> st_;
    uint32_t * dev_;
    const bool memop_;
    std::thread coordinator_, watchdog_;
};

// The generic copy task (F9, docs/v1-offload.md D3, D6): the pieces a request table names, read from their
// descriptors' extents into host units, by the I/O worker's threads (each thread its own blocking reads: two threads
// keep two reads in flight, M7). A descriptor row is (fd, direct, offset, nbytes) — nbytes 0 past a piece's extents —;
// an O_DIRECT descriptor is read as its aligned superset into the thread's aligned bounce buffer, then copied. The
// request table (`[1 + F, 2]`: n, then (piece, unit)) is the GPU's, in mapped memory: copied once per run.
struct CopyRole {
    const int64_t * desc;                             // [P, X, 4]
    int64_t pieces, extents;
    uint8_t * units;                                   // [U, unit]
    int64_t count, unit;
    const int32_t * req;                               // [1 + F, 2]
    int64_t width;
};

class Copy {
  public:
    static constexpr int64_t ALIGN = 4096;
    static constexpr int MAX_THREADS = 64;

    explicit Copy(std::vector<CopyRole> roles) : roles_(std::move(roles)) {
        int64_t most = 0;
        for (const auto & r : roles_) {
            for (int64_t i = 0; i < r.pieces * r.extents; ++i) {
                most = std::max(most, r.desc[i * 4 + 3]);
            }
        }
        bounce_bytes_ = (most + 3 * ALIGN) / ALIGN * ALIGN;
    }
    ~Copy() {
        for (void * b : bounce_) {
            std::free(b);
        }
    }

    static int run(void * arg, const holon_cpu_env * env) {
        auto * c = static_cast<Copy *>(arg);
        if (env->thread == 0) {
            c->err_.store(c->prepare(env->threads), std::memory_order_relaxed);
        }
        env->barrier(env);
        if (c->err_.load(std::memory_order_relaxed)) {
            return c->err_.load(std::memory_order_relaxed);
        }
        int err = 0;
        for (size_t i; (i = c->next_.fetch_add(1, std::memory_order_relaxed)) < c->work_.size();) {
            const auto [role, piece, unit] = c->work_[i];
            const CopyRole & r = c->roles_[role];
            uint8_t * dst = r.units + unit * r.unit;
            for (int64_t x = 0; x < r.extents && !err; ++x) {
                const int64_t * d = r.desc + (piece * r.extents + x) * 4;
                if (d[3]) {
                    err = c->read(int(d[0]), d[1] != 0, d[2], d[3], dst, c->bounce_[env->thread]);
                    dst += d[3];
                }
            }
        }
        return err;
    }

  private:
    int prepare(int threads) {
        if (threads > MAX_THREADS) {
            return 1;
        }
        while (int(bounce_.size()) < threads) {
            void * b = nullptr;
            if (posix_memalign(&b, ALIGN, size_t(bounce_bytes_))) {
                return 1;
            }
            bounce_.push_back(b);
        }
        work_.clear();
        for (size_t r = 0; r < roles_.size(); ++r) {
            const CopyRole & role = roles_[r];
            const int32_t n = role.req[0];
            if (n < 0 || n > role.width) {
                return 2;
            }
            for (int32_t i = 0; i < n; ++i) {
                const int32_t piece = role.req[2 + 2 * i], unit = role.req[3 + 2 * i];
                if (piece < 0 || piece >= role.pieces || unit < 0 || unit >= role.count) {
                    return 2;
                }
                work_.emplace_back(int(r), piece, unit);
            }
        }
        next_.store(0, std::memory_order_relaxed);
        return 0;
    }

    // `nbytes` at `offset` of `fd` into `dst`: pread loops (a short read is retried; EOF before the end is an error).
    int read(int fd, bool direct, int64_t offset, int64_t nbytes, uint8_t * dst, void * bounce) const {
        const int64_t lo = direct ? offset / ALIGN * ALIGN : offset;
        const int64_t hi = direct ? (offset + nbytes + ALIGN - 1) / ALIGN * ALIGN : offset + nbytes;
        uint8_t * buf = direct ? static_cast<uint8_t *>(bounce) : dst;
        int64_t got = 0;
        while (got < hi - lo) {
            const ssize_t n = pread(fd, buf + got, size_t(hi - lo - got), lo + got);
            if (n <= 0) {
                break;                                // the file's end (an O_DIRECT tail), or an error
            }
            got += n;
            if (direct && got % ALIGN) {
                break;
            }
        }
        if (got < offset + nbytes - lo) {
            return 3;
        }
        if (direct) {
            std::memcpy(dst, buf + (offset - lo), size_t(nbytes));
        }
        return 0;
    }

    std::vector<CopyRole> roles_;
    std::vector<void *> bounce_;
    int64_t bounce_bytes_ = 0;
    std::vector<std::tuple<int, int64_t, int64_t>> work_;
    std::atomic<size_t> next_{0};
    std::atomic<int> err_{0};
};

// The copy task's handle over its tensors (kept alive with it).
struct CopyTask {
    std::vector<at::Tensor> keep;
    std::unique_ptr<Copy> copy;

    explicit CopyTask(const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> & roles) {
        std::vector<CopyRole> out;
        for (const auto & [desc, units, req] : roles) {
            TORCH_CHECK(!desc.is_cuda() && desc.scalar_type() == at::kLong && desc.dim() == 3 && desc.size(2) == 4 &&
                        desc.is_contiguous(), "copy: descriptors [P, X, 4] int64 on the host");
            TORCH_CHECK(!units.is_cuda() && units.scalar_type() == at::kByte && units.dim() == 2 &&
                        units.is_contiguous(), "copy: units [U, unit] uint8 on the host");
            TORCH_CHECK(!req.is_cuda() && req.scalar_type() == at::kInt && req.dim() == 2 && req.size(1) == 2 &&
                        req.is_contiguous(), "copy: requests [1 + F, 2] int32 on the host");
            keep.insert(keep.end(), {desc, units, req});
            out.push_back({desc.data_ptr<int64_t>(), desc.size(0), desc.size(1), units.data_ptr<uint8_t>(),
                           units.size(0), units.size(1), req.data_ptr<int32_t>(), req.size(0) - 1});
        }
        copy = std::make_unique<Copy>(std::move(out));
    }

    Task native() {
        return Task{HOLON_CPU_TASK_ABI, reinterpret_cast<int64_t>(&Copy::run), reinterpret_cast<int64_t>(copy.get())};
    }
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
        .def("signal", &Worker::signal, "on the current stream (of `like`'s device): the slot's task runs once the "
             "stream reaches here")
        .def("wait", &Worker::wait, "on the current stream: later work waits for the slot's answer (fail-open)")
        .def("flag", &Worker::flag, "the sticky error bits: 1 a task failed, 2 a wait timed out")
        .def("served", &Worker::served, "requests answered so far")
        .def("stop", &Worker::stop, "join the threads (after a drain); a request still open is abandoned; True: the "
             "coordinator, in a task past the time bound, was left behind (its words and tasks must outlive it)")
        .def("kill", &Worker::kill, "tests: the coordinator stops answering");
    pybind11::class_<CopyTask>(m, "CopyTask")
        .def(pybind11::init<const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> &>(),
             "per role: descriptors [P, X, 4] int64 (fd, direct, offset, nbytes), units [U, unit] uint8, requests "
             "[1 + F, 2] int32 (n, then (piece, unit)): the copy task (F9) over them")
        .def("native", &CopyTask::native, "the task ABI's handle")
        .def("run", [](CopyTask & t, int threads) {
            Pool pool(std::vector<int>(size_t(std::max(threads, 1)), -1));     // unpinned threads, for this run
            return pool.run(&Copy::run, t.copy.get());
        }, pybind11::call_guard<pybind11::gil_scoped_release>(), "run now on `threads` threads: 0, or an error");
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
