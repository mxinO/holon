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
#include <nvtx3/nvToolsExt.h>
#include <linux/aio_abi.h>
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#endif
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <torch/extension.h>

#include <algorithm>
#include <cerrno>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
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

class IoEngine;

// What the worker's threads use: shared with them, so a coordinator stuck in a task can be left behind at a stop.
struct State {
    uint32_t * words;                                  // (slots + 1) lines; the last: the CPU's flag, the GPU's
    std::vector<std::pair<holon_cpu_task_fn, void *>> tasks;
    std::vector<int> cores;
    int64_t timeout_ns;
    std::atomic<bool> running{true}, dead{false}, exited{false};
    std::atomic<int64_t> served{0};
    std::shared_ptr<IoEngine> io;                     // the `io` pool's reads (completion-driven), or none

    Word word(size_t s) const { return Word{words + s * STRIDE}; }
    Word cpu_flag() const { return word(tasks.size()); }
    Word gpu_flag() const { return Word{words + tasks.size() * STRIDE + 1}; }
    uint32_t flag() const { return cpu_flag().load(__ATOMIC_ACQUIRE) | gpu_flag().load(__ATOMIC_ACQUIRE); }
};

// The generic copy task (F9, docs/v1-offload.md D3, D6; docs/v1-glm.md D8f): the pieces a request table names, read
// from their descriptors' extents into host rows. A descriptor row is (fd, direct, offset, nbytes) — nbytes 0 past a
// piece's extents. A role's rows are either canonical units (each piece's extents back to back: an O_DIRECT extent
// read as its aligned superset into a bounce buffer, then copied, unless offset, length and destination are all
// aligned) or landing rows (`runs` given: each O_DIRECT extent's aligned superset read straight into the row at the
// next aligned offset, where (offset in the row, nbytes) of its bytes go to `runs[row, x]`; no copy). A landing role
// with a keep pool takes requests (piece, row, keep): a piece with keep ≥ 0 is copied, canonical, into keep unit
// `keep` once its extents are in. The request table (`[1 + F, 2 or 3]`: n, then the rows) is the GPU's, in mapped
// memory: read once per run. Two ways to run: `run` (every thread of a pool its own blocking reads) and the I/O
// worker's completion-driven loop (`IoEngine`), which plans the same extents.
struct CopyRole {
    const int64_t * desc;                             // [P, X, 4]
    int64_t pieces, extents;
    uint8_t * units;                                  // [U, unit]: canonical units, or landing rows
    int64_t count, unit;
    const int32_t * req;                              // [1 + F, cols]
    int64_t width, cols;
    int64_t * runs;                                   // landing rows: [U, X, 2]; nullptr: canonical units
    uint8_t * keep;                                   // a landing role's keep pool [K, keep_unit], or nullptr
    int64_t keep_count, keep_unit;
};

constexpr int64_t ALIGN = 4096;

// One extent's read: [lo, hi) of `fd` lands at `into`; `need` bytes from `lo` must come (an O_DIRECT tail may end at
// the file's end); a bounced extent (into = a bounce buffer) then copies `nbytes` from into + skip to `dst`.
struct Extent {
    int fd;
    bool direct, bounce;
    int64_t lo, hi, need, skip, nbytes;
    uint8_t * into;
    uint8_t * dst;
    int piece;                                        // its piece in the plan
};

// A planned piece: its keep copies (src, dst, n), run once its extents are in.
struct Piece {
    std::vector<std::tuple<const uint8_t *, uint8_t *, int64_t>> keep;
};

struct Plan {
    std::vector<Extent> extents;
    std::vector<Piece> pieces;
    int64_t bounce_bytes = 0;                         // the largest bounced superset
};

// The request tables' pieces as extents: 0, or an error (2: a bad request; 4: a landing row too small or unaligned).
inline int plan_copy(const std::vector<CopyRole> & roles, Plan & plan) {
    plan.extents.clear();
    plan.pieces.clear();
    for (const CopyRole & r : roles) {
        const int32_t n = r.req[0];
        if (n < 0 || n > r.width) {
            return 2;
        }
        for (int32_t i = 0; i < n; ++i) {
            const int32_t * row = r.req + r.cols * (1 + i);
            const int32_t piece = row[0], unit = row[1], keep = r.cols > 2 ? row[2] : -1;
            if (piece < 0 || piece >= r.pieces || unit < 0 || unit >= r.count ||
                (keep >= 0 && (r.keep == nullptr || keep >= r.keep_count))) {
                return 2;
            }
            const int at = int(plan.pieces.size());
            plan.pieces.emplace_back();
            uint8_t * base = r.units + int64_t(unit) * r.unit;
            int64_t off = 0, canon = 0;
            for (int64_t x = 0; x < r.extents; ++x) {
                const int64_t * d = r.desc + (int64_t(piece) * r.extents + x) * 4;
                const int64_t offset = d[2], nbytes = d[3];
                if (r.runs) {
                    r.runs[(int64_t(unit) * r.extents + x) * 2] = 0;
                    r.runs[(int64_t(unit) * r.extents + x) * 2 + 1] = 0;
                }
                if (!nbytes) {
                    continue;
                }
                Extent e{int(d[0]), d[1] != 0, false, 0, 0, 0, 0, nbytes, nullptr, nullptr, at};
                e.lo = e.direct ? offset / ALIGN * ALIGN : offset;
                e.hi = e.direct ? (offset + nbytes + ALIGN - 1) / ALIGN * ALIGN : offset + nbytes;
                e.need = offset + nbytes - e.lo;
                e.skip = offset - e.lo;
                if (r.runs) {                                          // a landing row: the superset in place
                    if (off + (e.hi - e.lo) > r.unit || (e.direct && reinterpret_cast<uintptr_t>(base + off) % ALIGN)) {
                        return 4;
                    }
                    e.into = base + off;
                    r.runs[(int64_t(unit) * r.extents + x) * 2] = off + e.skip;
                    r.runs[(int64_t(unit) * r.extents + x) * 2 + 1] = nbytes;
                    if (keep >= 0) {
                        plan.pieces[at].keep.emplace_back(base + off + e.skip,
                                                          r.keep + int64_t(keep) * r.keep_unit + canon, nbytes);
                    }
                    off += (e.hi - e.lo + ALIGN - 1) / ALIGN * ALIGN;
                } else {                                               // a canonical unit
                    uint8_t * dst = base + canon;
                    e.bounce = e.direct && (e.skip || (e.hi - e.lo) != nbytes ||
                                            reinterpret_cast<uintptr_t>(dst) % ALIGN);
                    e.into = e.bounce ? nullptr : dst;
                    e.dst = dst;
                    if (e.bounce) {
                        plan.bounce_bytes = std::max(plan.bounce_bytes, e.hi - e.lo);
                    }
                }
                canon += nbytes;
                plan.extents.push_back(e);
            }
            if (r.keep && keep >= 0 && canon > r.keep_unit) {
                return 4;
            }
        }
    }
    return 0;
}

// After its read: a bounced extent's copy; true when the read covered what the extent needs.
inline bool land(const Extent & e, int64_t got) {
    if (got < e.need) {
        return false;
    }
    if (e.bounce) {
        std::memcpy(e.dst, e.into + e.skip, size_t(e.nbytes));
    }
    return true;
}

inline void keep_copies(const Piece & p) {
    for (const auto & [src, dst, n] : p.keep) {
        std::memcpy(dst, src, size_t(n));
    }
}

class Copy {
  public:
    static constexpr int MAX_THREADS = 64;

    explicit Copy(std::vector<CopyRole> roles) : roles_(std::move(roles)) {
        for (const auto & r : roles_) {
            for (int64_t i = 0; i < r.pieces * r.extents; ++i) {
                most_ = std::max(most_, r.desc[i * 4 + 3]);
            }
        }
        most_ = (most_ + 3 * ALIGN) / ALIGN * ALIGN;     // a bounced superset at most
    }
    ~Copy() {
        for (void * b : bounce_) {
            std::free(b);
        }
    }
    const std::vector<CopyRole> & roles() const { return roles_; }
    int64_t most() const { return most_; }

    // On every thread of a pool: each thread its own blocking reads, piece by piece (keep copies after a piece).
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
        const auto & ext = c->plan_.extents;
        for (size_t p; !err && (p = c->next_.fetch_add(1, std::memory_order_relaxed)) < c->plan_.pieces.size();) {
            for (size_t i = c->first_[p]; i < c->first_[p + 1] && !err; ++i) {
                Extent e = ext[i];
                if (e.bounce) {
                    e.into = static_cast<uint8_t *>(c->bounce_[env->thread]);
                }
                if (!land(e, read(e))) {
                    err = 3;
                }
            }
            if (!err) {
                keep_copies(c->plan_.pieces[p]);
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
            if (posix_memalign(&b, ALIGN, size_t(most_))) {
                return 1;
            }
            bounce_.push_back(b);
        }
        if (const int err = plan_copy(roles_, plan_)) {
            return err;
        }
        first_.assign(plan_.pieces.size() + 1, plan_.extents.size());   // each piece's extents (in order)
        for (size_t i = plan_.extents.size(); i-- > 0;) {
            first_[size_t(plan_.extents[i].piece)] = i;
        }
        for (size_t p = plan_.pieces.size(); p-- > 0;) {
            first_[p] = std::min(first_[p], first_[p + 1]);
        }
        next_.store(0, std::memory_order_relaxed);
        return 0;
    }

    // [lo, hi) of the extent's file into `into`: pread loops (a short read is retried; EOF ends an O_DIRECT tail).
    static int64_t read(const Extent & e) {
        int64_t got = 0;
        while (got < e.hi - e.lo) {
            const ssize_t n = pread(e.fd, e.into + got, size_t(e.hi - e.lo - got), e.lo + got);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                break;
            }
            got += n;
            if (e.direct && got % ALIGN) {
                break;
            }
        }
        return got;
    }

    std::vector<CopyRole> roles_;
    std::vector<void *> bounce_;
    int64_t most_ = 0;
    Plan plan_;
    std::vector<size_t> first_;
    std::atomic<size_t> next_{0};
    std::atomic<int> err_{0};
};

// ---- the I/O worker's asynchronous reads (docs/v1-glm.md D8f): io_uring (raw syscalls, no SQPOLL), else Linux
// native AIO; one submitting thread keeps up to `depth` chunks in flight and reaps completions without blocking.
class Backend {
  public:
    virtual ~Backend() = default;
    virtual bool prep(int fd, uint8_t * buf, int64_t len, int64_t off, uint64_t tag) = 0;   // false: queue full
    virtual void submit() = 0;
    virtual void reap(std::vector<std::pair<uint64_t, int64_t>> & out) = 0;              // (tag, result or −errno)
    virtual const char * name() const = 0;
};

#if __has_include(<linux/io_uring.h>)
class Uring final : public Backend {
  public:
    explicit Uring(unsigned entries) {
        io_uring_params p{};
        fd_ = int(syscall(__NR_io_uring_setup, entries, &p));
        if (fd_ < 0) {
            return;
        }
        entries_ = p.sq_entries;
        size_t sq = p.sq_off.array + p.sq_entries * sizeof(uint32_t), cq = p.cq_off.cqes +
                    p.cq_entries * sizeof(io_uring_cqe);
        const bool single = p.features & IORING_FEAT_SINGLE_MMAP;
        if (single) {
            sq = cq = std::max(sq, cq);
        }
        sq_ = mmap(nullptr, sq, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
        cq_ = single ? sq_ : mmap(nullptr, cq, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_,
                                  IORING_OFF_CQ_RING);
        sqes_ = static_cast<io_uring_sqe *>(mmap(nullptr, p.sq_entries * sizeof(io_uring_sqe), PROT_READ | PROT_WRITE,
                                                 MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES));
        if (sq_ == MAP_FAILED || cq_ == MAP_FAILED || sqes_ == MAP_FAILED) {
            close(fd_);
            fd_ = -1;
            return;
        }
        sizes_[0] = sq;
        sizes_[1] = single ? 0 : cq;
        auto * s = static_cast<uint8_t *>(sq_);
        auto * c = static_cast<uint8_t *>(cq_);
        sq_head_ = reinterpret_cast<unsigned *>(s + p.sq_off.head);
        sq_tail_ = reinterpret_cast<unsigned *>(s + p.sq_off.tail);
        sq_mask_ = *reinterpret_cast<unsigned *>(s + p.sq_off.ring_mask);
        sq_array_ = reinterpret_cast<unsigned *>(s + p.sq_off.array);
        cq_head_ = reinterpret_cast<unsigned *>(c + p.cq_off.head);
        cq_tail_ = reinterpret_cast<unsigned *>(c + p.cq_off.tail);
        cq_mask_ = *reinterpret_cast<unsigned *>(c + p.cq_off.ring_mask);
        cqes_ = reinterpret_cast<io_uring_cqe *>(c + p.cq_off.cqes);
    }
    ~Uring() override {
        if (fd_ < 0) {
            return;
        }
        munmap(sqes_, entries_ * sizeof(io_uring_sqe));
        if (sizes_[1]) {
            munmap(cq_, sizes_[1]);
        }
        munmap(sq_, sizes_[0]);
        close(fd_);
    }
    bool ok() const { return fd_ >= 0; }

    bool prep(int fd, uint8_t * buf, int64_t len, int64_t off, uint64_t tag) override {
        const unsigned tail = *sq_tail_;
        if (tail - __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE) >= entries_) {
            return false;
        }
        const unsigned i = tail & sq_mask_;
        io_uring_sqe & e = sqes_[i];
        std::memset(&e, 0, sizeof(e));
        e.opcode = IORING_OP_READ;
        e.fd = fd;
        e.addr = reinterpret_cast<uint64_t>(buf);
        e.len = unsigned(len);
        e.off = uint64_t(off);
        e.user_data = tag;
        sq_array_[i] = i;
        __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
        ++unsubmitted_;
        return true;
    }
    void submit() override {
        while (unsubmitted_) {
            const long n = syscall(__NR_io_uring_enter, fd_, unsubmitted_, 0, 0, nullptr, 0);
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EBUSY) {
                    return;                                           // (again on the next pass)
                }
                throw std::runtime_error("io_uring_enter failed");
            }
            unsubmitted_ -= unsigned(n);
        }
    }
    void reap(std::vector<std::pair<uint64_t, int64_t>> & out) override {
        unsigned head = *cq_head_;
        const unsigned tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        for (; head != tail; ++head) {
            const io_uring_cqe & c = cqes_[head & cq_mask_];
            out.emplace_back(c.user_data, int64_t(c.res));
        }
        __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
    }
    const char * name() const override { return "io_uring"; }

  private:
    int fd_ = -1;
    unsigned entries_ = 0, unsubmitted_ = 0, sq_mask_ = 0, cq_mask_ = 0;
    void * sq_ = MAP_FAILED;
    void * cq_ = MAP_FAILED;
    size_t sizes_[2] = {0, 0};
    io_uring_sqe * sqes_ = nullptr;
    unsigned *sq_head_ = nullptr, *sq_tail_ = nullptr, *sq_array_ = nullptr, *cq_head_ = nullptr, *cq_tail_ = nullptr;
    io_uring_cqe * cqes_ = nullptr;
};
#endif

class Aio final : public Backend {
  public:
    explicit Aio(unsigned depth) : depth_(depth) {
        if (syscall(__NR_io_setup, depth, &ctx_) < 0) {
            ctx_ = 0;
        }
        cbs_.resize(depth);
        free_.reserve(depth);
        for (unsigned i = 0; i < depth; ++i) {
            free_.push_back(&cbs_[i]);
        }
        events_.resize(depth);
    }
    ~Aio() override {
        if (ctx_) {
            syscall(__NR_io_destroy, ctx_);
        }
    }
    bool ok() const { return ctx_ != 0; }

    bool prep(int fd, uint8_t * buf, int64_t len, int64_t off, uint64_t tag) override {
        if (free_.empty()) {
            return false;
        }
        iocb * cb = free_.back();
        free_.pop_back();
        std::memset(cb, 0, sizeof(*cb));
        cb->aio_lio_opcode = IOCB_CMD_PREAD;
        cb->aio_fildes = uint32_t(fd);
        cb->aio_buf = reinterpret_cast<uint64_t>(buf);
        cb->aio_nbytes = uint64_t(len);
        cb->aio_offset = off;
        cb->aio_data = tag;
        queued_.push_back(cb);
        return true;
    }
    void submit() override {
        size_t at = 0;
        while (at < queued_.size()) {
            const long n = syscall(__NR_io_submit, ctx_, long(queued_.size() - at), queued_.data() + at);
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN) {
                    break;
                }
                throw std::runtime_error("io_submit failed");
            }
            at += size_t(n);
        }
        queued_.erase(queued_.begin(), queued_.begin() + long(at));
    }
    void reap(std::vector<std::pair<uint64_t, int64_t>> & out) override {
        timespec zero{0, 0};
        const long n = syscall(__NR_io_getevents, ctx_, 0L, long(depth_), events_.data(), &zero);
        for (long i = 0; i < n; ++i) {
            out.emplace_back(events_[size_t(i)].data, int64_t(events_[size_t(i)].res));
            free_.push_back(reinterpret_cast<iocb *>(events_[size_t(i)].obj));
        }
    }
    const char * name() const override { return "aio"; }

  private:
    unsigned depth_;
    aio_context_t ctx_ = 0;
    std::vector<iocb> cbs_;
    std::vector<iocb *> free_, queued_;
    std::vector<io_event> events_;
};

// The backend `want` names ("auto": io_uring where the kernel and the container allow it, else native AIO).
inline std::unique_ptr<Backend> make_backend(const std::string & want, unsigned depth) {
#if __has_include(<linux/io_uring.h>)
    if (want == "auto" || want == "uring") {
        auto u = std::make_unique<Uring>(depth);
        if (u->ok()) {
            return u;
        }
        TORCH_CHECK(want == "auto", "cpu_task: io_uring unavailable (", strerror(errno), ")");
    }
#else
    TORCH_CHECK(want != "uring", "cpu_task: built without io_uring");
#endif
    TORCH_CHECK(want == "auto" || want == "aio", "cpu_task: io backend auto, uring or aio, not ", want);
    auto a = std::make_unique<Aio>(depth);
    TORCH_CHECK(a->ok(), "cpu_task: neither io_uring nor native AIO is available (", strerror(errno), ")");
    return a;
}

// One request in the I/O worker: a slot's copy task planned, its extents read chunk by chunk.
struct IoRequest {
    bool urgent = false, failed = false;
    Plan plan;
    std::vector<int> pending;                         // per extent: its chunks in flight
    std::vector<int> left;                            // per piece: its extents not yet in
    std::vector<uint8_t *> buf;                       // per extent: its bounce buffer, while it is read
    size_t ext = 0;                                   // the next extent to start, and its next chunk's offset
    int64_t at = 0;
    size_t unfinished = 0;                            // extents not yet in
};

struct Chunk {                                        // a read in flight: its request's slot, extent and range
    size_t slot;
    size_t ext;
    int64_t from, len;                                // relative to the extent's `lo`
};

// The completion-driven loop of the `io` pool's one thread: new requests (urgent slots' first), chunks submitted up to
// `depth` in flight — an urgent request ahead of every chunk not yet submitted, the others' held while one is open
// —, completions reaped without blocking, a slot answered once its request's extents (and keep copies) are in.
class IoEngine {
  public:
    IoEngine(std::shared_ptr<State> st, std::vector<Copy *> copies, std::vector<bool> urgent, unsigned depth,
             int64_t chunk, std::unique_ptr<Backend> backend)
        : st_(std::move(st)), copies_(std::move(copies)), urgent_(std::move(urgent)), depth_(depth),
          chunk_(std::max<int64_t>(ALIGN, chunk / ALIGN * ALIGN)), be_(std::move(backend)), active_(copies_.size()) {
        int64_t most = ALIGN;
        for (auto * c : copies_) {
            most = std::max(most, c->most());
        }
        for (unsigned i = 0; i < depth_; ++i) {
            void * b = nullptr;
            TORCH_CHECK(!posix_memalign(&b, ALIGN, size_t(most)), "cpu_task: no bounce buffer");
            free_bufs_.push_back(static_cast<uint8_t *>(b));
            bufs_.push_back(free_bufs_.back());
        }
    }
    ~IoEngine() {
        for (auto * b : bufs_) {
            std::free(b);
        }
    }
    const char * backend() const { return be_->name(); }

    // One pass (`take`: new requests too; not while draining at a stop): true if anything moved.
    bool step(bool take = true);
    bool idle() const { return inflight_ == 0; }

  private:
    void start(size_t s);
    bool issue(size_t s, size_t ext, int64_t from, int64_t len);
    bool submit_from(size_t s);
    void complete(uint64_t tag, int64_t res);
    void extent_done(size_t s, size_t ext);
    void answer(size_t s);

    std::shared_ptr<State> st_;
    std::vector<Copy *> copies_;
    std::vector<bool> urgent_;
    unsigned depth_;
    int64_t chunk_;
    std::unique_ptr<Backend> be_;
    std::vector<std::unique_ptr<IoRequest>> active_;  // per slot
    std::vector<size_t> order_;                       // open requests' slots, by arrival
    std::vector<uint8_t *> bufs_, free_bufs_;
    unsigned inflight_ = 0;
    std::vector<Chunk> chunks_;                       // by tag
    std::vector<uint64_t> free_tags_;
    std::vector<std::pair<uint64_t, int64_t>> done_;
};

void IoEngine::start(size_t s) {
    auto r = std::make_unique<IoRequest>();
    r->urgent = urgent_[s];
    if (st_->flag() != 0 || plan_copy(copies_[s]->roles(), r->plan) != 0) {
        r->failed = true;
    }
    const size_t n = r->plan.extents.size();
    r->pending.assign(n, 0);
    r->buf.assign(n, nullptr);
    r->left.assign(r->plan.pieces.size(), 0);
    for (const Extent & e : r->plan.extents) {
        ++r->left[size_t(e.piece)];
    }
    r->unfinished = r->failed ? 0 : n;
    active_[s] = std::move(r);
    order_.push_back(s);
    if (active_[s]->unfinished == 0) {
        answer(s);                                    // nothing to read (or failed at once)
    }
}

bool IoEngine::issue(size_t s, size_t ext, int64_t from, int64_t len) {
    IoRequest & r = *active_[s];
    const Extent & e = r.plan.extents[ext];
    uint64_t tag;
    if (!free_tags_.empty()) {
        tag = free_tags_.back();
        free_tags_.pop_back();
        chunks_[tag] = Chunk{s, ext, from, len};
    } else {
        tag = chunks_.size();
        chunks_.push_back(Chunk{s, ext, from, len});
    }
    uint8_t * into = e.bounce ? r.buf[ext] : e.into;
    if (!be_->prep(e.fd, into + from, len, e.lo + from, tag)) {
        free_tags_.push_back(tag);
        return false;
    }
    ++r.pending[ext];
    ++inflight_;
    return true;
}

// The request's next chunks while the depth allows: false once it has none left to start (or no room, or no buffer).
bool IoEngine::submit_from(size_t s) {
    IoRequest & r = *active_[s];
    while (inflight_ < depth_ && r.ext < r.plan.extents.size() && !r.failed) {
        const Extent & e = r.plan.extents[r.ext];
        if (r.at == 0 && e.bounce) {
            if (free_bufs_.empty()) {
                return false;
            }
            r.buf[r.ext] = free_bufs_.back();
            free_bufs_.pop_back();
        }
        const int64_t span = e.hi - e.lo, len = std::min(chunk_, span - r.at);
        if (!issue(s, r.ext, r.at, len)) {
            return false;
        }
        r.at += len;
        if (r.at >= span) {
            ++r.ext;
            r.at = 0;
        }
    }
    return r.ext < r.plan.extents.size() && !r.failed;
}

void IoEngine::complete(uint64_t tag, int64_t res) {
    const Chunk c = chunks_[tag];
    free_tags_.push_back(tag);
    --inflight_;
    IoRequest & r = *active_[c.slot];
    --r.pending[c.ext];
    const Extent & e = r.plan.extents[c.ext];
    if (res == -EINTR || res == -EAGAIN) {             // again, the same range
        if (!issue(c.slot, c.ext, c.from, c.len)) {
            r.failed = true;
        }
    } else if (res < 0) {
        r.failed = true;
    } else if (res < c.len && c.from + res < e.need) {  // short of what the extent needs: the rest (none: EOF)
        if (res == 0 || (e.direct && res % ALIGN) || !issue(c.slot, c.ext, c.from + res, c.len - res)) {
            r.failed = true;
        }
    }
    const bool started = c.ext < r.ext;                // every chunk of the extent submitted
    if (r.pending[c.ext] == 0 && (started || r.failed)) {
        extent_done(c.slot, c.ext);
    }
}

void IoEngine::extent_done(size_t s, size_t ext) {
    IoRequest & r = *active_[s];
    Extent e = r.plan.extents[ext];
    if (r.buf[ext] && !r.failed) {
        e.into = r.buf[ext];
        land(e, e.need);                              // the bounced extent into its unit
    }
    if (r.buf[ext]) {
        free_bufs_.push_back(r.buf[ext]);
        r.buf[ext] = nullptr;
    }
    if (r.failed) {                                   // answered once nothing of it is in flight
        if (std::all_of(r.pending.begin(), r.pending.end(), [](int p) { return p == 0; })) {
            answer(s);
        }
        return;
    }
    if (--r.left[size_t(e.piece)] == 0) {
        keep_copies(r.plan.pieces[size_t(e.piece)]);
    }
    if (--r.unfinished == 0) {
        answer(s);
    }
}

void IoEngine::answer(size_t s) {
    IoRequest & r = *active_[s];
    for (auto *& b : r.buf) {
        if (b) {
            free_bufs_.push_back(b);
            b = nullptr;
        }
    }
    if (r.failed) {
        st_->cpu_flag().fetch_or(TASK, __ATOMIC_RELAXED);
    }
    uint32_t req = REQ;                                // not answered meanwhile by the watchdog
    st_->word(s).compare_exchange_strong(req, r.failed ? FAILED : DONE, __ATOMIC_RELEASE);
    st_->served.fetch_add(1, std::memory_order_relaxed);
    active_[s].reset();
    order_.erase(std::find(order_.begin(), order_.end(), s));
}

bool IoEngine::step(bool take) {
    bool moved = false;
    for (size_t s = 0; take && s < active_.size(); ++s) {      // new requests
        if (!active_[s] && st_->word(s).load(__ATOMIC_ACQUIRE) == REQ && !st_->dead.load(std::memory_order_relaxed)) {
            start(s);
            moved = true;
        }
    }
    const unsigned before = inflight_;
    bool urgent_open = false;                          // urgent requests' chunks first; the others' held meanwhile
    for (size_t s : std::vector<size_t>(order_)) {
        if (active_[s] && active_[s]->urgent) {
            urgent_open = true;
            submit_from(s);
        }
    }
    for (size_t s : std::vector<size_t>(order_)) {
        if (!urgent_open && active_[s] && !active_[s]->urgent) {
            submit_from(s);
        }
    }
    moved |= inflight_ != before;
    be_->submit();
    done_.clear();
    be_->reap(done_);
    for (const auto & [tag, res] : done_) {
        complete(tag, res);
        moved = true;
    }
    return moved;
}

// The `io` pool's coordinator: the engine's loop; at a stop, what is in flight lands before its buffers go.
void serve_io(std::shared_ptr<State> st) {
    pin_to(st->cores[0]);
    IoEngine & eng = *st->io;
    Backoff b;
    b.reset();
    while (st->running.load(std::memory_order_relaxed)) {
        if (eng.step()) {
            b.reset();
        } else if (!eng.idle()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));    // reads in flight: their completions soon
        } else {
            b.wait();
        }
    }
    while (!eng.idle()) {                              // (a read that never ends: `stop` leaves this thread behind)
        if (!eng.step(false)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    st->exited.store(true, std::memory_order_release);
}

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
                nvtxRangePushA("cpu_task");                // the task as nsys shows it (the GPU's k_signal / k_wait beside)
                const bool failed = st->flag() != 0 || pool.run(fn, arg) != 0;
                nvtxRangePop();
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
    // `depth` > 0: the `io` pool — its tasks copy tasks, read by one completion-driven thread (`IoEngine`) with up to
    // `depth` chunks of `chunk` bytes in flight on `backend` (auto, uring, aio); `urgent`: per slot, a hint that its
    // requests go before every chunk not yet submitted. 0: the tasks run on a pool of `cores`' threads.
    Worker(int64_t host, int64_t device, std::vector<Task> tasks, std::vector<int64_t> cores, int64_t timeout_ns,
           bool memop, int64_t depth, const std::string & backend, std::vector<bool> urgent, int64_t chunk)
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
        if (depth > 0) {
            std::vector<Copy *> copies;
            for (const auto & [fn, arg] : st_->tasks) {
                TORCH_CHECK(fn == &Copy::run, "cpu_task: the io pool reads copy tasks only");
                copies.push_back(static_cast<Copy *>(arg));
            }
            urgent.resize(copies.size(), false);
            st_->io = std::make_shared<IoEngine>(st_, std::move(copies), std::move(urgent), unsigned(depth), chunk,
                                                 make_backend(backend, unsigned(depth)));
            coordinator_ = std::thread(serve_io, st_);
        } else {
            coordinator_ = std::thread(serve, st_);
        }
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
    std::string backend() const { return st_->io ? st_->io->backend() : ""; }
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

// The copy task's handle over its tensors (kept alive with it).
struct CopyTask {
    std::vector<at::Tensor> keep;
    std::unique_ptr<Copy> copy;

    explicit CopyTask(const std::vector<std::vector<at::Tensor>> & roles) {
        std::vector<CopyRole> out;
        for (const auto & role : roles) {
            TORCH_CHECK(role.size() == 3 || role.size() == 5, "copy: a role is (descriptors, units, requests[, runs, "
                        "keep])");
            const at::Tensor & desc = role[0];
            const at::Tensor & units = role[1];
            const at::Tensor & req = role[2];
            TORCH_CHECK(!desc.is_cuda() && desc.scalar_type() == at::kLong && desc.dim() == 3 && desc.size(2) == 4 &&
                        desc.is_contiguous(), "copy: descriptors [P, X, 4] int64 on the host");
            TORCH_CHECK(!units.is_cuda() && units.scalar_type() == at::kByte && units.dim() == 2 &&
                        units.is_contiguous(), "copy: units [U, unit] uint8 on the host");
            TORCH_CHECK(!req.is_cuda() && req.scalar_type() == at::kInt && req.dim() == 2 &&
                        (req.size(1) == 2 || req.size(1) == 3) && req.is_contiguous(),
                        "copy: requests [1 + F, 2 or 3] int32 on the host");
            CopyRole r{desc.data_ptr<int64_t>(), desc.size(0), desc.size(1), units.data_ptr<uint8_t>(), units.size(0),
                       units.size(1), req.data_ptr<int32_t>(), req.size(0) - 1, req.size(1), nullptr, nullptr, 0, 0};
            keep.insert(keep.end(), {desc, units, req});
            if (role.size() == 5 && role[3].numel()) {
                const at::Tensor & runs = role[3];
                TORCH_CHECK(!runs.is_cuda() && runs.scalar_type() == at::kLong && runs.dim() == 3 &&
                            runs.size(0) == units.size(0) && runs.size(1) == desc.size(1) && runs.size(2) == 2 &&
                            runs.is_contiguous(), "copy: runs [U, X, 2] int64 on the host, a row's per extent");
                r.runs = runs.data_ptr<int64_t>();
                keep.push_back(runs);
            }
            if (role.size() == 5 && role[4].numel()) {
                const at::Tensor & k = role[4];
                TORCH_CHECK(!k.is_cuda() && k.scalar_type() == at::kByte && k.dim() == 2 && k.is_contiguous(),
                            "copy: keep [K, unit] uint8 on the host");
                r.keep = k.data_ptr<uint8_t>();
                r.keep_count = k.size(0);
                r.keep_unit = k.size(1);
                keep.push_back(k);
            }
            TORCH_CHECK(r.cols == 2 || (r.runs && r.keep), "copy: requests with a keep column need runs and a keep "
                        "pool");
            out.push_back(r);
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
        .def(pybind11::init<int64_t, int64_t, std::vector<Task>, std::vector<int64_t>, int64_t, bool, int64_t,
                            const std::string &, std::vector<bool>, int64_t>(),
             pybind11::arg("host"), pybind11::arg("device"), pybind11::arg("tasks"), pybind11::arg("cores"),
             pybind11::arg("timeout_ns"), pybind11::arg("memop"), pybind11::arg("depth") = 0,
             pybind11::arg("backend") = "auto", pybind11::arg("urgent") = std::vector<bool>{},
             pybind11::arg("chunk") = 256 * 1024,
             "host and device addresses of (slots + 1) 64-byte lines of mapped pinned memory, a task per slot (ABI, "
             "function, argument), the cores (the first runs the coordinator), the wait's time bound, memop or spin; "
             "depth > 0: an io pool of copy tasks, read asynchronously (backend auto / uring / aio; urgent per slot)")
        .def("backend", &Worker::backend, "the io pool's read interface (io_uring or aio), else empty")
        .def("signal", &Worker::signal, "on the current stream (of `like`'s device): the slot's task runs once the "
             "stream reaches here")
        .def("wait", &Worker::wait, "on the current stream: later work waits for the slot's answer (fail-open)")
        .def("flag", &Worker::flag, "the sticky error bits: 1 a task failed, 2 a wait timed out")
        .def("served", &Worker::served, "requests answered so far")
        .def("stop", &Worker::stop, "join the threads (after a drain); a request still open is abandoned; True: the "
             "coordinator, in a task past the time bound, was left behind (its words and tasks must outlive it)")
        .def("kill", &Worker::kill, "tests: the coordinator stops answering");
    pybind11::class_<CopyTask>(m, "CopyTask")
        .def(pybind11::init<const std::vector<std::vector<at::Tensor>> &>(),
             "per role: descriptors [P, X, 4] int64 (fd, direct, offset, nbytes), units [U, unit] uint8, requests "
             "[1 + F, 2] int32 (n, then (piece, unit)), and optionally runs [U, X, 2] int64 (landing rows: each "
             "extent's aligned superset in place, its (offset, nbytes) there) and a keep pool [K, unit] uint8 "
             "(requests [1 + F, 3]: (piece, row, keep)): the copy task (F9) over them")
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
