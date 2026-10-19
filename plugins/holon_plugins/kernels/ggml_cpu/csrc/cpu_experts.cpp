// `cpu_experts@1` (std.moe.kernels; docs/v1-offload.md D7): a layer's CPU jobs — whole experts, down(SwiGLU(gate·x,
// up·x)) — over GGUF blocks in place, with ggml-cpu's own activation quantization (`from_float` of the weights'
// `vec_dot_type`) and dot products (`vec_dot`, one per weight row and token), as spike x0's M3 measured them.
//
// A run is a `cpu_task@1` task: every thread of the worker's pool runs it, meeting at the pool's barrier between
//   phase 1  each (job, token) item's input row quantized to gate/up's vec_dot type      (items split across threads)
//   phase 2  gate and up rows (jobs × 2·inter), each row read once for all its job's tokens  (rows stolen, below)
//   phase 3  f32 SwiGLU per item, quantized to down's vec_dot type                          (items split)
//   phase 4  down rows (jobs × hidden) into the items' output rows                           (rows stolen)
// Rows, never K, are split across threads, and each (row, token) is one vec_dot call whatever the grouping, the batch
// or the thread that ran it: a token's output is bitwise the same alone or grouped and at any thread count (D7). Each
// thread takes 16-row chunks from the front of its share, then steals from the back of the others' (one slowed core
// would otherwise stall the batch, x0's M3).
//
// The type table (`traits`) is adapted from llama.cpp cbe28bb3 (ggml/src/ggml-cpu/ggml-cpu.c), MIT.
//
// The job table is `moe_resolve@1`'s, read in mapped host memory the GPU wrote: a malformed one fails the task (the
// worker answers with an error, fail-open), it never reads past a buffer.
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "ggml.h"
#include "holon_cpu_task.h"

extern "C" {
typedef void (*vec_dot_t)(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
typedef void (*from_float_t)(const float * x, void * y, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
#define DOT(name) void name(int, float *, size_t, const void *, size_t, const void *, size_t, int);
DOT(ggml_vec_dot_q2_0_q8_0)
DOT(ggml_vec_dot_q4_0_q8_0) DOT(ggml_vec_dot_q5_0_q8_0) DOT(ggml_vec_dot_q8_0_q8_0) DOT(ggml_vec_dot_iq4_nl_q8_0)
DOT(ggml_vec_dot_q2_K_q8_K) DOT(ggml_vec_dot_q3_K_q8_K) DOT(ggml_vec_dot_q4_K_q8_K) DOT(ggml_vec_dot_q5_K_q8_K)
DOT(ggml_vec_dot_q6_K_q8_K) DOT(ggml_vec_dot_iq1_s_q8_K) DOT(ggml_vec_dot_iq1_m_q8_K) DOT(ggml_vec_dot_iq2_xxs_q8_K)
DOT(ggml_vec_dot_iq2_xs_q8_K) DOT(ggml_vec_dot_iq2_s_q8_K) DOT(ggml_vec_dot_iq3_xxs_q8_K) DOT(ggml_vec_dot_iq3_s_q8_K)
DOT(ggml_vec_dot_iq4_xs_q8_K)
#undef DOT
}

namespace {

constexpr int PLANT_GATE_UP_ORDER = 1;   // judge-only: SwiGLU reads the merged halves swapped (cpu_gate_up_order)
constexpr int MUTATE_SPLIT_K = 2;        // L0's mutation: a grouped token's down dot split in two parts of K
constexpr int BAD_TABLE = 2;             // the task's error code: a malformed job table

// ggml-cpu's dot product and activation type per weight type: adapted from llama.cpp cbe28bb3
// (ggml/src/ggml-cpu/ggml-cpu.c `type_traits_cpu`), MIT, for the types the CPU experts serve.
struct Traits {
    vec_dot_t dot;
    ggml_type dot_type;
    from_float_t quantize;
};

Traits traits(int type) {
    const Traits q8_0{nullptr, GGML_TYPE_Q8_0, quantize_row_q8_0}, q8_k{nullptr, GGML_TYPE_Q8_K, quantize_row_q8_K};
    auto with = [](Traits t, vec_dot_t dot) { t.dot = dot; return t; };
    switch (type) {
        case GGML_TYPE_Q2_0: return with(q8_0, ggml_vec_dot_q2_0_q8_0);
        case GGML_TYPE_Q4_0: return with(q8_0, ggml_vec_dot_q4_0_q8_0);
        case GGML_TYPE_Q5_0: return with(q8_0, ggml_vec_dot_q5_0_q8_0);
        case GGML_TYPE_Q8_0: return with(q8_0, ggml_vec_dot_q8_0_q8_0);
        case GGML_TYPE_IQ4_NL: return with(q8_0, ggml_vec_dot_iq4_nl_q8_0);
        case GGML_TYPE_Q2_K: return with(q8_k, ggml_vec_dot_q2_K_q8_K);
        case GGML_TYPE_Q3_K: return with(q8_k, ggml_vec_dot_q3_K_q8_K);
        case GGML_TYPE_Q4_K: return with(q8_k, ggml_vec_dot_q4_K_q8_K);
        case GGML_TYPE_Q5_K: return with(q8_k, ggml_vec_dot_q5_K_q8_K);
        case GGML_TYPE_Q6_K: return with(q8_k, ggml_vec_dot_q6_K_q8_K);
        case GGML_TYPE_IQ1_S: return with(q8_k, ggml_vec_dot_iq1_s_q8_K);
        case GGML_TYPE_IQ1_M: return with(q8_k, ggml_vec_dot_iq1_m_q8_K);
        case GGML_TYPE_IQ2_XXS: return with(q8_k, ggml_vec_dot_iq2_xxs_q8_K);
        case GGML_TYPE_IQ2_XS: return with(q8_k, ggml_vec_dot_iq2_xs_q8_K);
        case GGML_TYPE_IQ2_S: return with(q8_k, ggml_vec_dot_iq2_s_q8_K);
        case GGML_TYPE_IQ3_XXS: return with(q8_k, ggml_vec_dot_iq3_xxs_q8_K);
        case GGML_TYPE_IQ3_S: return with(q8_k, ggml_vec_dot_iq3_s_q8_K);
        case GGML_TYPE_IQ4_XS: return with(q8_k, ggml_vec_dot_iq4_xs_q8_K);
        default: TORCH_CHECK(false, "cpu_experts: ggml type ", type, " has no CPU path here");
    }
}

// Per-thread row ranges [lo, hi), one word each: the owner takes chunks from the front, idle threads steal from the
// back.
class Rows {
  public:
    void reset(size_t total, int n) {
        for (int t = 0; t < n; ++t) {
            slots_[t].v.store(pack(total * t / n, total * (t + 1) / n), std::memory_order_relaxed);
        }
    }
    bool next(int t, int n, size_t & a, size_t & b) {
        if (take(t, true, a, b)) {
            return true;
        }
        for (int i = 1; i < n; ++i) {
            if (take((t + i) % n, false, a, b)) {
                return true;
            }
        }
        return false;
    }
    static constexpr int MAX_THREADS = 64;

  private:
    static constexpr size_t CHUNK = 16;
    static uint64_t pack(size_t lo, size_t hi) { return uint64_t(lo) << 32 | uint64_t(hi); }
    bool take(int t, bool front, size_t & a, size_t & b) {
        auto & v = slots_[t].v;
        uint64_t cur = v.load(std::memory_order_relaxed);
        while (true) {
            const size_t lo = cur >> 32, hi = cur & 0xffffffffu;
            if (lo >= hi) {
                return false;
            }
            const size_t n = std::min(CHUNK, hi - lo), nlo = front ? lo + n : lo, nhi = front ? hi : hi - n;
            if (v.compare_exchange_weak(cur, pack(nlo, nhi), std::memory_order_relaxed)) {
                a = front ? lo : nhi;
                b = front ? nlo : hi;
                return true;
            }
        }
    }
    struct alignas(64) Slot {
        std::atomic<uint64_t> v{0};
    };
    Slot slots_[MAX_THREADS];
};

// One instance: an expert class pair (gate/up and down types, hidden, inter) and the most tokens and jobs per run.
struct Experts {
    int hidden, inter, top_k, tokens, max_jobs;
    Traits gu, dn;
    size_t gu_row, dn_row, xq_row, aq_row;
    int dn_blocks;                                      // a down row's weight blocks (and its activation's)

    Experts(int64_t gu_type, int64_t dn_type, int64_t hidden_, int64_t inter_, int64_t top_k_, int64_t tokens_,
            int64_t jobs_)
        : hidden(int(hidden_)), inter(int(inter_)), top_k(int(top_k_)), tokens(int(tokens_)), max_jobs(int(jobs_)),
          gu(traits(int(gu_type))), dn(traits(int(dn_type))) {
        gu_row = ggml_row_size(ggml_type(gu_type), hidden);
        dn_row = ggml_row_size(ggml_type(dn_type), inter);
        xq_row = ggml_row_size(gu.dot_type, hidden);
        aq_row = ggml_row_size(dn.dot_type, inter);
        dn_blocks = int(inter / ggml_blck_size(ggml_type(dn_type)));
    }
};

// A run over fixed buffers: the job table, the input rows, the two host pools, the output rows.
struct Bound {
    const Experts * e;
    const int32_t * jobs;
    int width;                                         // a job row's entries: 4 + tokens
    const float * x;
    const uint8_t * gu_units;
    int64_t gu_count, gu_unit;
    const uint8_t * dn_units;
    int64_t dn_count, dn_unit;
    float * rows;
    int plant = 0;
    // scratch, sized for the most items (min(jobs · tokens, tokens · k)) at bind time; the job table's copy (the
    // shared mapped table may be rewritten while a task runs past a wait that gave up: never read it twice)
    std::vector<uint8_t> xq, aq;
    std::vector<int32_t> table;
    std::vector<float> h;
    std::vector<int> off;                              // each job's first item (njobs + 1)
    int njobs = 0, err = 0;
    Rows gu_rows, dn_rows;

    int items() const { return off[njobs]; }

    // Thread 0, before the others start: the table checked, the items counted, the steals reset.
    int prepare(int threads) {
        njobs = jobs[0];
        if (njobs < 0 || njobs > e->max_jobs || threads > Rows::MAX_THREADS) {
            return BAD_TABLE;
        }
        std::copy(jobs, jobs + int64_t(1 + njobs) * width, table.begin());
        const int64_t entries = int64_t(e->tokens) * e->top_k;
        off[0] = 0;
        for (int j = 0; j < njobs; ++j) {
            const int32_t * job = table.data() + int64_t(1 + j) * width;
            const int n = job[3];
            if (job[1] < 0 || job[1] >= gu_count || job[2] < 0 || job[2] >= dn_count || n < 1 || n > e->tokens) {
                return BAD_TABLE;
            }
            for (int i = 0; i < n; ++i) {
                if (job[4 + i] < 0 || job[4 + i] >= entries) {
                    return BAD_TABLE;
                }
            }
            off[j + 1] = off[j] + n;
        }
        if (size_t(items()) * e->xq_row > xq.size()) {
            return BAD_TABLE;
        }
        gu_rows.reset(size_t(njobs) * 2 * e->inter, threads);
        dn_rows.reset(size_t(njobs) * e->hidden, threads);
        return 0;
    }

    const int32_t * job(int j) const { return table.data() + int64_t(1 + j) * width; }

    static int run(void * arg, const holon_cpu_env * env) {
        auto * b = static_cast<Bound *>(arg);
        const Experts & e = *b->e;
        const int t = env->thread, n = env->threads;
        if (t == 0) {
            b->err = b->prepare(n);
        }
        env->barrier(env);
        if (b->err || b->njobs == 0) {
            return b->err;
        }
        const size_t items = b->items(), gu_rows = 2 * size_t(e.inter);
        uint8_t * xq = b->xq.data();
        uint8_t * aq = b->aq.data();
        float * h = b->h.data();
        size_t lo = items * t / n, hi = items * (t + 1) / n;
        for (size_t i = lo; i < hi; ++i) {                                  // phase 1
            e.gu.quantize(b->x + int64_t(b->entry(i) / e.top_k) * e.hidden, xq + i * e.xq_row, e.hidden);
        }
        env->barrier(env);
        size_t a, z;
        while (b->gu_rows.next(t, n, a, z)) {                               // phase 2
            for (size_t r = a; r < z; ++r) {
                const int j = int(r / gu_rows);
                const size_t rr = r % gu_rows;
                const uint8_t * w = b->gu_units + b->job(j)[1] * b->gu_unit + rr * e.gu_row;
                for (int i = b->off[j]; i < b->off[j + 1]; ++i) {
                    e.gu.dot(e.hidden, h + size_t(i) * gu_rows + rr, 0, w, 0, xq + i * e.xq_row, 0, 1);
                }
            }
        }
        env->barrier(env);
        thread_local std::vector<float> act;
        act.resize(e.inter);
        const bool swapped = b->plant & PLANT_GATE_UP_ORDER;
        for (size_t i = lo; i < hi; ++i) {                                  // phase 3
            const float * g = h + i * gu_rows + (swapped ? e.inter : 0);
            const float * u = h + i * gu_rows + (swapped ? 0 : e.inter);
            for (int c = 0; c < e.inter; ++c) {
                act[c] = g[c] / (1.0f + std::exp(-g[c])) * u[c];
            }
            e.dn.quantize(act.data(), aq + i * e.aq_row, e.inter);
        }
        env->barrier(env);
        while (b->dn_rows.next(t, n, a, z)) {                               // phase 4
            for (size_t r = a; r < z; ++r) {
                const int j = int(r / e.hidden);
                const size_t rr = r % e.hidden;
                const uint8_t * w = b->dn_units + b->job(j)[2] * b->dn_unit + rr * e.dn_row;
                const int32_t * entries = b->job(j) + 4 - b->off[j];
                for (int i = b->off[j]; i < b->off[j + 1]; ++i) {
                    float * out = b->rows + int64_t(entries[i]) * e.hidden + rr;
                    if ((b->plant & MUTATE_SPLIT_K) && i > b->off[j] && e.dn_blocks > 1) {   // L0's mutation
                        const int nb = e.dn_blocks / 2, n = e.inter / e.dn_blocks * nb;
                        float p, q;
                        e.dn.dot(n, &p, 0, w, 0, aq + i * e.aq_row, 0, 1);
                        e.dn.dot(e.inter - n, &q, 0, w + e.dn_row / e.dn_blocks * nb, 0,
                                 aq + i * e.aq_row + e.aq_row / e.dn_blocks * nb, 0, 1);
                        *out = p + q;
                    } else {
                        e.dn.dot(e.inter, out, 0, w, 0, aq + i * e.aq_row, 0, 1);
                    }
                }
            }
        }
        return 0;
    }

    // Item i's entry (t·k + j): its job's (i − off[j])-th.
    int entry(size_t i) const {
        const int j = int(std::upper_bound(off.begin(), off.begin() + njobs + 1, int(i)) - off.begin()) - 1;
        return job(j)[4 + (int(i) - off[j])];
    }
};

inline void relax() {
#if defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

// A synchronous run on `threads` threads spawned for it (tests, the oracle's comparisons, a CPU-only engine).
struct Local {
    int n;
    std::atomic<uint32_t> count{0}, gen{0};

    static void barrier(const holon_cpu_env * env) {
        auto * self = static_cast<Local *>(env->pool);
        const uint32_t g = self->gen.load(std::memory_order_acquire);
        if (self->count.fetch_add(1, std::memory_order_acq_rel) == uint32_t(self->n - 1)) {
            self->count.store(0, std::memory_order_relaxed);
            self->gen.fetch_add(1, std::memory_order_release);
        } else {
            while (self->gen.load(std::memory_order_acquire) == g) {
                relax();
            }
        }
    }

    static int run(int threads, holon_cpu_task_fn fn, void * arg) {
        Local pool{threads};
        std::vector<int> result(threads);
        std::vector<std::thread> others;
        for (int t = 1; t < threads; ++t) {
            others.emplace_back([&, t] {
                const holon_cpu_env env{t, threads, &Local::barrier, &pool};
                result[t] = fn(arg, &env);
            });
        }
        const holon_cpu_env env{0, threads, &Local::barrier, &pool};
        result[0] = fn(arg, &env);
        for (auto & t : others) {
            t.join();
        }
        return *std::max_element(result.begin(), result.end());
    }
};

void check(const at::Tensor & t, at::ScalarType dtype, int dims, const char * name) {
    TORCH_CHECK(!t.is_cuda() && t.scalar_type() == dtype && t.dim() == dims && t.is_contiguous(), "cpu_experts: ",
                name, " must be a contiguous host tensor of ", dtype, " with ", dims, " dimensions");
}

// The task's handle: the buffers it reads and writes (kept alive with it) and its run.
struct Task {
    std::shared_ptr<Experts> experts;
    std::vector<at::Tensor> keep;
    std::unique_ptr<Bound> bound;

    Task(std::shared_ptr<Experts> e, at::Tensor jobs, at::Tensor x, at::Tensor gu, at::Tensor dn, at::Tensor rows,
         int64_t plant)
        : experts(std::move(e)), keep{jobs, x, gu, dn, rows} {
        const Experts & ex = *experts;
        check(jobs, at::kInt, 2, "jobs");
        check(x, at::kFloat, 2, "x");
        check(gu, at::kByte, 2, "gate_up");
        check(dn, at::kByte, 2, "down");
        check(rows, at::kFloat, 2, "rows");
        TORCH_CHECK(jobs.size(0) >= 1 + ex.max_jobs && jobs.size(1) == 4 + ex.tokens && x.size(0) == ex.tokens &&
                    x.size(1) == ex.hidden && rows.size(0) == int64_t(ex.tokens) * ex.top_k && rows.size(1) == ex.hidden,
                    "cpu_experts: buffers of the instance's shape (jobs [1 + J, 4 + T], x [T, H], rows [T·k, H])");
        TORCH_CHECK(gu.size(1) >= int64_t(2) * ex.inter * ex.gu_row && dn.size(1) >= int64_t(ex.hidden) * ex.dn_row,
                    "cpu_experts: units smaller than an expert's piece");
        bound = std::make_unique<Bound>();
        Bound & b = *bound;
        b.e = experts.get();
        b.jobs = jobs.data_ptr<int32_t>();
        b.width = int(jobs.size(1));
        b.x = x.data_ptr<float>();
        b.gu_units = gu.data_ptr<uint8_t>();
        b.gu_count = gu.size(0);
        b.gu_unit = gu.size(1);
        b.dn_units = dn.data_ptr<uint8_t>();
        b.dn_count = dn.size(0);
        b.dn_unit = dn.size(1);
        b.rows = rows.data_ptr<float>();
        b.plant = int(plant);
        const size_t items = std::min(size_t(ex.max_jobs) * ex.tokens, size_t(ex.tokens) * ex.top_k);
        b.xq.resize(items * ex.xq_row);
        b.aq.resize(items * ex.aq_row);
        b.h.resize(items * 2 * size_t(ex.inter));
        b.off.resize(ex.max_jobs + 1);
        b.table.resize(int64_t(1 + ex.max_jobs) * b.width);
    }

    std::tuple<int64_t, int64_t, int64_t> native() {
        return {HOLON_CPU_TASK_ABI, reinterpret_cast<int64_t>(&Bound::run), reinterpret_cast<int64_t>(bound.get())};
    }

    int64_t run(int64_t threads) {
        TORCH_CHECK(threads >= 1 && threads <= Rows::MAX_THREADS, "cpu_experts: 1 to ", Rows::MAX_THREADS,
                    " threads");
        return Local::run(int(threads), &Bound::run, bound.get());
    }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Experts, std::shared_ptr<Experts>>(m, "Experts")
        .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
             "gate/up and down ggml types, hidden, inter, top_k, the most tokens and jobs per run")
        .def_static("supports", [](int64_t type) {
            try {
                traits(int(type));
                return true;
            } catch (const c10::Error &) {
                return false;
            }
        });
    pybind11::class_<Task>(m, "Task")
        .def(pybind11::init<std::shared_ptr<Experts>, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                            int64_t>(),
             "the instance; jobs, x, gate/up units, down units, rows (host tensors kept alive); plant bits")
        .def("native", &Task::native, "the task ABI's handle (ABI version, function, argument)")
        .def("run", &Task::run, pybind11::call_guard<pybind11::gil_scoped_release>(),
             "run now on `threads` threads; returns the task's error code (0: none)");
}
