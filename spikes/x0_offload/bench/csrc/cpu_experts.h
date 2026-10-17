// CPU expert compute for step 0 (M2, M3, M5): one MoE expert = down(SwiGLU(gate·x, up·x)) over GGUF blocks in
// place, with ggml-cpu's own `from_float` (activation quantization) and `vec_dot` (one row · one token).
//
// A batch is a layer's CPU jobs: J experts of one class, each with the same T tokens. Rows (never K) are split
// across the pool's threads, each thread taking 16-row chunks from the front of its own share and, once done,
// stealing from the back of the others' (`Rows`). Every (row, token) result is one ggml vec_dot whatever the thread
// count, the batch or which thread ran it: D7's contract holds by construction, and the bench checks it bitwise.
// Stealing keeps one slowed core from stalling the lockstep pool: on the desktops an IRQ- or sibling-loaded core
// intermittently ran 1.7x slower, which a static split turns into a 1.6x slower batch; a shared claim counter
// absorbed it but cost 10-20% when no core was slowed.
//
//   phase 1  quantize x per (job, token) to gate/up's vec_dot type
//   phase 2  gate and up rows (J x 2 x inter), each row read once for all T tokens
//   phase 3  f32 SwiGLU per (job, token), quantize to down's vec_dot type
//   phase 4  down rows (J x hidden)
#pragma once
#include <immintrin.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace x0 {

using Clock = std::chrono::steady_clock;

inline void pin_to(int core) {
  if (core < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// Pins the calling thread for a scope, restoring its previous affinity after.
class PinScope {
 public:
  explicit PinScope(int core) {
    pthread_getaffinity_np(pthread_self(), sizeof(old_), &old_);
    pin_to(core);
  }
  ~PinScope() { pthread_setaffinity_np(pthread_self(), sizeof(old_), &old_); }

 private:
  cpu_set_t old_;
};

// S6's backoff: hot spin while work flows, doze in 50 us sleeps after 100 ms idle.
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
        _mm_pause();
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

// Persistent spinning pool of n threads; the caller of run() is thread 0 and must sit on cores[0].
class Pool {
 public:
  explicit Pool(const std::vector<int>& cores) : n_(int(cores.size())) {
    if (n_ < 1) throw std::invalid_argument("pool: at least one core");
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
  int size() const { return n_; }

  template <class F>
  void run(F& f) {
    fn_ = [](void* c, int t) { (*static_cast<F*>(c))(t); };
    ctx_ = &f;
    pending_.store(n_ - 1, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    fn_(ctx_, 0);
    while (pending_.load(std::memory_order_acquire) != 0) _mm_pause();
  }

  // All n threads of a run() meet here.
  void barrier() {
    if (n_ == 1) return;
    const uint32_t g = bar_gen_.load(std::memory_order_acquire);
    if (bar_count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
      bar_count_.store(0, std::memory_order_relaxed);
      bar_gen_.fetch_add(1, std::memory_order_release);
    } else {
      while (bar_gen_.load(std::memory_order_acquire) == g) _mm_pause();
    }
  }

 private:
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
      fn_(ctx_, i);
      pending_.fetch_sub(1, std::memory_order_release);
      b.reset();
    }
  }

  const int n_;
  void (*fn_)(void*, int) = nullptr;
  void* ctx_ = nullptr;
  std::vector<std::thread> threads_;
  alignas(64) std::atomic<uint64_t> gen_{0};
  alignas(64) std::atomic<int> pending_{0};
  alignas(64) std::atomic<uint32_t> bar_count_{0};
  alignas(64) std::atomic<uint32_t> bar_gen_{0};
  alignas(64) std::atomic<bool> stop_{false};
};

// Per-thread row ranges [lo, hi), packed in one word each: the owner takes chunks from the front, idle threads
// steal from the back.
class Rows {
 public:
  void reset(size_t total, int n) {
    for (int t = 0; t < n; ++t) slots_[t].v.store(pack(total * t / n, total * (t + 1) / n), std::memory_order_relaxed);
  }
  bool next(int t, int n, size_t& a, size_t& b) {
    if (take(t, true, a, b)) return true;
    for (int k = 1; k < n; ++k)
      if (take((t + k) % n, false, a, b)) return true;
    return false;
  }

 private:
  static constexpr size_t kChunk = 16;
  static uint64_t pack(size_t lo, size_t hi) { return uint64_t(lo) << 32 | uint64_t(hi); }
  bool take(int t, bool front, size_t& a, size_t& b) {
    auto& v = slots_[t].v;
    uint64_t cur = v.load(std::memory_order_relaxed);
    while (true) {
      const size_t lo = cur >> 32, hi = cur & 0xffffffffu;
      if (lo >= hi) return false;
      const size_t nlo = front ? std::min(lo + kChunk, hi) : lo, nhi = front ? hi : std::max(hi - kChunk, lo);
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
  Slot slots_[64];
};

// One expert class: hidden, inter, gate/up type, down type. An expert blob is gate [inter rows] | up [inter rows] |
// down [hidden rows], each row `in` values of ggml blocks, exactly as the GGUF stores one expert's slices.
struct ExpertClass {
  int hidden = 0, inter = 0;
  ggml_type tgu = GGML_TYPE_COUNT, tdown = GGML_TYPE_COUNT;
  size_t gu_row = 0, down_row = 0, down_off = 0, bytes = 0, xq_row = 0, aq_row = 0;
  ggml_vec_dot_t dot_gu = nullptr, dot_down = nullptr;
  ggml_from_float_t q_gu = nullptr, q_down = nullptr;

  ExpertClass() = default;
  ExpertClass(int hidden_, int inter_, int tgu_, int tdown_)
      : hidden(hidden_), inter(inter_), tgu(ggml_type(tgu_)), tdown(ggml_type(tdown_)) {
    gu_row = ggml_row_size(tgu, hidden);
    down_row = ggml_row_size(tdown, inter);
    down_off = 2 * size_t(inter) * gu_row;
    bytes = down_off + size_t(hidden) * down_row;
    const auto* g = ggml_get_type_traits_cpu(tgu);
    const auto* d = ggml_get_type_traits_cpu(tdown);
    if (!g->vec_dot || !d->vec_dot) throw std::invalid_argument("no ggml-cpu vec_dot for this type");
    dot_gu = g->vec_dot;
    dot_down = d->vec_dot;
    q_gu = ggml_get_type_traits_cpu(g->vec_dot_type)->from_float;
    q_down = ggml_get_type_traits_cpu(d->vec_dot_type)->from_float;
    xq_row = ggml_row_size(g->vec_dot_type, hidden);
    aq_row = ggml_row_size(d->vec_dot_type, inter);
  }
};

struct Job {
  const uint8_t* w;  // the expert blob
  const float* x;    // [tokens, hidden]
  float* y;          // [tokens, hidden]
};

class Batch {
 public:
  // Computes `jobs` (all of class `c`, each with `tokens` tokens) on `pool`; the caller is the pool's thread 0.
  void run(Pool& pool, const ExpertClass& c, const Job* jobs, int njobs, int tokens) {
    const size_t items = size_t(njobs) * tokens;
    if (xq_.size() < items * c.xq_row) xq_.resize(items * c.xq_row);
    if (aq_.size() < items * c.aq_row) aq_.resize(items * c.aq_row);
    if (h_.size() < items * 2 * c.inter) h_.resize(items * 2 * c.inter);
    const int n = pool.size();
    auto split = [n](size_t total, int t, size_t& a, size_t& b) {
      a = total * t / n;
      b = total * (t + 1) / n;
    };
    if (n > 64) throw std::invalid_argument("batch: at most 64 threads");
    gu_.reset(size_t(njobs) * 2 * c.inter, n);
    down_.reset(size_t(njobs) * c.hidden, n);
    uint8_t* xq = xq_.data();
    uint8_t* aq = aq_.data();
    float* h = h_.data();
    auto body = [&](int t) {
      size_t a, b;
      split(items, t, a, b);  // phase 1
      for (size_t i = a; i < b; ++i)
        c.q_gu(jobs[i / tokens].x + (i % tokens) * c.hidden, xq + i * c.xq_row, c.hidden);
      pool.barrier();
      const size_t gu_rows = 2 * size_t(c.inter);
      while (gu_.next(t, n, a, b))  // phase 2
        for (size_t r = a; r < b; ++r) {
          const size_t j = r / gu_rows, rr = r % gu_rows;
          const uint8_t* w = jobs[j].w + rr * c.gu_row;
          for (int k = 0; k < tokens; ++k) {
            const size_t i = j * tokens + k;
            c.dot_gu(c.hidden, h + i * gu_rows + rr, 0, w, 0, xq + i * c.xq_row, 0, 1);
          }
        }
      pool.barrier();
      split(items, t, a, b);  // phase 3
      thread_local std::vector<float> act;
      act.resize(c.inter);
      for (size_t i = a; i < b; ++i) {
        const float* g = h + i * gu_rows;
        const float* u = g + c.inter;
        for (int e = 0; e < c.inter; ++e) act[e] = g[e] / (1.0f + std::exp(-g[e])) * u[e];
        c.q_down(act.data(), aq + i * c.aq_row, c.inter);
      }
      pool.barrier();
      while (down_.next(t, n, a, b))  // phase 4
        for (size_t r = a; r < b; ++r) {
          const size_t j = r / c.hidden, rr = r % c.hidden;
          const uint8_t* w = jobs[j].w + c.down_off + rr * c.down_row;
          for (int k = 0; k < tokens; ++k)
            c.dot_down(c.inter, jobs[j].y + size_t(k) * c.hidden + rr, 0, w, 0, aq + (j * tokens + k) * c.aq_row,
                       0, 1);
        }
    };
    pool.run(body);
  }

 private:
  std::vector<uint8_t> xq_, aq_;
  std::vector<float> h_;
  Rows gu_, down_;
};

}  // namespace x0
