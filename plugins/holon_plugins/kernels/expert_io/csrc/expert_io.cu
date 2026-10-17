// The offload store's in-graph kernels (std.moe.kernels: `moe_resolve@1`, `expert_fetch@1`, `moe_patch_rows@1`; the
// placement plugins' variants `moe_resolve.lfu@1`, `moe_resolve.lru@1`). Every call runs on the tensors' device and torch's current
// stream, reads its sizes from the device only, and is CUDA-graph capturable.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace {

constexpr int PADDING = 0, DECODE = 1;                 // row kinds (std.moe.kernels)
constexpr int RANGE = 1, OVERFLOW = 2, JOBS = 8;       // error bits: a corrupt slot; a delta record past the log;
                                                       // the CPU's share past its job table
constexpr int RESOLVE_THREADS = 256;                   // 8 warps: one per role for the scans
constexpr int FETCH_CTAS = 8, FETCH_THREADS = 256, FETCH_UNROLL = 8;   // step 0's M1: saturates PCIe
// The ring's copies run beside a prefill pass's compute, so they hold as few SMs as still keep PCIe near full (one
// small CTA per class, many reads in flight; A5000: more CTAs slow the pass's mmq by more than they gain)
constexpr int BESIDE_CTAS = 1, BESIDE_UNROLL = 16;

// The resolve's policies: `moe_resolve@1`, and the variants of the placement plugins kernels.expert_io serves
// (`moe_resolve.lfu@1`: F11, `moe_resolve.lru@1`: F4; their contracts and oracles are the plugins').
enum Policy { STATIC = 0, LFU = 1, LRU = 2 };
constexpr float LFU_DECAY = 0.915f;
// The judge's calibration bugs (judge/holon_judge/planted.py), zero otherwise: victims chosen without the call's
// needs, and the second admission of a layer gathered into the first one's victim slot.
constexpr int PLANT_VICTIM_NEEDED = 1, PLANT_SHARED_VICTIM = 2;

struct State {
    float * score;                                     // LFU: decayed counts [E]
    int32_t * last;                                    // LRU: the logical step each expert was last routed [E] …
    int32_t * clock;                                   // … and the layer's step [1]
    int plant;
};

__host__ __device__ inline int shared_words(int policy, int roles, int experts) {
    return policy == STATIC ? (roles + 2) * experts : (roles + 11) * experts + 1;   // + 1: the pairs' 8-byte alignment
}

// The layer's tables of one call (std.moe.kernels' terms).
struct Layer {
    int32_t * residency;                               // [E, R]: the map, written by the variants' admissions
    const int32_t * host;                              // [E, R]: host units
    const int32_t * limits;                            // [R]: the arena sizes
    const int32_t * ring;                              // [E, R]: this call's ring slots (-1: none)
    const int32_t * stage;                             // [R, F]: the slots staged misses are gathered into
    const int32_t * empty;                             // [R, P]: the layer's empty cache slots, the first n_empty
    int32_t * n_empty;
    int F, P;
    const int32_t * plan;                              // [E + 1]: the CPU's share of n candidates (F5b)
    int32_t * jobs;                                    // [J + 1, W]: the CPU's job table (mapped host memory)
    int J, W;
};

// One CTA per call. need[r][e]: 1 if role r of expert e is routed and missing (not resident, not in the ring); an
// admitted expert's entry becomes -2 - its new slot, a staged one's its staging rank. A warp per role ranks the misses
// by id (ballot scans), so staging slots and fetch entries are deterministic; then every entry gets its slot. First
// the CPU takes its share of the misses (F5b: `plan`, by (best routing rank, expert id), the last ones), which are
// neither fetched nor admitted and are written to the job table. The variants then update their state, rank their
// candidates (the call's decode misses, missing in every role of the map, not the CPU's) and victims (resident in
// every role, routed by no live entry of the call) in their policy's order; the first candidates take the layer's
// empty slots, the rest pair with the victims, and the admissions enter the map.
template <int P>
__global__ void k_resolve(const int32_t * __restrict__ ids, const uint8_t * __restrict__ kinds, Layer L,
                          int32_t * __restrict__ counts, int32_t * __restrict__ routed, int32_t * __restrict__ misses,
                          int32_t * __restrict__ errors, int32_t * __restrict__ delta, int32_t * __restrict__ n_delta,
                          int group, int cap, State st, int32_t * __restrict__ slots, int32_t * __restrict__ fetch,
                          int32_t * __restrict__ n_fetch, int32_t * __restrict__ cpu, int tokens, int k, int experts,
                          int roles, int staging) {
    extern __shared__ int32_t shared[];
    int32_t * need = shared;                           // [roles, experts]
    int32_t * best = need + roles * experts;           // [experts]: its least routing rank (INT_MAX: not routed); then
                                                       // its CPU job's index, or -1
    int32_t * ckeys = best + experts;                  // the CPU candidates' keys (rank · E + id), unordered
    int32_t * live = ckeys + experts;                  // [experts]: routed by a live entry of the call (variants)
    int32_t * dec = live + experts;                    // its DECODE entries
    int32_t * first = dec + experts;                   // its first DECODE entry (LRU's candidate order)
    int32_t * cand = first + experts;                  // candidates, by rank
    int32_t * vict = cand + experts;                   // victims, by rank
    // The candidates and the victims, unranked, as (order key, expert id) pairs packed so one comparison ranks them
    // (8-byte aligned: a word skipped when the tables before them hold an odd number).
    int64_t * cands = reinterpret_cast<int64_t *>(vict + experts + ((roles + 7) * experts & 1));
    int64_t * victs = cands + experts;
    __shared__ int has_decode, n_cand, n_vict, n_admit, base, spare, n_cpu, take;
    const int entries = tokens * k, tid = threadIdx.x;
    for (int i = tid; i < shared_words(P, roles, experts); i += blockDim.x) {   // `best`, `first` past every entry
        const bool high = (i >= roles * experts && i < (roles + 1) * experts) ||
                          (i >= (roles + 4) * experts && i < (roles + 5) * experts);
        shared[i] = high ? INT_MAX : 0;
    }
    if (tid == 0) {
        has_decode = n_cand = n_vict = n_cpu = 0;
        if (P != STATIC) {
            base = *n_delta;                           // the layer's log: only this call appends to it now
            spare = *L.n_empty;
        }
    }
    __syncthreads();
    for (int i = tid; i < entries; i += blockDim.x) {
        const int e = ids[i], kd = kinds[i / k];
        if (kd == PADDING || e < 0) {
            continue;
        }
        if (kd == DECODE) {
            atomicAdd(&counts[e], 1);
            if (P != STATIC) {
                atomicAdd(&dec[e], 1);
                atomicMin(&first[e], i);
                has_decode = 1;
            }
        }
        atomicOr(&routed[e], 1);                       // an earlier call's scope bit stays
        atomicMin(&best[e], i % k);
        if (P != STATIC) {
            live[e] = 1;
        }
        for (int r = 0; r < roles; ++r) {
            if (L.residency[e * roles + r] < 0 && L.ring[e * roles + r] < 0) {
                need[r * experts + e] = 1;
            }
        }
    }
    __syncthreads();
    // The CPU's share: candidates routed, missing in every role and held by the tier the device reads (-1).
    for (int e = tid; e < experts; e += blockDim.x) {
        bool held = best[e] != INT_MAX;
        for (int r = 0; r < roles; ++r) {
            held = held && L.residency[e * roles + r] == -1 && L.ring[e * roles + r] < 0;
        }
        if (held) {
            ckeys[atomicAdd(&n_cpu, 1)] = best[e] * experts + e;
        }
    }
    __syncthreads();
    if (tid == 0) {
        take = n_cpu ? min(L.plan[n_cpu], n_cpu) : 0;
        if (take > L.J) {                              // the caller sized the table short: take what fits, flag it
            take = L.J;
            atomicOr(errors, JOBS);
        }
    }
    __syncthreads();
    for (int i = tid; i < n_cpu; i += blockDim.x) {    // ranked by (best rank, id): the last `take` are the CPU's
        const int key = ckeys[i];
        int rank = 0;
        for (int y = 0; y < n_cpu; ++y) {
            rank += ckeys[y] < key;
        }
        if (rank >= n_cpu - take) {
            const int e = key % experts;
            best[e] = -2;                              // taken; its job index below
            for (int r = 0; r < roles; ++r) {
                need[r * experts + e] = 0;
            }
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        best[e] = best[e] == -2 ? -2 : -1;
    }
    __syncthreads();
    if (take && tid < 32) {                            // job indices by id (a warp's ballot scan); the jobs' heads
        int seen = 0;
        for (int b = 0; b < experts; b += 32) {
            const int e = b + tid;
            const bool mine = e < experts && best[e] == -2;
            const unsigned all = __ballot_sync(0xffffffffu, mine);
            if (mine) {
                best[e] = seen + __popc(all & ((1u << tid) - 1));
            }
            seen += __popc(all);
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        if (best[e] >= 0) {
            int32_t * job = L.jobs + int64_t(1 + best[e]) * L.W;
            int n = 0;
            for (int i = 0; i < entries; ++i) {        // its entries, ascending
                if (ids[i] == e && kinds[i / k] != PADDING) {
                    if (n < L.W - 4) {
                        job[4 + n] = i;
                    } else {
                        atomicOr(errors, JOBS);
                    }
                    ++n;
                }
            }
            n = min(n, L.W - 4);
            atomicOr(&routed[e], 4);                   // the CPU's (the route taps see it)
            job[0] = e;
            job[1] = L.host[e * roles];
            job[2] = L.host[e * roles + 1];
            job[3] = n;
        }
    }
    if (tid == 0) {
        L.jobs[0] = take;
    }
    __syncthreads();
    if (P != STATIC && has_decode) {                   // a step with decode rows: decay (or tick), count, classify
        const int now = P == LRU ? st.clock[0] + 1 : 0;
        for (int e = tid; e < experts; e += blockDim.x) {
            int key = 0;
            if (P == LFU) {                            // in f32, rounded each time: x0.915, then +1.0 per entry
                float s = __fmul_rn(st.score[e], LFU_DECAY);
                for (int c = 0; c < dec[e]; ++c) {
                    s = __fadd_rn(s, 1.0f);
                }
                st.score[e] = s;
                key = __float_as_int(s);               // non-negative: its bits order as the floats do
            } else {
                key = dec[e] ? now : st.last[e];
                if (dec[e]) {
                    st.last[e] = now;
                }
            }
            bool missing = true, whole = true;
            for (int r = 0; r < roles; ++r) {
                const int at = L.residency[e * roles + r];
                missing = missing && at < 0;
                whole = whole && at >= k + staging && at < L.limits[r];
            }
            if (live[e]) {                             // routed by the call that picks victims: the victims' scope
                routed[e] |= 2;
            }
            const bool is_cand = dec[e] && missing && best[e] < 0;    // not the CPU's
            const bool is_vict = !is_cand && whole && (!live[e] || (st.plant & PLANT_VICTIM_NEEDED));
            // Candidates hottest (LFU: score descending; LRU: routed first) first, victims coldest (least recent) first,
            // ties by expert id: ascending in the packed order.
            const int ckey = P == LFU ? INT_MAX - key : first[e];
            if (is_cand) {
                cands[atomicAdd(&n_cand, 1)] = (int64_t(ckey) << 32) | e;
            } else if (is_vict) {
                victs[atomicAdd(&n_vict, 1)] = (int64_t(key) << 32) | e;
            }
            // The order's key, in `first`: LFU's score; LRU's first decode entry (a candidate) or last stamp (a
            // victim).
            first[e] = P == LRU && is_cand ? first[e] : key;
        }
        __syncthreads();
        const int empties = min(n_cand, spare);        // the first candidates take the empty slots, unconditionally
        const int pairs = min(n_cand - empties, n_vict);
        if (P == LRU && tid == 0) {
            st.clock[0] = now;
        }
        if (empties + pairs) {                         // a call with no candidate (or nowhere to put it) admits nothing
            if (tid == 0) {
                n_admit = pairs;
            }
            // Ranks over the unranked lists (their order is not deterministic, the ranks are).
            for (int j = tid; j < n_cand + n_vict; j += blockDim.x) {
                const bool c = j < n_cand;
                const int64_t * list = c ? cands : victs;
                const int64_t mine = list[c ? j : j - n_cand];
                const int n = c ? n_cand : n_vict;
                int rank = 0;
#pragma unroll 8
                for (int y = 0; y < n; ++y) {
                    rank += list[y] < mine;
                }
                if (rank < (c ? empties + pairs : pairs)) {    // ranked past its use: never used
                    (c ? cand : vict)[rank] = int(mine & 0xffffffff);
                }
            }
            __syncthreads();
            if (P == LFU) {                            // admitted while strictly hotter; the first failing pair stops it
                for (int i = tid; i < pairs; i += blockDim.x) {
                    if (!(first[cand[empties + i]] > first[vict[i]])) {
                        atomicMin(&n_admit, i);
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                *n_delta = base + empties + n_admit;
                *L.n_empty = spare - empties;
            }
            for (int i = tid; i < empties + n_admit; i += blockDim.x) {   // the slots, read before any is rewritten
                const int a = cand[i];
                for (int r = 0; r < roles; ++r) {
                    const int b = i < empties ? -1 : vict[(st.plant & PLANT_SHARED_VICTIM) && i == empties + 1 ? 0
                                                                                                             : i - empties];
                    need[r * experts + a] = -2 - (b < 0 ? L.empty[r * L.P + spare - 1 - i] : L.residency[b * roles + r]);
                }
            }
            __syncthreads();
            for (int i = tid; i < empties + n_admit; i += blockDim.x) {
                const int a = cand[i];
                const int b = i < empties ? -2 - (spare - 1 - i)
                                          : vict[(st.plant & PLANT_SHARED_VICTIM) && i == empties + 1 ? 0 : i - empties];
                for (int r = 0; r < roles; ++r) {
                    L.residency[a * roles + r] = -2 - need[r * experts + a];
                    if (b >= 0) {
                        L.residency[b * roles + r] = -1;
                    }
                }
                if (base + i < cap) {
                    delta[(base + i) * 3] = group;
                    delta[(base + i) * 3 + 1] = a;
                    delta[(base + i) * 3 + 2] = b;
                } else {
                    atomicOr(errors, OVERFLOW);
                }
            }
            __syncthreads();
        }
    }
    const int warp = tid / 32, lane = tid % 32;
    if (warp < roles) {
        int32_t * row = need + warp * experts;
        int32_t * units = fetch + warp * 2 * L.F, * dst = units + L.F;
        const int32_t * stage = L.stage + warp * L.F;
        int seen = 0, staged = 0;
        for (int base_e = 0; base_e < experts; base_e += 32) {
            const int e = base_e + lane;
            const int v = e < experts ? row[e] : 0;
            const unsigned all = __ballot_sync(0xffffffffu, v != 0), mine = __ballot_sync(0xffffffffu, v == 1);
            const unsigned below = (1u << lane) - 1;
            const int rank = seen + __popc(all & below), srank = staged + __popc(mine & below);
            if (v != 0 && rank < L.F) {
                const int32_t unit = L.host[e * roles + warp];
                units[rank] = unit >= 0 ? unit : -2 - e;   // no device-readable tier holds it: moe_tiers@1's
                dst[rank] = v == 1 ? stage[min(srank, L.F - 1)] : -2 - v;
            }
            if (v == 1) {
                row[e] = srank;
            }
            seen += __popc(all);
            staged += __popc(mine);
        }
        if (lane == 0) {
            n_fetch[warp] = seen;
            atomicMax(misses, seen);
        }
    }
    __syncthreads();
    for (int i = tid; i < entries; i += blockDim.x) {
        const int e = ids[i], j = i % k;
        const bool live_entry = kinds[i / k] != PADDING && e >= 0;
        const bool on_cpu = live_entry && best[e] >= 0;
        cpu[i] = on_cpu ? i : -1;
        for (int r = 0; r < roles; ++r) {
            int32_t at = j;
            if (live_entry && !on_cpu) {
                at = L.residency[e * roles + r];
                const int32_t rg = L.ring[e * roles + r];
                if (at < 0 && rg < 0) {                // staged
                    at = L.stage[r * L.F + min(need[r * experts + e], L.F - 1)];
                } else {
                    at = at < 0 ? rg : at;
                    if (at < k + staging || at >= L.limits[r]) {      // corrupt: never read past the arena
                        at = j;
                        atomicOr(errors, RANGE);
                    }
                }
            }
            slots[r * entries + i] = at;
        }
    }
}

// Every CTA walks every fetched unit; its threads move UNROLL 16-byte words each per trip, all loads before all
// stores, so a CTA keeps blockDim * UNROLL * 16 bytes of reads in flight (zero-copy reads of host memory).
template <int UNROLL>
__global__ void k_fetch(const int32_t * __restrict__ fetch, const int32_t * __restrict__ n, const uint4 * __restrict__ src,
                        uint4 * __restrict__ dst, int64_t words, int staging) {
    const int count = min(*n, staging);
    const int64_t stride = int64_t(gridDim.x) * blockDim.x * UNROLL;
    for (int b = 0; b < count; ++b) {
        const uint4 * __restrict__ s = src + int64_t(fetch[b]) * words;
        uint4 * __restrict__ d = dst + int64_t(fetch[staging + b]) * words;
        for (int64_t base = int64_t(blockIdx.x) * blockDim.x * UNROLL + threadIdx.x; base < words;
             base += stride) {
            uint4 v[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int64_t i = base + int64_t(u) * blockDim.x;
                if (i < words) {
                    v[u] = s[i];
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int64_t i = base + int64_t(u) * blockDim.x;
                if (i < words) {
                    d[i] = v[u];
                }
            }
        }
    }
}

void check(const at::Tensor & t, const at::Tensor & like, at::ScalarType dtype, at::IntArrayRef shape, const char * name) {
    TORCH_CHECK(t.scalar_type() == dtype && t.device() == like.device() && t.is_contiguous() && t.sizes() == shape,
                name, ": contiguous ", shape, " of ", dtype, " on ", like.device(), ", got ", t.sizes(), " of ",
                t.scalar_type(), " on ", t.device());
}

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

// `state`: the policy's device state (LFU: score [E] f32; LRU: last [E], clock [1] int32).
void resolve(int64_t policy, const at::Tensor & ids, const at::Tensor & kinds, at::Tensor & residency,
             const at::Tensor & host, const at::Tensor & limits, at::Tensor & counts, at::Tensor & routed,
             const at::Tensor & ring, const at::Tensor & stage, const at::Tensor & empty, at::Tensor & n_empty,
             at::Tensor & misses, at::Tensor & errors, at::Tensor & delta, at::Tensor & n_delta, int64_t group,
             const std::vector<at::Tensor> & state, int64_t plant, const at::Tensor & plan, at::Tensor & jobs,
             at::Tensor & slots, at::Tensor & fetch, at::Tensor & n_fetch, at::Tensor & cpu, int64_t staging) {
    const c10::cuda::CUDAGuard guard(ids.device());
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && residency.dim() == 2 && stage.dim() == 2 && empty.dim() == 2 &&
                delta.dim() == 2, "ids [T, k] on CUDA");
    const int64_t tokens = ids.size(0), k = ids.size(1), experts = residency.size(0), roles = residency.size(1);
    const int64_t width = stage.size(1);
    TORCH_CHECK(roles <= RESOLVE_THREADS / 32 && width > 0, "at most ", RESOLVE_THREADS / 32, " roles, a stage list");
    check(ids, ids, at::kInt, {tokens, k}, "ids");
    check(kinds, ids, at::kByte, {tokens}, "kinds");
    check(residency, ids, at::kInt, {experts, roles}, "residency");
    check(host, ids, at::kInt, {experts, roles}, "host");
    check(limits, ids, at::kInt, {roles}, "limits");
    check(counts, ids, at::kInt, {experts}, "counts");
    check(routed, ids, at::kInt, {experts}, "routed");
    check(ring, ids, at::kInt, {experts, roles}, "ring");
    check(stage, ids, at::kInt, {roles, width}, "stage");
    check(empty, ids, at::kInt, {roles, empty.size(1)}, "empty");
    check(n_empty, ids, at::kInt, {1}, "n_empty");
    check(misses, ids, at::kInt, {1}, "misses");
    check(errors, ids, at::kInt, {1}, "errors");
    check(delta, ids, at::kInt, {delta.size(0), 3}, "delta");
    check(n_delta, ids, at::kInt, {1}, "n_delta");
    check(slots, ids, at::kInt, {roles, tokens, k}, "slots");
    check(fetch, ids, at::kInt, {roles, 2, width}, "fetch");
    check(n_fetch, ids, at::kInt, {roles}, "n_fetch");
    check(plan, ids, at::kInt, {experts + 1}, "plan");
    check(jobs, ids, at::kInt, {jobs.size(0), jobs.size(1)}, "jobs");
    check(cpu, ids, at::kInt, {tokens, k}, "cpu");
    TORCH_CHECK(jobs.dim() == 2 && jobs.size(1) >= 4, "jobs [J + 1, 4 + T']");
    State st{nullptr, nullptr, nullptr, int(plant)};
    if (policy == LFU) {
        TORCH_CHECK(state.size() == 1, "lfu: score");
        check(state[0], ids, at::kFloat, {experts}, "score");
        st.score = state[0].data_ptr<float>();
    } else if (policy == LRU) {
        TORCH_CHECK(state.size() == 2, "lru: last, clock");
        check(state[0], ids, at::kInt, {experts}, "last");
        check(state[1], ids, at::kInt, {1}, "clock");
        st.last = state[0].data_ptr<int32_t>();
        st.clock = state[1].data_ptr<int32_t>();
    } else {
        TORCH_CHECK(policy == STATIC && state.empty(), "policy ", policy);
    }
    Layer L{residency.data_ptr<int32_t>(), host.data_ptr<int32_t>(), limits.data_ptr<int32_t>(), ring.data_ptr<int32_t>(),
            stage.data_ptr<int32_t>(), empty.data_ptr<int32_t>(), n_empty.data_ptr<int32_t>(), int(width),
            int(empty.size(1)), plan.data_ptr<int32_t>(), jobs.data_ptr<int32_t>(), int(jobs.size(0) - 1),
            int(jobs.size(1))};
    const size_t smem = shared_words(int(policy), int(roles), int(experts)) * sizeof(int32_t);
    auto kernel = policy == LFU ? k_resolve<LFU> : policy == LRU ? k_resolve<LRU> : k_resolve<STATIC>;
    kernel<<<1, RESOLVE_THREADS, smem, stream()>>>(
        ids.data_ptr<int32_t>(), kinds.data_ptr<uint8_t>(), L, counts.data_ptr<int32_t>(), routed.data_ptr<int32_t>(),
        misses.data_ptr<int32_t>(), errors.data_ptr<int32_t>(), delta.data_ptr<int32_t>(), n_delta.data_ptr<int32_t>(),
        int(group), int(delta.size(0)), st, slots.data_ptr<int32_t>(), fetch.data_ptr<int32_t>(),
        n_fetch.data_ptr<int32_t>(), cpu.data_ptr<int32_t>(), tokens, k, experts, roles, staging);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// moe_tiers@1 (std.moe.kernels): one CTA after a layer call's resolve keeps the RAM tier exclusive and provides the
// pieces no device-readable tier holds. Decisions run in order (admissions in record order, then the SSD misses by id)
// on thread 0; the searches they need (a class's first free RAM unit, RAM's coldest expert of the layer's classes) are
// block-wide reductions every thread joins, in lockstep (the loop conditions read shared memory only).
struct Tiers {
    int32_t * residency, * host;                       // [G, E, R]
    const int32_t * piece, * classes;                  // [G, E, R], [G, R]
    int32_t * owner;                                   // [C, U]
    const int32_t * capacity;                          // [C]
    const float * heat;                                // [G, E] or null
    int32_t * io, * moves, * n_moves;                  // [R, 1 + F, 2], [R, 4, F], [R, 2]
    int groups, units;
};

constexpr int TIER_THREADS = 256;

// The first free unit of class c (-1: none), every thread joining.
__device__ int first_free(const Tiers & T, int c, int * red) {
    int best = INT_MAX;
    for (int u = threadIdx.x; u < T.capacity[c]; u += blockDim.x) {
        if (T.owner[c * T.units + u] < 0) {
            best = min(best, u);
            break;                                     // a thread's later units are larger
        }
    }
    red[threadIdx.x] = best;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w /= 2) {
        if (threadIdx.x < w) {
            red[threadIdx.x] = min(red[threadIdx.x], red[threadIdx.x + w]);
        }
        __syncthreads();
    }
    const int out = red[0];
    __syncthreads();
    return out == INT_MAX ? -1 : out;
}

// RAM's coldest whole expert of layer g's classes, strictly colder than `than`, not routed by the call in layer g and
// not placed by it (-1: none): its index g' * E + e', least (heat, index).
__device__ int coldest(const Tiers & T, int g, int roles, int experts, float than, const int32_t * flags,
                       float * hk, int * ik) {
    float bh = INFINITY;
    int bi = INT_MAX;
    for (int x = threadIdx.x; x < T.groups * experts; x += blockDim.x) {
        const int gg = x / experts, e = x % experts;
        bool ok = !(gg == g && flags[e] & (1 | 4));    // routed by the call (1), placed by it (4)
        for (int r = 0; r < roles && ok; ++r) {
            ok = T.classes[gg * roles + r] == T.classes[g * roles + r] && T.residency[x * roles + r] == -1;
        }
        if (ok) {
            const float h = T.heat[x];
            if (h < bh || (h == bh && x < bi)) {
                bh = h;
                bi = x;
            }
        }
    }
    hk[threadIdx.x] = bh;
    ik[threadIdx.x] = bi;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w /= 2) {
        if (threadIdx.x < w) {
            const float h = hk[threadIdx.x + w];
            const int i = ik[threadIdx.x + w];
            if (h < hk[threadIdx.x] || (h == hk[threadIdx.x] && i < ik[threadIdx.x])) {
                hk[threadIdx.x] = h;
                ik[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    const float h = hk[0];
    const int i = ik[0];
    __syncthreads();
    return i != INT_MAX && h < than ? i : -1;
}

// Expert e of layer g into RAM: `given` units (one per role) if not null, else free units of every class, else the
// coldest's (displaced to the SSD), else the SSD. Thread 0 writes; `units` gets the result (-1: the SSD).
__device__ void place(const Tiers & T, int g, int e, int roles, int experts, const int * given, int32_t * flags,
                      int * units, int * red, float * hk) {
    __shared__ int ok;
    if (given == nullptr) {
        if (threadIdx.x == 0) {
            ok = 1;
        }
        for (int r = 0; r < roles; ++r) {
            const int u = first_free(T, T.classes[g * roles + r], red);
            if (threadIdx.x == 0) {
                units[r] = u;
                ok = ok && u >= 0;
            }
            __syncthreads();
        }
        if (!ok) {
            const int x = T.heat ? coldest(T, g, roles, experts, T.heat[g * experts + e], flags, hk, red) : -1;
            if (threadIdx.x == 0) {
                for (int r = 0; r < roles; ++r) {
                    units[r] = x >= 0 ? T.host[x * roles + r] : -1;
                    if (x >= 0) {
                        T.residency[x * roles + r] = -2;
                        T.host[x * roles + r] = -1;
                    }
                }
            }
        }
    } else if (threadIdx.x == 0) {
        for (int r = 0; r < roles; ++r) {
            units[r] = given[r];
        }
    }
    if (threadIdx.x == 0) {
        const int self = g * experts + e;
        for (int r = 0; r < roles; ++r) {
            const int c = T.classes[g * roles + r];
            if (units[0] >= 0) {
                T.owner[c * T.units + units[r]] = self;
            }
            T.residency[self * roles + r] = units[0] >= 0 ? -1 : -2;
            T.host[self * roles + r] = units[0] >= 0 ? units[r] : -1;
        }
        if (units[0] >= 0) {
            flags[e] |= 4;
        }
    }
    __syncthreads();
}

__global__ void k_tiers(const int32_t * __restrict__ ids, const uint8_t * __restrict__ kinds, int g, int32_t * fetch,
                        const int32_t * n_fetch, const int32_t * stage, const int32_t * delta, const int32_t * start,
                        const int32_t * n_delta, Tiers T, int tokens, int k, int experts, int roles, int F) {
    extern __shared__ int32_t flags[];                 // [experts]: 1 routed, 2 by a DECODE row, 4 placed, 8 admitted,
    __shared__ int red[TIER_THREADS];                  // 16 an SSD miss
    __shared__ float hk[TIER_THREADS];
    __shared__ int units[8], src[8], staged[8], n_adm;
    const int tid = threadIdx.x, entries = tokens * k, n_rec = *n_delta - *start;
    for (int e = tid; e < experts; e += blockDim.x) {
        flags[e] = 0;
    }
    __syncthreads();
    for (int i = tid; i < entries; i += blockDim.x) {
        const int e = ids[i], kd = kinds[i / k];
        if (kd != PADDING && e >= 0) {
            atomicOr(&flags[e], kd == DECODE ? 3 : 1);
        }
    }
    for (int r = 0; r < roles; ++r) {
        for (int i = tid; i < min(n_fetch[r], F); i += blockDim.x) {
            const int s0 = fetch[(r * 2) * F + i];
            if (s0 <= -2) {
                atomicOr(&flags[-2 - s0], 16);
            }
        }
    }
    if (tid == 0) {
        n_adm = 0;
        for (int i = 0; i < n_rec; ++i) {
            const int * rec = delta + (*start + i) * 3;
            if (rec[1] >= 0) {
                flags[rec[1]] |= 8;
                ++n_adm;
            }
        }
        for (int r = 0; r < roles; ++r) {
            staged[r] = n_fetch[r] - n_adm;
            T.n_moves[r * 2] = T.n_moves[r * 2 + 1] = 0;
            T.io[r * (1 + F) * 2] = 0;
        }
    }
    __syncthreads();
    int j = 0;
    for (int i = 0; i < n_rec; ++i) {                  // 1. admissions (uniform: every thread reads the same record)
        const int * rec = delta + (*start + i) * 3;
        const int a = rec[1], b = rec[2];
        if (a < 0) {
            continue;
        }
        if (tid == 0) {
            for (int r = 0; r < roles; ++r) {
                const int slot = T.residency[(g * experts + a) * roles + r];
                int at = 0;
                while (at < min(n_fetch[r], F) && fetch[(r * 2 + 1) * F + at] != slot) {
                    ++at;
                }
                src[r] = fetch[(r * 2) * F + at];
                if (b >= 0) {                          // into a victim's slot: staged, then moved there
                    const int to = stage[r * F + staged[r] + j];
                    fetch[(r * 2 + 1) * F + at] = to;
                    const int m = T.n_moves[r * 2 + 1]++;
                    T.moves[(r * 4 + 2) * F + m] = to;
                    T.moves[(r * 4 + 3) * F + m] = slot;
                } else if (src[r] >= 0) {              // into an empty slot: its RAM unit freed
                    T.owner[T.classes[g * roles + r] * T.units + src[r]] = -1;
                }
            }
        }
        __syncthreads();
        if (b >= 0) {
            bool from_ram = true;
            for (int r = 0; r < roles; ++r) {
                from_ram = from_ram && src[r] >= 0;
            }
            __shared__ int given[8], victim[8];
            if (tid == 0) {
                for (int r = 0; r < roles; ++r) {
                    victim[r] = T.residency[(g * experts + a) * roles + r];
                    given[r] = src[r];
                    if (from_ram) {
                        T.owner[T.classes[g * roles + r] * T.units + src[r]] = -1;
                    }
                }
            }
            __syncthreads();
            place(T, g, b, roles, experts, from_ram ? given : nullptr, flags, units, red, hk);
            if (tid == 0 && units[0] >= 0) {
                for (int r = 0; r < roles; ++r) {
                    const int m = T.n_moves[r * 2]++;
                    T.moves[(r * 4) * F + m] = victim[r];
                    T.moves[(r * 4 + 1) * F + m] = units[r];
                }
            }
            ++j;
        }
        if (tid == 0) {
            for (int r = 0; r < roles; ++r) {
                T.host[(g * experts + a) * roles + r] = -1;
            }
        }
        __syncthreads();
    }
    for (int e = 0; e < experts; ++e) {                // 2. keeps, 3. the reads (uniform: shared flags)
        if (!(flags[e] & 16)) {
            continue;
        }
        bool kept = false;
        if (!(flags[e] & 8) && (flags[e] & 2)) {
            place(T, g, e, roles, experts, nullptr, flags, units, red, hk);
            kept = units[0] >= 0;
        }
        if (tid == 0) {
            for (int r = 0; r < roles; ++r) {
                int at = 0;
                while (fetch[(r * 2) * F + at] != -2 - e) {
                    ++at;
                }
                const int unit = kept ? units[r] : T.capacity[T.classes[g * roles + r]] + at;
                int32_t * io = T.io + r * (1 + F) * 2;
                const int n = io[0]++;
                io[2 + 2 * n] = T.piece[(g * experts + e) * roles + r];
                io[3 + 2 * n] = unit;
                fetch[(r * 2) * F + at] = unit;
            }
        }
        __syncthreads();
    }
}

void tiers(const at::Tensor & ids, const at::Tensor & kinds, int64_t group, at::Tensor & fetch, const at::Tensor & n_fetch,
           const at::Tensor & stage, const at::Tensor & delta, const at::Tensor & start, const at::Tensor & n_delta,
           at::Tensor & residency, at::Tensor & host, const at::Tensor & piece, const at::Tensor & classes,
           at::Tensor & owner, const at::Tensor & capacity, const c10::optional<at::Tensor> & heat, at::Tensor & io,
           at::Tensor & moves, at::Tensor & n_moves) {
    const c10::cuda::CUDAGuard guard(ids.device());
    const int64_t tokens = ids.size(0), k = ids.size(1), groups = residency.size(0), experts = residency.size(1),
                  roles = residency.size(2), F = stage.size(1);
    TORCH_CHECK(roles <= 8 && F > 0, "at most 8 roles, a stage list");
    check(ids, ids, at::kInt, {tokens, k}, "ids");
    check(kinds, ids, at::kByte, {tokens}, "kinds");
    check(fetch, ids, at::kInt, {roles, 2, F}, "fetch");
    check(n_fetch, ids, at::kInt, {roles}, "n_fetch");
    check(stage, ids, at::kInt, {roles, F}, "stage");
    check(delta, ids, at::kInt, {delta.size(0), 3}, "delta");
    check(start, ids, at::kInt, {1}, "start");
    check(n_delta, ids, at::kInt, {1}, "n_delta");
    check(residency, ids, at::kInt, {groups, experts, roles}, "residency");
    check(host, ids, at::kInt, {groups, experts, roles}, "host");
    check(piece, ids, at::kInt, {groups, experts, roles}, "piece");
    check(classes, ids, at::kInt, {groups, roles}, "classes");
    check(owner, ids, at::kInt, {owner.size(0), owner.size(1)}, "owner");
    check(capacity, ids, at::kInt, {owner.size(0)}, "capacity");
    check(io, ids, at::kInt, {roles, 1 + F, 2}, "io");
    check(moves, ids, at::kInt, {roles, 4, F}, "moves");
    check(n_moves, ids, at::kInt, {roles, 2}, "n_moves");
    const float * h = nullptr;
    if (heat.has_value()) {
        check(*heat, ids, at::kFloat, {groups, experts}, "heat");
        h = heat->data_ptr<float>();
    }
    Tiers T{residency.data_ptr<int32_t>(), host.data_ptr<int32_t>(), piece.data_ptr<int32_t>(),
            classes.data_ptr<int32_t>(), owner.data_ptr<int32_t>(), capacity.data_ptr<int32_t>(), h,
            io.data_ptr<int32_t>(), moves.data_ptr<int32_t>(), n_moves.data_ptr<int32_t>(), int(groups),
            int(owner.size(1))};
    k_tiers<<<1, TIER_THREADS, experts * sizeof(int32_t), stream()>>>(
        ids.data_ptr<int32_t>(), kinds.data_ptr<uint8_t>(), int(group), fetch.data_ptr<int32_t>(),
        n_fetch.data_ptr<int32_t>(), stage.data_ptr<int32_t>(), delta.data_ptr<int32_t>(), start.data_ptr<int32_t>(),
        n_delta.data_ptr<int32_t>(), T, int(tokens), int(k), int(experts), int(roles), int(F));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// moe_patch_rows@1: a token's input row (as f32) into the CPU's mapped rows when one of its entries is the CPU's; an
// entry's CPU output row into y. A CTA per row.
template <typename T>
__global__ void k_publish(const T * __restrict__ x, const int32_t * __restrict__ cpu, float * __restrict__ out, int k,
                          int hidden) {
    const int t = blockIdx.x;
    bool any = false;
    for (int j = 0; j < k; ++j) {
        any = any || cpu[t * k + j] >= 0;
    }
    if (any) {
        for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
            out[int64_t(t) * hidden + c] = float(x[int64_t(t) * hidden + c]);
        }
    }
}

__global__ void k_patch(float * __restrict__ y, const float * __restrict__ rows, const int32_t * __restrict__ cpu,
                        int hidden) {
    const int i = blockIdx.x, at = cpu[i];
    if (at >= 0) {
        for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
            y[int64_t(i) * hidden + c] = rows[int64_t(at) * hidden + c];
        }
    }
}

void publish(const at::Tensor & x, const at::Tensor & cpu, at::Tensor & out) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && cpu.dim() == 2 && cpu.size(0) == x.size(0),
                "x [T, H] contiguous on CUDA, cpu [T, k]");
    check(cpu, x, at::kInt, {x.size(0), cpu.size(1)}, "cpu");
    check(out, x, at::kFloat, {out.size(0), x.size(1)}, "out");
    TORCH_CHECK(out.size(0) >= x.size(0), "out has fewer rows than x");
    if (!x.size(0)) {
        return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x.scalar_type(), "publish", [&] {
        k_publish<scalar_t><<<x.size(0), 256, 0, stream()>>>(x.data_ptr<scalar_t>(), cpu.data_ptr<int32_t>(),
                                                             out.data_ptr<float>(), int(cpu.size(1)), int(x.size(1)));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void patch(at::Tensor & y, const at::Tensor & rows, const at::Tensor & cpu) {
    const c10::cuda::CUDAGuard guard(y.device());
    TORCH_CHECK(y.is_cuda() && y.dim() == 3 && y.scalar_type() == at::kFloat && y.is_contiguous(), "y [T, k, H] f32");
    check(cpu, y, at::kInt, {y.size(0), y.size(1)}, "cpu");
    check(rows, y, at::kFloat, {rows.size(0), y.size(2)}, "rows");
    if (!y.size(0)) {
        return;
    }
    k_patch<<<y.size(0) * y.size(1), 256, 0, stream()>>>(y.data_ptr<float>(), rows.data_ptr<float>(),
                                                          cpu.data_ptr<int32_t>(), int(y.size(2)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather(const at::Tensor & fetch, const at::Tensor & n, const at::Tensor & src, at::Tensor & dst, bool beside) {
    const c10::cuda::CUDAGuard guard(dst.device());
    TORCH_CHECK(dst.is_cuda() && dst.dim() == 2 && src.dim() == 2 && dst.size(1) == src.size(1) && dst.size(1) % 16 == 0,
                "src [U, unit], dst [S, unit] with unit a multiple of 16 bytes");
    TORCH_CHECK(fetch.dim() == 2 && fetch.size(0) == 2, "fetch [2, staging]");
    check(fetch, dst, at::kInt, {2, fetch.size(1)}, "fetch");
    check(n, dst, at::kInt, {1}, "n");
    check(src, dst, at::kByte, {src.size(0), src.size(1)}, "src");
    check(dst, dst, at::kByte, {dst.size(0), dst.size(1)}, "dst");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0,
                "src and dst 16-byte aligned");
    const auto * f = fetch.data_ptr<int32_t>();
    const auto * from = reinterpret_cast<const uint4 *>(src.data_ptr());
    auto * to = reinterpret_cast<uint4 *>(dst.data_ptr());
    if (beside) {
        k_fetch<BESIDE_UNROLL><<<BESIDE_CTAS, FETCH_THREADS, 0, stream()>>>(f, n.data_ptr<int32_t>(), from, to, dst.size(1) / 16,
                                                                          int(fetch.size(1)));
    } else {
        k_fetch<FETCH_UNROLL><<<FETCH_CTAS, FETCH_THREADS, 0, stream()>>>(f, n.data_ptr<int32_t>(), from, to, dst.size(1) / 16,
                                                                         int(fetch.size(1)));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// expert_prefetch@1 from the host's copy list: unit src[fetch[0, i]] -> slot dst[fetch[1, i]] for i < n, as one batch
// of copy-engine DMA on `stream`, a run of consecutive units into consecutive slots one copy. `src` is pinned host
// memory; the list is read before this returns.
void copy_batch(const at::Tensor & fetch, int64_t n, const at::Tensor & src, at::Tensor & dst, int64_t stream_handle) {
    TORCH_CHECK(!src.is_cuda() && src.is_pinned() && dst.is_cuda() && src.dim() == 2 && dst.dim() == 2 &&
                src.size(1) == dst.size(1) && src.is_contiguous() && dst.is_contiguous() && src.scalar_type() == at::kByte &&
                dst.scalar_type() == at::kByte, "src [U, unit] pinned host bytes, dst [S, unit] CUDA bytes");
    TORCH_CHECK(!fetch.is_cuda() && fetch.scalar_type() == at::kInt && fetch.dim() == 2 && fetch.size(0) == 2 &&
                n <= fetch.size(1), "fetch [2, F] int32 on the host, n <= F");
    const c10::cuda::CUDAGuard guard(dst.device());
    const auto f = fetch.contiguous();
    const int32_t * u = f.data_ptr<int32_t>();
    const int32_t * s = u + f.size(1);
    const int64_t unit = dst.size(1);
    const auto * from = static_cast<const uint8_t *>(src.data_ptr());
    auto * to = static_cast<uint8_t *>(dst.data_ptr());
    std::vector<void *> dsts;
    std::vector<const void *> srcs;
    std::vector<size_t> sizes;
    for (int64_t i = 0; i < n;) {
        int64_t run = 1;
        while (i + run < n && u[i + run] == u[i] + run && s[i + run] == s[i] + run) {
            ++run;
        }
        TORCH_CHECK(u[i] >= 0 && u[i] + run <= src.size(0) && s[i] >= 0 && s[i] + run <= dst.size(0),
                    "copy ", i, " outside src or dst");
        dsts.push_back(to + int64_t(s[i]) * unit);
        srcs.push_back(from + int64_t(u[i]) * unit);
        sizes.push_back(size_t(run * unit));
        i += run;
    }
    if (dsts.empty()) {
        return;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attrs = {};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t index = 0;
#if CUDART_VERSION >= 13000
    C10_CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attrs, &index, 1, stream));
#else
    size_t failed = 0;
    C10_CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), const_cast<void **>(srcs.data()), sizes.data(), dsts.size(), &attrs,
                                        &index, 1, &failed, stream));
#endif
#else
    for (size_t i = 0; i < dsts.size(); ++i) {
        C10_CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyHostToDevice, stream));
    }
#endif
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resolve", &resolve, "moe_resolve@1 (policy 0) or a variant into slots, fetch, n_fetch (and the counts, misses, "
          "errors, delta log, map and state in place)");
    m.def("tiers", &tiers, "moe_tiers@1: the call's admissions and SSD misses through the exclusive RAM tier");
    m.def("publish", &publish, "moe_patch_rows@1's publish: x's rows with a CPU entry into the CPU's mapped rows");
    m.def("patch", &patch, "moe_patch_rows@1: y[t, j] = rows[cpu[t, j]] where it is >= 0");
    m.def("copy_batch", &copy_batch, "expert_prefetch@1 from the host's list: copy-engine DMA on a stream");
    m.def("fetch", &gather, "expert_fetch@1 (beside: expert_prefetch@1's, on few SMs): dst[fetch[1, i]] = src[fetch[0, i]] "
          "for i < min(n, staging)");
}
