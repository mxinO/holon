// The offload store's in-graph kernels (std.moe.kernels: `moe_resolve@1`, `expert_fetch@1`; the placement plugins'
// variants `moe_resolve.lfu@1`, `moe_resolve.lru@1`). Every call runs on the tensors' device and torch's current
// stream, reads its sizes from the device only, and is CUDA-graph capturable.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace {

constexpr int PADDING = 0, DECODE = 1;                 // row kinds (std.moe.kernels)
constexpr int RANGE = 1, OVERFLOW = 2;                 // error bits: a corrupt slot; a delta record past the log
constexpr int RESOLVE_THREADS = 256;                   // 8 warps: one per role for the scans
constexpr int FETCH_CTAS = 8, FETCH_THREADS = 256, FETCH_UNROLL = 8;   // step 0's M1: saturates PCIe

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
    return (roles + (policy == STATIC ? 0 : 6)) * experts;
}

// One CTA per call. need[r][e]: 1 if role r of expert e is routed and missing; an admitted expert's entry becomes
// -2 - its new slot, a staged one's its staging rank. A warp per role ranks the misses by id (ballot scans), so
// staging slots and fetch entries are deterministic; then every entry gets its slot. The variants first update
// their state, then pair candidates (the call's decode misses, missing in every role) with victims (resident in
// every role, routed by no live entry of the call) in their policy's order and admit the pairs into the map.
template <int P>
__global__ void k_resolve(const int32_t * __restrict__ ids, const uint8_t * __restrict__ kinds,
                          int32_t * __restrict__ residency, const int32_t * __restrict__ host,
                          const int32_t * __restrict__ limits, int32_t * __restrict__ counts,
                          int32_t * __restrict__ misses, int32_t * __restrict__ errors, int32_t * __restrict__ delta,
                          int32_t * __restrict__ n_delta, int group, int cap, State st,
                          int32_t * __restrict__ slots, int32_t * __restrict__ fetch, int32_t * __restrict__ n_fetch,
                          int tokens, int k, int experts, int roles, int staging) {
    extern __shared__ int32_t shared[];
    int32_t * need = shared;                           // [roles, experts]
    int32_t * live = need + roles * experts;           // [experts]: routed by a live entry of the call (variants)
    int32_t * dec = live + experts;                    // its DECODE entries
    int32_t * first = dec + experts;                   // its first DECODE entry (LRU's candidate order)
    int32_t * cand = first + experts;                  // candidates, by rank
    int32_t * vict = cand + experts;                   // victims, by rank
    int32_t * kind = vict + experts;                   // 1: a candidate, 2: a victim
    __shared__ int has_decode, n_cand, n_vict, n_admit, base, now;
    const int entries = tokens * k, tid = threadIdx.x;
    for (int i = tid; i < shared_words(P, roles, experts); i += blockDim.x) {
        shared[i] = 0;
    }
    if (tid == 0) {
        has_decode = n_cand = n_vict = 0;
    }
    __syncthreads();
    if (P != STATIC) {
        for (int e = tid; e < experts; e += blockDim.x) {
            first[e] = INT_MAX;
        }
        __syncthreads();
    }
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
        if (P != STATIC) {
            live[e] = 1;
        }
        for (int r = 0; r < roles; ++r) {
            if (residency[e * roles + r] < 0) {
                need[r * experts + e] = 1;
            }
        }
    }
    __syncthreads();
    if (P != STATIC && has_decode) {                   // a step with decode rows: decay (or tick), then count
        if (P == LRU && tid == 0) {
            now = st.clock[0] = st.clock[0] + 1;
        }
        __syncthreads();
        for (int e = tid; e < experts; e += blockDim.x) {
            if (P == LFU) {                            // in f32, rounded each time: x0.915, then +1.0 per entry
                float s = __fmul_rn(st.score[e], LFU_DECAY);
                for (int c = 0; c < dec[e]; ++c) {
                    s = __fadd_rn(s, 1.0f);
                }
                st.score[e] = s;
            } else if (dec[e]) {
                st.last[e] = now;
            }
        }
        __syncthreads();
        for (int e = tid; e < experts; e += blockDim.x) {
            bool missing = true, whole = true;
            for (int r = 0; r < roles; ++r) {
                const int at = residency[e * roles + r];
                missing = missing && at < 0;
                whole = whole && at >= k + staging && at < limits[r];
            }
            if (dec[e] && missing) {
                kind[e] = 1;
                atomicAdd(&n_cand, 1);
            } else if (whole && (!live[e] || (st.plant & PLANT_VICTIM_NEEDED))) {
                kind[e] = 2;
                atomicAdd(&n_vict, 1);
            }
            // The order's key, in `first`: LFU's score (non-negative: its bits order as the floats do); LRU's first
            // decode entry (a candidate) or last stamp (a victim).
            first[e] = P == LFU ? __float_as_int(st.score[e]) : kind[e] == 1 ? first[e] : st.last[e];
        }
        __syncthreads();
        for (int e = tid; e < experts; e += blockDim.x) {   // ranks: candidates hottest (oldest route) first, victims
            const int ke = kind[e], key = first[e];         // coldest (least recent) first; ties by expert id
            if (!ke) {
                continue;
            }
            int rank = 0;
            for (int x = 0; x < experts; ++x) {
                const int kx = first[x];
                const bool lower = kx < key || (kx == key && x < e);
                rank += kind[x] == ke && (ke == 1 && P == LFU ? kx > key || (kx == key && x < e) : lower);
            }
            (ke == 1 ? cand : vict)[rank] = e;
        }
        __syncthreads();
        if (tid == 0) {
            n_admit = min(n_cand, n_vict);
        }
        __syncthreads();
        if (P == LFU) {                                // admitted while strictly hotter; the first failing pair stops it
            for (int i = tid; i < min(n_cand, n_vict); i += blockDim.x) {
                if (!(first[cand[i]] > first[vict[i]])) {
                    atomicMin(&n_admit, i);
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            base = atomicAdd(n_delta, n_admit);
        }
        __syncthreads();
        for (int i = tid; i < n_admit; i += blockDim.x) {   // the victim's slots, read before any is rewritten
            const int a = cand[i], b = vict[(st.plant & PLANT_SHARED_VICTIM) && i == 1 ? 0 : i];
            for (int r = 0; r < roles; ++r) {
                need[r * experts + a] = -2 - residency[b * roles + r];
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
        for (int i = tid; i < n_admit; i += blockDim.x) {
            const int a = cand[i], b = vict[(st.plant & PLANT_SHARED_VICTIM) && i == 1 ? 0 : i];
            for (int r = 0; r < roles; ++r) {
                residency[a * roles + r] = -2 - need[r * experts + a];
                residency[b * roles + r] = -1;
            }
        }
        __syncthreads();
    }
    const int warp = tid / 32, lane = tid % 32;
    if (warp < roles) {
        int32_t * row = need + warp * experts;
        int32_t * units = fetch + warp * 2 * staging, * dst = units + staging;
        int seen = 0, staged = 0;
        for (int base_e = 0; base_e < experts; base_e += 32) {
            const int e = base_e + lane;
            const int v = e < experts ? row[e] : 0;
            const unsigned all = __ballot_sync(0xffffffffu, v != 0), mine = __ballot_sync(0xffffffffu, v == 1);
            const unsigned below = (1u << lane) - 1;
            const int rank = seen + __popc(all & below), srank = staged + __popc(mine & below);
            if (v != 0 && rank < staging) {
                units[rank] = host[e * roles + warp];
                dst[rank] = v == 1 ? k + min(srank, staging - 1) : -2 - v;
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
        for (int r = 0; r < roles; ++r) {
            int32_t at = live_entry ? residency[e * roles + r] : j;
            if (live_entry && at < 0) {
                at = k + min(need[r * experts + e], staging - 1);
            } else if (live_entry && (at < k + staging || at >= limits[r])) {     // corrupt: never read past the arena
                at = j;
                atomicOr(errors, RANGE);
            }
            slots[r * entries + i] = at;
        }
    }
}

// Every CTA walks every fetched unit; its threads move FETCH_UNROLL 16-byte words each per trip, all loads before all
// stores, so a CTA keeps blockDim * FETCH_UNROLL * 16 bytes of reads in flight (zero-copy reads of host memory).
__global__ void k_fetch(const int32_t * __restrict__ fetch, const int32_t * __restrict__ n, const uint4 * __restrict__ src,
                        uint4 * __restrict__ dst, int64_t words, int staging) {
    const int count = min(*n, staging);
    const int64_t stride = int64_t(gridDim.x) * blockDim.x * FETCH_UNROLL;
    for (int b = 0; b < count; ++b) {
        const uint4 * __restrict__ s = src + int64_t(fetch[b]) * words;
        uint4 * __restrict__ d = dst + int64_t(fetch[staging + b]) * words;
        for (int64_t base = int64_t(blockIdx.x) * blockDim.x * FETCH_UNROLL + threadIdx.x; base < words;
             base += stride) {
            uint4 v[FETCH_UNROLL];
#pragma unroll
            for (int u = 0; u < FETCH_UNROLL; ++u) {
                const int64_t i = base + int64_t(u) * blockDim.x;
                if (i < words) {
                    v[u] = s[i];
                }
            }
#pragma unroll
            for (int u = 0; u < FETCH_UNROLL; ++u) {
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
             const at::Tensor & host, const at::Tensor & limits, at::Tensor & counts, at::Tensor & misses,
             at::Tensor & errors, at::Tensor & delta, at::Tensor & n_delta, int64_t group,
             const std::vector<at::Tensor> & state, int64_t plant, at::Tensor & slots, at::Tensor & fetch,
             at::Tensor & n_fetch) {
    const c10::cuda::CUDAGuard guard(ids.device());
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && residency.dim() == 2 && fetch.dim() == 3 && delta.dim() == 2,
                "ids [T, k] on CUDA");
    const int64_t tokens = ids.size(0), k = ids.size(1), experts = residency.size(0), roles = residency.size(1);
    const int64_t staging = fetch.size(2);
    TORCH_CHECK(roles <= RESOLVE_THREADS / 32 && staging > 0, "at most ", RESOLVE_THREADS / 32, " roles, staging > 0");
    check(ids, ids, at::kInt, {tokens, k}, "ids");
    check(kinds, ids, at::kByte, {tokens}, "kinds");
    check(residency, ids, at::kInt, {experts, roles}, "residency");
    check(host, ids, at::kInt, {experts, roles}, "host");
    check(limits, ids, at::kInt, {roles}, "limits");
    check(counts, ids, at::kInt, {experts}, "counts");
    check(misses, ids, at::kInt, {1}, "misses");
    check(errors, ids, at::kInt, {1}, "errors");
    check(delta, ids, at::kInt, {delta.size(0), 3}, "delta");
    check(n_delta, ids, at::kInt, {1}, "n_delta");
    check(slots, ids, at::kInt, {roles, tokens, k}, "slots");
    check(fetch, ids, at::kInt, {roles, 2, staging}, "fetch");
    check(n_fetch, ids, at::kInt, {roles}, "n_fetch");
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
    const size_t smem = shared_words(int(policy), int(roles), int(experts)) * sizeof(int32_t);
    auto kernel = policy == LFU ? k_resolve<LFU> : policy == LRU ? k_resolve<LRU> : k_resolve<STATIC>;
    kernel<<<1, RESOLVE_THREADS, smem, stream()>>>(
        ids.data_ptr<int32_t>(), kinds.data_ptr<uint8_t>(), residency.data_ptr<int32_t>(), host.data_ptr<int32_t>(),
        limits.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), misses.data_ptr<int32_t>(), errors.data_ptr<int32_t>(),
        delta.data_ptr<int32_t>(), n_delta.data_ptr<int32_t>(), int(group), int(delta.size(0)), st,
        slots.data_ptr<int32_t>(), fetch.data_ptr<int32_t>(), n_fetch.data_ptr<int32_t>(), tokens, k, experts, roles,
        staging);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather(const at::Tensor & fetch, const at::Tensor & n, const at::Tensor & src, at::Tensor & dst) {
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
    k_fetch<<<FETCH_CTAS, FETCH_THREADS, 0, stream()>>>(
        fetch.data_ptr<int32_t>(), n.data_ptr<int32_t>(), reinterpret_cast<const uint4 *>(src.data_ptr()),
        reinterpret_cast<uint4 *>(dst.data_ptr()), dst.size(1) / 16, int(fetch.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resolve", &resolve, "moe_resolve@1 (policy 0) or a variant into slots, fetch, n_fetch (and the counts, misses, "
          "errors, delta log, map and state in place)");
    m.def("fetch", &gather, "expert_fetch@1: dst[fetch[1, i]] = src[fetch[0, i]] for i < min(n, staging)");
}
