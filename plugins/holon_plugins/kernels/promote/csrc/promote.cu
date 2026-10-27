// placement.promote's resolve contract `moe_resolve.promote@1` on CUDA (F10, docs/dev/v1-offload.md D3): around the store's
// own `moe_resolve@1` launch (whichever provider serves it), one CTA before it stamps the call's routes and promotes
// its decode misses into empty slots and spares (the map then names their new slots), and one CTA after it lists the
// promoted pieces' fetches after the staged ones and refills the absent spares from the layer's least recently routed
// residents. The contract and its oracle are placement.promote's (reference.py). Every
// call runs on torch's current stream, reads its sizes from the device only, and is CUDA-graph capturable.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int PADDING = 0, DECODE = 1;                 // row kinds (std.moe.kernels)
constexpr int OVERFLOW = 2;                            // error bit: a delta record past the log
constexpr int THREADS = 256;
constexpr int MAX_SPARES = 8;

// The layer's state (placement.promote's `device_state`) and the call's tables.
struct Layer {
    int32_t * residency;                               // [E, R]
    const int32_t * limits;                            // [R]
    const int32_t * empty;                             // [R, P]: the layer's empty cache slots, the first n_empty
    int32_t * n_empty;
    int P;
    int32_t * last;                                    // [E]
    int32_t * clock;                                   // [1]
    int32_t * spare;                                   // [S, 1 + R]: owner + 1 (0: absent), then its slots
    int S;
};

__device__ inline bool cached(const Layer & L, int e, int roles, int lo) {
    bool whole = true;
    for (int r = 0; r < roles; ++r) {
        const int at = L.residency[e * roles + r];
        whole = whole && at >= lo && at < L.limits[r];
    }
    return whole;
}

// Before the resolve. `out`: [0] how many promoted experts are fetched (-1: the call has no decode entry), then they,
// in promotion order (an owner taking its spare back is not: nothing wrote its slots since it left them).
__global__ void k_promote(const int32_t * __restrict__ ids, const uint8_t * __restrict__ kinds, Layer L,
                          int32_t * __restrict__ routed, int32_t * __restrict__ errors, int32_t * __restrict__ delta,
                          int32_t * __restrict__ n_delta, int cap, int group, int32_t * __restrict__ out, int tokens,
                          int k, int experts, int roles, int staging) {
    extern __shared__ int32_t shared[];
    int32_t * first = shared;                          // [E]: its first DECODE entry (INT_MAX: none)
    int32_t * live = first + experts;                  // [E]: routed by a live entry
    int32_t * key = live + experts;                    // [E]: a candidate's first entry
    int32_t * order = key + experts;                   // [E]: the candidates, by first entry
    __shared__ int has_decode, n_cand, from[MAX_SPARES];
    const int tid = threadIdx.x, entries = tokens * k, lo = k + staging;
    for (int e = tid; e < experts; e += blockDim.x) {
        first[e] = INT_MAX;
        live[e] = 0;
    }
    if (tid == 0) {
        has_decode = n_cand = 0;
    }
    __syncthreads();
    for (int i = tid; i < entries; i += blockDim.x) {
        const int e = ids[i], kd = kinds[i / k];
        if (kd == PADDING || e < 0) {
            continue;
        }
        live[e] = 1;
        if (kd == DECODE) {
            atomicMin(&first[e], i);
            has_decode = 1;
        }
    }
    __syncthreads();
    if (!has_decode) {
        if (tid == 0) {
            out[0] = -1;
        }
        return;
    }
    for (int e = tid; e < experts; e += blockDim.x) {
        if (live[e]) {
            routed[e] |= 2;                            // the victims' scope (one thread per expert)
        }
    }
    const int now = L.clock[0] + 1;                    // 1. stamp
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        bool missing = true;
        for (int r = 0; r < roles; ++r) {
            missing = missing && L.residency[e * roles + r] < 0;
        }
        if (first[e] != INT_MAX) {
            L.last[e] = now;
        }
        key[e] = first[e] != INT_MAX && missing ? first[e] : INT_MAX;
        if (key[e] != INT_MAX) {
            atomicAdd(&n_cand, 1);
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {  // 2. the candidates in order (first entries are distinct)
        if (key[e] != INT_MAX) {
            int rank = 0;
            for (int y = 0; y < experts; ++y) {
                rank += key[y] < key[e];
            }
            order[rank] = e;
        }
    }
    __syncthreads();
    if (tid == 0) {                                    // a few promotions: one thread, in order
        L.clock[0] = now;
        const int n_spare = L.S, n_empty = *L.n_empty, base = *n_delta;
        int present = 0;
        for (int s = 0; s < n_spare; ++s) {
            present += L.spare[s * (1 + roles)] > 0;
        }
        const int n_prom = min(n_cand, n_empty + present);
        for (int s = 0; s < n_spare; ++s) {
            from[s] = 0;                               // used by an owner (1) or another (2)
        }
        for (int i = 0; i < n_prom; ++i) {             // owners first: their own spares back, no record
            for (int s = 0; s < n_spare; ++s) {
                if (L.spare[s * (1 + roles)] == order[i] + 1) {
                    from[s] = 1;
                }
            }
        }
        int took = 0, s = 0, records = 0, fetched = 0;
        for (int i = 0; i < n_prom; ++i) {
            const int a = order[i];
            int b = -1;
            const int32_t * slots = nullptr;
            for (int x = 0; x < n_spare; ++x) {
                if (L.spare[x * (1 + roles)] == a + 1) {
                    slots = L.spare + x * (1 + roles) + 1;
                }
            }
            if (slots == nullptr && took < n_empty) {  // the empty slots, from the top
                const int j = n_empty - 1 - took++;
                for (int r = 0; r < roles; ++r) {
                    L.residency[a * roles + r] = L.empty[r * L.P + j];
                }
                b = -2 - j;
            } else if (slots == nullptr) {             // the next spare no promoted expert owns
                while (from[s] || L.spare[s * (1 + roles)] == 0) {
                    ++s;
                }
                from[s] = 2;
                b = L.spare[s * (1 + roles)] - 1;
                slots = L.spare + s * (1 + roles) + 1;
            }
            if (slots != nullptr) {
                for (int r = 0; r < roles; ++r) {
                    L.residency[a * roles + r] = slots[r];
                }
            }
            if (b != -1) {
                if (base + records < cap) {
                    delta[(base + records) * 3] = group;
                    delta[(base + records) * 3 + 1] = a;
                    delta[(base + records) * 3 + 2] = b;
                } else {
                    atomicOr(errors, OVERFLOW);
                }
                ++records;
                out[1 + fetched++] = a;                // an owner's bytes are still in its spare: no fetch
            }
        }
        for (int x = 0; x < n_spare; ++x) {
            if (from[x]) {
                for (int c = 0; c <= roles; ++c) {
                    L.spare[x * (1 + roles) + c] = 0;
                }
            }
        }
        *L.n_empty = n_empty - took;
        *n_delta = base + records;
        out[0] = fetched;
    }
}

// After the resolve: the promoted pieces' fetches after the staged ones, then 4. the refill (at most the residents but
// one, none in a layer holding every expert).
__global__ void k_settle(const int32_t * __restrict__ out, Layer L, const int32_t * __restrict__ host,
                         int32_t * __restrict__ fetch, int32_t * __restrict__ n_fetch, int32_t * __restrict__ misses,
                         int F, int k, int experts, int roles, int staging) {
    extern __shared__ int32_t shared[];
    int32_t * key = shared;                            // [E]: a resident's last stamp (-1: not a resident)
    __shared__ int absent[MAX_SPARES], n_absent, n_res;
    const int tid = threadIdx.x, n_prom = out[0], lo = k + staging;   // the fetched promotions
    if (n_prom < 0) {
        return;
    }
    for (int i = tid; i < n_prom * roles; i += blockDim.x) {
        const int r = i / n_prom, j = i % n_prom, a = out[1 + j], at = n_fetch[r] + j;
        if (at < F) {
            const int32_t unit = host[a * roles + r];
            fetch[r * 2 * F + at] = unit >= 0 ? unit : -2 - a;
            fetch[r * 2 * F + F + at] = L.residency[a * roles + r];
        }
    }
    __syncthreads();
    if (tid < roles) {
        n_fetch[tid] += n_prom;
        atomicMax(misses, n_fetch[tid]);
    }
    if (tid == 0) {
        n_absent = n_res = 0;
        for (int s = 0; s < L.S; ++s) {
            if (L.spare[s * (1 + roles)] == 0) {
                absent[n_absent++] = s;
            }
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        key[e] = cached(L, e, roles, lo) ? L.last[e] : -1;
        if (key[e] >= 0) {
            atomicAdd(&n_res, 1);
        }
    }
    __syncthreads();
    const int n_take = n_res == experts ? 0 : min(n_absent, n_res - 1);
    for (int e = tid; e < experts && n_take > 0; e += blockDim.x) {   // least recently routed, ties by expert id
        if (key[e] >= 0) {
            int rank = 0;
            for (int y = 0; y < experts; ++y) {
                rank += key[y] >= 0 && (key[y] < key[e] || (key[y] == key[e] && y < e));
            }
            if (rank < n_take) {
                int32_t * row = L.spare + absent[rank] * (1 + roles);
                row[0] = e + 1;
                for (int r = 0; r < roles; ++r) {
                    row[1 + r] = L.residency[e * roles + r];
                    L.residency[e * roles + r] = -1;
                }
            }
        }
    }
}

void check(const at::Tensor & t, const at::Tensor & like, at::ScalarType dtype, std::vector<int64_t> shape,
           const char * name) {
    TORCH_CHECK(t.device() == like.device() && t.scalar_type() == dtype && t.is_contiguous() && t.sizes() == shape,
                name, ": ", shape, " ", dtype, ", contiguous, on ", like.device());
}

Layer layer(const at::Tensor & ids, at::Tensor & residency, const at::Tensor & limits, const at::Tensor & empty,
            at::Tensor & n_empty, at::Tensor & last, at::Tensor & clock, at::Tensor & spare) {
    const int64_t experts = residency.size(0), roles = residency.size(1);
    TORCH_CHECK(residency.dim() == 2 && empty.dim() == 2 && spare.dim() == 2 && spare.size(0) <= MAX_SPARES,
                "residency [E, R], empty [R, P], spare [S <= ", MAX_SPARES, ", 1 + R]");
    check(residency, ids, at::kInt, {experts, roles}, "residency");
    check(limits, ids, at::kInt, {roles}, "limits");
    check(empty, ids, at::kInt, {roles, empty.size(1)}, "empty");
    check(n_empty, ids, at::kInt, {1}, "n_empty");
    check(last, ids, at::kInt, {experts}, "last");
    check(clock, ids, at::kInt, {1}, "clock");
    check(spare, ids, at::kInt, {spare.size(0), 1 + roles}, "spare");
    return Layer{residency.data_ptr<int32_t>(), limits.data_ptr<int32_t>(), empty.data_ptr<int32_t>(),
                 n_empty.data_ptr<int32_t>(), int(empty.size(1)), last.data_ptr<int32_t>(), clock.data_ptr<int32_t>(),
                 spare.data_ptr<int32_t>(), int(spare.size(0))};
}

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void promote(const at::Tensor & ids, const at::Tensor & kinds, at::Tensor & residency, const at::Tensor & limits,
             const at::Tensor & empty, at::Tensor & n_empty, at::Tensor & routed, at::Tensor & errors,
             at::Tensor & delta, at::Tensor & n_delta, int64_t group, at::Tensor & last, at::Tensor & clock,
             at::Tensor & spare, at::Tensor & out, int64_t staging) {
    const c10::cuda::CUDAGuard guard(ids.device());
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && delta.dim() == 2, "ids [T, k] on CUDA, delta [D, 3]");
    const int64_t tokens = ids.size(0), k = ids.size(1), experts = residency.size(0);
    check(ids, ids, at::kInt, {tokens, k}, "ids");
    check(kinds, ids, at::kByte, {tokens}, "kinds");
    check(routed, ids, at::kInt, {experts}, "routed");
    check(errors, ids, at::kInt, {1}, "errors");
    check(delta, ids, at::kInt, {delta.size(0), 3}, "delta");
    check(n_delta, ids, at::kInt, {1}, "n_delta");
    check(out, ids, at::kInt, {1 + experts}, "out");
    const Layer L = layer(ids, residency, limits, empty, n_empty, last, clock, spare);
    k_promote<<<1, THREADS, 4 * experts * sizeof(int32_t), stream()>>>(
        ids.data_ptr<int32_t>(), kinds.data_ptr<uint8_t>(), L, routed.data_ptr<int32_t>(), errors.data_ptr<int32_t>(),
        delta.data_ptr<int32_t>(), n_delta.data_ptr<int32_t>(), int(delta.size(0)), int(group), out.data_ptr<int32_t>(),
        int(tokens), int(k), int(experts), int(residency.size(1)), int(staging));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void settle(const at::Tensor & out, at::Tensor & residency, const at::Tensor & host, const at::Tensor & limits,
            const at::Tensor & empty, at::Tensor & n_empty, at::Tensor & fetch, at::Tensor & n_fetch,
            at::Tensor & misses, at::Tensor & last, at::Tensor & clock, at::Tensor & spare,
            int64_t top_k, int64_t staging) {
    const c10::cuda::CUDAGuard guard(out.device());
    const int64_t experts = residency.size(0), roles = residency.size(1);
    TORCH_CHECK(out.is_cuda() && fetch.dim() == 3, "out on CUDA, fetch [R, 2, F]");
    const int64_t width = fetch.size(2);
    check(out, out, at::kInt, {1 + experts}, "out");
    check(host, out, at::kInt, {experts, roles}, "host");
    check(fetch, out, at::kInt, {roles, 2, width}, "fetch");
    check(n_fetch, out, at::kInt, {roles}, "n_fetch");
    check(misses, out, at::kInt, {1}, "misses");
    const Layer L = layer(out, residency, limits, empty, n_empty, last, clock, spare);
    k_settle<<<1, THREADS, experts * sizeof(int32_t), stream()>>>(
        out.data_ptr<int32_t>(), L, host.data_ptr<int32_t>(), fetch.data_ptr<int32_t>(), n_fetch.data_ptr<int32_t>(),
        misses.data_ptr<int32_t>(), int(width), int(top_k), int(experts), int(roles), int(staging));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("promote", &promote, "moe_resolve.promote@1 before the resolve: stamp, promotions (map, spares, log)");
    m.def("settle", &settle, "moe_resolve.promote@1 after the resolve: the promoted pieces' fetches, the spares' refill");
}
