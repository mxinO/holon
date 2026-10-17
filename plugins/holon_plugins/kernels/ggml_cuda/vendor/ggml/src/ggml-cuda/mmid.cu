#include "common.cuh"
#include "mmid.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

// Holon: the helper is rewritten to build a compact list of the active slots in O(tokens·n_expert_used) instead of
// one block per slot scanning every id (an arena of S slots made the scan and mmq's grid S-sized; mmq now launches
// over ggml_cuda_mm_ids_nz entries, mmid.cuh). One block radix-sorts the (slot, token·n_expert_used + iex) pairs, each
// a 32-bit key `slot << pair_bits | pair`, on the slot bits only: the sort is stable, so the order is ggml's, slots
// ascending and each slot's pairs in token order. Every pair is its own compact row.
template <int threads, int items>
__launch_bounds__(threads, 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_z, const int n_pairs, const int pair_bits, const int slot_bits, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse) {
    using block_sort = cub::BlockRadixSort<uint32_t, threads, items>;
    using block_scan = cub::BlockScan<int, threads>;
    extern __shared__ __align__(16) char data_mm_ids_helper[]; // the sort's, then the scan's temporary storage
    __shared__ uint32_t last_slot[threads];

    const int tid = threadIdx.x;
    uint32_t keys[items]; // compact rows tid·items + q: a blocked arrangement, before and after the sort
#pragma unroll
    for (int q = 0; q < items; ++q) {
        const int i = tid*items + q;
        keys[q] = i < n_pairs ? uint32_t(ids[(i / n_expert_used)*si1 + i % n_expert_used]) << pair_bits | i : UINT32_MAX;
    }
    block_sort(*(typename block_sort::TempStorage *) data_mm_ids_helper).Sort(keys, pair_bits, pair_bits + slot_bits);
    last_slot[tid] = keys[items - 1] >> pair_bits;
    __syncthreads();

    // a row starts a slot's range if its slot differs from the row before it
    const uint32_t slot_before = tid > 0 ? last_slot[tid - 1] : UINT32_MAX;
    int n_heads = 0;
#pragma unroll
    for (int q = 0; q < items; ++q) {
        n_heads += tid*items + q < n_pairs && keys[q] >> pair_bits != (q > 0 ? keys[q - 1] >> pair_bits : slot_before);
    }
    int z;
    int n_active;
    block_scan(*(typename block_scan::TempStorage *) data_mm_ids_helper).ExclusiveSum(n_heads, z, n_active);

#pragma unroll
    for (int q = 0; q < items; ++q) {
        const int r = tid*items + q;
        if (r >= n_pairs) {
            break;
        }
        const int slot = keys[q] >> pair_bits;
        const int pair = keys[q] & ((1u << pair_bits) - 1);
        if (uint32_t(slot) != (q > 0 ? keys[q - 1] >> pair_bits : slot_before)) {
            if (z < n_z) {
                expert_bounds[z] = r;
                expert_bounds[n_z + 1 + z] = slot;
            } else if (z == n_z) {
                expert_bounds[z] = r; // the end of the last listed slot's rows
            }
            ++z;
        }
        ids_dst[r] = pair;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[pair] = r;
        } else {
            ids_src1[r] = (pair / n_expert_used)*sis1 + (pair % n_expert_used) % nchannels_y;
        }
    }
    // the entries past the active slots are empty and name slot 0, which mmq never reads for them
    for (int i = min(n_active, n_z + 1) + tid; i <= n_z; i += threads) {
        expert_bounds[i] = n_pairs;
        if (i < n_z) {
            expert_bounds[n_z + 1 + i] = 0;
        }
    }
}

template <int threads, int items>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_z, const int n_pairs, const int pair_bits, const int slot_bits, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const size_t nbytes_shared = std::max(sizeof(typename cub::BlockRadixSort<uint32_t, threads, items>::TempStorage),
                                          sizeof(typename cub::BlockScan<int, threads>::TempStorage));
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<threads, items>), nbytes_shared);
    mm_ids_helper<threads, items><<<1, threads, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_z, n_pairs, pair_bits, slot_bits, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_z, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(int64_t(n_tokens)*n_expert_used <= GGML_CUDA_MM_IDS_MAX_PAIRS && "too many (token, expert) pairs for mm_ids_helper");
    const int n_pairs = n_tokens*n_expert_used;
    int pair_bits = 0;
    while ((1 << pair_bits) < n_pairs) {
        ++pair_bits;
    }
    int slot_bits = 1;
    while ((int64_t(1) << slot_bits) < n_experts) {
        ++slot_bits;
    }
    GGML_ASSERT(pair_bits + slot_bits <= 32 && "too few bits in mm_ids_helper's keys");

#define LAUNCH_MM_IDS_HELPER(threads, items)                                                                            \
    launch_mm_ids_helper<threads, items>(ids, ids_src1, ids_dst, expert_bounds, n_z, n_pairs, pair_bits, slot_bits,   \
                                         n_expert_used, nchannels_y, si1, sis1, write_inverse, stream)
    if (n_pairs <= 128) {
        LAUNCH_MM_IDS_HELPER(32, 4);
    } else if (n_pairs <= 512) {
        LAUNCH_MM_IDS_HELPER(128, 4);
    } else if (n_pairs <= 1024) {
        LAUNCH_MM_IDS_HELPER(256, 4);
    } else if (n_pairs <= 2048) {
        LAUNCH_MM_IDS_HELPER(512, 4);
    } else if (n_pairs <= 4096) {
        LAUNCH_MM_IDS_HELPER(1024, 4);
    } else {
        static_assert(GGML_CUDA_MM_IDS_MAX_PAIRS == 1024*16, "the largest instance holds GGML_CUDA_MM_IDS_MAX_PAIRS");
        LAUNCH_MM_IDS_HELPER(1024, 16);
    }
#undef LAUNCH_MM_IDS_HELPER
}
