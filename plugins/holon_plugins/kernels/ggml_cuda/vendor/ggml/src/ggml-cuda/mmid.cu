#include "common.cuh"
#include "mmid.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

// Holon: the helper is rewritten (docs/v1-offload.md D4). Both paths order the compact rows as ggml does, slots
// ascending and each slot's (token, iex) pairs in order, give every pair its own row (a slot repeated inside a row
// included), put pairs with an id outside [0, n_experts) last (no slot lists them), and report a violated
// precondition in `errors` (GGML_CUDA_MM_IDS_ERR_*, or'd; NULL: none):
// - per slot (ggml_cuda_mm_ids_per_slot: the launch lists every slot, a small scan): one warp per slot scanning every
//   pair, as ggml's; z is the slot;
// - compact: one block radix-sorts the pairs, keys `slot << pair_bits | pair` sorted on the slot bits (stable), and
//   lists the active slots; mmq launches over ggml_cuda_mm_ids_nz of them, which bounds it by the call's slots
//   instead of the stack's.

static __device__ __forceinline__ void mm_ids_error(int32_t * errors, const int bit) {
    if (errors) {
        atomicOr(errors, bit);
    }
}

// pair (token·n_expert_used + iex) -> its compact row r: ids_dst[r], and ids_src1[r] (its src1 row) or, with
// write_inverse, ids_src1[pair] = r
static __device__ __forceinline__ void mm_ids_write_row(
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, const int pair, const int r, const int n_expert_used,
        const int nchannels_y, const int sis1, const bool write_inverse) {
    ids_dst[r] = pair;
    if (write_inverse) {
        ids_src1[pair] = r;
    } else {
        ids_src1[r] = (pair / n_expert_used)*sis1 + (pair % n_expert_used) % nchannels_y;
    }
}

// Block b < n_experts lists slot b's pairs; block n_experts the pairs with an id out of range, after all the others.
template <int n_expert_used_template>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void mm_ids_helper_per_slot(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_pairs, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse, const int max_cols, int32_t * __restrict__ errors) {
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;
    const int lane = threadIdx.x;
    const unsigned lanes_below = (1u << lane) - 1;
    extern __shared__ int pairs_mm_ids_helper[]; // this block's pairs, in order

    int nex_prev = 0;   // pairs before this block's (this lane's share)
    int it_compact = 0; // this block's pairs
    for (int p0 = 0; p0 < n_pairs; p0 += WARP_SIZE) {
        const int p = p0 + lane;
        const int id = p < n_pairs ? ids[(p / n_expert_used)*si1 + p % n_expert_used] : INT_MAX;
        const bool valid = 0 <= id && id < n_experts;
        const bool mine = p < n_pairs && (expert < n_experts ? id == expert : !valid);
        nex_prev += valid && id < expert;
        const unsigned ballot = __ballot_sync(0xFFFFFFFF, mine);
        if (mine) {
            pairs_mm_ids_helper[it_compact + __popc(ballot & lanes_below)] = p;
        }
        it_compact += __popc(ballot);
    }
    nex_prev = warp_reduce_sum<WARP_SIZE>(nex_prev);
    __syncwarp();

    for (int i = lane; i < it_compact; i += WARP_SIZE) {
        mm_ids_write_row(ids_src1, ids_dst, pairs_mm_ids_helper[i], nex_prev + i, n_expert_used, nchannels_y, sis1, write_inverse);
    }
    if (lane != 0) {
        return;
    }
    expert_bounds[expert] = nex_prev; // block n_experts: the end of the last slot's rows
    if (expert < n_experts) {
        expert_bounds[n_experts + 1 + expert] = expert;
        if (it_compact > max_cols) {
            mm_ids_error(errors, GGML_CUDA_MM_IDS_ERR_COLUMNS);
        }
    } else if (it_compact > 0) {
        mm_ids_error(errors, GGML_CUDA_MM_IDS_ERR_RANGE);
    }
}

template <int threads, int items>
__launch_bounds__(threads, 1)
static __global__ void mm_ids_helper_compact(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_z, const int n_pairs, const int pair_bits, const int slot_bits, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const int max_cols, int32_t * __restrict__ errors) {
    using block_sort = cub::BlockRadixSort<uint32_t, threads, items>;
    using block_scan = cub::BlockScan<int, threads>;
    extern __shared__ __align__(16) char data_mm_ids_helper[]; // the sort's, then the scan's temporary storage
    __shared__ uint32_t last_slot[threads];
    __shared__ int n_valid;   // the rows with an id in range: the pairs past them have slot n_experts

    const int tid = threadIdx.x;
    uint32_t keys[items]; // compact rows tid·items + q: a blocked arrangement, before and after the sort
#pragma unroll
    for (int q = 0; q < items; ++q) {
        const int i = tid*items + q;
        const int id = i < n_pairs ? ids[(i / n_expert_used)*si1 + i % n_expert_used] : n_experts;
        keys[q] = uint32_t(0 <= id && id < n_experts ? id : n_experts) << pair_bits | i;
    }
    if (tid == 0) {
        n_valid = n_pairs;
    }
    block_sort(*(typename block_sort::TempStorage *) data_mm_ids_helper).Sort(keys, pair_bits, pair_bits + slot_bits);
    last_slot[tid] = keys[items - 1] >> pair_bits;
    __syncthreads();

    // a row starts a slot's range if its slot differs from the row before it
    const uint32_t slot_before = tid > 0 ? last_slot[tid - 1] : UINT32_MAX;
    int n_heads = 0;
#pragma unroll
    for (int q = 0; q < items; ++q) {
        const uint32_t slot = keys[q] >> pair_bits;
        const bool head = tid*items + q < n_pairs && slot != (q > 0 ? keys[q - 1] >> pair_bits : slot_before);
        n_heads += head && slot < uint32_t(n_experts);
        if (head && slot == uint32_t(n_experts)) {
            n_valid = tid*items + q;
        }
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
        const uint32_t slot = keys[q] >> pair_bits;
        if (slot < uint32_t(n_experts) && slot != (q > 0 ? keys[q - 1] >> pair_bits : slot_before)) {
            if (z < n_z) {
                expert_bounds[z] = r;
                expert_bounds[n_z + 1 + z] = slot;
            } else if (z == n_z) {
                expert_bounds[z] = r; // the end of the last listed slot's rows
            }
            ++z;
        }
        mm_ids_write_row(ids_src1, ids_dst, keys[q] & ((1u << pair_bits) - 1), r, n_expert_used, nchannels_y, sis1, write_inverse);
    }
    __syncthreads();
    // the entries past the active slots are empty and name slot 0, which mmq never reads for them
    for (int i = min(n_active, n_z + 1) + tid; i <= n_z; i += threads) {
        expert_bounds[i] = n_valid;
        if (i < n_z) {
            expert_bounds[n_z + 1 + i] = 0;
        }
    }
    __syncthreads();
    for (int i = tid; i < min(n_active, n_z); i += threads) {
        if (expert_bounds[i + 1] - expert_bounds[i] > max_cols) {
            mm_ids_error(errors, GGML_CUDA_MM_IDS_ERR_COLUMNS);
        }
    }
    if (tid == 0 && n_active > n_z) {
        mm_ids_error(errors, GGML_CUDA_MM_IDS_ERR_SLOTS);
    }
    if (tid == 0 && n_valid < n_pairs) {
        mm_ids_error(errors, GGML_CUDA_MM_IDS_ERR_RANGE);
    }
}

template <int threads, int items>
static void launch_mm_ids_helper_compact(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_z, const int n_pairs, const int pair_bits, const int slot_bits, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const int max_cols, int32_t * errors,
        cudaStream_t stream) {
    const size_t nbytes_shared = std::max(sizeof(typename cub::BlockRadixSort<uint32_t, threads, items>::TempStorage),
                                          sizeof(typename cub::BlockScan<int, threads>::TempStorage));
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper_compact<threads, items>), nbytes_shared);
    mm_ids_helper_compact<threads, items><<<1, threads, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_experts, n_z, n_pairs, pair_bits, slot_bits, n_expert_used, nchannels_y, si1,
         sis1, write_inverse, max_cols, errors);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_z, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse, const int max_cols, int32_t * errors, cudaStream_t stream) {
    GGML_ASSERT(int64_t(n_tokens)*n_expert_used <= GGML_CUDA_MM_IDS_MAX_PAIRS && "too many (token, expert) pairs for mm_ids_helper");
    const int n_pairs = n_tokens*n_expert_used;

    if (ggml_cuda_mm_ids_per_slot(n_experts, n_pairs, n_z)) {
#define LAUNCH_MM_IDS_HELPER(k)                                                                                           \
        mm_ids_helper_per_slot<k><<<n_experts + 1, WARP_SIZE, n_pairs*sizeof(int), stream>>>(ids, ids_src1, ids_dst,     \
            expert_bounds, n_experts, n_pairs, n_expert_used, nchannels_y, si1, sis1, write_inverse, max_cols, errors)
        switch (n_expert_used) { // a constant divisor for the common ones
            case  2: LAUNCH_MM_IDS_HELPER( 2); break;
            case  4: LAUNCH_MM_IDS_HELPER( 4); break;
            case  6: LAUNCH_MM_IDS_HELPER( 6); break;
            case  8: LAUNCH_MM_IDS_HELPER( 8); break;
            case 10: LAUNCH_MM_IDS_HELPER(10); break;
            case 16: LAUNCH_MM_IDS_HELPER(16); break;
            default: LAUNCH_MM_IDS_HELPER( 0); break;
        }
#undef LAUNCH_MM_IDS_HELPER
        return;
    }

    int pair_bits = 0;
    while ((1 << pair_bits) < n_pairs) {
        ++pair_bits;
    }
    int slot_bits = 1; // slots 0..n_experts, n_experts marking an id out of range
    while ((int64_t(1) << slot_bits) <= n_experts) {
        ++slot_bits;
    }
    GGML_ASSERT(pair_bits + slot_bits <= 32 && "too few bits in mm_ids_helper's keys");

#define LAUNCH_MM_IDS_HELPER(threads, items)                                                                              \
    launch_mm_ids_helper_compact<threads, items>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_z, n_pairs, pair_bits, \
        slot_bits, n_expert_used, nchannels_y, si1, sis1, write_inverse, max_cols, errors, stream)
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
