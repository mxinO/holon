#pragma once

// Holon: the most (token, expert) pairs one call sorts (mm_ids_helper's largest instance: 1024 threads x 16)
#define GGML_CUDA_MM_IDS_MAX_PAIRS 16384

// Holon: violated preconditions the helper reports (or'd into its `errors` word); the rows it names are not computed
#define GGML_CUDA_MM_IDS_ERR_SLOTS   1 // more distinct slots than the call's bound (ids_slots)
#define GGML_CUDA_MM_IDS_ERR_RANGE   2 // an id outside [0, the stack's slots)
#define GGML_CUDA_MM_IDS_ERR_COLUMNS 4 // a slot with more pairs than mmq's column tiles (an id repeated in a row)

// Holon: the per-slot helper (ggml's: one warp per slot, scanning every pair; z is the slot) for a launch that lists
// every slot and a small scan, where it is the faster one (measured on sm_86, docs/v1-offload.md D4; its shared
// memory holds 4096 pairs); else the compact one
static inline bool ggml_cuda_mm_ids_per_slot(const int64_t n_experts, const int64_t n_pairs, const int64_t n_z) {
    return n_z == n_experts && n_experts*n_pairs <= 1024*1024 && n_pairs <= 4096;
}

// Holon: at most this many more z entries than active slots: the launch lists every slot (the per-slot helper)
#define GGML_CUDA_MM_IDS_SPARE_Z 64

// Holon: the launch's z entries: one per active slot, at most min(slots, pairs, n_slots_used) (a slot needs a
// (token, expert) pair, and a call's ids name at most n_slots_used distinct slots: an arena of n_experts slots serves
// one layer's experts at a time); every slot instead where that is at most GGML_CUDA_MM_IDS_SPARE_Z more and the
// per-slot helper applies (a stack of a layer's own experts: the launch ggml makes)
static inline int ggml_cuda_mm_ids_nz(const int64_t n_experts, const int64_t n_tokens, const int64_t n_expert_used,
                                      const int64_t n_slots_used) {
    const int64_t n_pairs = n_tokens*n_expert_used;
    const int64_t n_z = std::min({n_experts, n_pairs, n_slots_used});
    const bool every_slot = n_experts - n_z <= GGML_CUDA_MM_IDS_SPARE_Z && ggml_cuda_mm_ids_per_slot(n_experts, n_pairs, n_experts);
    return (int) (every_slot ? n_experts : n_z);
}

// Holon: orders the (slot, token·n_expert_used + iex) pairs of `ids` (slots ascending, then token order) into compact
// rows r: ids_dst[r] is the pair's token·n_expert_used + iex; ids_src1[r] its src1 row (or, with write_inverse,
// ids_src1[pair] = r). expert_bounds holds n_z + 1 bounds, the first compact row of each listed slot in ascending
// order and then the end of their rows, followed by the n_z slots themselves (0 past the active ones);
// n_z = ggml_cuda_mm_ids_nz. Pairs mmq does not compute (an id out of range, slots past n_z, a slot's pairs past
// max_cols, mmq's column tiles) set their GGML_CUDA_MM_IDS_ERR_* bit in *errors (if not NULL).
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_z, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse,
        int max_cols, int32_t * errors, cudaStream_t stream);
