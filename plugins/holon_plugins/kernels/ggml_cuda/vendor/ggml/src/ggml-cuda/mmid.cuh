#pragma once

// Holon: the most (token, expert) pairs one call sorts (mm_ids_helper's largest instance: 1024 threads x 16)
#define GGML_CUDA_MM_IDS_MAX_PAIRS 16384

// Holon: the compact launch's z entries: at most one per active slot, and a slot needs a (token, expert) pair
static inline int ggml_cuda_mm_ids_nz(const int64_t n_experts, const int64_t n_tokens, const int64_t n_expert_used) {
    return (int) std::min(n_experts, n_tokens*n_expert_used);
}

// Holon: sorts the (slot, token·n_expert_used + iex) pairs of `ids` (slots ascending, then token order) into compact
// rows r: ids_dst[r] is the pair's token·n_expert_used + iex; ids_src1[r] its src1 row (or, with write_inverse,
// ids_src1[pair] = r). expert_bounds holds n_z + 1 bounds, the first compact row of each active slot in ascending
// order and then n_pairs, followed by the n_z slots themselves (0 past the active ones); n_z = ggml_cuda_mm_ids_nz.
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
