// The C API over the vendored llama.cpp kernels: plain pointers, the caller's stream and workspace.
//
// `type` is a ggml type id (enum ggml_type). Pointers are device pointers on the current device; rows are
// contiguous. Weights `w` are `out` rows of ggml blocks covering `in` values each (gguf.native). Activations `x` are
// [rows, in] bf16 (x_bf16) or f32; outputs `y` are [rows, out] (or [n, hidden]) bf16 (y_bf16) or f32. `stream` is a
// cudaStream_t. The workspace holds llama.cpp's scratch (q8_1 activations, stream-k fixup); a call that would
// overflow it aborts.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes of one row of `in` values of `type`; 0 if `in` is not a whole number of blocks.
size_t holon_ggml_row_bytes(int type, int64_t in);

// llama.cpp's dispatch for a matmul with `rows` activation rows on the current device.
int holon_ggml_use_mmvq(int type, int64_t rows);
int holon_ggml_use_mmq(int type, int64_t rows);

// How llama.cpp's mmq runs `out` x `rows` here (its J choice and stream-k grid, mirrored for tests): 0 = tiling,
// 1 = stream-k with one block per tile (no fixup), 2 = stream-k with the fixup pass. Sets the tile size I x J.
int holon_ggml_mmq_schedule(int type, int64_t out, int64_t rows, int * tile_i, int * tile_j);

// Workspace bytes for calls with `in` values per row and at most `rows` rows (any type), on the current device.
size_t holon_ggml_workspace_bytes(int64_t in, int64_t rows);

// y = x @ W.T. mmvq: rows <= 8, y bf16 or f32. mmq: y f32.
void holon_ggml_mmvq(int type, const void * w, int64_t out, int64_t in, const void * x, int x_bf16, int64_t rows,
                     void * y, int y_bf16, void * workspace, size_t workspace_bytes, void * stream);
void holon_ggml_mmq(int type, const void * w, int64_t out, int64_t in, const void * x, int x_bf16, int64_t rows,
                    float * y, void * workspace, size_t workspace_bytes, void * stream);

// y[i] = dequantized row ids[i] of the table (rows of `hidden` values).
void holon_ggml_get_rows(int type, const void * table, int64_t hidden, const int32_t * ids, int64_t n, void * y,
                         int y_bf16, void * stream);

// y = the `n` dequantized values of `src` (llama.cpp's convert).
void holon_ggml_dequantize(int type, const void * src, int64_t n, void * y, int y_bf16, void * stream);

// Mixture of experts (llama.cpp's MUL_MAT_ID). `w` is a stack of `experts` matrices of `out` rows each (E·out rows of
// blocks of `in` values); `ids` [tokens, k] int32, contiguous, every id in [0, experts) and unique within its row;
// `x` [tokens, rows_per_token, in] with rows_per_token 1 (every slot reads x[t, 0]) or k (slot j reads x[t, j]);
// `y` [tokens, k, out_y] f32: slot j of token t is expert ids[t, j] applied to its input row.

// Whether llama.cpp runs `tokens` tokens of `type` on mmvq (else mmq) here: get_mmvq_mmid_max_batch.
int holon_ggml_use_mmvq_id(int type, int64_t tokens);

// Workspace bytes for expert calls of at most `tokens` tokens (any type, any path) on the current device.
size_t holon_ggml_workspace_bytes_id(int64_t in, int64_t tokens, int64_t k, int64_t experts);

// mmvq: tokens <= the mmvq limit. swiglu: `w` is a merged stack (per expert out/2 gate rows, then out/2 up rows) and
// y = silu(gate) * up with out_y = out/2 (llama.cpp's fused gate); else out_y = out.
void holon_ggml_mmvq_id(int type, const void * w, int64_t experts, int64_t out, int64_t in, const void * x, int x_bf16,
                        int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k, float * y, int swiglu,
                        void * workspace, size_t workspace_bytes, void * stream);
// mmq: any tokens (one call is sized by the workspace), out_y = out.
void holon_ggml_mmq_id(int type, const void * w, int64_t experts, int64_t out, int64_t in, const void * x, int x_bf16,
                       int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k, float * y,
                       void * workspace, size_t workspace_bytes, void * stream);

// y[r, i] = silu(x[r, i]) * x[r, n + i] for `rows` rows of 2n f32 values (gate, then up), y f32.
void holon_ggml_swiglu(const float * x, int64_t rows, int64_t n, float * y, void * stream);

// Softmax routing: `logits` [tokens, experts] f32 -> the k largest probabilities `weights` [tokens, k] f32 (divided by
// their sum, clamped as llama.cpp, when `renormalize`) and their `ids` [tokens, k] int32. experts: a power of 2 <= 512.
void holon_ggml_topk_moe(const float * logits, int64_t tokens, int64_t experts, int64_t k, int renormalize,
                         float * weights, int32_t * ids, void * stream);

// out[t] = sum_j weights[t, j] * y[t, j] in f32, slot by slot, stored bf16: y [tokens, k, hidden] f32, out bf16.
void holon_ggml_moe_combine(const float * y, const float * weights, int64_t tokens, int64_t k, int64_t hidden,
                            void * out, void * stream);

#ifdef __cplusplus
}
#endif
