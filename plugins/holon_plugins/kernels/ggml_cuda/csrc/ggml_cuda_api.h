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

#ifdef __cplusplus
}
#endif
