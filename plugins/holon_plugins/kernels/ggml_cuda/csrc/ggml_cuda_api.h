// The C API over the vendored llama.cpp kernels: plain pointers, the caller's stream and workspace.
//
// `type` is a ggml type id (enum ggml_type). Pointers are device pointers on the current device; rows are
// contiguous. Weights `w` are `out` rows of ggml blocks covering `in` values each (gguf.native). Activations `x` are
// [rows, in] bf16 (x_bf16) or f32; outputs `y` are [rows, out] (or [n, hidden]) bf16 (y_bf16) or f32. `stream` is a
// cudaStream_t. The workspace holds llama.cpp's scratch (quantized activations, stream-k fixup); a call that would
// overflow it aborts. `prec` is mmq's activation precision, a ggml_prec: GGML_PREC_Q8 (30, q8_1) or GGML_PREC_Q4 (40,
// native W4A4: FP4 types on Blackwell, f32 activations only). `scale`s are NVFP4's f32 per-tensor or per-expert
// factors (scale2), fused into mmvq's epilogue; NULL: none.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes of one row of `in` values of `type`; 0 if `in` is not a whole number of blocks.
size_t holon_ggml_row_bytes(int type, int64_t in);

// llama.cpp's dispatch for a matmul with `rows` activation rows: mmvq on a device of compute capability `cc`
// (100 * major + 10 * minor, given explicitly: a host function, so any device may answer for another class), mmq on
// the current device.
int holon_ggml_use_mmvq(int type, int cc, int64_t rows);
int holon_ggml_use_mmq(int type, int64_t rows);

// The activation precision ggml_cuda_mul_mat_q will use for `type` asked for `prec` here (its exported
// ggml_cuda_mmq_get_prec_src1 on the dst our mmq calls build): the caller's check before launch and graph capture.
int holon_ggml_mmq_prec(int type, int prec);

// How llama.cpp's mmq runs `out` x `rows` at `prec` here (its J choice and stream-k grid, mirrored for tests): 0 =
// tiling, 1 = stream-k with one block per tile (no fixup), 2 = stream-k with the fixup pass. Sets the tile size I x J.
int holon_ggml_mmq_schedule(int type, int prec, int64_t out, int64_t rows, int * tile_i, int * tile_j);

// Workspace bytes for calls with `in` values per row and at most `rows` rows (any type, any precision), on the
// current device.
size_t holon_ggml_workspace_bytes(int64_t in, int64_t rows);

// Tests only: the most bytes one call took from its workspace since the last read (then 0), to hold the sizes above
// to what llama.cpp allocates on every call. One process-wide counter: not per device, not thread-safe.
size_t holon_ggml_workspace_high_water(void);

// y = x @ W.T, times `scale[0]` (NVFP4 only; NULL: none) in f32 before one rounding to y's type, bf16 or f32. mmvq:
// rows <= 8 (the scale in llama.cpp's fused epilogue at one row; above it in the unfused kernel's epilogue, after the
// full reduction: docs/m07-perf.md D11); `x_q8_1`, if not NULL, is x already quantized as mmvq quantizes it
// (holon_ggml_quantize_q8_1), read instead of quantizing x (D6); `gate`, if not NULL, a second weight of W's shape:
// y = silu(x @ gate.T · scale) · (x @ W.T · scale) from the f32 sums, at one row (llama.cpp's fused gate, D6).
// mmq: any rows (a call is sized by the workspace), the stream-k fixup's partials kept in f32 until the rounding (D4).
void holon_ggml_mmvq(int type, const void * w, int64_t out, int64_t in, const void * x, int x_bf16,
                     const void * x_q8_1, int64_t rows, void * y, int y_bf16, const float * scale, const void * gate,
                     void * workspace, size_t workspace_bytes, void * stream);
void holon_ggml_mmq(int type, int prec, const void * w, int64_t out, int64_t in, const void * x, int x_bf16,
                    int64_t rows, void * y, int y_bf16, const float * scale, void * workspace, size_t workspace_bytes,
                    void * stream);

// Tests only: native mmq's FP4 activations of x [rows, in] f32 (`type` MXFP4, or NVFP4 on Blackwell), as it
// quantizes them: block_fp4_mmq [in padded to 512 / 256][rows] into `blocks`, NVFP4's per-row scale into `scale`.
void holon_ggml_quantize_fp4(int type, const float * x, int64_t rows, int64_t in, void * blocks, float * scale,
                             void * stream);

// mmvq's q8_1 activations of x [rows, in] (f32 or bf16) for a matmul of `type`, as it quantizes them (block_q8_1 per
// 32 values, rows of `in` padded to 512) into `y`: the input several mmvq calls share (`x_q8_1`, docs/m07-perf.md
// D6), and the quantize's own time in benchmarks (D8).
void holon_ggml_quantize_q8_1(int type, const void * x, int x_bf16, int64_t rows, int64_t in, void * y, void * stream);

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

// Workspace bytes for expert calls of at most `tokens` tokens whose ids name at most `slots` distinct slots (any
// type, any path) on the current device.
size_t holon_ggml_workspace_bytes_id(int64_t in, int64_t tokens, int64_t k, int64_t slots);

// mmvq: tokens <= the mmvq limit. swiglu: `w` is a merged stack (per expert out/2 gate rows, then out/2 up rows) and
// y = silu(gate · gate_scale[e]) * up · up_scale[e] with out_y = out/2 (llama.cpp's fused gate); else out_y = out and
// y = x @ W_e.T · up_scale[e]. Scales: `experts` f32 values each, or NULL (NVFP4 only).
void holon_ggml_mmvq_id(int type, const void * w, int64_t experts, int64_t out, int64_t in, const void * x, int x_bf16,
                        int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k, float * y, int swiglu,
                        const float * gate_scale, const float * up_scale, void * workspace, size_t workspace_bytes,
                        void * stream);
// mmq: any tokens (one call is sized by the workspace), out_y = out. `ids` name at most `slots` distinct slots of the
// stack (it bounds the launch; the rows of further slots are not computed).
void holon_ggml_mmq_id(int type, int prec, const void * w, int64_t experts, int64_t out, int64_t in, const void * x,
                       int x_bf16, int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k, float * y,
                       int64_t slots, void * workspace, size_t workspace_bytes, void * stream);

// y[r, i] = silu(x[r, i]) * x[r, n + i] for `rows` rows of 2n f32 values (gate, then up), y f32; with `ids` (one
// expert per row), gate and up first times gate_scale[ids[r]] and up_scale[ids[r]].
void holon_ggml_swiglu(const float * x, int64_t rows, int64_t n, const float * gate_scale, const float * up_scale,
                       const int32_t * ids, float * y, void * stream);

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
