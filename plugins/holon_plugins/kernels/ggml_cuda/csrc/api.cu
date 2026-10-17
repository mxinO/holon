// The C API (ggml_cuda_api.h): llama.cpp's own entry points, called with ggml tensors built on the stack over the
// caller's pointers, a context that borrows the caller's stream and a pool over the caller's workspace.
//
// The dispatch rules (dense and MUL_MAT_ID), the fused gate and the mmq schedule mirror llama.cpp cbe28bb3
// (ggml/src/ggml-cuda/ggml-cuda.cu, mmq.cuh, mmq.cu), MIT License, Copyright (c) 2023-2026 The ggml authors; the
// license is in ../vendor/LICENSE.
#include "ggml_cuda_api.h"

#include "common.cuh"
#include "convert.cuh"
#include "getrows.cuh"
#include "mmid.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "moe-weighted-reduction.cuh"
#include "quantize.cuh"
#include "topk-moe.cuh"
#include "unary.cuh"

namespace {

constexpr size_t POOL_ALIGN = 256;

// Test-only (L0 holds every call to the workspace formulas): the most bytes one call took from its workspace since
// the last holon_ggml_workspace_high_water(). One counter for the process, not per device, not thread-safe; serving
// never reads it, and the update is one max per call.
size_t high_water = 0;

// llama.cpp's scratch allocator, as a bump allocator over the caller's workspace: no cudaMalloc, so a call is safe
// inside CUDA graph capture. It lives for one call (llama.cpp frees everything before returning).
struct workspace_pool : ggml_cuda_pool {
    char * base;
    size_t size;
    size_t used = 0;

    workspace_pool(void * base, size_t size) : base((char *) base), size(size) {
        GGML_ASSERT((uintptr_t) base % POOL_ALIGN == 0);
    }

    ~workspace_pool() override { high_water = std::max(high_water, used); }

    void * alloc(size_t nbytes, size_t * actual_size) override {
        const size_t n = GGML_PAD(nbytes, POOL_ALIGN);
        if (used + n > size) {
            GGML_ABORT("ggml_cuda: workspace overflow (%zu + %zu > %zu bytes)", used, n, size);
        }
        void * ptr = base + used;
        used += n;
        *actual_size = n;
        return ptr;
    }

    void free(void *, size_t) override {}
};

// The backend context for one call: the caller's stream (graph capture records it) and workspace. NULL, torch's
// default stream, is passed as cudaStreamLegacy: llama.cpp would replace a NULL stream with a private one.
struct call_context {
    ggml_backend_cuda_context ctx;

    call_context(void * stream, void * workspace, size_t workspace_bytes) : ctx(ggml_cuda_get_device()) {
        ctx.streams[ctx.device][ctx.curr_stream_no] = stream ? (cudaStream_t) stream : cudaStreamLegacy;
        ctx.pools[ctx.device][ctx.curr_stream_no] = std::make_unique<workspace_pool>(workspace, workspace_bytes);
    }
};

// A contiguous [nrows, ncols] ggml tensor over `data`. Zeroed op_params (default precision); no buffer.
ggml_tensor matrix(const ggml_type type, const void * data, const int64_t ncols, const int64_t nrows) {
    ggml_tensor t = {};
    t.type = type;
    t.data = const_cast<void *>(data);
    t.ne[0] = ncols;
    t.ne[1] = nrows;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = ggml_row_size(type, ncols);
    t.nb[2] = t.nb[3] = t.nb[1] * nrows;
    return t;
}

// A contiguous [n2, n1, n0] ggml tensor over `data`.
ggml_tensor tensor3(const ggml_type type, const void * data, const int64_t n0, const int64_t n1, const int64_t n2) {
    ggml_tensor t = matrix(type, data, n0, n1);
    t.ne[2] = n2;
    t.nb[3] = t.nb[2] * n2;
    return t;
}

// Rows [first, first + rows) of each of the `experts` matrices of `out` rows in the stack `w`.
ggml_tensor experts_view(const ggml_type type, const void * w, const int64_t experts, const int64_t out,
                         const int64_t in, const int64_t first, const int64_t rows) {
    const size_t row = ggml_row_size(type, in);
    ggml_tensor t = tensor3(type, (const char *) w + first*row, in, rows, experts);
    t.nb[2] = out*row;
    t.nb[3] = t.nb[2] * experts;
    return t;
}

ggml_type act(const int bf16) { return bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32; }

// mmq's output with the activation precision asked of it: llama.cpp's graph sets it in dst's op_params[3]
// (GGML_PREC_Q8 or GGML_PREC_Q4, read by ggml_cuda_mmq_get_prec_src1). Left zero it is the default, native W4A4 for
// the FP4 types on Blackwell, so every mmq call and its pre-launch check (holon_ggml_mmq_prec) pass the spec's.
ggml_tensor with_prec(ggml_tensor dst, const int prec) {
    ggml_set_op_params_i32(&dst, 3, prec);
    return dst;
}

// A per-tensor (dense) or per-expert scale for mmvq's fused epilogue: `n` f32 values, or none.
ggml_tensor scale_vector(const float * scale, const int64_t n) { return matrix(GGML_TYPE_F32, scale, n, 1); }

// SwiGLU after mmq, computed as llama.cpp's fused mmvq gate computes it: up * silu(gate), ggml_cuda_op_silu_single;
// with `ids`, each half first scaled by its row's expert scale (NVFP4's scale2, as mmvq's x_scale and gate_scale).
__global__ void swiglu_f32(const float * __restrict__ x, const float * __restrict__ gate_scale,
                           const float * __restrict__ up_scale, const int32_t * __restrict__ ids,
                           float * __restrict__ y, const int64_t n, const int64_t size) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    const float * row = x + 2*(i/n)*n;
    float gate = row[i%n], up = row[n + i%n];
    if (ids) {
        const int32_t e = ids[i/n];
        up *= up_scale[e];
        gate *= gate_scale[e];
    }
    y[i] = up * ggml_cuda_op_silu_single(gate);
}

int device_cc() { return ggml_cuda_info().devices[ggml_cuda_get_device()].cc; }

} // namespace

size_t holon_ggml_row_bytes(int type, int64_t in) {
    const ggml_type t = (ggml_type) type;
    return in % ggml_blck_size(t) == 0 ? ggml_row_size(t, in) : 0;
}

int holon_ggml_use_mmvq(int type, int cc, int64_t rows) { return ggml_cuda_should_use_mmvq((ggml_type) type, cc, rows); }

int holon_ggml_use_mmq(int type, int64_t rows) { return ggml_cuda_should_use_mmq((ggml_type) type, device_cc(), rows, 0); }

int holon_ggml_mmq_prec(int type, int prec) {
    ggml_tensor src0 = {};
    src0.type = (ggml_type) type;
    const ggml_tensor dst = with_prec(matrix(GGML_TYPE_F32, nullptr, 1, 1), prec);
    return ggml_cuda_mmq_get_prec_src1(&src0, &dst, device_cc());
}

int holon_ggml_mmq_schedule(int type, int prec, int64_t out, int64_t rows, int * tile_i, int * tile_j) {
    const auto & dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const bool fallback = ggml_cuda_mmq_needs_fallback(out);          // as ggml_cuda_mul_mat_q
    const ggml_prec prec_src1 = (ggml_prec) holon_ggml_mmq_prec(type, prec);
    int J_best = 0, ntx_best = INT_MAX;
    for (int J = 8; J <= 128 && ntx_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) type, J, fallback, dev.cc, prec_src1);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, dev.cc) <= dev.smpbo && (rows + J - 1)/J < ntx_best) {
            J_best = J;
            ntx_best = (rows + J - 1)/J;
        }
    }
    const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) type, J_best, fallback, dev.cc, prec_src1);
    *tile_i = c.I;
    *tile_j = c.J;
    if (!c.stream_k) {
        return 0;
    }
    const int64_t ntiles = (out + c.I - 1)/c.I * ntx_best;           // as launch_mul_mat_q
    const int64_t waves = (ntiles + dev.nsm - 1)/dev.nsm;
    const int64_t nblocks = 100*ntiles/(dev.nsm*waves) >= 90 ? ntiles : dev.nsm;
    return ntiles % nblocks == 0 ? 1 : 2;
}

size_t holon_ggml_workspace_high_water(void) {
    const size_t used = high_water;
    high_water = 0;
    return used;
}

namespace {

// `fixup_tiles` partial f32 tiles per SM for the stream-k fixup: a dense call keeps two (each block's last, and the
// end of its first, held in f32 for the fixup's single rounding: mmq.cuh, mmq_store), a call with ids none (it tiles).
size_t workspace_bytes(int64_t in, int64_t rows, int fixup_tiles) {
    const int64_t in_padded = GGML_PAD(in, MATRIX_ROW_PADDING);
    // mmq's q8_1 activations plus its tile padding (J <= 128 rows, rounded up to a load chunk of <= 512 ints);
    // mmvq's block_q8_1 and native FP4's block_fp4_mmq (half the bytes) need less
    const size_t q8_1 = (rows*in_padded/QK8_1_MMQ + 128) * sizeof(block_q8_1_mmq) + 512*sizeof(int);
    // native NVFP4's per-row activation scale, reserved on every workspace (one size per workspace name)
    const size_t row_scale = rows * sizeof(float);
    // stream-k fixup: partial I x J f32 tiles per SM (I, J <= 128 on every config we build)
    const size_t fixup = size_t(ggml_cuda_info().devices[ggml_cuda_get_device()].nsm) * fixup_tiles*128*128 * sizeof(float);
    return GGML_PAD(q8_1, POOL_ALIGN) + GGML_PAD(row_scale, POOL_ALIGN) + GGML_PAD(fixup, POOL_ALIGN);
}

} // namespace

size_t holon_ggml_workspace_bytes(int64_t in, int64_t rows) { return workspace_bytes(in, rows, 2); }

void holon_ggml_mmvq(int type, const void * w, int64_t out, int64_t in, const void * x, int x_bf16,
                     const void * x_q8_1, int64_t rows, void * y, int y_bf16, const float * scale, const void * gate,
                     void * workspace, size_t workspace_bytes, void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_tensor src0 = matrix((ggml_type) type, w, in, out);
    const ggml_tensor src1 = x_q8_1 ? matrix(GGML_TYPE_Q8_1, x_q8_1, in, rows) : matrix(act(x_bf16), x, in, rows);
    ggml_tensor dst = matrix(act(y_bf16), y, out, rows);
    // the fused gate as llama.cpp builds it for a dense GLU: src0 the up rows, `gate` the gate rows; NVFP4's
    // per-tensor scale2 on both (a merged gate/up holds one)
    const ggml_tensor x_scale = scale_vector(scale, 1), gate_t = matrix((ggml_type) type, gate, in, out);
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.x_scale = scale ? &x_scale : nullptr;
    if (gate) {
        fusion.gate = &gate_t;
        fusion.glu_op = GGML_GLU_OP_SWIGLU;
        fusion.gate_scale = fusion.x_scale;
    }
    ggml_cuda_mul_mat_vec_q(c.ctx, &src0, &src1, nullptr, &dst, scale || gate ? &fusion : nullptr);
}

void holon_ggml_mmq(int type, int prec, const void * w, int64_t out, int64_t in, const void * x, int x_bf16,
                    int64_t rows, void * y, int y_bf16, const float * scale, void * workspace, size_t workspace_bytes,
                    void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_tensor src0 = matrix((ggml_type) type, w, in, out);
    const ggml_tensor src1 = matrix(act(x_bf16), x, in, rows);
    ggml_tensor dst = with_prec(matrix(act(y_bf16), y, out, rows), prec);
    ggml_cuda_mul_mat_q(c.ctx, &src0, &src1, nullptr, &dst, scale);
}

void holon_ggml_quantize_fp4(int type, const float * x, int64_t rows, int64_t in, void * blocks, float * scale,
                             void * stream) {
    const ggml_tensor src1 = matrix(GGML_TYPE_F32, x, in, rows);
    // as ggml_cuda_mul_mat_q quantizes a dense src1 for native FP4 (its float8 loads when aligned)
    quantize_mmq_fp4_cuda(x, nullptr, blocks, scale, (ggml_type) type, ggml_cuda_is_aligned(&src1, 32), in, in,
                          rows*in, rows*in, GGML_PAD(in, MATRIX_ROW_PADDING), rows, 1, 1, (cudaStream_t) stream);
    CUDA_CHECK(cudaGetLastError());
}

void holon_ggml_quantize_q8_1(int type, const void * x, int x_bf16, int64_t rows, int64_t in, void * y, void * stream) {
    // as ggml_cuda_mul_mat_vec_q quantizes a contiguous src1
    quantize_row_q8_1_cuda(x, act(x_bf16), nullptr, y, (ggml_type) type, in, in, rows*in, rows*in,
                           GGML_PAD(in, MATRIX_ROW_PADDING), rows, 1, 1, (cudaStream_t) stream);
    CUDA_CHECK(cudaGetLastError());
}

void holon_ggml_get_rows(int type, const void * table, int64_t hidden, const int32_t * ids, int64_t n, void * y,
                         int y_bf16, void * stream) {
    const size_t row = ggml_row_size((ggml_type) type, hidden);
    const size_t y_row = hidden * ggml_type_size(act(y_bf16));
    get_rows_cuda(table, (ggml_type) type, ids, y, act(y_bf16), hidden, row, row, row,
                  n, 1, 1, sizeof(int32_t), n*sizeof(int32_t), n*sizeof(int32_t), y_row, n*y_row, n*y_row,
                  (cudaStream_t) stream);
}

void holon_ggml_dequantize(int type, const void * src, int64_t n, void * y, int y_bf16, void * stream) {
    if (y_bf16) {
        const to_bf16_cuda_t convert = ggml_get_to_bf16_cuda((ggml_type) type);
        GGML_ASSERT(convert != nullptr);
        convert(src, (nv_bfloat16 *) y, n, (cudaStream_t) stream);
    } else {
        const to_fp32_cuda_t convert = ggml_get_to_fp32_cuda((ggml_type) type);
        GGML_ASSERT(convert != nullptr);
        convert(src, (float *) y, n, (cudaStream_t) stream);
    }
}

int holon_ggml_use_mmvq_id(int type, int64_t tokens) {
    return tokens <= get_mmvq_mmid_max_batch((ggml_type) type, device_cc());
}

size_t holon_ggml_workspace_bytes_id(int64_t in, int64_t tokens, int64_t k, int64_t slots) {
    // as ggml_cuda_mul_mat_q allocates with ids: ids_src1, ids_dst (padded by J - 1 <= 127 for its unconditional
    // reads), expert_bounds and the slots of the compact launch's z entries, the helper's scratch (mmid.cuh), then
    // q8_1 as a dense call over the tokens·k (token, slot) rows; no stream-k fixup (mmq with ids always tiles); mmvq
    // with ids needs less
    const size_t ids_src1 = GGML_PAD(tokens*k*sizeof(int32_t), POOL_ALIGN);
    const size_t ids_dst = GGML_PAD((tokens*k + 127)*sizeof(int32_t), POOL_ALIGN);
    const int64_t n_z = std::min(slots, tokens*k) + GGML_CUDA_MM_IDS_SPARE_Z; // ggml_cuda_mm_ids_nz's bound
    const size_t bounds = GGML_PAD((2*n_z + 1)*sizeof(int32_t), POOL_ALIGN);
    const size_t helper = GGML_PAD(ggml_cuda_mm_ids_scratch_bytes(tokens*k), POOL_ALIGN); // the device-wide sort's
    return ids_src1 + ids_dst + bounds + helper + workspace_bytes(in, tokens*k, 0);
}

void holon_ggml_mmvq_id(int type, const void * w, int64_t experts, int64_t out, int64_t in, const void * x, int x_bf16,
                        const void * x_q8_1, int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k,
                        float * y, int swiglu,
                        const float * gate_scale, const float * up_scale, void * workspace, size_t workspace_bytes,
                        void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_type t = (ggml_type) type;
    const int64_t out_y = swiglu ? out/2 : out;
    // the fused gate as llama.cpp builds it: src0 the up rows, `gate` the gate rows, both at the stack's expert stride
    const ggml_tensor src0 = experts_view(t, w, experts, out, in, out - out_y, out_y);
    const ggml_tensor gate = experts_view(t, w, experts, out, in, 0, out_y);
    const ggml_tensor src1 = x_q8_1 ? tensor3(GGML_TYPE_Q8_1, x_q8_1, in, rows_per_token, tokens)
                                    : tensor3(act(x_bf16), x, in, rows_per_token, tokens);
    const ggml_tensor ids_t = matrix(GGML_TYPE_I32, ids, k, tokens);
    ggml_tensor dst = tensor3(GGML_TYPE_F32, y, out_y, k, tokens);
    // NVFP4's per-expert scales in the fused epilogue: up's (or the stack's) as x_scale, gate's as gate_scale
    const ggml_tensor x_scale = scale_vector(up_scale, experts), g_scale = scale_vector(gate_scale, experts);
    ggml_cuda_mm_fusion_args_host fusion{};
    if (swiglu) {
        fusion.gate = &gate;
        fusion.glu_op = GGML_GLU_OP_SWIGLU;
        fusion.gate_scale = gate_scale ? &g_scale : nullptr;
    }
    fusion.x_scale = up_scale ? &x_scale : nullptr;
    ggml_cuda_mul_mat_vec_q(c.ctx, &src0, &src1, &ids_t, &dst, swiglu || up_scale ? &fusion : nullptr);
}

void holon_ggml_mmq_id(int type, int prec, const void * w, int64_t experts, int64_t out, int64_t in, const void * x,
                       int x_bf16, int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k, float * y,
                       int64_t slots, int32_t * errors, void * workspace, size_t workspace_bytes, void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_tensor src0 = experts_view((ggml_type) type, w, experts, out, in, 0, out);
    const ggml_tensor src1 = tensor3(act(x_bf16), x, in, rows_per_token, tokens);
    const ggml_tensor ids_t = matrix(GGML_TYPE_I32, ids, k, tokens);
    ggml_tensor dst = with_prec(tensor3(GGML_TYPE_F32, y, out, k, tokens), prec);
    ggml_cuda_mul_mat_q(c.ctx, &src0, &src1, &ids_t, &dst, nullptr, slots, errors);
}

void holon_ggml_swiglu(const float * x, int64_t rows, int64_t n, const float * gate_scale, const float * up_scale,
                       const int32_t * ids, float * y, void * stream) {
    const int64_t size = rows*n;
    swiglu_f32<<<(size + CUDA_GLU_BLOCK_SIZE - 1)/CUDA_GLU_BLOCK_SIZE, CUDA_GLU_BLOCK_SIZE, 0, (cudaStream_t) stream>>>(
        x, gate_scale, up_scale, ids, y, n, size);
    CUDA_CHECK(cudaGetLastError());
}

void holon_ggml_topk_moe(const float * logits, int64_t tokens, int64_t experts, int64_t k, int renormalize,
                         float * weights, int32_t * ids, void * stream) {
    call_context c(stream, nullptr, 0);
    const ggml_tensor logits_t = matrix(GGML_TYPE_F32, logits, experts, tokens);
    ggml_tensor weights_t = tensor3(GGML_TYPE_F32, weights, 1, k, tokens);   // llama.cpp's [1, k, tokens] view
    ggml_tensor ids_t = matrix(GGML_TYPE_I32, ids, k, tokens);
    ggml_tensor clamp = {};                                                   // the sum's clamp (llama-graph.cpp)
    const float clamp_min = 6.103515625e-5f;
    memcpy(clamp.op_params, &clamp_min, sizeof(clamp_min));
    ggml_cuda_topk_moe_args args;
    args.softmax = true;
    ggml_cuda_op_topk_moe(c.ctx, &logits_t, &weights_t, &ids_t, renormalize ? &clamp : nullptr, nullptr, nullptr, args);
}

int64_t holon_ggml_moe_router_rows(void) { return HOLON_ROUTER_MAX_ROWS; }

int64_t holon_ggml_moe_router_tiles(int64_t tokens) { return (tokens + HOLON_ROUTER_ROWS - 1) / HOLON_ROUTER_ROWS; }

void holon_ggml_moe_router(const void * x, int x_bf16, const float * router, int64_t tokens, int64_t hidden,
                           int64_t experts, int64_t k, int renormalize, float * logits, uint32_t * done,
                           float * weights, int32_t * ids, void * x_q8_1, void * stream) {
    GGML_ASSERT((uintptr_t) x % 8 == 0 && (uintptr_t) router % 16 == 0);
    // the sum's clamp, as holon_ggml_topk_moe's (llama-graph.cpp)
    ggml_cuda_holon_moe_router(x, x_bf16, router, tokens, hidden, experts, k, renormalize ? 6.103515625e-5f : -INFINITY,
                               logits, done, weights, ids, x_q8_1, (cudaStream_t) stream);
}

void holon_ggml_moe_combine(const float * y, const float * weights, int64_t tokens, int64_t k, int64_t hidden,
                            const void * residual, void * out, void * stream) {
    call_context c(stream, nullptr, 0);
    const ggml_tensor experts = tensor3(GGML_TYPE_F32, y, hidden, k, tokens);
    const ggml_tensor weights_t = matrix(GGML_TYPE_F32, weights, k, tokens);
    ggml_tensor dst = matrix(GGML_TYPE_BF16, out, hidden, tokens);
    const ggml_tensor residual_t = matrix(GGML_TYPE_BF16, residual, hidden, tokens);
    ggml_cuda_op_moe_weighted_reduction(c.ctx, &experts, nullptr, &weights_t, &dst, residual ? &residual_t : nullptr);
}
