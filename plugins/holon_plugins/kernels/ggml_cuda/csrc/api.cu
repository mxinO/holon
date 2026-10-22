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

} // namespace

int holon_mmq_plant = 0;                        // mmq.cuh's plant hook: off unless the judge sets it
long long holon_mmq_fixups[2] = {0, 0};

namespace {

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

// moe_router@1's logits on prompt chunks (docs/v1-moe-prefill.md D3): logits[t, e] = sum over h of x[t, h] *
// router[e, h] from x's exact values (bf16 or f32), in an order fixed per logit: ks chains of f32 fma, chain g over
// hidden values [g, g + 1) * hidden / ks ascending, then the chains' sums added in g order, ks = router_chains(experts)
// (more chains for fewer experts, to fill the GPU without moving more partial sums than logits need). So a row's
// logits do not depend on the call's rows nor on where its tile falls. router_partials runs the chains, a block of 64
// threads per (64 rows, 64 experts, chain): each thread stages one row of x and one of the router, ROUTER_BK hidden
// values at a time (the next stage's loads in flight meanwhile), and computes an 8 x 8 corner of the tile from shared
// memory; router_sum adds the chains.
constexpr int ROUTER_BM = 64, ROUTER_BN = 64, ROUTER_BK = 32, ROUTER_THREADS = 64;

static int router_chains(const int64_t experts) { return (int) std::max<int64_t>(1, 2048 / experts); }

// ROUTER_BK consecutive values of x as f32
static __device__ __forceinline__ void router_load(const nv_bfloat16 * p, float * v) {
#pragma unroll
    for (int q = 0; q < ROUTER_BK/8; ++q) {
        const uint4 w = ((const uint4 *) p)[q];
        const __nv_bfloat162 * b = (const __nv_bfloat162 *) &w;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[8*q + 2*i] = __low2float(b[i]);
            v[8*q + 2*i + 1] = __high2float(b[i]);
        }
    }
}

static __device__ __forceinline__ void router_load(const float * p, float * v) {
#pragma unroll
    for (int q = 0; q < ROUTER_BK/4; ++q) {
        const float4 w = ((const float4 *) p)[q];
        v[4*q] = w.x; v[4*q + 1] = w.y; v[4*q + 2] = w.z; v[4*q + 3] = w.w;
    }
}

template <typename T, typename W>
__global__ void __launch_bounds__(ROUTER_THREADS) router_partials(
        const T * __restrict__ x, const W * __restrict__ router, float * __restrict__ partials, const int tokens,
        const int hidden, const int experts, const int ks) {
    __shared__ __align__(16) float xs[ROUTER_BK][ROUTER_BM];
    __shared__ __align__(16) float rs[ROUTER_BK][ROUTER_BN];
    const int tid = threadIdx.x, tx = tid % 8, ty = tid / 8;
    const int t0 = blockIdx.x * ROUTER_BM, e0 = blockIdx.y * ROUTER_BN, g = blockIdx.z;
    const int k0 = g * (hidden / ks), k1 = k0 + hidden / ks;
    const T * xp = x + (int64_t) min(t0 + tid, tokens - 1)*hidden;              // rows past the call: unused
    const W * rp = router + (int64_t) (e0 + tid)*hidden;
    float xv[ROUTER_BK], rv[ROUTER_BK];
    router_load(xp + k0, xv);
    router_load(rp + k0, rv);
    float acc[8][8] = {{0.0f}};
    for (int h0 = k0; h0 < k1; h0 += ROUTER_BK) {
#pragma unroll
        for (int k = 0; k < ROUTER_BK; ++k) {
            xs[k][tid] = xv[k];
            rs[k][tid] = rv[k];
        }
        __syncthreads();
        if (h0 + ROUTER_BK < k1) {                                          // the next stage's, in flight meanwhile
            router_load(xp + h0 + ROUTER_BK, xv);
            router_load(rp + h0 + ROUTER_BK, rv);
        }
#pragma unroll
        for (int k = 0; k < ROUTER_BK; ++k) {
            const float4 a0 = *(const float4 *) &xs[k][8*ty], a1 = *(const float4 *) &xs[k][8*ty + 4];
            const float4 b0 = *(const float4 *) &rs[k][8*tx], b1 = *(const float4 *) &rs[k][8*tx + 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int t = t0 + 8*ty + i;
        if (t < tokens) {
            float * out = partials + ((int64_t) g*tokens + t)*experts + e0 + 8*tx;
            ((float4 *) out)[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            ((float4 *) out)[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
}

__global__ void router_sum(const float * __restrict__ partials, float * __restrict__ logits, const int64_t n,
                           const int ks) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float sum = partials[i];
    for (int g = 1; g < ks; ++g) {
        sum += partials[g*n + i];
    }
    logits[i] = sum;
}

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

int holon_ggml_mmq_plant(int bits) {
    const int before = holon_mmq_plant;
    holon_mmq_plant = bits;
    return before;
}

void holon_ggml_mmq_fixups(long long * launched, long long * planted) {
    *launched = holon_mmq_fixups[0];
    *planted = holon_mmq_fixups[1];
    holon_mmq_fixups[0] = holon_mmq_fixups[1] = 0;
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
                        const float * gate_scale, const float * up_scale, int null_slots, void * workspace,
                        size_t workspace_bytes, void * stream) {
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
    fusion.null_slots = null_slots;           // an arena's null slots: their entries store 0, computing nothing
    ggml_cuda_mul_mat_vec_q(c.ctx, &src0, &src1, &ids_t, &dst, swiglu || up_scale || null_slots ? &fusion : nullptr);
}

void holon_ggml_mmq_id(int type, int prec, const void * w, int64_t experts, int64_t out, int64_t in, const void * x,
                       int x_bf16, int64_t rows_per_token, const int32_t * ids, int64_t tokens, int64_t k, float * y,
                       int64_t slots, int32_t * errors, int swiglu, const float * gate_scale, const float * up_scale,
                       int64_t ids_j_tokens, void * workspace, size_t workspace_bytes, void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_tensor src0 = experts_view((ggml_type) type, w, experts, out, in, 0, out);
    const ggml_tensor src1 = tensor3(act(x_bf16), x, in, rows_per_token, tokens);
    const ggml_tensor ids_t = matrix(GGML_TYPE_I32, ids, k, tokens);
    ggml_tensor dst = with_prec(tensor3(GGML_TYPE_F32, y, swiglu ? out/2 : out, k, tokens), prec);
    ggml_cuda_mul_mat_q(c.ctx, &src0, &src1, &ids_t, &dst, nullptr, slots, errors, swiglu, gate_scale, up_scale,
                        ids_j_tokens);
}

int holon_ggml_mmq_glu(int type, int prec, int64_t out) {
    const int cc = device_cc();
    const ggml_prec prec_src1 = (ggml_prec) holon_ggml_mmq_prec(type, prec);
    for (int J = 8; J <= 128; J += 8) {                                   // every tile width a call may pick
        const bool fallback = ggml_cuda_mmq_needs_fallback(out);
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) type, J, fallback, cc, prec_src1);
        if (c.type == GGML_TYPE_COUNT) {
            continue;
        }
        const size_t smem = mmq_get_nbytes_shared(c, cc);
        if (fallback || !c.use_mma_data_layout(cc) || out % c.I != 0 ||
                (smem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo && size_t(c.I/2*c.J)*sizeof(float) > smem - c.J*sizeof(int))) {
            return 0;
        }
    }
    return 1;
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

template <typename W>
static void router_logits(const void * x, int x_bf16, const W * router, int64_t tokens, int64_t hidden,
                          int64_t experts, float * partials, float * logits, cudaStream_t stream, int ks) {
    const dim3 grid((tokens + ROUTER_BM - 1)/ROUTER_BM, experts/ROUTER_BN, ks);
    if (x_bf16) {
        router_partials<<<grid, ROUTER_THREADS, 0, stream>>>(
            (const nv_bfloat16 *) x, router, partials, tokens, hidden, experts, ks);
    } else {
        router_partials<<<grid, ROUTER_THREADS, 0, stream>>>(
            (const float *) x, router, partials, tokens, hidden, experts, ks);
    }
}

void holon_ggml_moe_router_logits(const void * x, int x_bf16, const void * router, int router_bf16, int64_t tokens,
                                  int64_t hidden, int64_t experts, float * partials, float * logits, void * stream) {
    const int ks = router_chains(experts);
    GGML_ASSERT(hidden % (ks*ROUTER_BK) == 0 && experts % ROUTER_BN == 0 && tokens > 0);
    GGML_ASSERT((uintptr_t) x % 16 == 0 && (uintptr_t) router % 16 == 0 && (uintptr_t) partials % 16 == 0);
    if (router_bf16) {       // the router's exact values as f32 (router_load): the same chains, the same logits
        router_logits(x, x_bf16, (const nv_bfloat16 *) router, tokens, hidden, experts, partials, logits,
                      (cudaStream_t) stream, ks);
    } else {
        router_logits(x, x_bf16, (const float *) router, tokens, hidden, experts, partials, logits,
                      (cudaStream_t) stream, ks);
    }
    const int64_t n = tokens*experts;
    router_sum<<<(n + 255)/256, 256, 0, (cudaStream_t) stream>>>(partials, logits, n, ks);
    CUDA_CHECK(cudaGetLastError());
}

int64_t holon_ggml_moe_router_partials(int64_t experts) { return router_chains(experts); }

int64_t holon_ggml_moe_router_rows(void) { return HOLON_ROUTER_MAX_ROWS; }

int64_t holon_ggml_moe_router_tiles(int64_t tokens) { return (tokens + HOLON_ROUTER_ROWS - 1) / HOLON_ROUTER_ROWS; }

void holon_ggml_moe_router(const void * x, int x_bf16, const void * router, int router_bf16, int64_t tokens,
                           int64_t hidden, int64_t experts, int64_t k, int renormalize, float * logits, uint32_t * done,
                           float * weights, int32_t * ids, void * x_q8_1, void * stream) {
    GGML_ASSERT((uintptr_t) x % (x_bf16 ? 8 : 16) == 0 && (uintptr_t) router % (router_bf16 ? 8 : 16) == 0);
    // the sum's clamp, as holon_ggml_topk_moe's (llama-graph.cpp)
    ggml_cuda_holon_moe_router(x, x_bf16, router, router_bf16, tokens, hidden, experts, k,
                               renormalize ? 6.103515625e-5f : -INFINITY,
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
