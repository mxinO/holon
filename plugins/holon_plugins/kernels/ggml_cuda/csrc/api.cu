// The C API (ggml_cuda_api.h): llama.cpp's own entry points, called with ggml tensors built on the stack over the
// caller's pointers, a context that borrows the caller's stream and a pool over the caller's workspace.
#include "ggml_cuda_api.h"

#include "common.cuh"
#include "convert.cuh"
#include "getrows.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"

namespace {

constexpr size_t POOL_ALIGN = 256;

// llama.cpp's scratch allocator, as a bump allocator over the caller's workspace: no cudaMalloc, so a call is safe
// inside CUDA graph capture. It lives for one call (llama.cpp frees everything before returning).
struct workspace_pool : ggml_cuda_pool {
    char * base;
    size_t size;
    size_t used = 0;

    workspace_pool(void * base, size_t size) : base((char *) base), size(size) {
        GGML_ASSERT((uintptr_t) base % POOL_ALIGN == 0);
    }

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

ggml_type act(const int bf16) { return bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32; }

int device_cc() { return ggml_cuda_info().devices[ggml_cuda_get_device()].cc; }

} // namespace

size_t holon_ggml_row_bytes(int type, int64_t in) {
    const ggml_type t = (ggml_type) type;
    return in % ggml_blck_size(t) == 0 ? ggml_row_size(t, in) : 0;
}

int holon_ggml_use_mmvq(int type, int64_t rows) { return ggml_cuda_should_use_mmvq((ggml_type) type, device_cc(), rows); }

int holon_ggml_use_mmq(int type, int64_t rows) { return ggml_cuda_should_use_mmq((ggml_type) type, device_cc(), rows, 0); }

int holon_ggml_mmq_schedule(int type, int64_t out, int64_t rows, int * tile_i, int * tile_j) {
    const auto & dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const bool fallback = out % 128 != 0;                              // as ggml_cuda_mul_mat_q
    int J_best = 0, ntx_best = INT_MAX;                                // as mul_mat_q_switch_J
    for (int J = 8; J <= 128 && ntx_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) type, J, fallback, dev.cc);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, dev.cc) <= dev.smpbo && (rows + J - 1)/J < ntx_best) {
            J_best = J;
            ntx_best = (rows + J - 1)/J;
        }
    }
    const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) type, J_best, fallback, dev.cc);
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

size_t holon_ggml_workspace_bytes(int64_t in, int64_t rows) {
    const int64_t in_padded = GGML_PAD(in, MATRIX_ROW_PADDING);
    // mmq's q8_1 activations plus its tile padding (J_max <= 512 rows); mmvq's block_q8_1 need less
    const size_t q8_1 = (rows*in_padded/QK8_1_MMQ + 512) * sizeof(block_q8_1_mmq);
    // stream-k fixup: one partial I x J f32 tile per SM (I, J <= 128 on every config we build)
    const size_t fixup = size_t(ggml_cuda_info().devices[ggml_cuda_get_device()].nsm) * 128*128 * sizeof(float);
    return GGML_PAD(q8_1, POOL_ALIGN) + GGML_PAD(fixup, POOL_ALIGN);
}

void holon_ggml_mmvq(int type, const void * w, int64_t out, int64_t in, const void * x, int x_bf16, int64_t rows,
                     void * y, int y_bf16, void * workspace, size_t workspace_bytes, void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_tensor src0 = matrix((ggml_type) type, w, in, out);
    const ggml_tensor src1 = matrix(act(x_bf16), x, in, rows);
    ggml_tensor dst = matrix(act(y_bf16), y, out, rows);
    ggml_cuda_mul_mat_vec_q(c.ctx, &src0, &src1, nullptr, &dst);
}

void holon_ggml_mmq(int type, const void * w, int64_t out, int64_t in, const void * x, int x_bf16, int64_t rows,
                    float * y, void * workspace, size_t workspace_bytes, void * stream) {
    call_context c(stream, workspace, workspace_bytes);
    const ggml_tensor src0 = matrix((ggml_type) type, w, in, out);
    const ggml_tensor src1 = matrix(act(x_bf16), x, in, rows);
    ggml_tensor dst = matrix(GGML_TYPE_F32, y, out, rows);
    ggml_cuda_mul_mat_q(c.ctx, &src0, &src1, nullptr, &dst);
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
