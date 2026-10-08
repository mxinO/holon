// What the vendored llama.cpp kernels need from the rest of ggml (ggml-base, ggml-cuda.cu), and nothing more:
// type traits and tensor helpers, logging/abort, the device table, and the backend context's destructor.
//
// Adapted from ggml in llama.cpp 51ce9c11 (ggml/src/ggml.c, ggml/src/ggml-cuda/ggml-cuda.cu), MIT License,
// Copyright (c) 2023-2026 The ggml authors; the license is in ../vendor/LICENSE.
#include "common.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace {

struct type_traits {
    const char * name;
    int64_t      blck;
    size_t       size;
    bool         quantized;
};

type_traits traits(const ggml_type type) {
#define PLAIN(T, name, ctype) case GGML_TYPE_##T: return {name, 1, sizeof(ctype), false};
#define QUANT(T, name, qk, block) case GGML_TYPE_##T: return {name, qk, sizeof(block), true};
    switch (type) {
        PLAIN(F32, "f32", float) PLAIN(F16, "f16", ggml_fp16_t) PLAIN(BF16, "bf16", ggml_bf16_t)
        PLAIN(F64, "f64", double) PLAIN(I8, "i8", int8_t) PLAIN(I16, "i16", int16_t) PLAIN(I32, "i32", int32_t)
        PLAIN(I64, "i64", int64_t)
        QUANT(Q4_0, "q4_0", QK4_0, block_q4_0) QUANT(Q4_1, "q4_1", QK4_1, block_q4_1)
        QUANT(Q5_0, "q5_0", QK5_0, block_q5_0) QUANT(Q5_1, "q5_1", QK5_1, block_q5_1)
        QUANT(Q8_0, "q8_0", QK8_0, block_q8_0) QUANT(Q8_1, "q8_1", QK8_1, block_q8_1)
        QUANT(Q1_0, "q1_0", QK1_0, block_q1_0) QUANT(Q2_0, "q2_0", QK2_0, block_q2_0)
        QUANT(Q2_K, "q2_K", QK_K, block_q2_K) QUANT(Q3_K, "q3_K", QK_K, block_q3_K)
        QUANT(Q4_K, "q4_K", QK_K, block_q4_K) QUANT(Q5_K, "q5_K", QK_K, block_q5_K)
        QUANT(Q6_K, "q6_K", QK_K, block_q6_K) QUANT(Q8_K, "q8_K", QK_K, block_q8_K)
        QUANT(IQ2_XXS, "iq2_xxs", QK_K, block_iq2_xxs) QUANT(IQ2_XS, "iq2_xs", QK_K, block_iq2_xs)
        QUANT(IQ2_S, "iq2_s", QK_K, block_iq2_s) QUANT(IQ3_XXS, "iq3_xxs", QK_K, block_iq3_xxs)
        QUANT(IQ3_S, "iq3_s", QK_K, block_iq3_s) QUANT(IQ1_S, "iq1_s", QK_K, block_iq1_s)
        QUANT(IQ1_M, "iq1_m", QK_K, block_iq1_m) QUANT(IQ4_NL, "iq4_nl", QK4_NL, block_iq4_nl)
        QUANT(IQ4_XS, "iq4_xs", QK_K, block_iq4_xs) QUANT(TQ1_0, "tq1_0", QK_K, block_tq1_0)
        QUANT(TQ2_0, "tq2_0", QK_K, block_tq2_0) QUANT(MXFP4, "mxfp4", QK_MXFP4, block_mxfp4)
        QUANT(NVFP4, "nvfp4", QK_NVFP4, block_nvfp4)
        default: GGML_ABORT("ggml_cuda: unknown ggml type %d", (int) type);
    }
#undef PLAIN
#undef QUANT
}

} // namespace

// ggml-base

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
    abort();
}

void ggml_log_internal(ggml_log_level level, const char * format, ...) {
    if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO) {
        return;
    }
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

const char * ggml_type_name(ggml_type type) { return traits(type).name; }
int64_t ggml_blck_size(ggml_type type) { return traits(type).blck; }
size_t ggml_type_size(ggml_type type) { return traits(type).size; }
bool ggml_is_quantized(ggml_type type) { return traits(type).quantized; }

size_t ggml_row_size(ggml_type type, int64_t ne) {
    GGML_ASSERT(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

int64_t ggml_nelements(const ggml_tensor * t) { return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]; }

size_t ggml_element_size(const ggml_tensor * t) { return ggml_type_size(t->type); }

size_t ggml_nbytes(const ggml_tensor * t) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0) {
            return 0;
        }
    }
    size_t nbytes = t->ne[0] * t->nb[0] / ggml_blck_size(t->type);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        nbytes += (t->ne[i] - 1) * t->nb[i];
    }
    return nbytes;
}

bool ggml_is_contiguous(const ggml_tensor * t) {
    size_t next = ggml_type_size(t->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] != 1 && t->nb[i] != next) {
            return false;
        }
        next *= i == 0 ? t->ne[0] / ggml_blck_size(t->type) : t->ne[i];
    }
    return true;
}

bool ggml_is_contiguously_allocated(const ggml_tensor * t) {
    return ggml_nbytes(t) == ggml_nelements(t) * ggml_type_size(t->type) / ggml_blck_size(t->type);
}

bool ggml_is_matrix(const ggml_tensor * t) { return t->ne[2] == 1 && t->ne[3] == 1; }

bool ggml_are_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

bool ggml_are_same_stride(const ggml_tensor * a, const ggml_tensor * b) {
    return a->nb[0] == b->nb[0] && a->nb[1] == b->nb[1] && a->nb[2] == b->nb[2] && a->nb[3] == b->nb[3];
}

// Our tensors have no ggml buffer: they are never compute buffers, so llama.cpp leaves their padding alone.
ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer == nullptr);
    return GGML_BACKEND_BUFFER_USAGE_ANY;
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const ggml_tensor * t) {
    GGML_ASSERT(buffer == nullptr);
    return ggml_nbytes(t);
}

// ggml-cuda runtime

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    ggml_abort(file, line, "CUDA error in %s: %s (%s)", func, msg, stmt);
    abort();
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_set_device(int device) {
    if (device != ggml_cuda_get_device()) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

// The true device facts, which llama.cpp's kernel choices depend on (cc, nsm, smpbo, warp size).
const ggml_cuda_device_info & ggml_cuda_info() {
    static const ggml_cuda_device_info info = [] {
        ggml_cuda_device_info info = {};
        CUDA_CHECK(cudaGetDeviceCount(&info.physical_device_count));
        GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);
        info.device_count = info.physical_device_count;
        for (int id = 0; id < info.device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = info.devices[id];
            d.cc = 100*prop.major + 10*prop.minor;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlockOptin;
            d.warp_size = prop.warpSize;
            d.total_vram = prop.totalGlobalMem;
            d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
            d.physical_device = id;
            d.physical_share_count = 1;
        }
        return info;
    }();
    return info;
}

// The context never owns streams or handles here: the caller's stream is borrowed per call and the pool is ours.
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    GGML_ABORT("ggml_cuda: the pool is bound per call, never created by llama.cpp");
}
