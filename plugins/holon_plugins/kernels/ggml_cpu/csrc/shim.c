// Adapted from llama.cpp cbe28bb3 (ggml/src/ggml.c, ggml/src/ggml-cpu/ggml-cpu.c), MIT: the few symbols of ggml-base
// and the CPU backend that the vendored quantization and dot-product sources use — the block sizes of the types the
// CPU experts read (ggml.c's type traits), ggml_abort, and the FP4 scale tables (ggml-cpu.c's, filled at load).
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static size_t block_size(enum ggml_type type, int64_t * blck) {
    *blck = QK_K;
    switch (type) {
        case GGML_TYPE_F32: *blck = 1; return sizeof(float);
        case GGML_TYPE_F16: *blck = 1; return sizeof(ggml_fp16_t);
        case GGML_TYPE_Q4_0: *blck = QK4_0; return sizeof(block_q4_0);
        case GGML_TYPE_Q5_0: *blck = QK5_0; return sizeof(block_q5_0);
        case GGML_TYPE_Q8_0: *blck = QK8_0; return sizeof(block_q8_0);
        case GGML_TYPE_IQ4_NL: *blck = QK4_NL; return sizeof(block_iq4_nl);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_Q8_K: return sizeof(block_q8_K);
        case GGML_TYPE_IQ1_S: return sizeof(block_iq1_s);
        case GGML_TYPE_IQ1_M: return sizeof(block_iq1_m);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS: return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ3_S: return sizeof(block_iq3_s);
        case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        default: *blck = 0; return 0;
    }
}

int64_t ggml_blck_size(enum ggml_type type) {
    int64_t blck;
    block_size(type, &blck);
    return blck;
}

size_t ggml_type_size(enum ggml_type type) {
    int64_t blck;
    return block_size(type, &blck);
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    int64_t blck;
    const size_t size = block_size(type, &blck);
    if (!blck || ne % blck) {
        ggml_abort(__FILE__, __LINE__, "ggml_row_size: type %d, %lld values", (int) type, (long long) ne);
    }
    return size * ne / blck;
}

const char * ggml_type_name(enum ggml_type type) {
    static char name[16];
    snprintf(name, sizeof(name), "type %d", (int) type);
    return name;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

__attribute__((constructor)) static void fill_tables(void) {
    for (int i = 0; i < (1 << 8); ++i) {
        ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
        ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
    }
}
