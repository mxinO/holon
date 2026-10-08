// Torch bindings over the C API. Only ggml_cuda_api.h is included, so torch's -D__CUDA_NO_HALF* flags never meet
// llama.cpp's headers. Every call runs on the tensors' device and torch's current stream (CUDA-graph capturable).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "ggml_cuda_api.h"

namespace {

int bf16(const at::Tensor & t, const char * name) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat, name, ": bfloat16 or float32");
    return t.scalar_type() == at::kBFloat16;
}

void check(const at::Tensor & t, const at::Tensor & like, at::IntArrayRef shape, const char * name) {
    TORCH_CHECK(t.device() == like.device() && t.is_contiguous() && t.sizes() == shape, name, ": contiguous ", shape,
                " on ", like.device(), ", got ", t.sizes(), " on ", t.device());
}

// Weights: `out` rows of the blocks of `in` values, uint8.
void check_blocks(int64_t type, const at::Tensor & w, const at::Tensor & like, int64_t out, int64_t in) {
    const size_t row = holon_ggml_row_bytes(type, in);
    TORCH_CHECK(row > 0, "ggml type ", type, ": ", in, " values are not whole blocks");
    TORCH_CHECK(w.scalar_type() == at::kByte, "blocks: uint8");
    check(w, like, {out, int64_t(row)}, "blocks");
}

void * stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void matmul(bool vec, int64_t type, const at::Tensor & w, const at::Tensor & x, at::Tensor & y, at::Tensor & ws) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t rows = x.size(0), in = x.size(1), out = y.size(1);
    check(x, x, {rows, in}, "x");
    check(y, x, {rows, out}, "y");
    check_blocks(type, w, x, out, in);
    TORCH_CHECK(ws.scalar_type() == at::kByte && ws.device() == x.device() && ws.is_contiguous(), "workspace: uint8");
    TORCH_CHECK(rows > 0, "no rows");
    if (vec) {
        holon_ggml_mmvq(type, w.data_ptr(), out, in, x.data_ptr(), bf16(x, "x"), rows, y.data_ptr(), bf16(y, "y"),
                        ws.data_ptr(), ws.numel(), stream());
    } else {
        TORCH_CHECK(y.scalar_type() == at::kFloat, "mmq: y float32");
        holon_ggml_mmq(type, w.data_ptr(), out, in, x.data_ptr(), bf16(x, "x"), rows, y.data_ptr<float>(),
                       ws.data_ptr(), ws.numel(), stream());
    }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_bytes", [](int64_t type, int64_t in) { return holon_ggml_row_bytes(type, in); });
    m.def("use_mmvq", [](int64_t device, int64_t type, int64_t rows) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_use_mmvq(type, rows) != 0;
    });
    m.def("use_mmq", [](int64_t device, int64_t type, int64_t rows) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_use_mmq(type, rows) != 0;
    });
    m.def("workspace_bytes", [](int64_t device, int64_t in, int64_t rows) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_workspace_bytes(in, rows);
    });
    m.def("mmvq", [](int64_t type, const at::Tensor & w, const at::Tensor & x, at::Tensor & y, at::Tensor & ws) {
        matmul(true, type, w, x, y, ws);
    });
    m.def("mmq", [](int64_t type, const at::Tensor & w, const at::Tensor & x, at::Tensor & y, at::Tensor & ws) {
        matmul(false, type, w, x, y, ws);
    });
    m.def("get_rows", [](int64_t type, const at::Tensor & table, const at::Tensor & ids, at::Tensor & y) {
        const c10::cuda::CUDAGuard guard(y.device());
        const int64_t n = y.size(0), hidden = y.size(1);
        check(ids, y, {n}, "ids");
        TORCH_CHECK(ids.scalar_type() == at::kInt, "ids: int32");
        check_blocks(type, table, y, table.size(0), hidden);
        TORCH_CHECK(n > 0, "no ids");
        holon_ggml_get_rows(type, table.data_ptr(), hidden, ids.data_ptr<int32_t>(), n, y.data_ptr(), bf16(y, "y"),
                            stream());
    });
    m.def("dequantize", [](int64_t type, const at::Tensor & src, at::Tensor & y) {
        const c10::cuda::CUDAGuard guard(y.device());
        check_blocks(type, src, y, y.size(0), y.size(1));
        holon_ggml_dequantize(type, src.data_ptr(), y.numel(), y.data_ptr(), bf16(y, "y"), stream());
    });
}
