// Torch bindings over the C API. Only ggml_cuda_api.h is included, so torch's -D__CUDA_NO_HALF* flags never meet
// llama.cpp's headers. Every call runs on the tensors' device and torch's current stream (CUDA-graph capturable).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

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

// An optional f32 scale of `n` values on x's device (NVFP4's scale2: per tensor or per expert), as a pointer.
const float * scale(const std::optional<at::Tensor> & s, const at::Tensor & like, int64_t n, const char * name) {
    if (!s) {
        return nullptr;
    }
    TORCH_CHECK(s->scalar_type() == at::kFloat && s->device() == like.device() && s->is_contiguous() && s->numel() == n,
                name, ": ", n, " contiguous float32 on ", like.device());
    return s->data_ptr<float>();
}

// The block_q8_1 bytes of `rows` activation rows of `in` values padded to 512 (as mmvq quantizes them).
int64_t q8_1_bytes(int64_t rows, int64_t in) { return rows * ((in + 511) / 512 * 512) / 32 * 36; }

void matmul(bool vec, int64_t type, int64_t prec, const at::Tensor & w, const at::Tensor & x, at::Tensor & y,
            const std::optional<at::Tensor> & s, at::Tensor & ws, const std::optional<at::Tensor> & gate = std::nullopt,
            const std::optional<at::Tensor> & xq = std::nullopt) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t rows = x.size(0), in = x.size(1), out = y.size(1);
    check(x, x, {rows, in}, "x");
    check(y, x, {rows, out}, "y");
    check_blocks(type, w, x, out, in);
    TORCH_CHECK(ws.scalar_type() == at::kByte && ws.device() == x.device() && ws.is_contiguous(), "workspace: uint8");
    TORCH_CHECK(rows > 0, "no rows");
    if (vec) {
        if (gate) {
            TORCH_CHECK(rows == 1, "mmvq: a fused gate only at one row");
            check_blocks(type, *gate, x, out, in);
        }
        if (xq) {                                   // x quantized already: at least its rows' bytes, uint8
            TORCH_CHECK(xq->scalar_type() == at::kByte && xq->device() == x.device() && xq->is_contiguous() &&
                        xq->numel() >= q8_1_bytes(rows, in), "xq: ", q8_1_bytes(rows, in), " contiguous uint8");
        }
        holon_ggml_mmvq(type, w.data_ptr(), out, in, x.data_ptr(), bf16(x, "x"), xq ? xq->data_ptr() : nullptr, rows,
                        y.data_ptr(), bf16(y, "y"), scale(s, x, 1, "scale"), gate ? gate->data_ptr() : nullptr,
                        ws.data_ptr(), ws.numel(), stream());
    } else {
        holon_ggml_mmq(type, prec, w.data_ptr(), out, in, x.data_ptr(), bf16(x, "x"), rows, y.data_ptr(), bf16(y, "y"),
                       scale(s, x, 1, "scale"), ws.data_ptr(), ws.numel(), stream());
    }
}

void check_f32(const at::Tensor & t, const char * name) { TORCH_CHECK(t.scalar_type() == at::kFloat, name, ": float32"); }

// x [tokens, 1 or k, in], ids [tokens, k] int32, y [tokens, k, out_y] f32; the stack `w` [E·out, row_bytes] with
// out = 2·out_y for the fused SwiGLU (mmvq only), else out_y.
void experts(bool vec, int64_t type, int64_t prec, const at::Tensor & w, const at::Tensor & x, const at::Tensor & ids,
             at::Tensor & y, bool swiglu, const std::optional<at::Tensor> & gate_scale,
             const std::optional<at::Tensor> & up_scale, at::Tensor & ws, int64_t slots = 0,
             const std::optional<at::Tensor> & errors = std::nullopt) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.dim() == 3 && ids.dim() == 2 && y.dim() == 3, "x, ids, y: 3, 2 and 3 dimensions");
    const int64_t tokens = ids.size(0), k = ids.size(1), rows = x.size(1), in = x.size(2), out_y = y.size(2);
    const int64_t out = swiglu ? 2*out_y : out_y;
    TORCH_CHECK(rows == 1 || rows == k, "x: 1 or k rows per token, got ", rows);
    check(x, x, {tokens, rows, in}, "x");
    check(ids, x, {tokens, k}, "ids");
    TORCH_CHECK(ids.scalar_type() == at::kInt, "ids: int32");
    check(y, x, {tokens, k, out_y}, "y");
    check_f32(y, "y");
    TORCH_CHECK(out > 0 && w.dim() == 2 && w.size(0) % out == 0, "blocks: experts x ", out, " rows");
    check_blocks(type, w, x, w.size(0), in);
    TORCH_CHECK(ws.scalar_type() == at::kByte && ws.device() == x.device() && ws.is_contiguous(), "workspace: uint8");
    TORCH_CHECK(tokens > 0, "no tokens");
    if (vec) {
        TORCH_CHECK(holon_ggml_use_mmvq_id(type, tokens), "mmvq with ids: ", tokens, " tokens is past its limit");
        const int64_t n = w.size(0) / out;
        TORCH_CHECK(swiglu || !gate_scale, "mmvq with ids: a gate scale needs the fused SwiGLU");
        holon_ggml_mmvq_id(type, w.data_ptr(), n, out, in, x.data_ptr(), bf16(x, "x"), rows, ids.data_ptr<int32_t>(),
                           tokens, k, y.data_ptr<float>(), swiglu, scale(gate_scale, x, n, "gate_scale"),
                           scale(up_scale, x, n, "up_scale"), ws.data_ptr(), ws.numel(), stream());
    } else {
        TORCH_CHECK(!swiglu && !gate_scale && !up_scale, "mmq with ids: no fused SwiGLU or scales");
        holon_ggml_mmq_id(type, prec, w.data_ptr(), w.size(0) / out, out, in, x.data_ptr(), bf16(x, "x"), rows,
                          ids.data_ptr<int32_t>(), tokens, k, y.data_ptr<float>(), slots,
                          errors ? errors->data_ptr<int32_t>() : nullptr, ws.data_ptr(), ws.numel(), stream());
    }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_bytes", [](int64_t type, int64_t in) { return holon_ggml_row_bytes(type, in); });
    m.def("use_mmvq", [](int64_t cc, int64_t type, int64_t rows) { return holon_ggml_use_mmvq(type, cc, rows) != 0; });
    m.def("use_mmq", [](int64_t device, int64_t type, int64_t rows) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_use_mmq(type, rows) != 0;
    });
    m.def("mmq_prec", [](int64_t device, int64_t type, int64_t prec) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_mmq_prec(type, prec);
    });
    m.def("mmq_schedule", [](int64_t device, int64_t type, int64_t prec, int64_t out, int64_t rows) {
        const c10::cuda::CUDAGuard guard(device);
        int tile_i, tile_j;
        const int kind = holon_ggml_mmq_schedule(type, prec, out, rows, &tile_i, &tile_j);
        return std::make_tuple(kind, tile_i, tile_j);
    });
    m.def("workspace_bytes", [](int64_t device, int64_t in, int64_t rows) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_workspace_bytes(in, rows);
    });
    m.def("workspace_high_water", [] { return holon_ggml_workspace_high_water(); });
    m.def("mmvq", [](int64_t type, const at::Tensor & w, const at::Tensor & x, at::Tensor & y,
                     const std::optional<at::Tensor> & s, at::Tensor & ws, const std::optional<at::Tensor> & gate,
                     const std::optional<at::Tensor> & xq) { matmul(true, type, 0, w, x, y, s, ws, gate, xq); },
          py::arg("type"), py::arg("w"), py::arg("x"), py::arg("y"), py::arg("scale"), py::arg("ws"),
          py::arg("gate") = py::none(), py::arg("xq") = py::none());
    m.def("mmq", [](int64_t type, int64_t prec, const at::Tensor & w, const at::Tensor & x, at::Tensor & y,
                    const std::optional<at::Tensor> & s, at::Tensor & ws) { matmul(false, type, prec, w, x, y, s, ws); });
    m.def("quantize_fp4", [](int64_t type, const at::Tensor & x, at::Tensor & blocks, at::Tensor & scale) {
        const c10::cuda::CUDAGuard guard(x.device());
        TORCH_CHECK(x.dim() == 2 && x.size(0) > 0, "x: [rows, in]");
        const int64_t rows = x.size(0), in = x.size(1), padded = (in + 511) / 512 * 512;
        check(x, x, {rows, in}, "x");
        check_f32(x, "x");
        check(blocks, x, {padded / 256, rows, 144}, "blocks");
        check(scale, x, {rows}, "scale");
        TORCH_CHECK(blocks.scalar_type() == at::kByte, "blocks: uint8");
        check_f32(scale, "scale");
        holon_ggml_quantize_fp4(type, x.data_ptr<float>(), rows, in, blocks.data_ptr(), scale.data_ptr<float>(),
                                stream());
    });
    m.def("quantize_q8_1", [](int64_t type, const at::Tensor & x, at::Tensor & y) {
        const c10::cuda::CUDAGuard guard(x.device());
        TORCH_CHECK(x.dim() == 2 && x.size(0) > 0, "x: [rows, in]");
        const int64_t rows = x.size(0), in = x.size(1);
        check(x, x, {rows, in}, "x");
        check(y, x, {q8_1_bytes(rows, in)}, "y");                  // block_q8_1: 32 int8 values and a half2
        TORCH_CHECK(y.scalar_type() == at::kByte, "y: uint8");
        holon_ggml_quantize_q8_1(type, x.data_ptr(), bf16(x, "x"), rows, in, y.data_ptr(), stream());
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
    m.def("use_mmvq_id", [](int64_t device, int64_t type, int64_t tokens) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_use_mmvq_id(type, tokens) != 0;
    });
    m.def("workspace_bytes_id", [](int64_t device, int64_t in, int64_t tokens, int64_t k, int64_t slots) {
        const c10::cuda::CUDAGuard guard(device);
        return holon_ggml_workspace_bytes_id(in, tokens, k, slots);
    });
    m.def("mmvq_id", [](int64_t type, const at::Tensor & w, const at::Tensor & x, const at::Tensor & ids, at::Tensor & y,
                        bool swiglu, const std::optional<at::Tensor> & gate_scale,
                        const std::optional<at::Tensor> & up_scale, at::Tensor & ws) {
        experts(true, type, 0, w, x, ids, y, swiglu, gate_scale, up_scale, ws);
    });
    m.def("mmq_id", [](int64_t type, int64_t prec, const at::Tensor & w, const at::Tensor & x, const at::Tensor & ids,
                       at::Tensor & y, at::Tensor & ws, int64_t slots, const std::optional<at::Tensor> & errors) {
        TORCH_CHECK(slots > 0, "mmq with ids: slots > 0");
        TORCH_CHECK(!errors || (errors->scalar_type() == at::kInt && errors->device() == x.device() && errors->numel() >= 1),
                    "errors: one int32 on x's device");
        experts(false, type, prec, w, x, ids, y, false, std::nullopt, std::nullopt, ws, slots, errors);
    });
    m.def("swiglu", [](const at::Tensor & x, at::Tensor & y, const std::optional<at::Tensor> & gate_scale,
                       const std::optional<at::Tensor> & up_scale, const std::optional<at::Tensor> & ids) {
        const c10::cuda::CUDAGuard guard(y.device());
        const int64_t n = y.size(-1), rows = n > 0 ? y.numel() / n : 0;
        TORCH_CHECK(x.dim() == y.dim() && x.size(-1) == 2*n && x.numel() == 2*y.numel(), "x: y's shape, last dim 2n");
        check(x, y, x.sizes(), "x");
        check(y, y, y.sizes(), "y");
        check_f32(x, "x");
        check_f32(y, "y");
        TORCH_CHECK(rows > 0, "no rows");
        TORCH_CHECK(!gate_scale == !ids && !up_scale == !ids, "swiglu: gate and up scales with the rows' ids, or none");
        const float * gs = nullptr, * us = nullptr;
        if (ids) {
            check(*ids, y, {rows}, "ids");                         // [tokens, k], one expert per row, flattened
            TORCH_CHECK(ids->scalar_type() == at::kInt, "ids: int32");
            TORCH_CHECK(gate_scale->numel() == up_scale->numel(), "gate_scale, up_scale: one value per expert each");
            gs = scale(gate_scale, y, gate_scale->numel(), "gate_scale");
            us = scale(up_scale, y, up_scale->numel(), "up_scale");
        }
        holon_ggml_swiglu(x.data_ptr<float>(), rows, n, gs, us, ids ? ids->data_ptr<int32_t>() : nullptr,
                          y.data_ptr<float>(), stream());
    }, py::arg("x"), py::arg("y"), py::arg("gate_scale") = py::none(), py::arg("up_scale") = py::none(),
       py::arg("ids") = py::none());
    m.def("topk_moe", [](const at::Tensor & logits, at::Tensor & weights, at::Tensor & ids, bool renormalize) {
        const c10::cuda::CUDAGuard guard(logits.device());
        TORCH_CHECK(logits.dim() == 2 && ids.dim() == 2, "logits, ids: 2 dimensions");
        const int64_t tokens = logits.size(0), experts = logits.size(1), k = ids.size(1);
        check(logits, logits, {tokens, experts}, "logits");
        check(weights, logits, {tokens, k}, "weights");
        check(ids, logits, {tokens, k}, "ids");
        check_f32(logits, "logits");
        check_f32(weights, "weights");
        TORCH_CHECK(ids.scalar_type() == at::kInt, "ids: int32");
        TORCH_CHECK(experts > 0 && experts <= 512 && (experts & (experts - 1)) == 0, "experts: a power of 2 <= 512");
        TORCH_CHECK(k > 0 && k <= experts && tokens > 0, "1 <= k <= experts, tokens > 0");
        holon_ggml_topk_moe(logits.data_ptr<float>(), tokens, experts, k, renormalize, weights.data_ptr<float>(),
                            ids.data_ptr<int32_t>(), stream());
    });
    m.def("moe_combine", [](const at::Tensor & y, const at::Tensor & weights, at::Tensor & out,
                            const std::optional<at::Tensor> & residual) {
        const c10::cuda::CUDAGuard guard(y.device());
        TORCH_CHECK(y.dim() == 3, "y: 3 dimensions");
        const int64_t tokens = y.size(0), k = y.size(1), hidden = y.size(2);
        check(y, y, {tokens, k, hidden}, "y");
        check(weights, y, {tokens, k}, "weights");
        check(out, y, {tokens, hidden}, "out");
        check_f32(y, "y");
        check_f32(weights, "weights");
        TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out: bfloat16");
        if (residual) {
            check(*residual, y, {tokens, hidden}, "residual");
            TORCH_CHECK(residual->scalar_type() == at::kBFloat16, "residual: bfloat16");
        }
        TORCH_CHECK(tokens > 0 && hidden > 0, "no tokens");
        holon_ggml_moe_combine(y.data_ptr<float>(), weights.data_ptr<float>(), tokens, k, hidden,
                               residual ? residual->data_ptr() : nullptr, out.data_ptr(), stream());
    }, py::arg("y"), py::arg("weights"), py::arg("out"), py::arg("residual") = py::none());
}
