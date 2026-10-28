"""Correctness of the S2 kernels vs gguf-py, for every ggml type in the S2 files -> results/correctness_sm*.json.

dequantize: ours vs gguf.quants.dequantize (fp32 bit-exact fraction, max abs error; bf16 output exact fraction).
The vendored legacy/k-quant dequant computes in fp16, so it passes at <= 1e-3 * max|w| rather than bit-exact.
matmuls: NMSE vs x.float() @ dequant_ref(W).T, budget 5e-4 (llama.cpp test-backend-ops MUL_MAT threshold);
"fallback" = our dequant to bf16 + torch bf16 GEMM, "bf16_ref" = gguf-py dequant rounded to bf16 + torch GEMM.
"""
import json
import os
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ggml_ops  # noqa: E402
from models import find_tensor  # noqa: E402

TYPES = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ2_XS"]
MMQ = {"Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
ROWS = 2048  # rows of each real tensor used (gguf-py's numpy i-quant dequant is slow)
NMSE_BUDGET = 5e-4
EXPERTS, TOP_K, MOE_TOKENS = 8, 4, 64


def nmse(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((y.float() - ref) ** 2).sum() / (ref**2).sum())


def main() -> None:
    ext = ggml_ops.load_ext()
    torch.manual_seed(0)
    dev = "cuda"
    report = {"device": torch.cuda.get_device_name(), "nmse_budget": NMSE_BUDGET, "types": {}}
    for name in TYPES:
        key, t = find_tensor(name)
        qtype = int(t.tensor_type)
        data = np.ascontiguousarray(t.data[:ROWS])
        ref = torch.from_numpy(gguf.quants.dequantize(data, t.tensor_type)).to(dev)
        rows, cols = ref.shape
        w = torch.from_numpy(data).to(dev)
        r = {"tensor": f"{key}:{t.name}", "shape": [rows, cols]}

        out32 = torch.empty(rows, cols, device=dev)
        ext.dequantize(w, qtype, out32)
        out16 = torch.empty(rows, cols, device=dev, dtype=torch.bfloat16)
        ext.dequantize(w, qtype, out16)
        r["dequant_f32_exact_frac"] = float((out32 == ref).float().mean())
        r["dequant_f32_max_abs_err"] = float((out32 - ref).abs().max())
        r["dequant_f32_max_abs_w"] = float(ref.abs().max())
        r["dequant_bf16_exact_frac"] = float((out16 == ref.to(torch.bfloat16)).float().mean())

        w16 = ref.to(torch.bfloat16)
        r["dequant_fp32_exact"] = r["dequant_f32_exact_frac"] == 1.0
        for n in (1, 8, 32, 512):
            x = torch.randn(n, cols, device=dev, dtype=torch.bfloat16)
            yref = x.float() @ ref.T
            xq = torch.empty(ggml_ops.q8_1_words(n, cols), dtype=torch.int32, device=dev)
            y = torch.empty(n, rows, device=dev, dtype=torch.bfloat16)
            if n <= 32:
                ext.mul_mat_vec(w, x, qtype, y, xq)
                r[f"mmvq_n{n}_nmse"] = nmse(y, yref)
            if n >= 32 and name in MMQ:
                ext.mul_mat(w, x, qtype, y, xq)
                r[f"mmq_n{n}_nmse"] = nmse(y, yref)
            r[f"bf16_ref_gemm_n{n}_nmse"] = nmse(x @ w16.T, yref)
            r[f"fallback_n{n}_nmse"] = nmse(x @ out16.T, yref)

        # MoE: split the real tensor's rows into EXPERTS experts.
        er = rows // EXPERTS
        we = w.reshape(EXPERTS, er, -1)
        refe = ref.reshape(EXPERTS, er, cols)
        x = torch.randn(MOE_TOKENS, cols, device=dev, dtype=torch.bfloat16)
        topk = torch.stack([torch.randperm(EXPERTS, device=dev)[:TOP_K] for _ in range(MOE_TOKENS)]).int()
        yref = torch.einsum("tc,tkrc->tkr", x.float(), refe[topk.long()]).reshape(-1, er)
        xq = torch.empty(ggml_ops.q8_1_words(MOE_TOKENS, cols), dtype=torch.int32, device=dev)
        y = torch.empty(MOE_TOKENS * TOP_K, er, device=dev, dtype=torch.bfloat16)
        ext.moe_vec(x, we, topk, qtype, y, xq)
        r["moe_vec_nmse"] = nmse(y, yref)
        if name in MMQ:
            sorted_ids, expert_ids, npad = ggml_ops.moe_align(topk, ext.moe_block_size(qtype), EXPERTS)
            y.zero_()
            ext.moe(x, we, sorted_ids, expert_ids, npad, qtype, TOP_K, y, xq)
            r["moe_mmq_nmse"] = nmse(y, yref)

        worst = max(v for k, v in r.items() if k.endswith("nmse"))
        r["pass"] = worst <= NMSE_BUDGET and r["dequant_f32_max_abs_err"] <= 1e-3 * r["dequant_f32_max_abs_w"]
        report["types"][name] = r
        print(name, json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()}))

    sm = "".join(map(str, torch.cuda.get_device_capability()))
    (HERE / "results" / f"correctness_{os.environ.get('S2_TAG') or 'sm' + sm}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
