"""Per-kernel timing of our extension on the S2 shapes -> results/ours_ops_sm*.json.

Methodology mirrors llama.cpp `test-backend-ops perf`: the op is repeated back to back on the same weights (here
R times inside one CUDA graph, timed with CUDA events), so small weights may be L2-resident in both harnesses.
Timed ops include the q8_1 activation quantization, as in llama.cpp's MUL_MAT. MoE routing (moe_align) is
precomputed and not timed.
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import gguf  # noqa: E402

import ggml_ops  # noqa: E402
from models import find_tensor  # noqa: E402

TYPES = ["IQ3_XXS", "IQ3_S", "IQ4_XS", "Q3_K", "Q4_K", "Q8_0"]
MMQ = {"Q3_K", "Q4_K", "Q8_0"}
DENSE = [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288), (4096, 2048), (512, 2048), (2048, 4096)]
NS = [1, 8, 32, 512]
EXPERTS, TOP_K = 128, 8
MOE = {"gate_up": (768, 2048, True), "down": (2048, 768, False)}  # (rows, k, input broadcast over slots)


def time_us(fn, target_ms: float = 20.0) -> float:
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    reps = max(1, min(500, int(target_ms * 1000 / max(s.elapsed_time(e) * 1000, 1.0))))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / reps)
    return best


def main() -> None:
    ext = ggml_ops.load_ext()
    dev = "cuda"
    torch.manual_seed(0)
    rows_out = []
    for name in TYPES:
        qtype = int(gguf.GGMLQuantizationType[name])
        _, t = find_tensor(name)
        raw = torch.from_numpy(t.data.copy().reshape(-1)).to(dev)
        for m, k in DENSE:
            w = ggml_ops.tile_weight(raw, qtype, (m, k))
            wdq = torch.empty(m, k, dtype=torch.bfloat16, device=dev)
            for n in NS:
                x = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
                y = torch.empty(n, m, dtype=torch.bfloat16, device=dev)
                xq = torch.empty(ggml_ops.q8_1_words(n, k), dtype=torch.int32, device=dev)
                case = {"type": name, "kind": "dense", "m": m, "k": k, "n": n}
                if n <= 32:
                    rows_out.append(case | {"path": "mmvq", "us": time_us(lambda: ext.mul_mat_vec(w, x, qtype, y, xq))})
                if name in MMQ and n >= 8:
                    rows_out.append(case | {"path": "mmq", "us": time_us(lambda: ext.mul_mat(w, x, qtype, y, xq))})

                def fallback():
                    ext.dequantize(w, qtype, wdq)
                    torch.matmul(x, wdq.T, out=y)

                rows_out.append(case | {"path": "dequant_gemm", "us": time_us(fallback)})
                print(rows_out[-1], flush=True)
            del w, wdq
        for label, (m, k, bcast) in MOE.items():
            w = ggml_ops.tile_weight(raw, qtype, (EXPERTS, m, k))
            for n in NS:
                topk = torch.stack([torch.randperm(EXPERTS, device=dev)[:TOP_K] for _ in range(n)]).int()
                if bcast:  # one input row per token, shared by its TOP_K experts
                    x = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
                    ids, kk = topk, TOP_K
                else:  # one input row per (token, slot)
                    x = torch.randn(n * TOP_K, k, dtype=torch.bfloat16, device=dev)
                    ids, kk = topk.reshape(-1, 1).contiguous(), 1
                y = torch.empty(n * TOP_K, m, dtype=torch.bfloat16, device=dev)
                xq = torch.empty(ggml_ops.q8_1_words(x.shape[0], k), dtype=torch.int32, device=dev)
                case = {"type": name, "kind": f"moe_{label}", "m": m, "k": k, "n": n}
                rows_out.append(case | {"path": "moe_vec", "us": time_us(lambda: ext.moe_vec(x, w, ids, qtype, y, xq))})
                print(rows_out[-1], flush=True)
                if name in MMQ:
                    sid, eid, npad = ggml_ops.moe_align(ids, ext.moe_block_size(qtype), EXPERTS)
                    rows_out.append(case | {"path": "moe_mmq", "us": time_us(
                        lambda: ext.moe(x, w, sid, eid, npad, qtype, kk, y, xq))})
                    print(rows_out[-1], flush=True)
            del w
    sm = "".join(map(str, torch.cuda.get_device_capability()))
    out = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "act_dtype": "bf16", "rows": rows_out}
    (HERE / "results" / f"ours_ops_{os.environ.get('S2_TAG') or 'sm' + sm}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
