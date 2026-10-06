"""CUDA-graph capture of the decode kernels + run-to-run byte identity -> results/graph_sm*.json.

For each type and path, one graph is captured over static buffers (x, q8_1 workspace, y, routing). We check:
eager == graph replay (bytewise), 20 replays bytewise identical, and a different input replayed through the same
graph equals its eager result (the graph really reads the static input buffer).
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

TYPES = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ2_XS"]
MMQ = {"Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
M, K, EXPERTS, TOP_K, REPLAYS = 2048, 4096, 16, 4, 20


def check(run, x: torch.Tensor, y: torch.Tensor) -> dict:
    x0, x1 = torch.randn_like(x), torch.randn_like(x)
    x.copy_(x0)
    run()
    torch.cuda.synchronize()
    eager0 = y.clone()
    x.copy_(x1)
    run()
    eager1 = y.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    x.copy_(x0)
    outs = []
    for _ in range(REPLAYS):
        y.zero_()
        g.replay()
        outs.append(y.clone())
    x.copy_(x1)
    g.replay()
    torch.cuda.synchronize()
    return {
        "graph_eq_eager": bool(torch.equal(outs[0].view(torch.int16), eager0.view(torch.int16))),
        "replays_identical": all(torch.equal(o.view(torch.int16), outs[0].view(torch.int16)) for o in outs),
        "new_input_eq_eager": bool(torch.equal(y.view(torch.int16), eager1.view(torch.int16))),
    }


def main() -> None:
    ext = ggml_ops.load_ext()
    dev = "cuda"
    torch.manual_seed(0)
    res = {}
    for name in TYPES:
        qtype = int(gguf.GGMLQuantizationType[name])
        _, t = find_tensor(name)
        raw = torch.from_numpy(t.data.copy().reshape(-1)).to(dev)
        w = ggml_ops.tile_weight(raw, qtype, (M, K))
        we = ggml_ops.tile_weight(raw, qtype, (EXPERTS, M // EXPERTS * 4, K))
        r = {}
        for n in (1, 8, 32):
            x = torch.empty(n, K, dtype=torch.bfloat16, device=dev)
            y = torch.empty(n, M, dtype=torch.bfloat16, device=dev)
            xq = torch.empty(ggml_ops.q8_1_words(n, K), dtype=torch.int32, device=dev)
            r[f"mmvq_n{n}"] = check(lambda: ext.mul_mat_vec(w, x, qtype, y, xq), x, y)
            if name in MMQ and n == 32:
                r["mmq_n32"] = check(lambda: ext.mul_mat(w, x, qtype, y, xq), x, y)
            wdq = torch.empty(M, K, dtype=torch.bfloat16, device=dev)

            def fallback():
                ext.dequantize(w, qtype, wdq)
                torch.matmul(x, wdq.T, out=y)

            r[f"dequant_gemm_n{n}"] = check(fallback, x, y)
            em = we.shape[1]
            topk = torch.stack([torch.randperm(EXPERTS, device=dev)[:TOP_K] for _ in range(n)]).int()
            ye = torch.empty(n * TOP_K, em, dtype=torch.bfloat16, device=dev)
            r[f"moe_vec_n{n}"] = check(lambda: ext.moe_vec(x, we, topk, qtype, ye, xq), x, ye)
            if name in MMQ:
                sid, eid, npad = ggml_ops.moe_align(topk, ext.moe_block_size(qtype), EXPERTS)
                r[f"moe_mmq_n{n}"] = check(lambda: ext.moe(x, we, sid, eid, npad, qtype, TOP_K, ye, xq), x, ye)
        res[name] = r
        ok = all(all(v.values()) for v in r.values())
        print(name, "OK" if ok else json.dumps(r))
    sm = "".join(map(str, torch.cuda.get_device_capability()))
    out = {"device": torch.cuda.get_device_name(), "all_ok": all(all(all(v.values()) for v in r.values())
                                                                  for r in res.values()), "types": res}
    (HERE / "results" / f"graph_{os.environ.get('S2_TAG') or 'sm' + sm}.json").write_text(json.dumps(out, indent=1))
    print("all_ok", out["all_ok"])


if __name__ == "__main__":
    main()
