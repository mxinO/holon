"""Run-to-run byte identity (question 5).

For each path: fixed inputs, 20 graph replays and 5 eager runs (re-planning each time where a host plan exists)
must be byte-identical to the first output. Also recorded (informational, bounded comparisons in the design):
whether a sequence's output is identical alone vs inside a batch (batch invariance) and eager vs graph.

  CUDA_VISIBLE_DEVICES=0 python determinism.py
"""

import torch

from common import GEOMS, PagedKV, make_paged_kv, save
from fi_kernels import XQA, FIDecode, FIPagedPrefill, FIRaggedPrefill
from triton_attn import TritonOnPagedKV, pick_splits

HAS_XQA = torch.cuda.get_device_capability(0)[0] in (9, 10, 12)
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
PAGE = 16


def builders(g, nq: int):
    def fi(cls_args, meta):
        def build(kv):
            k = FIDecode(g, kv, *cls_args, meta=meta) if nq == 1 else FIPagedPrefill(g, kv, nq, meta)
            k.plan()
            return k
        return build

    def tri(kv):
        B = kv.lens.numel()
        return TritonOnPagedKV(g.hq, g.hkv, g.d, PAGE, B, nq, pick_splits(B, g.hkv, int(kv.lens.max()), NUM_SMS))

    out = {}
    if nq == 1:
        if g.group != 6:
            out["fi_decode_cudacore"] = fi((False,), "host")
        out["fi_decode_tc"] = fi((True,), "host")
        out["fi_decode_tc_devmeta"] = fi((True,), "device")
    else:
        out["fi_paged_prefill"] = fi((), "host")
        out["fi_paged_prefill_devmeta"] = fi((), "device")
    out["triton"] = tri
    if HAS_XQA:
        out["xqa"] = lambda kv: XQA(g, kv, nq)
    return out


def check(g, name, build, lens, nq) -> dict:
    kv = make_paged_kv(g, lens, PAGE, seed=7)[0]
    q = torch.randn(len(lens) * nq, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
    k = build(kv)
    if getattr(k, "meta", None) == "device":
        k.prepare()
    first = k.run(q, kv).clone()
    eager_same = True
    for _ in range(5):
        if hasattr(k, "plan"):
            k.plan()
        eager_same &= torch.equal(k.run(q, kv), first)
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        k.run(q, kv)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        k.run(q, kv)
    graph.replay()
    torch.cuda.synchronize()
    g_first = k.out.clone()
    graph_same = True
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        graph_same &= torch.equal(k.out, g_first)
    # Batch invariance: sequence 0 alone, same physical pages.
    kv1 = PagedKV(kv.k, kv.v, kv.page_table[:1], kv.lens[:1], PAGE)
    k1 = build(kv1)
    if getattr(k1, "meta", None) == "device":
        k1.prepare()
    alone = k1.run(q[:nq], kv1)
    row = {"geom": g.name, "kernel": name, "nq": nq, "lens": lens, "eager_run_to_run": bool(eager_same),
           "graph_run_to_run": bool(graph_same), "eager_vs_graph_identical": bool(torch.equal(first, g_first))}
    if len(lens) > 1:
        row["batch_invariant_seq0"] = bool(torch.equal(alone, first[:nq]))
    print(row, flush=True)
    return row


def main():
    rows = []
    for gname in ["qwen3_8b", "qwen3_8_27b"]:
        g = GEOMS[gname]
        for lens in ([32768], [8192, 100, 3000, 17, 32768, 5000, 64, 1024]):
            for nq in (1, 3):
                for name, build in builders(g, nq).items():
                    rows.append(check(g, name, build, lens, nq))
        # Ragged prefill (eager).
        lens = [4096, 300, 77]
        q = torch.randn(sum(lens), g.hq, g.d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(sum(lens), g.hkv, g.d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        r = FIRaggedPrefill(g, lens)
        first = r.run(q, k, v).clone()
        same = all(torch.equal(r.run(q, k, v), first) for _ in range(10))
        rows.append({"geom": gname, "kernel": "fi_ragged_prefill", "lens": lens, "eager_run_to_run": same})
        print(rows[-1], flush=True)
    print("saved", save("determinism", {"rows": rows}))


if __name__ == "__main__":
    main()
