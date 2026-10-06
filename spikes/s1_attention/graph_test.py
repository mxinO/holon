"""CUDA-graph replay with changing lengths (questions 2 and 3).

plan "host":  FlashInfer CUDA-graph wrappers; plan() on the host with exact lengths between replays
              (non-spec decode at depth 1).
plan "upper": FlashInfer CUDA-graph wrappers (split-KV on); plan() on the host with an UPPER BOUND on each
              length (what the host mirror knows for spec Seqs at depth 1), and the exact CSR metadata written
              on the device inside the graph from the SeqTable lengths.
plan "none":  nothing on the host between replays; only the device page table / lengths change. FlashInfer
              FA2 with disable_split_kv and CSR written in-graph, the Triton prototype and (SM90+) XQA.
hazard:      the FlashInfer CUDA-graph wrapper with disable_split_kv=True launches padded CTAs that read request
             indices the plan never wrote; we poison the pinned plan buffer to show they are consumed.

  CUDA_VISIBLE_DEVICES=0 python graph_test.py
"""

import subprocess
import sys
import time

import torch

from common import GEOMS, PagedKV, csr_from_table, err, make_paged_kv, reference, save
from fi_kernels import XQA, FIDecode, FIPagedPrefill, sm_scale
from triton_attn import TritonOnPagedKV, pick_splits

HAS_XQA = torch.cuda.get_device_capability(0)[0] in (9, 10, 12)
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
B, PAGE, MAX_LEN = 8, 16, 8192
TOL = 2e-2


def length_schedule(nq: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(0)
    first = torch.randint(nq, 600, (B,), generator=g).tolist()
    steps = [first]
    for _ in range(3):                                   # decode growth, +1 .. +nq per step
        steps.append([n + int(torch.randint(1, nq + 1, (1,), generator=g)) for n in steps[-1]])
    steps.append(torch.randint(nq, MAX_LEN + 1, (B,), generator=g).tolist())   # jump far beyond step 0
    steps.append([nq] * B)                               # shrink to the minimum
    steps.append([MAX_LEN] * B)                          # the bound
    return steps


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def replay_test(g, name: str, nq: int, build, plan: str) -> dict:
    """build(kv) -> (object with run(q, kv) [and plan(kv)], per-step device prep or None)."""
    kv = make_paged_kv(g, [MAX_LEN] * B, PAGE, seed=3)[0]       # page table sized for the bound
    sched = length_schedule(nq)
    kv.lens.copy_(torch.tensor(sched[0], dtype=torch.int32))
    q = torch.randn(B * nq, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
    k, prep = build(kv)

    def step():
        if prep is not None:
            prep()
        return k.run(q, kv)

    graph = capture(step)
    out = k.out
    results = []
    for lens in sched:
        kv.lens.copy_(torch.tensor(lens, dtype=torch.int32))      # the SeqTable update (device)
        q.normal_()
        if plan == "host":
            k.plan(kv)
        elif plan == "upper":   # host mirror: actual + up to nq-1 unaccepted draft tokens, capped by the bound
            slack = torch.randint(0, nq, (B,), dtype=torch.int32, device="cuda")
            k.plan(PagedKV(kv.k, kv.v, kv.page_table, torch.clamp(kv.lens + slack, max=MAX_LEN), PAGE))
        graph.replay()
        torch.cuda.synchronize()
        e = err(out, reference(q, kv, nq, sm_scale(g)))
        results.append({"max_len": max(lens), "min_len": min(lens), **e})
    ok = all(r["max_abs"] < TOL and not r["nan"] for r in results)
    row = {"geom": g.name, "kernel": name, "nq": nq, "plan": plan, "pass": ok,
           "worst_max_abs": max(r["max_abs"] for r in results), "steps": results}
    print(row["kernel"], plan, g.name, nq, "PASS" if ok else "FAIL", row["worst_max_abs"], flush=True)
    return row


def plan_cost_us(g, batch: int, ctx: int, kind: str) -> float:
    """Host wall time of one FlashInfer plan() from host (pinned) CSR arrays (what plan_bounds pays per step)."""
    kv = make_paged_kv(g, [ctx] * batch, PAGE)[0]
    k = FIDecode(g, kv, kind == "tc") if kind != "verify" else FIPagedPrefill(g, kv, 3)
    csr = tuple(t.cpu().pin_memory() for t in csr_from_table(kv))
    k.plan(csr=csr)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        k.plan(csr=csr)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 50 * 1e6


HAZARD_CHILD = r"""
import sys, torch
sys.path.insert(0, %r)
from common import GEOMS, make_paged_kv, reference, err
from fi_kernels import FIDecode, sm_scale
poison = int(sys.argv[1])
g = GEOMS["qwen3_8b"]
kv = make_paged_kv(g, [1000], 16)[0]
k = FIDecode(g, kv, True, "graph_nosplit")
k.w._pin_memory_int_workspace_buffer.fill_(poison)
k.plan()
pi = k.w._plan_info
q = torch.randn(1, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
o = k.run(q, kv)
torch.cuda.synchronize()
print("padded_batch_size", pi[0], "max_abs", err(o, reference(q, kv, 1, sm_scale(g)))["max_abs"])
"""


def hazard() -> list[dict]:
    rows = []
    for poison in [0, 0x7F]:
        p = subprocess.run([sys.executable, "-c", HAZARD_CHILD % str(sys.path[0]), str(poison)],
                           capture_output=True, text=True, timeout=600)
        tail = (p.stdout + p.stderr).strip().splitlines()
        rows.append({"poison_byte": poison, "returncode": p.returncode, "output": tail[-1][:300] if tail else ""})
        print("hazard", rows[-1], flush=True)
    return rows


def plan_costs(gnames) -> list[dict]:
    rows = []
    for gname in gnames:
        g = GEOMS[gname]
        for batch in [1, 8, 64]:
            for kind in ["tc", "verify"] + (["cudacore"] if g.group != 6 else []):
                c = plan_cost_us(g, batch, 8192, kind)
                rows.append({"geom": gname, "batch": batch, "ctx": 8192, "kind": kind, "us": c})
                print("plan", gname, batch, kind, f"{c:.1f}us", flush=True)
    return rows


def main():
    if sys.argv[1:] == ["plan_cost"]:     # re-measure only the host plan() cost
        import json
        from common import RESULTS, class_name
        path = RESULTS / f"graph_test_{class_name()}.json"
        res = json.loads(path.read_text())
        res["plan_cost_us"] = plan_costs(["qwen3_8b", "qwen3_8_27b"])
        path.write_text(json.dumps(res, indent=1))
        return
    res = {"replay": [], "plan_cost_us": [], "hazard": []}
    for gname in ["qwen3_8b", "qwen3_8_27b"]:
        g = GEOMS[gname]

        def fi_dec(use_tc, meta):
            def build(kv):
                k = FIDecode(g, kv, use_tc, meta)
                k.plan()
                return k, (k.prepare if meta in ("device", "upper") else None)
            return build

        def fi_verify(nq, meta):
            def build(kv):
                k = FIPagedPrefill(g, kv, nq, meta)
                k.plan()
                return k, (k.prepare if meta in ("device", "upper") else None)
            return build

        def tri(nq):
            def build(kv):
                return TritonOnPagedKV(g.hq, g.hkv, g.d, PAGE, B, nq, pick_splits(B, g.hkv, MAX_LEN, NUM_SMS)), None
            return build

        def xqa(nq):
            def build(kv):
                return XQA(g, kv, nq), None
            return build

        cases = []
        if g.group != 6:
            cases += [("fi_decode_cudacore", 1, fi_dec(False, "host"), "host"),
                      ("fi_decode_cudacore", 1, fi_dec(False, "upper"), "upper")]
        cases += [("fi_decode_tc", 1, fi_dec(True, "host"), "host"),
                  ("fi_paged_prefill", 3, fi_verify(3, "host"), "host"),
                  ("fi_decode_tc", 1, fi_dec(True, "upper"), "upper"),
                  ("fi_paged_prefill", 3, fi_verify(3, "upper"), "upper"),
                  ("fi_paged_prefill", 5, fi_verify(5, "upper"), "upper"),
                  ("fi_decode_tc_devmeta", 1, fi_dec(True, "device"), "none"),
                  ("fi_paged_prefill_devmeta", 3, fi_verify(3, "device"), "none"),
                  ("triton", 1, tri(1), "none"), ("triton", 3, tri(3), "none"), ("triton", 5, tri(5), "none")]
        if HAS_XQA:
            cases += [("xqa", 1, xqa(1), "none"), ("xqa", 3, xqa(3), "none")]
        for name, nq, build, plan in cases:
            try:
                res["replay"].append(replay_test(g, name, nq, build, plan))
            except Exception as e:
                res["replay"].append({"geom": gname, "kernel": name, "nq": nq, "error": str(e)[:300]})
                print(name, gname, nq, "ERROR", str(e)[:300], flush=True)
    res["plan_cost_us"] = plan_costs(["qwen3_8b", "qwen3_8_27b"])
    res["hazard"] = hazard()
    print("saved", save("graph_test", res))


if __name__ == "__main__":
    main()
