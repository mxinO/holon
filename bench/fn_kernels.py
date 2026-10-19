"""Per-call latency of FN's decode-step kernels (docs/v1-flash-next.md acceptance 3: `qsa_select` <= 20 us per layer
at 32K context, 8K blocks, <= 8 query rows, on the RTX 5080), each replayed from a CUDA graph of `--calls` calls as
the decode step runs them. Query rows at the end of a --ctx context: one sequence (a decode row and its MTP drafts)
or one per sequence (a batch), page 16, Qwen3.8-Flash-Next's shapes."""
import argparse
import json
import pathlib

import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_std.mixers.qsa import kernels as qsa

PAGE, HI, DK, HQ, HKV, D = 16, 4, 128, 24, 2, 256

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=32 * 1024)
ap.add_argument("--rows", default="1,2,4,8")
ap.add_argument("--calls", type=int, default=100)
ap.add_argument("--out", default=None)
args = ap.parse_args()

ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
select = ctx.select_kernel(qsa.SELECT, site="bench", heads=HI, head_dim=DK, ratio=qsa.RATIO, budget=qsa.BUDGET,
                           page=PAGE)
attend = ctx.select_kernel(qsa.ATTEND, site="bench", heads=HQ, kv_heads=HKV, head_dim=D, ratio=qsa.RATIO,
                           width=qsa.width(qsa.BUDGET, qsa.RATIO), page=PAGE, layout="kv.hnd", q_dtype="bfloat16",
                           kv_dtype="bfloat16")
facts = dict(ratio=qsa.RATIO, page=PAGE)
select = select.impl(ctx, heads=HI, head_dim=DK, budget=qsa.BUDGET, **facts)
attend = attend.impl(ctx, heads=HQ, kv_heads=HKV, head_dim=D, width=qsa.width(qsa.BUDGET, qsa.RATIO),
                     layout="kv.hnd", q_dtype="bfloat16", kv_dtype="bfloat16", **facts)
pages = -(-(args.ctx + 8) // PAGE)


def timed(fn) -> float:
    """us per call, from a graph of --calls calls (after a warm-up), median of 5 replays."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.calls):
            fn()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / args.calls)
    return sorted(times)[2]


def case(rows: int, seqs: int) -> dict:
    """`rows` queries at the end of the context: of one sequence (consecutive positions) or one each."""
    table = torch.randperm(seqs * pages, device="cuda", dtype=torch.int32).view(seqs, pages)
    if seqs == 1:
        seq, positions = torch.zeros(rows, dtype=torch.int32, device="cuda"), args.ctx - rows + torch.arange(rows)
    else:
        seq, positions = torch.arange(rows, dtype=torch.int32, device="cuda"), torch.full((rows,), args.ctx - 1)
    positions = positions.cuda()
    pooled = torch.randn(seqs * pages, PAGE // qsa.RATIO, DK, device="cuda")
    q = torch.randn(rows, HI, DK, device="cuda").bfloat16()
    blocks = select(q, pooled, table, seq, positions)
    kv = torch.randn(seqs * pages, 2, HKV, PAGE, D, device="cuda").bfloat16()
    qa = torch.randn(rows, HQ, D, device="cuda").bfloat16()
    return {"rows": rows, "sequences": seqs, "ctx": args.ctx, "blocks": (args.ctx + 1) // qsa.RATIO,
            "qsa_select_us": round(timed(lambda: select(q, pooled, table, seq, positions)), 2),
            "qsa_attend_us": round(timed(lambda: attend(qa, kv, blocks, table, seq, positions)), 2)}


results = []
for rows in map(int, args.rows.split(",")):
    for seqs in sorted({1, rows}):
        results.append(case(rows, seqs))
        print(results[-1], flush=True)
if args.out:
    pathlib.Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "calls": args.calls,
                                                  "results": results}, indent=1) + "\n")
