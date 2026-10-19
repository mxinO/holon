"""Per-call latency of FN's QSA decode kernels (docs/v1-flash-next.md acceptance 3: `qsa_select` <= 20 us per layer at
4 rows of one sequence, 32K context, on the RTX 5080; 8 rows reported), replayed from a CUDA graph as the decode step
runs them, on real indexer inputs, with the caches out of L2.

- **Inputs:** a QSA layer's raw index keys and last 8 queries from N4's 32K oracle run (`spikes/fn0/n4_oracle.py
  --capture`, the `recall-32k` prompt: 31,878 tokens, 7,969 blocks); the pooled keys built from them by
  `qsa_index@1`. Query rows: the last positions of one sequence (a decode row and its drafts), or one row each of
  separate sequences (separate copies of the keys). `qsa_attend` reads random bf16 KV and applies the output gate.
- **Cold caches:** --buffers copies of every per-layer buffer (pooled keys, KV, page tables; > 64 MB in all), the graph
  cycling through them as the layers of a step would, so no call finds its keys in L2.
"""
import argparse
import json
import pathlib

import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_std.mixers.qsa import kernels as qsa
from holon_std.mixers.qsa import reference

PAGE, RATIO, BUDGET, HQ, HKV, D, ROT, THETA, EPS = 16, 4, 2048, 24, 2, 256, 64, 1e7, 1e-6

ap = argparse.ArgumentParser()
ap.add_argument("--capture", required=True, help="spikes/fn0/n4_oracle.py --capture's file")
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--tokens", type=int, default=None, help="the prompt's first this many tokens (default: all)")
ap.add_argument("--rows", default="1,2,4,8")
ap.add_argument("--buffers", type=int, default=24)
ap.add_argument("--rounds", type=int, default=4, help="passes over the buffers per graph")
ap.add_argument("--out", default=None)
args = ap.parse_args()

layer = torch.load(args.capture)["layers"][args.layer]
raw_keys, queries = layer["raw"][:args.tokens].bfloat16().cuda(), layer["q"].bfloat16().cuda()   # [L, Dk], [8, Hi, Dk]
ctx_len, hi, dk = len(raw_keys), queries.shape[1], queries.shape[2]
pages = -(-ctx_len // PAGE)
build = BuildContext(PluginSet(discover()), torch.device("cuda"))


def kernel(op: str, **facts):
    return build.select_kernel(op, site="bench", **facts).impl(build, **facts)


index = kernel(qsa.INDEX, head_dim=dk, ratio=RATIO, page=PAGE, rotary_dim=ROT, max_positions=ctx_len, eps=EPS,
               dtype="bfloat16")
select = kernel(qsa.SELECT, heads=hi, head_dim=dk, ratio=RATIO, budget=BUDGET, page=PAGE)
attend = kernel(qsa.ATTEND, heads=HQ, kv_heads=HKV, head_dim=D, ratio=RATIO, width=qsa.width(BUDGET, RATIO),
                page=PAGE, layout="kv.hnd", q_dtype="bfloat16", kv_dtype="bfloat16")

# the pooled keys of the prompt, built once by the kernel and copied
table0 = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
raw = torch.empty(pages, PAGE, dk, dtype=torch.bfloat16, device="cuda")
pooled0 = torch.zeros(pages, PAGE // RATIO, dk, device="cuda")
cos_sin = reference.cos_sin(ctx_len, ROT, THETA).cuda()
index(raw_keys, raw, pooled0, table0, torch.zeros(ctx_len, dtype=torch.int32, device="cuda"),
      torch.arange(ctx_len, device="cuda"), layer["weight"].cuda(), cos_sin)


def timed(calls) -> float:
    """us per call of a graph that runs `calls` (a list of thunks) --rounds times, median of 5 replays."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.rounds):
            for c in calls:
                c()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / (args.rounds * len(calls)))
    return sorted(times)[2]


def case(rows: int, seqs: int) -> dict:
    """`rows` queries at the end of the prompt: of one sequence (consecutive positions) or one per sequence."""
    if seqs == 1:
        seq, positions, q = (torch.zeros(rows, dtype=torch.int32, device="cuda"),
                             torch.arange(ctx_len - rows, ctx_len, device="cuda"), queries[-rows:])
    else:
        seq, positions, q = (torch.arange(rows, dtype=torch.int32, device="cuda"),
                             torch.full((rows,), ctx_len - 1, device="cuda"), queries[-rows:])
    sets = []
    for _ in range(args.buffers):          # per layer: its own pooled keys, KV and tables
        table = torch.arange(seqs * pages, dtype=torch.int32, device="cuda").view(seqs, pages)
        sets.append((table, pooled0.repeat(seqs, 1, 1),
                     torch.randn(seqs * pages, 2, HKV, PAGE, D, device="cuda").bfloat16()))
    qg = torch.randn(rows, HQ, 2 * D, device="cuda").bfloat16()       # gqa's gated q layout: q, then the output gate
    qa, gate = qg[..., :D].contiguous(), qg[..., D:]
    blocks = [select(q, pooled, table, seq, positions) for table, pooled, _ in sets]
    nbytes = sum(p.nbytes + kv.nbytes for _, p, kv in sets)
    sel = timed([lambda t=t, p=p: select(q, p, t, seq, positions) for t, p, _ in sets])
    att = timed([lambda t=t, kv=kv, b=b: attend(qa, kv, b, t, seq, positions, gate)
                 for (t, _, kv), b in zip(sets, blocks, strict=True)])
    return {"rows": rows, "sequences": seqs, "tokens": ctx_len, "blocks": ctx_len // RATIO,
            "buffers_mb": round(nbytes / 2**20), "qsa_select_us": round(sel, 2), "qsa_attend_us": round(att, 2)}


results = []
for rows in map(int, args.rows.split(",")):
    for seqs in sorted({1, rows}):
        results.append(case(rows, seqs))
        print(results[-1], flush=True)
if args.out:
    pathlib.Path(args.out).write_text(json.dumps(
        {"gpu": torch.cuda.get_device_name(), "inputs": f"N4 recall-32k, QSA layer {args.layer}, {ctx_len} tokens",
         "buffers": args.buffers, "rounds": args.rounds, "results": results}, indent=1) + "\n")
