"""Per-call speed of IQ1_M's mmq tile loader against IQ2_S's (docs/v1-moe-prefill.md step 3c, D9): mmq with ids on
random-byte stacks (realistic scales) at FN's MoE shape (E = 512, top-10, H = 2,560, I = 640), the merged gate/up
stack [E, 2·I, H] with bf16 activations [T, 1, H], as `LinearExperts` calls it (slots = E + k). Down (in = I = 640)
is not a multiple of 256, so neither type can hold it. Skewed routing as spikes/moe_prefill/gemm.py's; the two types
interleaved per round, CUDA-event timed, median of `--iters` per round, median over rounds.

At top-10 a call takes at most 3,276 tokens (`ID_PAIRS` 32,768 pairs), so 3,072 (FN's `pass_rows`) is the largest.
Run with `plugins` on PYTHONPATH (it takes plugins/tests' `random_blocks`):

    python spikes/moe_prefill/loader_speed.py [--tokens 512 1024 2048 3072] [--rounds 5] [--json out]
"""
from __future__ import annotations

import argparse
import functools
import json
import statistics

import torch


def timed(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def routing(tokens: int, experts: int, k: int, s: float, gen: torch.Generator) -> torch.Tensor:
    logp = -s * torch.log(torch.arange(1, experts + 1, dtype=torch.float64))
    logp = logp[torch.randperm(experts, generator=gen)]
    gumbel = -torch.log(-torch.log(torch.rand(tokens, experts, generator=gen, dtype=torch.float64)))
    return (logp + gumbel).topk(k, dim=1).indices.to(torch.int32)


def main() -> None:
    from holon_plugins.kernels.ggml_cuda import extension
    from holon_plugins.kernels.ggml_cuda import plugin as ggml
    from tests.test_ggml_cuda import random_blocks

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 3072])
    ap.add_argument("--types", nargs="+", default=["iq1_m", "iq2_s"])
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability()
    lib = extension.load(10 * major + minor)
    jt = ggml.IDS_J_TOKENS.get(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1], ggml.IDS_J_TOKENS_DEFAULT)   # D2's batched-decode rule
    e, k, n_in, n_out = a.experts, a.top_k, a.hidden, 2 * a.inter
    slots, prec = e + k, ggml.PREC["q8_1"]
    errors = torch.zeros(1, dtype=torch.int32, device=dev)
    stacks = {t: torch.from_numpy(random_blocks(t, e * n_out, n_in, seed=1)).to(dev) for t in a.types}
    gen = torch.Generator().manual_seed(0)
    work = {}
    for t in a.tokens:
        assert t * k <= ggml.ID_PAIRS, f"{t} tokens x top-{k} is past the id helper's {ggml.ID_PAIRS} pairs"
        ids = routing(t, e, k, 0.5, gen).to(dev)
        x = (1e-3 * torch.randn(t, 1, n_in, generator=gen)).bfloat16().to(dev)
        y = torch.empty(t, k, n_out, device=dev)
        ws = torch.empty(lib.workspace_bytes_id(0, n_in, t, k, slots), dtype=torch.uint8, device=dev)
        work[t] = (x, ids, y, ws)
    ms = {(name, t): [] for name in a.types for t in a.tokens}
    for _ in range(a.rounds):
        for t in a.tokens:
            for name in a.types:
                x, ids, y, ws = work[t]
                fn = functools.partial(lib.mmq_id, ggml.TYPES[name], prec, stacks[name], x, ids, y, ws, slots, errors, jt)
                ms[name, t].append(timed(fn, a.iters))
    assert int(errors.item()) == 0, int(errors.item())
    rows = []
    for t in a.tokens:
        row = {"tokens": t, **{f"{n}_ms": round(statistics.median(ms[n, t]), 4) for n in a.types},
               **{f"{n}_rounds": [round(v, 4) for v in ms[n, t]] for n in a.types}}
        if len(a.types) == 2:
            row["ratio"] = round(statistics.median(ms[a.types[0], t]) / statistics.median(ms[a.types[1], t]), 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(), "experts": e, "top_k": k, "in": n_in, "out": n_out,
                       "stack": "gate_up (merged)", "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
