"""GLM step 7 (mode (ii)'s decode look-ahead, Maya's `fast_ahead_route`): how well layer l's FFN input predicts the
experts layers l + 1 … l + 4 route to, through their own routers (sigmoid scores plus the selection bias, top-k as
the router does). Decode rows of the bench's natural-text prompts (all-resident, eager): per distance d, the recall
of the actual top-8 by the predicted top-8 and top-16, and the share of a token's routes so covered.
With `--map` (mode (ii)'s flags: exclusive tiers), against the store's tier map at each decode step's start: a token's
SSD reads (routed experts on the SSD only), those the top-8 predictions d ≤ D layers early name (Maya's look-ahead,
`STRATA_GLM_AHEAD=D`), the reads it would issue (predicted SSD-only experts) and the wasted ones (not routed by that
layer at that token; and of those, routed by it in the next 32 tokens after all), and the RAM hits and host MiB a
token the gather reads (RAM units and SSD landing rows).

    python spikes/glm7/lookahead.py <maya-s -00001 shard> '<engine flags>' <requests.json> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import shlex

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("requests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--map", action="store_true")
    a = ap.parse_args()
    from holon.api import SamplingParams
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from holon_std.moe.units import MoE

    p = argparse.ArgumentParser()
    engine_flags(p)
    cfg = engine_config(a.model, p.parse_args(shlex.split(a.flags)))
    import dataclasses
    e = Engine(dataclasses.replace(cfg, max_seqs=1, cuda_graphs=False, prefix_cache=False))
    moes = sorted((m for m in e.executor.model.modules() if isinstance(m, MoE)), key=lambda m: m.layer)
    order = {m.layer: i for i, m in enumerate(moes)}
    step: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    route = MoE._route

    def recorded(self, x):
        out = route(self, x)
        if x.shape[0] == 1:                                    # a decode row
            step[order[self.layer]] = (x.float().clone(), out[1].reshape(-1).clone())
        return out
    MoE._route = recorded
    hits = collections.Counter()
    total = collections.Counter()
    store = e.ctx.shared_objects.get("std.moe.offload") if a.map else None
    if store is not None:
        assert store.layers == [m.layer for m in moes]
        mib = [sum(c.unit_bytes for c in store.groups[g].values()) / 2**20 for g in store.layers]
    io = collections.Counter()
    pending: dict[tuple[int, int], int] = {}                  # a wasted read (layer, expert) -> its token
    later = 0
    tok = 0
    reqs = json.load(open(a.requests))["requests"]
    spans = [r["ids"] for r in reqs if r["key"].startswith("1024")][:a.prompts]

    def score(i: int, x: torch.Tensor) -> torch.Tensor:
        m = moes[i]
        s = torch.sigmoid(x @ m.router.float().T)
        return s + m.bias.float() if m.bias is not None else s

    for ids in spans:
        e.add(ids, SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True))
        while e.has_work():
            step.clear()
            res = store.residency.device.cpu() if store is not None else None
            ssd = (res == -2).all(-1) if res is not None else None
            e.step()
            if len(step) != len(moes):
                continue
            tok += 1
            if ssd is not None:
                for i in range(len(moes)):
                    routed = set(step[i][1].tolist())
                    for x in routed:                   # a wasted read routed within 32 tokens after all
                        if tok - pending.pop((i, x), -99) <= 32:
                            later += 1
                    disk = {x for x in routed if ssd[i, x]}
                    io["disk"] += len(disk)
                    ram = sum(1 for x in routed if (res[i, x] == -1).all())   # host bytes the gather reads
                    io["ram"] += ram
                    io["host_mib"] += (ram + len(disk)) * mib[i]
                    for big in (1, 2, 4):
                        named = set()
                        for d in range(1, big + 1):
                            if i - d >= 0:
                                named |= set(score(i, step[i - d][0])[0].topk(8).indices.tolist())
                        issue = {x for x in named if ssd[i, x]}
                        io[f"covered_D{big}"] += len(disk & issue)
                        io[f"issued_D{big}"] += len(issue)
                        io[f"wasted_D{big}"] += len(issue - routed)
                        if big == 4:
                            pending |= {(i, x): tok for x in issue - routed}
            for i in range(len(moes)):
                x = step[i][0]
                for d in range(1, 5):
                    if i + d >= len(moes):
                        break
                    actual = set(step[i + d][1].tolist())
                    pred = score(i + d, x)[0]
                    top = pred.topk(16).indices.tolist()
                    hits[(d, 8)] += len(actual & set(top[:8]))
                    hits[(d, 16)] += len(actual & set(top))
                    total[d] += len(actual)
    MoE._route = route
    out = {f"d{d}_top{k}": hits[(d, k)] / max(total[d], 1) for d in range(1, 5) for k in (8, 16)}
    if store is not None:
        out["map"] = {k: v / tok for k, v in sorted(io.items())} | {"tokens": tok,
                                                                    "wasted_D4_routed_within_32": later / tok}
    out["routes"] = dict(total)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    e.close()


if __name__ == "__main__":
    main()
