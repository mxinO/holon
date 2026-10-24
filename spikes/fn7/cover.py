"""FN's cover for the plant gap (§22.5, §22.6): the PLE block and the QSA selection of the real model, checked against
their float32 oracles (`qwen4_exp.reference.ple_block`, `std.mixers.qsa.reference`) on the engine's own inputs.

The engine serves one long prompt in prompt steps of the deployment's size. On its PLE layer it records, per step,
the block's inputs (the PLE key and value, the residual streams R) and what it added to R. On its first QSA layer it
records every step's raw index keys, and the sparse steps' indexer queries and the blocks each row selected.

The oracles then take the whole prompt in one piece, from a fresh state:
- **PLE** (float32, on the GPU): the relative error of each row's addition to R, at the conv window's rows after
  each step start (where the engine's carried state is read) and elsewhere. A state that is not carried, or is
  rounded, shows at the step starts; a wrong gate shows everywhere.
- **QSA** (float32 keys, float64 scores, D2's tie rule): the pooled keys built from every raw key, then the selection
  of sampled sparse rows, against the engine's. A pooled key rotated at the wrong position changes the selections.

Placement and tiers do not enter: the check reads one engine's numbers against the math. Seconds past the prompt.
With `--plant` the run is under `holon_judge.planted`'s plant, and must fail.
Usage: cover.py <model> '<engine flags>' [--ctx 8192] [--rows 256] [--plant NAME] [--out <json>]"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLE_TOL, QSA_TOL = 1e-5, 0.98       # a row's PLE error (clean 8e-7 on the A5000); rows selecting the same blocks


class _PLE:
    """A PLE unit's block, recorded per prompt step (on the host)."""

    def __init__(self, unit) -> None:
        self.unit, self.block, self.steps = unit, unit._block, []
        unit._block = self

    def __call__(self, key, value, r, *rest):
        before = r.clone()
        out = self.block(key, value, r, *rest)
        if len(r) > 1:
            self.steps.append((key.cpu(), value.cpu(), before.cpu(), (r - before).cpu()))
        return out


class _QSA:
    """A QSA unit's raw index keys (every prompt step) and its sparse steps' queries and selections (on the host)."""

    def __init__(self, unit) -> None:
        self.unit, self.indexer, self.select = unit, unit._indexer, unit._select
        self.keys, self.sparse, self.last = [], [], None
        unit._indexer, unit._select = self._index, self._choose

    def _index(self, x, positions):
        q, k = self.indexer(x, positions)
        if len(positions) > 1:
            self.keys.append((k.contiguous().cpu(), positions.cpu()))
            self.last = (q.cpu(), positions.cpu())
        return q, k

    def _choose(self, q, pooled, table, seq, positions):
        blocks = self.select(q, pooled, table, seq, positions)
        if len(positions) > 1:
            self.sparse.append((*self.last, blocks.cpu()))
        return blocks


def ple_errors(rec: _PLE, step: int) -> dict:
    from holon_plugins.models.qwen4_exp.reference import ple_block

    u = rec.unit
    key, value, before, added = (torch.cat(t) for t in zip(*rec.steps, strict=True))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = before.to(dev).float()
    window, taps = u.window, u.conv.shape[1]
    state = torch.zeros(1, window, r.shape[1], device=dev)
    zero = torch.zeros(1, dtype=torch.bool, device=dev)
    ple_block(key.to(dev), value.to(dev), r, u.norm_key.to(dev), u.norm_query.to(dev), u.norm_conv.to(dev),
              u.conv.to(dev), state, torch.zeros(1, dtype=torch.long, device=dev),
              torch.tensor([0, len(r)], device=dev), zero, u.n, window // (taps - 1), u.eps)
    want = (r - before.to(dev).float()).cpu()
    err = (added.float() - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-12)
    starts = torch.zeros(len(err), dtype=torch.bool)
    for s in range(step, len(err), step):
        starts[s:s + window] = True
    return {"rows": len(err), "max": float(err.max()), "median": float(err.median()),
            "max_at_step_starts": float(err[starts].max()) if starts.any() else None,
            "max_elsewhere": float(err[~starts].max())}


def qsa_agreement(rec: _QSA, eps: float, rows: int) -> dict:
    from holon_std.mixers.qsa import reference

    u = rec.unit
    k, at = (torch.cat(t) for t in zip(*rec.keys, strict=True))
    q, positions, blocks = (torch.cat(t) for t in zip(*rec.sparse, strict=True))
    page, ratio = u.page, u.ratio
    pages = -(-int(at.max() + 1) // page)
    raw = torch.zeros(pages, page, u.dk, dtype=k.dtype)
    pooled = torch.zeros(pages, page // ratio, u.dk)
    table, seq = torch.arange(pages, dtype=torch.int32)[None], torch.zeros(len(k), dtype=torch.int32)
    reference.qsa_index(k, raw, pooled, table, seq, at, u.index_k_norm.cpu(), u.cos_sin.cpu(), ratio, eps)
    pick = torch.linspace(0, len(q) - 1, min(rows, len(q))).long()
    want = reference.qsa_select(q[pick], pooled, table, seq[pick], positions[pick], ratio, u.budget)
    same = [set(w[w >= 0].tolist()) == set(g[g >= 0].tolist()) for w, g in zip(want, blocks[pick], strict=True)]
    overlap = [len(set(w[w >= 0].tolist()) & set(g[g >= 0].tolist())) / max(len(set(w[w >= 0].tolist())), 1)
               for w, g in zip(want, blocks[pick], strict=True)]
    return {"rows": len(pick), "same": sum(same) / len(same), "overlap": sum(overlap) / len(overlap)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--plant")
    ap.add_argument("--prompts", default=str(ROOT / "judge/prompts/v1-long.jsonl"))
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import engine
    from holon.api import SamplingParams
    from holon_judge import planted
    from holon_judge.recording import PromptSet
    from holon_plugins.models.qwen4_exp.units import PLE
    from holon_std.mixers.qsa import units as qsa_units

    init, eps = qsa_units.QSA.__init__, {}

    def qsa_init(self, ctx, **params):                       # the indexer's eps, which the unit does not keep
        init(self, ctx, **params)
        eps[id(self)] = params.get("eps", 1e-6)
    ids = max(PromptSet.load(a.prompts).prompts, key=lambda p: len(p.ids)).ids[:a.ctx]
    with (planted.PLANTS[a.plant]() if a.plant else contextlib.nullcontext()), \
            mock_init(qsa_units.QSA, qsa_init):
        e = engine(a.model, a.flags, max_seqs=1, max_model_len=max(a.ctx + 64, 4096), prefix_cache=False)
        step = e.cfg.max_step_tokens
        modules = list(e.executor.model.modules())
        ples = [_PLE(m) for m in modules if isinstance(m, PLE)]
        qsa = next(_QSA(m) for m in modules if isinstance(m, qsa_units.QSA))
        e.generate([ids], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        e.close()
    out = {"ctx": len(ids), "step": step, "plant": a.plant,
           "ple": [ple_errors(p, step) for p in ples], "qsa": qsa_agreement(qsa, eps[id(qsa.unit)], a.rows)}
    out["pass"] = all(p["max"] <= PLE_TOL for p in out["ple"]) and out["qsa"]["same"] >= QSA_TOL
    text = json.dumps(out)
    print(text, flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(text + "\n")


@contextlib.contextmanager
def mock_init(cls, init):
    old, cls.__init__ = cls.__init__, init
    try:
        yield
    finally:
        cls.__init__ = old


if __name__ == "__main__":
    main()
