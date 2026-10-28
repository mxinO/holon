"""GLM's cover (docs/dev/v1-glm.md acceptance 1(f), §6.2): the real model's own inputs, recorded as the engine serves
one long prompt in 2,000-token steps (≥ 4 step boundaries, none a multiple of 64) and then ≥ 256 decode rows, checked
against the float32 / float64 oracles (`reference.py`) of each part:

- **DSA** (one layer): each sampled sparse row's selection — the pools its attention read — against the oracle's
  scores on the engine's own indexer queries, head weights and pooled keys (float64): equal, or the differing pools
  within `EPS` of the top-k's edge (fp32 error); the pooled keys against the oracle's pooling of every stored raw key
  and gate; one decode `mla_attend` call and 32 rows of one prompt call against the oracle on their own latents.
- **Router** (one MoE layer): sampled rows' ids and weights against `moe_topk`'s oracle on the f64 logits of the
  layer's own input: the picks equal up to near-ties of score + bias, the weights where the picks are equal.
- **KDA** (one layer): each call's state at its start against the oracle carried from token 0 (f32, token by token,
  on the engine's own projected q, k, v, gates and β), and each row's output; reported at the rows just past a step
  start, elsewhere in the prompt, and at the decode rows.
- **mHC** (one block's pair): sampled rows' `comb` itself, x and R after the (pending) write-back, and the block's last
  write-back, against `hc_mhc_mix` / `hc_mhc_writeback`.

Reports beside the checks: the clamp's bind rate per path (dense and shared from their SwiGLU inputs, routed from the
sampled rows' experts dequantized from the file), the real comb logits' spread (|s2·m|), ties at the selection
boundary per 1,000 rows (the 512th and 513th oracle scores equal; at 0.0 apart). With `--plant` the run is under
`holon_judge.planted`'s plant, which must fail its detector (`DETECTORS`); `BOUNDS` were written FN's way after
the clean and planted runs.

Usage: cover.py <model> '<engine flags>' [--ctx 8500] [--step 2000] [--decode 256] [--plant NAME] [--out <json>]"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYERS = "11,5,5,5"     # the DSA, MoE, KDA and mHC layers checked (GLM-5.3-Flash: DSA at 3, 7, …; MoE from 3)
EPS = 2e-6            # a selection's top-k edge: fp32 error of the scores (test_dsa_kernels' ε)
TIE = 1e-5            # the router's near-ties of score + bias (fast-math expf)
SAMPLE = 64           # sampled rows a prompt call (DSA, router, mHC)
# metric -> bound (§6.2, FN's way): ≈ 10× the clean runs' maximum (RTX PRO 6000 all-resident, RTX 5090 mode (i); step
# 5's evidence), each ≥ 10× below its plant's effect; a clean 0 bounded at the check's own tolerance
# The attend's two, step 7 (q̃ one bf16 part, chunks of 256): ≈ 10× the RTX 5090's clean 6.9e-4 / 5.7e-4; its plants
# (a chunk combined unscaled, chunks dropped) 0.66 / 0.34 and 0.83 / 0.89, ≥ 57× (step 5's 4e-5 / 5e-5: the 4-dot form)
BOUNDS = {"dsa.worst_gap": EPS, "dsa.pooled_err": 1.2e-6, "dsa.attend_prompt_rel_l2": 7e-3,
          "dsa.attend_decode_rel_l2": 6e-3, "router.picks_off": 0.0, "router.weight_rel_err": 1.4e-5,
          "kda.out_err_at_step_starts": 0.14, "kda.out_err_elsewhere": 0.22, "kda.out_err_decode": 0.06,
          "kda.state_err_at_step_starts": 0.016, "mhc.comb_abs_err": 1e-5, "mhc.x_rel_err": 4e-3,
          "mhc.r_rel_err": 1e-7, "mhc.writeback_rel_err": 4e-7}
# each plant's detector (§6.3)
DETECTORS = {"dsa_decode_select_shift": "dsa.worst_gap", "kda_state_not_carried": "kda.out_err_at_step_starts",
             "kda_gate_softplus": "kda.out_err_decode", "mhc_comb_transposed": "mhc.r_rel_err",
             "mhc_sinkhorn_short": "mhc.comb_abs_err", "dsa_pool_ape_off": "dsa.pooled_err",
             "router_bias_in_weights": "router.weight_rel_err", "dsa_attend_chunk_unscaled": "dsa.attend_prompt_rel_l2",
             "dsa_attend_chunk_dropped": "dsa.attend_prompt_rel_l2"}


def _pick(n: int, k: int) -> torch.Tensor:
    return torch.linspace(0, n - 1, min(n, k)).round().long().unique() if n > k else torch.arange(n)


def _pages(table, seq, positions, page: int) -> torch.Tensor:
    """The one sequence's units in logical page order, up to its last position (`max_seqs` 1)."""
    return table[int(seq[0]), :int(positions.max()) // page + 1].long()


class DSA:
    """One DSA layer's index inputs, sampled selections, the blocks its attention read, two `mla_attend` calls."""

    def __init__(self, unit) -> None:
        self.u, self.index, self.select, self.attend = unit, unit._index, unit._select, unit._attend
        self.keys, self.rows, self.blocks, self.calls, self.pooled = [], [], {}, {}, None
        unit._index, unit._select, unit._attend = self._index, self._select, self._attend

    def _index(self, k, gate, raw, gates, pooled, ape, table, seq, positions):
        self.index(k, gate, raw, gates, pooled, ape, table, seq, positions)
        self.keys.append((k.cpu(), gate.cpu(), positions.cpu()))

    def _select(self, q, pooled, table, seq, positions, w):
        blocks = self.select(q, pooled, table, seq, positions, w)
        snap = pooled[_pages(table, seq, positions, self.u.page)].cpu()
        pick = _pick(len(q), SAMPLE) if len(q) > 1 else torch.arange(1)
        self.rows.append((q[pick].cpu(), w[pick].cpu(), positions[pick].cpu(), snap))
        self.pooled = snap
        return blocks

    def _attend(self, q, kv, blocks, table, seq, positions):
        out = self.attend(q, kv, blocks, table, seq, positions)
        for p, b in zip(positions.tolist(), blocks.cpu(), strict=True):
            self.blocks[p] = b
        kind = "decode" if len(q) == 1 else "prompt"
        if kind == "decode" or "prompt" not in self.calls:
            pick = torch.arange(1) if kind == "decode" else _pick(len(q), 32)
            pages = _pages(table, seq, positions, kv.shape[1])
            self.calls[kind] = (q[pick].cpu(), kv[pages].cpu(), blocks[pick].cpu(), positions[pick].cpu(),
                                out[pick].cpu())
        return out


class Router:
    def __init__(self, unit) -> None:
        self.u, self.rows = unit, []
        unit._route = self._route

    def _route(self, x):
        weights, ids, xin = type(self.u)._route(self.u, x)                  # the class's: a plant patches it
        pick = _pick(len(x), SAMPLE)
        self.rows.append((x[pick].float().cpu(), weights[pick].cpu(), ids[pick].cpu()))
        return weights, ids, xin


class KDA:
    def __init__(self, unit) -> None:
        self.u, self.calls = unit, []
        unit._rule_call = self._rule_call

    def _rule_call(self, rule, handle, mixed, f, b, ssm_state, slots, has_init, snapshot=None):
        before = ssm_state[slots[0]].float().cpu() if bool(has_init[0]) else None
        o = type(self.u)._rule_call(self.u, rule, handle, mixed, f, b, ssm_state, slots, has_init, snapshot)
        self.calls.append((mixed.cpu(), f.cpu(), b.cpu(), o.cpu(), before, getattr(rule, "lower_bound", -5.0)))
        return o


class MHC:
    """One block's mHC pair: its mixes and its last write-back, sampled rows."""

    def __init__(self, block) -> None:
        self.mixes, self.writebacks = [], []
        for hc in (block.attn_hc, block.ffn_hc):
            def mix(r, pending=None, hc=hc):
                pick = _pick(len(r), SAMPLE)
                before = r[pick].clone()
                pend = None if pending is None else tuple(p[pick].clone() for p in pending)
                out = type(hc).mix(hc, r, pending)
                self.mixes.append((hc, before.cpu(), None if pend is None else [p.cpu() for p in pend],
                                   [o[pick].cpu() for o in out], r[pick].cpu()))
                return out
            hc.mix = mix
        hc = block.ffn_hc

        def writeback(r, y, post, comb):
            pick = _pick(len(r), SAMPLE)
            before = r[pick].clone()
            type(hc).writeback(hc, r, y, post, comb)
            self.writebacks.append((before.cpu(), y[pick].cpu(), post[pick].cpu(), comb[pick].cpu(), r[pick].cpu()))
        hc.writeback = writeback


class Clamp:
    """The dense / shared SwiGLU inputs' bind counts (gate > L, |up| > L), per path."""

    def __init__(self, unit, path: str, counts: dict) -> None:
        self.unit, self.path, self.counts, self.act = unit, path, counts, unit._act
        unit._act = self

    def __call__(self, gate_up):
        g, u = gate_up.float().chunk(2, dim=-1)
        L, c = self.unit.limit, self.counts.setdefault(self.path, [0, 0, 0, 0, 0])     # gate, up, rows hit, rows
        c[0] += int((g > L).sum())
        c[1] += int((u.abs() > L).sum())
        c[2] += int(((g > L) | (u.abs() > L)).any(-1).sum())
        c[3] += g.numel() // g.shape[-1]
        c[4] = g.shape[-1]                                                               # values a row
        return self.act(gate_up)


def dsa_report(rec: DSA) -> dict:
    from holon_std.mixers.qsa import reference
    u = rec.u
    ratio, top, page = u.ratio, u.budget // u.ratio, u.page
    gaps, same, ties, zero_ties, rows = [], 0, 0, 0, 0
    for q, w, positions, pooled in rec.rows:
        for t in range(len(q)):
            p = int(positions[t])
            n = (p + 1) // ratio
            if n <= top or p not in rec.blocks:
                continue
            first = torch.arange(n) * ratio
            keys = pooled[first // page, (first % page) // ratio].double()
            scores = ((keys @ q[t].double().T).relu() * w[t].double()).sum(-1) / math.sqrt(q.shape[-1])
            got = rec.blocks[p][:top].long()
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[got[(got >= 0) & (got < n)]] = True
            want = scores.float().sort(descending=True, stable=True)
            edge = want.values[top - 1:top + 1]
            ties += int(edge[0] == edge[1])
            zero_ties += int(edge[0] == edge[1] == 0)
            gap = 0.0 if chosen.sum() == top and bool(chosen[want.indices[:top]].all()) else \
                float((scores[~chosen].max() - scores[chosen].min()) / scores.abs().max())
            gaps.append(max(gap, 0.0))
            same += gap <= EPS
            rows += 1
    out = {"rows": rows, "same": same / max(rows, 1), "worst_gap": max(gaps, default=0.0),
           "ties_per_1000": 1000 * ties / max(rows, 1), "zero_ties_per_1000": 1000 * zero_ties / max(rows, 1)}
    k, g, at = (torch.cat(t) for t in zip(*rec.keys, strict=True))     # the pooling, from every stored key and gate
    pages = -(-int(at.max() + 1) // page)
    raw, gates = torch.zeros(pages, page, u.dk, dtype=k.dtype), torch.zeros(pages, page, u.dk, dtype=k.dtype)
    pooled = torch.zeros(pages, page // ratio, u.dk)
    table, seq = torch.arange(pages, dtype=torch.int32)[None], torch.zeros(len(k), dtype=torch.int32)
    reference.dsa_index(k, g, raw, gates, pooled, u.ape.cpu(), table, seq, at, ratio)
    n = (int(at.max()) + 1) // ratio
    blocks = torch.arange(n)
    mine = rec.pooled[blocks // (page // ratio), blocks % (page // ratio)]
    want = pooled[blocks // (page // ratio), blocks % (page // ratio)]
    out["pooled_err"] = float(((mine - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)).max())
    for kind, (q, kv, b, positions, got) in rec.calls.items():
        table = torch.arange(len(kv), dtype=torch.int32)[None]
        want = reference.mla_attend(q.float(), kv, b, table, torch.zeros(len(q), dtype=torch.int32), positions,
                                    ratio, u.scale)
        out[f"attend_{kind}_rel_l2"] = float(((got.float() - want).norm(dim=(1, 2)) / want.norm(dim=(1, 2))).max())
    return out


def router_report(rec: Router, top_k: int, renormalize: bool, scale: float) -> dict:
    from holon_std.moe import reference
    u = rec.u
    rows, picks_off, weight_err = 0, 0, 0.0
    for x, weights, ids in rec.rows:
        logits = (x.double() @ u.router.double().cpu().T).float()
        want_w, want_ids = reference.moe_topk(logits, top_k, renormalize, "sigmoid", u.bias.cpu(), scale)
        choice = torch.sigmoid(logits.double()) + u.bias.double().cpu()
        got, want = choice.gather(1, ids.long()).sort(1).values, choice.gather(1, want_ids.long()).sort(1).values
        off = ((got - want).abs() > TIE * want.abs().clamp_min(1)).any(1)
        picks_off += int(off.sum())
        same = (ids.long().sort(1).values == want_ids.long().sort(1).values).all(1)
        if same.any():
            order_g, order_w = ids.long().argsort(1), want_ids.long().argsort(1)
            a = weights.gather(1, order_g)[same]
            b = want_w.gather(1, order_w)[same]
            weight_err = max(weight_err, float(((a - b).abs() / b.abs().clamp_min(1e-12)).max()))
        rows += len(x)
    return {"rows": rows, "picks_off": picks_off / max(rows, 1), "weight_rel_err": weight_err}


def kda_report(rec: KDA, step: int) -> dict:
    from holon_std.mixers.gdn import reference
    u = rec.u
    dev = "cuda"
    h, d, w = u.h, u.d, u.h * u.d
    state = torch.zeros(1, h, d, d, device=dev)
    starts, elsewhere, decode, state_err, seen = [], [], [], [], 0
    for mixed, f, b, o, before, lb in rec.calls:
        t = len(mixed)
        if before is not None:
            state_err.append(float((before.to(dev) - state[0]).norm() / state[0].norm().clamp_min(1e-12)))
        q, k, v = (x.reshape(t, h, d).to(dev) for x in mixed.split([w, w, w], dim=-1))
        want = reference.kda(q, k, v, f.to(dev), b.to(dev), u.A_log.float(), u.dt_bias.float(), state,
                             torch.zeros(1, dtype=torch.int32, device=dev),
                             torch.tensor([0, t], dtype=torch.int32, device=dev),
                             torch.tensor([seen > 0], device=dev), lb).float()
        err = ((o.to(dev).float() - want).flatten(1).norm(dim=1) / want.flatten(1).norm(dim=1).clamp_min(1e-12)).cpu()
        if t == 1:
            decode.append(float(err[0]))
        else:
            starts.append(float(err[:64].max()) if seen else 0.0)
            elsewhere.append(float(err[64:].max()) if t > 64 else 0.0)
        seen += t
    return {"calls": len(rec.calls), "out_err_at_step_starts": max(starts[1:], default=0.0),
            "out_err_elsewhere": max(elsewhere, default=0.0), "out_err_decode": max(decode, default=0.0),
            "state_err_at_step_starts": max(state_err, default=0.0)}


def mhc_report(rec: MHC) -> dict:
    from holon_std.hyper import reference
    dev = "cuda"
    comb_err, x_err, r_err, wb_err, spread = 0.0, 0.0, 0.0, 0.0, []
    for hc, before, pend, (x, _post, comb), after in rec.mixes:
        k, r = hc._mix, before.to(dev).clone()
        n = round(math.sqrt(len(hc.base) + 1)) - 1                       # streams (n² + 2n mix weights)
        args = (None, None, None) if pend is None else tuple(p.to(dev) for p in pend)
        wx, wpost, wcomb = reference.hc_mhc_mix(r, hc.fn, hc.base, hc.scale, *args, n, k.eps, k.hc_eps, k.iters,
                                                k.dtype)
        comb_err = max(comb_err, float((comb.to(dev) - wcomb).abs().max()))
        x_err = max(x_err, float((x.to(dev).float() - wx.float()).norm() / wx.float().norm()))
        r_err = max(r_err, float((after.to(dev) - r).norm() / r.norm()))
        z = before.to(dev).clone()                     # the comb logits as the mix sees them (after any write-back)
        if pend is not None:
            reference.hc_mhc_writeback(z, *args)
        m = torch.nn.functional.linear(z * torch.rsqrt(z.square().mean(-1, keepdim=True) + k.eps), hc.fn.float())
        spread.append((m[:, 2 * n:] * hc.scale[2]).abs().flatten().cpu())
    for before, y, post, comb, after in rec.writebacks:
        r = before.to(dev).clone()
        reference.hc_mhc_writeback(r, y.to(dev), post.to(dev), comb.to(dev))
        wb_err = max(wb_err, float((after.to(dev) - r).norm() / r.norm()))
    s = torch.cat(spread)
    q = torch.tensor([0.5, 0.9, 0.99])
    return {"comb_abs_err": comb_err, "x_rel_err": x_err, "r_rel_err": r_err, "writeback_rel_err": wb_err,
            "comb_logit_spread": {"p50_p90_p99": [round(float(v), 3) for v in s.quantile(q)],
                                  "max": float(s.max()), "frac_ge_8": float((s >= 8).float().mean())}}


def routed_clamp(path: pathlib.Path, rec: Router, layer: int, limit: float, rows: int = 16) -> list:
    """The routed gate / up of sampled rows' picked experts, from the file's expert weights (gguf-py)."""
    import gguf
    files = sorted(path.parent.glob(path.name.replace("-00001-of-", "-*-of-")))
    tensors = {t.name: t for f in files for t in gguf.GGUFReader(f).tensors}
    gate, up = tensors[f"blk.{layer}.ffn_gate_exps.weight"], tensors[f"blk.{layer}.ffn_up_exps.weight"]
    x, _, ids = rec.rows[0]
    pick = _pick(len(x), rows)
    cache, counts = {}, [0, 0, 0, 0, 0]

    def weight(t, e):
        if (t.name, e) not in cache:
            cache[(t.name, e)] = torch.from_numpy(gguf.quants.dequantize(t.data[e], t.tensor_type)).float()
        return cache[(t.name, e)]
    for r in pick.tolist():
        for e in ids[r].tolist():
            g, u = weight(gate, e) @ x[r], weight(up, e) @ x[r]
            counts[0] += int((g > limit).sum())
            counts[1] += int((u.abs() > limit).sum())
            counts[2] += int(((g > limit) | (u.abs() > limit)).any())
            counts[3] += 1
            counts[4] = len(g)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--ctx", type=int, default=8500)
    ap.add_argument("--step", type=int, default=2000)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--plant")
    ap.add_argument("--layers", default=LAYERS, help="DSA,MoE,KDA,mHC layer indices")
    ap.add_argument("--prompts", default=str(ROOT / "judge/prompts/v1-long-glm5_next.jsonl"))
    ap.add_argument("--out")
    a = ap.parse_args()
    import gguf
    from holon.api import SamplingParams
    from holon_judge import planted
    from holon_judge.engine import from_flags
    from holon_judge.recording import PromptSet
    from holon_std.ffn.units import SwiGLU
    from holon_std.hyper.units import MHCBlock
    from holon_std.mixers.gdn.units import KDA as KDAUnit
    from holon_std.mixers.qsa.units import DSAMLA
    from holon_std.moe.units import MoE

    fields = gguf.GGUFReader(a.model).fields
    value = lambda key: fields[f"glm5-next.{key}"].contents()           # noqa: E731
    top_k, renormalize, scale = value("expert_used_count"), bool(value("expert_weights_norm")), \
        value("expert_weights_scale")
    ids = max(PromptSet.load(a.prompts).prompts, key=lambda p: len(p.ids)).ids[:a.ctx]
    counts: dict = {}
    DSA_LAYER, MOE_LAYER, KDA_LAYER, MHC_LAYER = map(int, a.layers.split(","))
    with planted.PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        e = from_flags(a.model, a.flags, {"max_seqs": 1, "max_step_tokens": a.step, "prefix_cache": False,
                                          "max_model_len": a.ctx + a.decode + 64,
                                          "cuda_graphs": False})               # (eager: the hooks copy to the host)
        layers = e.executor.model.layers
        assert isinstance(layers[DSA_LAYER].mixer, DSAMLA) and isinstance(layers[KDA_LAYER].mixer, KDAUnit)
        dsa, kda = DSA(layers[DSA_LAYER].mixer), KDA(layers[KDA_LAYER].mixer)
        router = Router(next(p for p in layers[MOE_LAYER].ffn.parts if isinstance(p, MoE)))
        mhc = MHC(layers[MHC_LAYER])
        assert isinstance(layers[MHC_LAYER], MHCBlock)
        Clamp(layers[0].ffn, "dense", counts)
        Clamp(next(p for p in layers[MOE_LAYER].ffn.parts if isinstance(p, SwiGLU)), "shared", counts)
        e.generate([ids], SamplingParams(temperature=0.0, max_tokens=a.decode, ignore_eos=True))
        e.close()
    out = {"ctx": len(ids), "step": a.step, "layers": a.layers, "decode": a.decode, "plant": a.plant, "flags": a.flags,
           "gpu": torch.cuda.get_device_name(), "dsa": dsa_report(dsa),
           "router": router_report(router, top_k, renormalize, scale), "kda": kda_report(kda, a.step),
           "mhc": mhc_report(mhc)}
    limit = float(value("swiglu_clamp_exp")[MOE_LAYER])
    counts["routed"] = routed_clamp(pathlib.Path(a.model), router, MOE_LAYER, limit)
    out["clamp_bind"] = {p: {"gate_elements": c[0] / (c[3] * c[4]), "up_elements": c[1] / (c[3] * c[4]),
                             "rows": c[2] / c[3], "rows_seen": c[3]} for p, c in counts.items()}
    out["failed"] = [m for m, b in BOUNDS.items() if _metric(out, m) > b]
    out["pass"] = not out["failed"]
    if a.plant in DETECTORS:
        m = DETECTORS[a.plant]
        out["detector"] = {"metric": m, "value": _metric(out, m), "bound": BOUNDS[m],
                           "times_bound": _metric(out, m) / BOUNDS[m] if BOUNDS[m] else math.inf}
    text = json.dumps(out)
    print(text, flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def _metric(out: dict, name: str) -> float:
    part, key = name.split(".")
    return out[part][key]


if __name__ == "__main__":
    main()
