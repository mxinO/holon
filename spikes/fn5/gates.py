"""FN step 5 (docs/v1-flash-next.md acceptance 2, §6.2): the full model's decode checks on the RTX PRO 6000,
all-resident (experts, PLE table and MTP head in VRAM), one engine at a time.

- `divergence`: the absolute decode check. Each `v1-long` prompt (4K-32K tokens, its chat prompt up to `gen_start`) is
  one session; an arm decodes `--tokens` greedily with top-k logprobs, and a reference engine of the same QSA mode
  (prefix cache off, eager, whole prompts) re-scores prompt + generated tokens teacher-forced; KL(reference || arm) at
  the generated positions (`holon_judge.bounded.divergence`'s, session-bootstrap SEs; `--cuts n`: n sessions per
  prompt, cut at n lengths). Arms: `sparse` (as built), `mtp` (sparse, speculating with the MTP head, 3 drafts),
  `dense` (the control: `holon_judge.qwen4exp.forced_dense` in the arm and its reference), and, with `--plant`,
  `sparse+plant` and `mtp+plant` (`qsa_decode_select_shift`, decode and verify only, so the sparse reference serves
  them too). Gate per arm: KL p99 <= max(2 x dense, dense + 2.5 SE of dense's p99); pre-registered: the unplanted arms
  pass and both planted arms fail, and if a planted arm passes, the statistic is tightened to p95 (the per-position
  KLs are kept for the next step, a paired difference).
- `mtp`: accepted tokens per verify pass (accepted drafts + 1, Strata's count) at 3 drafts on `v1-qwen3_5`'s chat and
  agent prompts (their chat prompts, `--tokens` greedy each); acceptance 2's gate is >= 2.2.
- `launches`: CUDA kernels per decode token and per 4-row verify pass, counted with the profiler on eager steps (a
  graph replays the same kernels), the MTP head's draft sub-steps apart.

    python spikes/fn5/gates.py divergence --model <...-00001-of-00002.gguf> --pack <mtp-q2_0.gguf> --out <json>
    python spikes/fn5/gates.py mtp --model <...> --pack <...> --out <json>
    python spikes/fn5/gates.py launches --model <...> --pack <...> --out <json>
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import pathlib
import subprocess
import time

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge import bounded, planted, qwen4exp
from holon_judge.compare import kl_topk
from holon_judge.recording import PromptSet, Recording, tail_from_topk

ROOT = pathlib.Path(__file__).resolve().parents[2]
DRAFTS = 3


def config(a, **kw) -> EngineConfig:
    base = {"max_model_len": a.max_model_len, "max_step_tokens": a.max_step_tokens, "max_seqs": a.max_seqs}
    if kw.pop("mtp", False):
        base |= {"max_draft_tokens": DRAFTS, "plugin_options": {"spec.k": str(DRAFTS), "mtp.path": a.pack},
                 "slots": {"drafter": "drafters.mtp", "draft_policy": "std.spec.fixed_k",
                           "verifier": "std.spec.greedy"}}
    return EngineConfig(a.model, **(base | kw))


def build(a, mode=contextlib.nullcontext, **kw) -> Engine:
    with mode():
        return Engine(config(a, **kw))


def free(engine: Engine) -> None:
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def prompts(path: str, categories: tuple[str, ...] = ()) -> list[list[int]]:
    """Each prompt's chat prompt: its ids up to `gen_start` (the set's own rows, checked against its sha256)."""
    PromptSet.load(str(ROOT / path))
    rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]
    return [r["ids"][:r["gen_start"]] for r in rows if not categories or r["category"] in categories]


def generate(engine: Engine, ids: list[list[int]], tokens: int, k: int | None) -> list[bounded.Turn]:
    params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True, logprobs=k)
    return [(p, out, lps) for p, (out, lps) in zip(ids, bounded._generate(engine, ids, params), strict=True)]


def sessions(ids: list[list[int]], cuts: int, low: int = 2560) -> list[list[int]]:
    """`cuts` sessions per prompt: its chat prompt cut at `cuts` lengths evenly spaced from `low` (past the dense
    limit, so they decode sparse) to its whole length; a prompt of at most `low` tokens is one session."""
    if cuts == 1:
        return ids
    return [p[:low + (len(p) - low) * i // (cuts - 1)] for p in ids if len(p) > low for i in range(cuts)] + \
        [p for p in ids if len(p) <= low]


def divergence(a) -> dict:
    ids = sessions(prompts("judge/prompts/v1-long.jsonl"), a.cuts)
    arms = {"sparse": ({}, None), "mtp": ({"mtp": True}, None), "dense": ({}, qwen4exp.forced_dense)}
    if a.plant:
        arms |= {"sparse+plant": ({}, a.plant), "mtp+plant": ({"mtp": True}, a.plant)}
    turns, out = {}, {"arms": {}}
    for name, (kw, mode) in arms.items():
        ctx = (mode if callable(mode) else planted.PLANTS[mode]) if mode else contextlib.nullcontext
        t0 = time.time()
        kept = pathlib.Path(a.out).with_suffix(f".{name}.turns.pt")     # a re-run reuses an arm's generation
        with ctx():
            if kept.exists():
                turns[name], stats = torch.load(kept, weights_only=False)
            else:
                engine = Engine(config(a, **kw))
                turns[name] = generate(engine, ids, a.tokens, a.k)
                stats = engine.stats()
                free(engine)
                torch.save((turns[name], stats), kept)
        out["arms"][name] = {"wall_s": round(time.time() - t0, 1),
                             "spec": {k: v for k, v in stats.items() if k.startswith("spec")}}
        print(name, out["arms"][name], flush=True)
    kls = {}
    for ref_mode, names in ((contextlib.nullcontext, [n for n in arms if n != "dense"]),
                            (qwen4exp.forced_dense, ["dense"])):
        with ref_mode():
            reference = Engine(config(a, prefix_cache=False, overlap=False, cuda_graphs=False, max_seqs=1,
                                      max_step_tokens=max(len(p) for p in ids) + a.tokens + 1))
            for name in names:
                kl, session = kls[name] = per_position(reference, turns[name], len(ids), a.k)
                values, ses = bounded._quantiles(kl, session, torch.tensor(QUANTILES, dtype=torch.float64))
                names_q = [f"p{round(q * 100)}" for q in QUANTILES]
                out["arms"][name] |= {"kl_mean": kl.mean().item(), "n": len(kl),
                                      **dict(zip(names_q, values.tolist(), strict=True)),
                                      **{f"{q}_se": v for q, v in zip(names_q, ses.tolist(), strict=True)}}
                print(name, {k: v for k, v in out["arms"][name].items() if k[0] == "p"}, flush=True)
            free(reference)
    torch.save(kls, pathlib.Path(a.out).with_suffix(".kl.pt"))
    # The pre-registered statistic: p99; tightened to p95 if a plant passes it (acceptance 2)
    dense = out["arms"]["dense"]
    for stat in ("p99", "p95"):
        bound = max(2 * dense[stat], dense[stat] + 2.5 * dense[f"{stat}_se"])
        out[f"gate_{stat}"] = {"bound": bound, **{n: out["arms"][n][stat] <= bound for n in arms if n != "dense"}}
    expect = {n: "+plant" not in n for n in arms if n != "dense"}
    for stat in ("p99", "p95"):
        out["statistic"] = stat
        out["pre_registered"] = {"expected": expect, "as_expected": all(out[f"gate_{stat}"][n] == v
                                                                       for n, v in expect.items())}
        if all(not out[f"gate_{stat}"][n] for n in expect if "+plant" in n):
            break
    return out


QUANTILES = (0.5, 0.95, 0.99)


def per_position(reference, turns: list[bounded.Turn], sessions: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(reference || arm) at every generated position and its session (`bounded.divergence`'s, unreduced)."""
    ref, session = bounded._reference(reference, turns, sessions, k)
    top_ids = torch.tensor([x[:k] for _, _, lps in turns for lp in lps for x in lp.top_ids])
    top = torch.tensor([x[:k] for _, _, lps in turns for lp in lps for x in lp.top], dtype=torch.float32)
    kl, _ = kl_topk(ref, Recording(top_ids, top, top[:, 0], tail_from_topk(top), ref.offsets, {}))
    return kl, session


def mtp(a) -> dict:
    ids = prompts("judge/prompts/v1-qwen3_5.jsonl", ("chat", "agent"))
    engine = build(a, mtp=True)
    generate(engine, ids, a.tokens, None)
    stats = engine.stats()
    free(engine)
    rows = {int(k.rsplit("_", 1)[1]): v for k, v in stats.items() if k.startswith(f"spec_rows_k{DRAFTS}_accepting_")}
    passes = sum(rows.values())
    per_pass = 1 + sum(n * v for n, v in rows.items()) / passes
    return {"prompts": len(ids), "rows_by_accepted": rows, "passes": passes, "tokens_per_pass": per_pass,
            "gate": per_pass >= 2.2, "spec": {k: v for k, v in stats.items() if k.startswith("spec")}}


def launches(a) -> dict:
    """Kernels per eager step, by step kind, with the profiler: a decode token (no drafter), and a verify pass of
    1 + 3 rows with the MTP head's sub-steps (MTP on)."""
    from torch.profiler import ProfilerActivity, profile

    ids = prompts("judge/prompts/v1-qwen3_5.jsonl", ("chat",))[:1]
    out = {}
    for name, kw in (("decode", {}), ("mtp", {"mtp": True})):
        engine = build(a, cuda_graphs=False, overlap=False, max_seqs=1, **kw)
        engine.add(ids[0], SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True))
        counts = []
        for step in range(64):
            if not engine.has_work():
                break
            if step < 4:                                    # the prefill and the first steps
                engine.step()
                continue
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                engine.step()
                torch.cuda.synchronize()
            counts.append(sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA))
        free(engine)
        out[name] = {"per_step": counts, "median": sorted(counts)[len(counts) // 2]}
        print(name, out[name], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("divergence", "mtp", "launches"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True, help="Strata's MTP pack (mtp-q2_0.gguf)")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--plant", help="the planted arms' plant (divergence)")
    ap.add_argument("--cuts", type=int, default=1, help="sessions per v1-long prompt (divergence): its prompt cut at "
                    "this many lengths from 2,560 tokens to the whole")
    ap.add_argument("--max-model-len", type=int, default=33792)
    ap.add_argument("--max-step-tokens", type=int, default=8192)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    result = {"divergence": divergence, "mtp": mtp, "launches": launches}[a.what](a)
    rec = {"name": f"fn5_{a.what}", "date": datetime.date.today().isoformat(),
           "gpu": torch.cuda.get_device_name(), "model": pathlib.Path(a.model).name, "pack": pathlib.Path(a.pack).name,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip(),
           "args": {k: v for k, v in vars(a).items() if k not in ("model", "pack", "out")},
           "wall_s": round(time.time() - t0, 1), **result}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "arms"}, indent=1, default=str))


if __name__ == "__main__":
    main()
