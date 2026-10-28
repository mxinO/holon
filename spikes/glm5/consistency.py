"""GLM's self-consistency gate (docs/dev/v1-glm.md acceptance 1(c), §6.2): Holon's paths against each other on the full
model, no golden (FN's gate, spikes/fn7/consistency.py, without MTP: D7 defers it). On FN's four `v1-long` prompts
(natural text and recall, 5K-20K tokens), retokenized for GLM (`judge/prompts/v1-long-glm5_next.jsonl`):

- `decode`: the prompts scored teacher-forced in 1-token steps (every quantized matmul on mmvq, the fused router,
  KDA's recurrent kernel, mHC's decode mix, DSA's decode index / select / attend) against the reference prefill;
- `chunked`: scored in 250-token steps (straddling DSA's 4-token pools, KDA's 64-token chunks and the mixers' sub-batch
  boundaries) against the same reference.

The reference is one prefill per prompt at `--prefill-step` (the deployment's step; a prompt longer than it runs in
steps of it). Per pair: KL(reference || path) mean, p50 and p99 over positions, the signed mean ΔNLL per prompt and per
1,024-position band. Bars (FN's, §6.2): KL mean ≤ 0.13, KL p50 ≤ 0.007; a band's |ΔNLL| ≤ 0.75 is reported, not
gated. Pre-registered must-fail plants (§6.3): `dsa_decode_select_shift` (decode and verify rows only: the decode
pair's p50) and `kda_state_not_carried` (every multi-token KDA call from a zero state: the chunked pair) — a plant's
verdict counts only where its effect is ≥ 10× the clean run's value of that statistic.

    python spikes/glm5/consistency.py --model <...-00001-of-00003.gguf> --out <json> [--plant P]
        [--holon-flags '<engine flags>'] [--prefill-step N] [--parts decode,chunked]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import pathlib
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPTS = ("summary-6k", "recall-8k", "recall-16k", "chat-multiturn-20k")
SET = ROOT / "judge/prompts/v1-long-glm5_next.jsonl"
K = 20
BARS = {"kl_mean": 0.13, "kl_p50": 0.007, "band_dnll": 0.75}


def engine(a, **kw):
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import engine as build
    return build(a.model, a.holon_flags, max_model_len=a.max_len, max_seqs=1, prefix_cache=False, **kw)


def scored(a, prompts, step: int):
    from holon_judge import engine as judge
    from holon_judge.engine import score
    judge.cpu_split = lambda e: None                 # (c) compares the engine with itself: no floor, no CPU split
    e = engine(a, max_step_tokens=step)
    try:
        return score(e, prompts, K)
    finally:
        e.close()


def compare(ref, rec) -> dict:
    from holon_judge.compare import kl_topk
    kl = kl_topk(ref, rec)[0].double()
    d = (ref.target_lp - rec.target_lp).double()                   # path's NLL minus the reference's
    o = ref.offsets.tolist()
    per, bands = {}, []
    for i, p in enumerate(PROMPTS):
        s, e = o[i], o[i + 1]
        b = [d[s + lo:min(s + lo + 1024, e)].mean().item() for lo in range(0, e - s, 1024)]
        bands += b
        per[p] = {"dnll": d[s:e].mean().item(), "kl_mean": kl[s:e].mean().item(), "kl_p50": kl[s:e].median().item(),
                  "worst_band": max(b, key=abs)}
    out = {"n": len(kl), "kl_mean": kl.mean().item(), "kl_p50": kl.median().item(), "kl_p99": kl.quantile(0.99).item(),
           "dnll": d.mean().item(), "worst_band_dnll": max(bands, key=abs), "prompts": per}
    out["pass"] = out["kl_mean"] <= BARS["kl_mean"] and out["kl_p50"] <= BARS["kl_p50"]
    out["band_within"] = abs(out["worst_band_dnll"]) <= BARS["band_dnll"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--plant", help="a judge plant, e.g. dsa_decode_select_shift (must fail)")
    ap.add_argument("--holon-flags", default="", help="engine flags over the defaults (a deployment's)")
    ap.add_argument("--prefill-step", type=int, default=0, help="the reference's step (default: whole prompts)")
    ap.add_argument("--parts", default="decode,chunked")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from holon_judge import planted
    from holon_judge.recording import PromptSet
    full = PromptSet.load(SET)
    keep = {p.id: p for p in full.prompts}
    prompts = PromptSet(full.name, full.sha256, [keep[p] for p in PROMPTS])
    a.max_len = max(a.max_len, max(len(p.ids) for p in prompts.prompts) + 8)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {"name": "glm5_consistency", "date": datetime.date.today().isoformat(), "gpu": torch.cuda.get_device_name(),
           "model": pathlib.Path(a.model).name, "plant": a.plant, "prompts": PROMPTS, "bars": BARS, "commit": commit,
           "flags": a.holon_flags, "prefill_step": a.prefill_step or a.max_len}
    parts = a.parts.split(",")
    with planted.PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        prefill = scored(a, prompts, a.prefill_step or a.max_len)
        if "decode" in parts:
            out["decode"] = compare(prefill, scored(a, prompts, 1))
        if "chunked" in parts:
            out["chunked"] = compare(prefill, scored(a, prompts, 250))
    out["pass"] = all(out[k]["pass"] for k in parts)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "prompts"})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
