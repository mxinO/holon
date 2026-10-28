"""The self-consistency gate (FN: docs/dev/v1-flash-next.md §18 (c); GLM: docs/dev/v1-glm.md acceptance 1(c), §6.2):
Holon's paths against each other on the full model, no golden. On four `v1-long` prompts (natural text and recall,
5K-20K tokens; `--prompts` the set tokenized for the model, e.g. `judge/prompts/v1-long-glm5_next.jsonl`):

- `decode`: the prompts scored teacher-forced in 1-token steps (every quantized matmul on mmvq, GDN's recurrent
  kernel, the GEMV HC mix, QSA's decode kernels) against one prefill step per prompt;
- `chunked`: scored in 250-token steps (straddling QSA's 4-token blocks and PLE's window) against the same prefill;
- `mtp`: each prompt's chat prompt decoded greedily for `--tokens` with MTP (3 drafts) and without, top-20 logprobs:
  KL between the two at each position up to their first differing token, and where that is.

Per pair: KL(reference || path) mean, p50 and p99 over positions, the signed mean ΔNLL of the scored tokens (path
minus reference, nats) per prompt and per 1,024-position band. Bars, set before the gate's first run from the bf16
noise between correct paths (§17.2, the fixed mix, IQ2_XS and IQ3_S on all of `v1-long`: KL mean 0.043 / 0.063, p50
0.0023 / 0.0032, worst band |ΔNLL| 0.50): KL mean ≤ 0.13, KL p50 ≤ 0.007; every band's |ΔNLL| ≤ 0.75 is reported
(`band_within`), not gated (owner, 2026-10-21). `--plant qsa_decode_select_shift` (decode and verify only) must fail.
`mtp`'s bars (the lead's call, 2026-10-24, set before the final tree's run, docs/dev/v1-flash-next.md §22.8): about 5×
the clean pairs' (KL mean 0.0020 / 0.0044, p99 0.023 / 0.063 on the RTX 5080 / 5090, 5346d3a) — KL mean ≤ 0.02 and p99
≤ 0.3 over at least 100 agreeing positions, on 8 chat prompts (the four's and `MTP_MORE`); `--plant
linear_rows_swap` (verify passes only) and `shexp_gate_off` must fail. The pair runs at the reference's step
(`--prefill-step`, the deployment's); it needs `--pack` (the MTP head) and runs only on a model that has one (GLM's
gate: `--parts decode,chunked`, D7 defers MTP; its plants `dsa_decode_select_shift` and `kda_state_not_carried`).

The engines run 8 sequences with the prefix cache on unless `--holon-flags` say otherwise (GLM's gate: `--max-seqs 1
--no-prefix-cache`); the decode and chunked steps win over the flags' `--max-step-tokens`.

    python judge/consistency.py --model <...-00001-of-00002.gguf> [--pack <mtp-q2_0.gguf>] --out <json> [--plant P]
        [--holon-flags '<engine flags>'] [--prefill-step N] [--parts decode,chunked,mtp] [--prompts <set.jsonl>]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import pathlib
import subprocess

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ("summary-6k", "recall-8k", "recall-16k", "chat-multiturn-20k")
DRAFTS, K = 3, 20
BARS = {"kl_mean": 0.13, "kl_p50": 0.007, "band_dnll": 0.75}
MTP_BARS = {"kl_mean": 0.02, "kl_p99": 0.3, "positions": 100}
MTP_MORE = ("chat-explain", "code-lru", "agent-csv", "chat-zh")   # v1-qwen3_5's, beside the four: 8 prompts in all


def engine(a, **kw):
    from holon_judge.engine import from_flags
    if kw.pop("mtp", False):
        kw |= {"max_draft_tokens": DRAFTS, "plugin_options": {"spec.k": str(DRAFTS), "mtp.path": a.pack},
               "slots": {"drafter": "drafters.mtp", "draft_policy": "std.spec.fixed_k", "verifier": "std.spec.greedy"}}
    return from_flags(a.model, a.holon_flags, {"max_model_len": a.max_len, "max_seqs": 8}, kw)


def scored(a, prompts, step: int):
    from holon_judge import engine as judge
    from holon_judge.engine import score
    # An F5b deployment's split names the path-aware floor's CPU experts (one token per step, a frozen placement);
    # (c) compares the engine with itself, with no floor (§18): its split is not recorded.
    judge.cpu_split = lambda e: None
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
    out["pass"] = out["kl_mean"] <= BARS["kl_mean"] and out["kl_p50"] <= BARS["kl_p50"]   # judged by KL (owner,
    out["band_within"] = abs(out["worst_band_dnll"]) <= BARS["band_dnll"]                 # 2026-10-21); reported
    return out


def mtp_vs_plain(a, chats: list[list[int]]) -> dict:
    from holon.api import SamplingParams
    from holon_judge.bounded import _generate
    params = SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True, logprobs=K)
    runs = {}
    for name, kw in (("plain", {}), ("mtp", {"mtp": True})):
        e = engine(a, max_step_tokens=a.prefill_step or 8192, **kw)
        runs[name] = _generate(e, chats, params)
        e.close()
    from holon_judge.compare import kl_topk
    from holon_judge.recording import Recording, tail_from_topk

    def rows(lps, n):                     # top-k ids and logprobs at the first n + 1 positions
        flat = [(i[:K], x[:K]) for lp in lps for i, x in zip(lp.top_ids, lp.top, strict=True)][:n + 1]
        return torch.tensor([i for i, _ in flat]), torch.tensor([x for _, x in flat], dtype=torch.float32)

    first, ids, lps = [], {"plain": [], "mtp": []}, {"plain": [], "mtp": []}
    for (t0, l0), (t1, l1) in zip(runs["plain"], runs["mtp"], strict=True):
        n = next((i for i, (x, y) in enumerate(zip(t0, t1, strict=True)) if x != y), len(t0) - 1)
        first.append(n)
        for name, lp in (("plain", l0), ("mtp", l1)):
            i, x = rows(lp, n)
            ids[name].append(i)
            lps[name].append(x)
    offsets = torch.tensor([0, *(len(x) for x in lps["plain"])]).cumsum(0)
    rec = {name: Recording(torch.cat(ids[name]), torch.cat(lps[name]), torch.cat(lps[name])[:, 0],
                           tail_from_topk(torch.cat(lps[name])), offsets, {}) for name in ids}
    kl = kl_topk(rec["plain"], rec["mtp"])[0].double()
    out = {"tokens": a.tokens, "first_difference": first, "positions": len(kl), "kl_mean": kl.mean().item(),
           "kl_p99": kl.quantile(0.99).item(), "bars": MTP_BARS}
    out["pass"] = out["kl_mean"] <= MTP_BARS["kl_mean"] and out["kl_p99"] <= MTP_BARS["kl_p99"] and \
        out["positions"] >= MTP_BARS["positions"]
    return out


def chats(a) -> list[list[int]]:
    """The MTP pair's 8 chat prompts (§22.8): the four's and `MTP_MORE`."""
    def rows(path):
        return {r["id"]: r for r in map(json.loads, pathlib.Path(path).read_text().splitlines())}
    four, more = rows(a.prompts), rows(a.mtp_prompts)
    return [four[p]["ids"][:four[p]["gen_start"]] for p in PROMPTS] + \
        [more[p]["ids"][:more[p]["gen_start"]] for p in MTP_MORE]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", help="the MTP head (the mtp pair's)")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--plant", help="a judge plant, e.g. qsa_decode_select_shift (must fail)")
    ap.add_argument("--prompts", default=str(ROOT / "judge/prompts/v1-long.jsonl"), help="the four prompts' set")
    ap.add_argument("--mtp-prompts", default=str(ROOT / "judge/prompts/v1-qwen3_5.jsonl"), help="MTP_MORE's set")
    ap.add_argument("--holon-flags", default="", help="engine flags over the defaults (an offload deployment's)")
    ap.add_argument("--prefill-step", type=int, default=0, help="the reference's step (default: whole prompts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="decode,chunked,mtp", help="the pairs to run (a plant: its pair)")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if "mtp" in parts and not a.pack:
        ap.error("the mtp pair needs --pack")
    from holon_judge import planted
    from holon_judge.recording import PromptSet
    full = PromptSet.load(a.prompts)
    keep = {p.id: p for p in full.prompts}
    prompts = PromptSet(full.name, full.sha256, [keep[p] for p in PROMPTS])
    a.max_len = max(a.max_len, max(len(p.ids) for p in prompts.prompts) + a.tokens + 8)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {"name": "consistency", "date": datetime.date.today().isoformat(), "gpu": torch.cuda.get_device_name(),
           "model": pathlib.Path(a.model).name, "plant": a.plant, "prompts": PROMPTS,
           "set": pathlib.Path(a.prompts).name, "bars": BARS, "commit": commit, "flags": a.holon_flags,
           "prefill_step": a.prefill_step or a.max_len}
    with planted.PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        if "decode" in parts or "chunked" in parts:
            prefill = scored(a, prompts, a.prefill_step or a.max_len)
        if "decode" in parts:
            out["decode"] = compare(prefill, scored(a, prompts, 1))
        if "chunked" in parts:
            out["chunked"] = compare(prefill, scored(a, prompts, 250))
        if "mtp" in parts:
            out["mtp"] = mtp_vs_plain(a, chats(a))
    out["pass"] = all(out[k]["pass"] for k in parts)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "prompts"})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
