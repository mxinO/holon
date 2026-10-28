"""Noise-floor tables: every recorded run of one model vs its fp32 golden (and selected bf16-vs-bf16 pairs),
with per-category / context-vs-generated breakdowns and a validation of the top-k KL approximation.

    python noise_floor.py --dir out/qwen3-8b --out results/noise_floor_qwen3-8b.json
Run files in --dir: fp32.safetensors (golden) and any of the CANDIDATES below.
"""

import argparse
import json
import os

import torch

from compare import compare, kl_full, kl_topk, position_metrics, proposed_budgets, summarize
from golden import Recording, load_prompts

CANDIDATES = ["bf16", "bf16-r2", "bf16-eager", "bf16-chunk256", "fp16", "bug-eps", "bug-rope", "fp32-bug-eps",
              "fp32-bug-rope"]
PAIRS = [("bf16", "bf16-r2"), ("bf16", "bf16-eager"), ("bf16", "bf16-chunk256")]  # run-to-run, kernel/path changes
KS = [5, 10, 20, 50, 100]


def kl_validation(g: Recording, c: Recording) -> dict:
    """Exact full-vocab KL vs the bucketed top-k approximation, on the positions with stored full logprobs.
    The first position of each sequence (predicting from <|im_start|> alone, near-uniform) is excluded: the
    stride-32 subset over-samples it 24/1033 while it is 24/32783 of the real set."""
    keep = ~torch.isin(g.full_pos, g.offsets[:-1])
    g = Recording(None, None, None, None, {}, g.full_pos[keep], g.full_lp[keep])
    c = Recording(None, None, None, None, {}, c.full_pos[keep], c.full_lp[keep])
    exact = kl_full(g.full_lp, c.full_lp)
    q = torch.tensor([0.5, 0.99], dtype=torch.float64)
    out = {"n": len(exact), "exact": dict(zip(["mean", "p50", "p99"], [exact.double().mean().item(),
                                                                        *torch.quantile(exact.double(), q).tolist()]))}
    for k in KS:
        g_lp, g_ids = g.full_lp.topk(k, -1)
        c_lp, c_ids = c.full_lp.topk(k, -1)
        approx, _ = kl_topk(g_ids, g_lp, c_ids, c_lp)
        sig = exact > 1e-6
        ratio = (approx[sig] / exact[sig]).double()
        tail = (1 - g_lp.double().exp().sum(-1))
        out[f"k{k}"] = dict(mean=approx.double().mean().item(), p50=torch.quantile(approx.double(), q)[0].item(),
                            p99=torch.quantile(approx.double(), q)[1].item(),
                            ratio_p50=ratio.median().item(), ratio_p10=torch.quantile(ratio, 0.1).item(),
                            tail_mass_p50=tail.median().item(), tail_mass_p99=torch.quantile(tail, 0.99).item())
    return out


def bootstrap_se(m: dict[str, torch.Tensor], reps: int = 200) -> dict:
    """Sequence-level bootstrap SE of the floor metrics (positions within a sequence are correlated)."""
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.nonzero(m["seq"] == s)[:, 0] for s in m["seq"].unique()]
    q = torch.tensor([0.5, 0.99], dtype=torch.float64)
    stats = []
    for _ in range(reps):
        idx = torch.cat([seqs[i] for i in torch.randint(len(seqs), (len(seqs),), generator=gen)])
        kl = torch.quantile(m["kl"][idx].double(), q)
        stats.append([m["agree"][idx].double().mean(), kl[0], kl[1], m["dnll"][idx].mean()])
    sd = torch.tensor(stats).std(0).tolist()
    return dict(zip(["top1", "kl_p50", "kl_p99", "dnll"], sd))


def breakdown(g: Recording, c: Recording, prompts: list[dict]) -> dict:
    m = position_metrics(g, c)
    k = g.meta["k"]
    cats = [p["category"] for p in prompts]
    seq_cat = [cats[s] for s in m["seq"].tolist()]
    # position t predicts token t+1, which is model-generated iff t+1 >= gen_start
    starts = torch.tensor([p["gen_start"] for p in prompts])
    local = torch.arange(len(m["seq"])) - g.offsets[m["seq"]]
    gen = local + 1 >= starts[m["seq"]]
    out = {c_: summarize(m, k, torch.tensor([x == c_ for x in seq_cat])).dict() for c_ in sorted(set(cats))}
    out["context"] = summarize(m, k, ~gen).dict()
    out["generated"] = summarize(m, k, gen).dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts/v1.jsonl"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts, _ = load_prompts(a.prompts)
    path = lambda name: os.path.join(a.dir, f"{name}.safetensors")
    runs = {n: Recording.load(path(n)) for n in ["fp32", *CANDIDATES] if os.path.exists(path(n))}
    g = runs["fp32"]
    res = {"golden": g.meta, "golden_topk_bytes": sum(t.numel() * t.element_size() for t in
                                                      (g.topk_ids, g.topk_lp, g.target_lp, g.offsets)),
           "vs_fp32": {}, "budgets": {}, "pairs": {}, "breakdown": {}, "kl_validation": {}, "runs": {}}
    for name, c in runs.items():
        if name == "fp32":
            continue
        res["runs"][name] = {k: c.meta[k] for k in ("dtype", "device", "attn", "override", "wall_s")}
        res["vs_fp32"][name] = compare(g, c).dict()
        res["vs_fp32"][name]["bootstrap_se"] = bootstrap_se(position_metrics(g, c))
        print(f"fp32 vs {name:11s}", {k: f"{v:.4g}" for k, v in res["vs_fp32"][name].items()
                                      if isinstance(v, float)}, res["vs_fp32"][name]["bootstrap_se"])
        if name in ("bf16", "fp16"):
            res["breakdown"][name] = breakdown(g, c, prompts)
        if g.full_lp is not None and c.full_lp is not None and name in ("bf16", "fp16", "bug-eps"):
            res["kl_validation"][name] = kl_validation(g, c)
    floor = compare(g, runs["bf16"])
    budgets = proposed_budgets(floor)
    res["budgets"] = {"proposed": budgets.__dict__, "verdicts": {}}
    for name, c in runs.items():
        if name != "fp32":
            r = compare(g, c, budgets)
            res["budgets"]["verdicts"][name] = dict(passed=r.passed, failures=r.failures)
            print(f"budget {name:13s} {'PASS' if r.passed else 'FAIL'} {'; '.join(r.failures)}")
    for x, y in PAIRS:
        if x in runs and y in runs:
            res["pairs"][f"{x}|{y}"] = r = compare(runs[x], runs[y]).dict()
            r["bitwise_identical"] = all(torch.equal(getattr(runs[x], t), getattr(runs[y], t))
                                         for t in ("topk_ids", "topk_lp", "target_lp"))
            print(f"{x} vs {y}", {k: f"{v:.4g}" for k, v in r.items() if isinstance(v, (float, bool))})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
