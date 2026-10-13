"""L2 comparator: a candidate recording vs the fp32 golden → metrics, per category and overall, and a verdict.

Budgets come from the floor, the same comparison for HF in the model's activation dtype (design §10.3, calibrated in
spike S5): pass iff KL p99 ≤ 2× floor (the discriminating metric), KL p50 ≤ 2× floor, top-1 agreement ≥ floor − 0.25
points and the candidate's ΔPPL against the floor, paired at the same positions, within 0.25% + 2.5 standard errors
(sequences as clusters: ΔPPL is noisy — S5 found it does not discriminate, and M0.3 measured standard errors of
0.3–2%, so a margin without noise failed runs whose top-1 and KL matched the floor). The standard error is capped at
√2 × the floor's own (floor vs golden: the noise between two correct paths), so an error confined to a few
sequences cannot widen its own budget. The guard is coarse: on v1-qwen3_5 (24 sequences) two correct bf16 paths
differ by a paired SE of ≈ 0.9%, so only shifts beyond ≈ 2.5% fail it; top-1 and KL carry the verdict. The verdict is
on all positions; per-category numbers are for diagnosis.

KL(golden ‖ candidate) is computed over k+1 buckets: the golden's top-k ids plus one tail bucket holding the rest of
each distribution. Merging outcomes never increases KL (log-sum inequality), so where the candidate's top-k contains
all of the golden's ids this is a lower bound on the full-vocabulary KL; at k = 20 it captures 95–100% of bf16-scale
KL (S5). A golden id missing from the candidate's top-k is imputed as min(candidate k-th prob, candidate tail /
(m+1)), m the number of missing ids: a feasible value that leaves the candidate's tail bucket non-empty, but it can
under- or over-state the true probability, so at those positions the KL is an estimate. `missing` reports how often
that happens (~1.5% of golden ids for bf16); the floor goes through the same estimator.
"""
from __future__ import annotations

import dataclasses
import math

import torch

from .recording import Recording

TOP1_DELTA = 0.0025       # top-1 agreement ≥ floor − 0.25 points (≈ 2.5 sequence-bootstrap SE in S5)
KL_FACTOR = 2.0           # KL p50/p99 ≤ 2× floor: the floor varies 5× across models, so δ_KL is relative
DPPL_DELTA = 0.0025       # |ΔPPL vs floor| ≤ 0.25% + DPPL_Z standard errors: a coarse guard (S5)
DPPL_Z = 2.5              # as bounded's margins (design §7.10)
CONFIDENT_GAP = 0.1       # nats between the golden's top-2: positions that are not near-ties


@dataclasses.dataclass(frozen=True)
class Metrics:
    n: int
    top1: float               # argmax agreement
    top1_confident: float     # argmax agreement where the golden's top-2 gap ≥ CONFIDENT_GAP
    kl_mean: float
    kl_p50: float
    kl_p99: float
    kl_max: float             # heavy-tailed; reported, never budgeted
    dppl: float               # relative perplexity change, exp(mean ΔNLL) − 1
    dppl_se: float            # standard error with sequences as clusters (positions within one are correlated);
                              # nan with a single sequence
    missing: float            # share of golden top-k ids absent from the candidate's top-k (imputed)


@dataclasses.dataclass(frozen=True)
class Budgets:
    top1_min: float
    kl_p50_max: float
    kl_p99_max: float
    dppl_max: float

    @classmethod
    def from_floor(cls, floor: Metrics, paired_se: float) -> Budgets:
        """`paired_se`: the standard error of the candidate's ΔPPL against the floor (`paired_dppl`), capped at √2 ×
        the floor's own (a candidate cannot inflate it); nan (one sequence) counts as 0. A floor bitwise equal to its
        golden would cap it at 0 (budget 0.25%); real floors run in the activation dtype against an fp32 golden."""
        known = [x for x in (paired_se, math.sqrt(2) * floor.dppl_se) if not math.isnan(x)]
        return cls(floor.top1 - TOP1_DELTA, KL_FACTOR * floor.kl_p50, KL_FACTOR * floor.kl_p99,
                   DPPL_DELTA + DPPL_Z * min(known, default=0.0))

    def failures(self, m: Metrics, dppl_vs_floor: float) -> list[str]:
        checks = [(m.top1 >= self.top1_min, f"top-1 {m.top1:.4%} < {self.top1_min:.4%}"),
                  (m.kl_p50 <= self.kl_p50_max, f"KL p50 {m.kl_p50:.3e} > {self.kl_p50_max:.3e}"),
                  (m.kl_p99 <= self.kl_p99_max, f"KL p99 {m.kl_p99:.3e} > {self.kl_p99_max:.3e}"),
                  (abs(dppl_vs_floor) <= self.dppl_max,
                   f"|ΔPPL vs floor| {abs(dppl_vs_floor):.3%} > {self.dppl_max:.3%}")]
        return [msg for ok, msg in checks if not ok]


@dataclasses.dataclass(frozen=True)
class Result:
    golden: dict
    candidate: dict
    overall: Metrics
    categories: dict[str, Metrics]
    floor: Metrics | None = None
    budgets: Budgets | None = None
    failures: tuple[str, ...] = ()
    dppl_vs_floor: tuple[float, float] | None = None   # the candidate's ΔPPL against the floor, its SE
    drafter: dict | None = None    # the L0-drafter rule's numbers (`compare_drafter`): its verdict replaces L2's

    @property
    def passed(self) -> bool | None:
        return None if self.budgets is None else not self.failures

    def to_json(self) -> dict:
        out = dataclasses.asdict(self)
        out["passed"] = self.passed
        return out

    def table(self) -> str:
        head = f"{'':10s} {'n':>6s} {'top-1':>8s} {'top-1 conf':>10s} {'KL p50':>9s} {'KL p99':>9s} " \
               f"{'KL max':>9s} {'ΔPPL ± SE':>17s} {'missing':>7s}"
        rows = [head, _row("overall", self.overall)]
        if self.floor is not None:
            rows.append(_row("floor", self.floor))
        if self.dppl_vs_floor is not None:
            d, se = self.dppl_vs_floor
            rows.append(f"{'vs floor':10s} {'':6s} {'':8s} {'':10s} {'':9s} {'':9s} {'':9s} "
                        f"{f'{d:+.3%} ± {se:.3%}':>17s}")
        if self.budgets is not None:
            b = self.budgets
            rows.append(f"{'budget':10s} {'':6s} {'≥' + format(b.top1_min, '.3%'):>8s} {'':10s} "
                        f"{'≤' + format(b.kl_p50_max, '.2e'):>9s} {'≤' + format(b.kl_p99_max, '.2e'):>9s} "
                        f"{'':9s} {'≤' + format(b.dppl_max, '.3%'):>17s}")
        rows += [_row(name, m) for name, m in self.categories.items()]
        if (d := self.drafter) is not None:
            rows += [f"L0-drafter (§5): {d['n_confident']} positions of margin > {d['margin']:.3e} (the floor's p99 "
                     f"|Δ margin|): top-1 {d['top1']:.3%} vs floor {d['floor_top1']:.3%} (≥ {d['top1_min']:.3%}); "
                     f"KL mean {d['kl_mean']:.3e} (≤ {d['kl_mean_max']:.3e}); confident disagreements by row of a "
                     f"draft (block row, chain step): {d['confident_disagreements_by_row']}",
                     "L2's rule, reported only: " + ("pass" if not d["l2_failures"] else "; ".join(d["l2_failures"]))]
        where = f"{self.candidate.get('model')} {self.candidate.get('dtype')} vs {self.golden.get('dtype')} golden"
        if self.passed is None:
            verdict = "no floor given: metrics only"
        else:
            verdict = "PASS" if self.passed else "FAIL: " + "; ".join(self.failures)
        return "\n".join([f"L2 {where}, prompts {self.golden.get('prompts')}", *rows, verdict])


def _row(name: str, m: Metrics) -> str:
    return (f"{name:10s} {m.n:6d} {m.top1:8.3%} {m.top1_confident:10.3%} {m.kl_p50:9.2e} {m.kl_p99:9.2e} "
            f"{m.kl_max:9.2e} {f'{m.dppl:+.3%} ± {m.dppl_se:.3%}':>17s} {m.missing:7.2%}")


def kl_topk(golden: Recording, cand: Recording) -> tuple[torch.Tensor, torch.Tensor]:
    """Per position: bucketed KL(golden ‖ candidate) and the number of imputed golden ids."""
    g_lp, g_tail = golden.topk_lp.double(), golden.tail_lp.double()
    c_p, c_rest = cand.topk_lp.double().exp(), cand.tail_lp.double().exp()
    match = golden.topk_ids[:, :, None] == cand.topk_ids[:, None, :]           # [N, k, k_cand]
    found = match.any(-1)
    n_missing = (~found).sum(-1)
    q_found = c_p.gather(1, match.int().argmax(-1))
    q_missing = torch.minimum(c_p[:, -1:], c_rest[:, None] / (n_missing[:, None] + 1))
    q = torch.where(found, q_found, q_missing)
    # The candidate's tail bucket, without subtracting from 1 (which cancels when it is tiny): its own tail, minus
    # the mass imputed to golden ids, plus its top-k ids that are not among the golden's.
    q_tail = c_rest - (q * ~found).sum(-1) + (c_p * ~match.any(1)).sum(-1)
    p, p_tail = g_lp.exp(), g_tail.exp()
    log_q, log_q_tail = q.clamp_min(1e-30).log(), q_tail.clamp_min(1e-30).log()
    kl = (p * (g_lp - log_q)).sum(-1) + torch.where(p_tail > 0, p_tail * (g_tail - log_q_tail), 0.0)
    return kl.clamp_min(0), n_missing


def _summarize(rows: dict[str, torch.Tensor], k: int) -> Metrics:
    n = len(rows["kl"])
    agree, kl, dnll = rows["agree"], rows["kl"], rows["dnll"]
    # ΔNLL as a ratio estimator over sequences: Σ_s ΔNLL_s / Σ_s n_s, with the cluster-robust standard error.
    _, seq = rows["seq"].unique(return_inverse=True)
    s = int(seq.max()) + 1
    d_s = torch.zeros(s, dtype=torch.float64).index_add_(0, seq, dnll)
    n_s = torch.zeros(s, dtype=torch.float64).index_add_(0, seq, torch.ones_like(dnll))
    mean = d_s.sum().item() / n
    se = math.sqrt(s / (s - 1) * ((d_s - mean * n_s) ** 2).sum().item()) / n if s > 1 else math.nan
    dppl = math.expm1(mean)
    p50, p99 = torch.quantile(kl, torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
    confident = agree[rows["gap"] >= CONFIDENT_GAP]
    return Metrics(n=n, top1=agree.double().mean().item(),
                   top1_confident=confident.double().mean().item() if len(confident) else math.nan,
                   kl_mean=kl.mean().item(), kl_p50=p50, kl_p99=p99, kl_max=kl.max().item(),
                   dppl=dppl, dppl_se=se * (1 + dppl), missing=rows["missing"].sum().item() / (n * k))


def metrics(golden: Recording, cand: Recording) -> tuple[Metrics, dict[str, Metrics]]:
    """Overall and per-category metrics of `cand` against `golden`."""
    if golden.meta["prompts_sha256"] != cand.meta["prompts_sha256"] or not torch.equal(golden.offsets, cand.offsets):
        raise ValueError("recordings are of different prompt sets")
    kl, missing = kl_topk(golden, cand)
    rows = {"agree": golden.topk_ids[:, 0] == cand.topk_ids[:, 0],
            "gap": golden.topk_lp[:, 0] - golden.topk_lp[:, 1],
            "kl": kl, "missing": missing,
            "dnll": golden.target_lp.double() - cand.target_lp.double(),
            "seq": golden.seq_index()}
    seq_cats = golden.meta["categories"]
    names = list(dict.fromkeys(seq_cats))
    row_cat = torch.tensor([names.index(c) for c in seq_cats])[rows["seq"]]
    by_cat = {name: _summarize({key: v[row_cat == j] for key, v in rows.items()}, golden.k)
              for j, name in enumerate(names)}
    return _summarize(rows, golden.k), by_cat


_PROVENANCE = ("source", "model", "revision", "dtype", "device", "hardware", "threads", "prompts", "prompts_sha256",
               "weight_source", "max_step_tokens", "q8_1", "kv", "moe_block", "experts", "expert_calls")


def _provenance(rec: Recording) -> dict:
    return {key: rec.meta.get(key) for key in _PROVENANCE}


def _path_failures(floor: dict, cand: dict) -> list[str]:
    """A floor emulates the candidate's KV format (none: the model dtype), or it would hide KV bugs (a quantized floor)
    or fail a correct quantized cache (an exact one). A Tier B floor emulates one kernel path (`q8_1`) for every
    expert matmul: the candidate's expert calls while scoring must all have taken it (an MoE floor's candidate must
    have counted them)."""
    if floor.get("kv") != cand.get("kv"):
        return [f"the floor's KV format is {floor.get('kv') or 'the model dtype'}, the candidate's "
                f"{cand.get('kv') or 'the model dtype'}"]
    path, calls = floor.get("q8_1"), cand.get("expert_calls")
    if not path or not floor.get("moe_block"):
        return []
    if not calls or not calls.get(path):
        return [f"no expert calls counted on {path}, the floor's path"]
    return [f"{n} expert calls on {other}, the floor emulates {path}" for other, n in calls.items()
            if other != path and n]


def paired_dppl(golden: Recording, cand: Recording, floor: Recording) -> tuple[float, float]:
    """The candidate's perplexity change against the floor's at the same positions, exp(mean(NLL_c − NLL_f)) − 1,
    and its standard error with sequences as clusters (the ratio estimator `_summarize` uses for ΔPPL)."""
    d = floor.target_lp.double() - cand.target_lp.double()
    _, seq = golden.seq_index().unique(return_inverse=True)
    s, n = int(seq.max()) + 1, len(d)
    d_s = torch.zeros(s, dtype=torch.float64).index_add_(0, seq, d)
    n_s = torch.zeros(s, dtype=torch.float64).index_add_(0, seq, torch.ones_like(d))
    mean = d_s.sum().item() / n
    se = math.sqrt(s / (s - 1) * ((d_s - mean * n_s) ** 2).sum().item()) / n if s > 1 else math.nan
    return math.expm1(mean), se * math.exp(mean)


def compare(golden: Recording, cand: Recording, floor: Recording | None = None) -> Result:
    """The L2 check. Without a floor only the metrics are reported; with one, budgets and a verdict."""
    overall, categories = metrics(golden, cand)
    floor_m = budgets = paired = None
    failures: list[str] = []
    if floor is not None:
        floor_m = metrics(golden, floor)[0]
        paired = paired_dppl(golden, cand, floor)
        budgets = Budgets.from_floor(floor_m, paired[1])
        failures = budgets.failures(overall, paired[0]) + _path_failures(floor.meta, cand.meta)
    return Result(_provenance(golden), _provenance(cand), overall, categories, floor_m, budgets, tuple(failures),
                  paired)


def _margins(golden: Recording, rec: Recording) -> torch.Tensor:
    """Per position, `rec`'s margin between the golden's top-2 ids (nan where `rec`'s top-k lacks either)."""
    out = []
    for j in (0, 1):
        match = rec.topk_ids == golden.topk_ids[:, j:j + 1]
        lp = torch.where(match, rec.topk_lp, torch.nan).nansum(-1).where(match.any(-1), torch.nan)
        out.append(lp)
    return out[0] - out[1]


def compare_drafter(golden: Recording, cand: Recording, floor: Recording) -> Result:
    """The L0-drafter check (docs/m04-spec.md §5): pass iff the candidate's top-1 agreement at confident positions —
    those whose golden top-2 margin exceeds the floor's 99th-percentile perturbation of that margin (how far a correct
    bf16 path moves it) — is at least the floor's − 0.25 points there, and its mean KL at most 2× the floor's. Draft
    heads are judged on few positions, where L2's all-position top-1 rule turns on a row or two at near-ties; L2's
    metrics and verdict are reported alongside."""
    result = compare(golden, cand, floor)
    margin = golden.topk_lp[:, 0] - golden.topk_lp[:, 1]
    shift = (_margins(golden, floor) - margin).abs()
    threshold = torch.quantile(shift[~shift.isnan()].double(), 0.99).item()
    confident = margin > threshold
    agree = lambda rec: (golden.topk_ids[confident, 0] == rec.topk_ids[confident, 0]).double().mean().item()  # noqa
    within = torch.arange(len(margin)) - golden.offsets[golden.seq_index()]       # a row's place in its sequence
    missed = confident & (golden.topk_ids[:, 0] != cand.topk_ids[:, 0])
    numbers = {"margin": threshold, "n_confident": int(confident.sum()), "top1": agree(cand),
               "confident_disagreements_by_row": torch.bincount(within[missed], minlength=int(within.max()) + 1
                                                                ).tolist(),
               "floor_top1": agree(floor), "top1_min": agree(floor) - TOP1_DELTA, "kl_mean": result.overall.kl_mean,
               "kl_mean_max": KL_FACTOR * result.floor.kl_mean, "l2_failures": list(result.failures)}
    failures = [msg for ok, msg in (
        (numbers["top1"] >= numbers["top1_min"], f"top-1 at confident positions {numbers['top1']:.4%} < "
                                                 f"{numbers['top1_min']:.4%}"),
        (numbers["kl_mean"] <= numbers["kl_mean_max"], f"KL mean {numbers['kl_mean']:.3e} > "
                                                       f"{numbers['kl_mean_max']:.3e}")) if not ok]
    return dataclasses.replace(result, failures=tuple(failures), drafter=numbers)
