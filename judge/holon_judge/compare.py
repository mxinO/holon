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
            rows += [f"L0-drafter (§5, §12): {d['n_confident']} of {d['n']} positions confident (top-1 margin > "
                     f"{d['logit_shift_p99']:.3e}, the floor's p99 |Δlogit|)"]
            rows += [f"  top-1 {w}: {d[w]['top1']:.3%} vs floor {d[w]['floor_top1']:.3%} − {d[w]['margin']:.3%}"
                     for w in ("overall", "confident")]
            rows += [f"  {name}: {d[name][0]:.3e} ≤ {d[name][1]:.3e}" for name in ("kl_p50", "kl_p99", "kl_mean")]
            rows += [f"  confident disagreements by row of a draft: {d['confident_disagreements_by_row']}",
                     "  L2's rule, reported only: " + ("pass" if not d["l2_failures"] else "; ".join(d["l2_failures"]))]
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
    # (A masked distribution's refused tokens hold -inf logprobs: p = 0 there contributes nothing.)
    kl = torch.where(p > 0, p * (g_lp - log_q), 0.0).sum(-1) + torch.where(p_tail > 0, p_tail * (g_tail - log_q_tail),
                                                                           0.0)
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
               "weight_source", "max_step_tokens", "q8_1", "act", "kv", "moe_block", "experts", "calls", "head_calls")


def _provenance(rec: Recording) -> dict:
    return {key: rec.meta.get(key) for key in _PROVENANCE}


def _path_failures(floor: dict, cand: dict) -> list[str]:
    """A floor emulates the candidate's KV format (none: the model dtype), or it would hide KV bugs (a quantized floor)
    or fail a correct quantized cache (an exact one). A Tier B floor emulates one kernel path (`q8_1`) in one
    activation mode (`act`, default q8_1) for every quantized matmul: the candidate's calls while scoring must all
    have taken that `path/mode` (docs/m06-lowp.md D3: a run is scored against the floor of the mode it ran), and be
    counted. A quantized LM head's calls (`head_calls`) may take either path — its rows are the prompt-logprob slices
    and the sampled rows, not the step's — but must take the floor's mode. A recording made before M0.6 counted
    expert calls by path (`expert_calls`), all q8_1."""
    if floor.get("kv") != cand.get("kv"):
        return [f"the floor's KV format is {floor.get('kv') or 'the model dtype'}, the candidate's "
                f"{cand.get('kv') or 'the model dtype'}"]
    if not floor.get("q8_1"):
        return []
    want = f"{floor['q8_1']}/{floor.get('act') or 'q8_1'}"
    calls = cand.get("calls") or {f"{path}/q8_1": n for path, n in (cand.get("expert_calls") or {}).items()}
    return ([] if calls.get(want) else [f"no calls counted on {want}, the floor's path and mode"]) + [
        f"{n} calls on {other}, the floor emulates {want}" for other, n in calls.items() if other != want and n] + [
        f"{n} LM-head calls on {k}, the floor's mode is {want.split('/')[1]}"   # the head: any path, the floor's mode
        for k, n in (cand.get("head_calls") or {}).items() if n and k.split("/")[1] != want.split("/")[1]]


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


def compare(golden: Recording, cand: Recording, floor: Recording | None = None,
            ceiling: float | None = None) -> Result:
    """The L2 check. Without a floor only the metrics are reported; with one, budgets and a verdict: the path and mode
    check first (a run scored against another mode's floor fails whatever its numbers), then the budgets. `ceiling`:
    Tier A's sanity ceiling on KL p99 (fp32 throughout, M0.2 §4.6), also checked."""
    overall, categories = metrics(golden, cand)
    floor_m = budgets = paired = None
    failures: list[str] = []
    if floor is not None:
        floor_m = metrics(golden, floor)[0]
        paired = paired_dppl(golden, cand, floor)
        budgets = Budgets.from_floor(floor_m, paired[1])
        failures = _path_failures(floor.meta, cand.meta) + budgets.failures(overall, paired[0])
    if ceiling is not None and overall.kl_p99 > ceiling:
        failures.append(f"KL p99 {overall.kl_p99:.3e} > the sanity ceiling {ceiling:.0e}")
    return Result(_provenance(golden), _provenance(cand), overall, categories, floor_m, budgets, tuple(failures),
                  paired)


def _logit_shift(golden: Recording, floor: Recording) -> torch.Tensor:
    """Per position, the floor's largest |Δlogit| over the golden's top-k ids its own top-k holds: their logprob
    differences, centred (a logprob is a logit less the row's log-normaliser, which centring removes)."""
    match = golden.topk_ids[:, :, None] == floor.topk_ids[:, None, :]                  # [N, k golden, k floor]
    found = match.any(-1)
    d = (floor.topk_lp[:, None, :] * match).sum(-1) - golden.topk_lp                      # where found
    centre = (d * found).sum(-1, keepdim=True) / found.sum(-1, keepdim=True).clamp_min(1)
    return ((d - centre).abs() * found).amax(-1)


def compare_drafter(golden: Recording, cand: Recording, floor: Recording) -> Result:
    """The L0-drafter check (docs/m04-spec.md §5, as decided in §12): a draft head is judged on its top-1 agreement
    with the golden, overall and at confident positions — the golden's top-1 margin above the floor's 99th-percentile
    |Δlogit| (`_logit_shift`) — each at least the floor's less max(0.25 points, 2.5 paired standard errors of the
    difference, prompts as clusters, capped at √2 × the floor's own: `bounded`'s guard), and on KL p50, p99 and
    mean, each at most 2× the floor's. A fixed 0.25-point bar alone fails a correct head at the floor's rate of
    near-tie flips too often at the few thousand positions a draft head has. L2's verdict is reported alongside, and
    the confident disagreements by row of a draft (block row, chain step)."""
    from .bounded import SE_FACTOR, Mean

    result = compare(golden, cand, floor)
    margin = golden.topk_lp[:, 0] - golden.topk_lp[:, 1]
    threshold = torch.quantile(_logit_shift(golden, floor).double(), 0.99).item()
    confident = margin > threshold
    ids = [str(i).split("@")[0] for i in golden.meta.get("seq_ids", range(len(golden.offsets) - 1))]   # cuts → prompt
    prompt = torch.tensor([list(dict.fromkeys(ids)).index(i) for i in ids])[golden.seq_index()]
    agree = {name: (golden.topk_ids[:, 0] == rec.topk_ids[:, 0]).double() for name, rec in (("cand", cand),
                                                                                           ("floor", floor))}
    numbers, failures = {"logit_shift_p99": threshold, "n": len(margin), "n_confident": int(confident.sum())}, []
    for where, rows in (("overall", torch.ones_like(confident)), ("confident", confident)):
        c, f = agree["cand"][rows], agree["floor"][rows]
        # The paired error capped at √2 × the floor's own, so a candidate's own disagreements cannot widen its bar
        se = min(Mean.of(c - f, prompt[rows]).se, math.sqrt(2) * Mean.of(f, prompt[rows]).se)
        bar = max(TOP1_DELTA, SE_FACTOR * se)
        numbers[where] = {"top1": c.mean().item(), "floor_top1": f.mean().item(), "margin": bar}
        if c.mean().item() < f.mean().item() - bar:
            failures.append(f"top-1 {where} {c.mean().item():.4%} < {f.mean().item() - bar:.4%}")
    for name in ("kl_p50", "kl_p99", "kl_mean"):
        ours, theirs = getattr(result.overall, name), getattr(result.floor, name)
        numbers[name] = (ours, KL_FACTOR * theirs)
        if ours > KL_FACTOR * theirs:
            failures.append(f"{name} {ours:.3e} > {KL_FACTOR * theirs:.3e}")
    within = torch.arange(len(margin)) - golden.offsets[golden.seq_index()]               # a row's place in its draft
    missed = confident & (agree["cand"] == 0)
    numbers["confident_disagreements_by_row"] = torch.bincount(within[missed], minlength=int(within.max()) + 1).tolist()
    numbers["l2_failures"] = list(result.failures)
    return dataclasses.replace(result, failures=tuple(failures), drafter=numbers)
