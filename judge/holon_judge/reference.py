"""L2's reference form (docs/v1-glm.md D15a): a lock judged against a bf16 GPU reference, for a model with no fp32
golden and no HF floor (the owner's rule: no fp32 goldens, no big-model CPU runs).

- **The reference** is Holon's own engine running the model all-resident on a card that holds it, scored by `score`
  (the recording a candidate makes), `kind: reference` in its meta, with `oracle`: the model's tiny-model HF match
  (a tracked test or evidence file). A self-reference bounds drift; it does not decide correctness, which the
  oracle and the model's other gates carry.
- **The budgets** are absolute numbers fixed before any candidate is scored (`calibrate`, written beside the
  reference as `<stem>.budgets.json`), from pairs of recordings that should agree — the source pairs (e.g. the
  reference card against another card at equal settings, the other side in its plainest fit) and consistency pairs,
  reported only. Prompts are the unit of the noise (a block bootstrap over prompts: a route flip's cascade correlates
  the positions within one). With the worst source pair: KL p50 and p99 at most `KL_FACTOR` times its own (KL mean
  is reported only: any change of bits saturates it), top-1 at least its own less max(`TOP1_DELTA`, 2.5·√2·SE), and
  the ΔPPL one-sided — the candidate's perplexity at most the reference's × (1 + `DPPL_DELTA` + 2.5·√2·SE); the
  two-sided value is reported. A KL budget is also at least the 99.5th percentile of that pair's bootstrap draws:
  with 24 or 8 prompts a resample's KL sits near 2× the point estimate, so 2× alone fails ≈ 5% of correct runs per
  statistic, the quantile ≈ 1% (the coordinator's call, 2026-10-27, before the source pair was measured). The
  false-fail rate (the source pairs' bootstrap draws judged by the budgets) is
  recorded with them.
- **The comparison** (`judge`) is `compare`'s metrics with the reference in the golden's place and the budgets in the
  floor's; no kernel-path check (two Holon runs differ in path by card, which a cross-card source pair carries), the
  setting check (KV format, CPU split) kept.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from collections.abc import Sequence

import torch

from .compare import (
    DPPL_DELTA,
    DPPL_Z,
    KL_FACTOR,
    TOP1_DELTA,
    Metrics,
    _provenance,
    _row,
    _setting_failures,
    metrics,
    position_rows,
)
from .recording import Recording

SAMPLES = 2000            # bootstrap draws over prompts
KL_QUANTILE = 0.995       # a KL budget is at least this quantile of the worst source pair's draws (≈ 1% false-fail)
SEED = 0
ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256(path: str | pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def budgets_path(reference: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(reference).with_suffix(".budgets.json")


def oracle_failures(meta: dict) -> list[str]:
    """B-m3: a reference names the model's tiny-model HF match, a file tracked in this repository."""
    oracle = meta.get("oracle")
    if meta.get("kind") != "reference":
        return ["not a reference recording (meta kind)"]
    if not oracle or not (ROOT / oracle).is_file():
        return [f"the reference names no tiny-model HF match that exists here (oracle {oracle!r})"]
    return []


def _draws(rows: dict[str, torch.Tensor], samples: int = SAMPLES) -> dict[str, torch.Tensor]:
    """The statistics over `samples` resamples of the prompts (with replacement), each prompt's positions whole."""
    seq = rows["seq"]
    order = torch.argsort(seq, stable=True)
    counts = torch.bincount(seq)
    starts = torch.cumsum(counts, 0) - counts
    g = torch.Generator().manual_seed(SEED)
    out: dict[str, list[float]] = {k: [] for k in ("top1", "kl_p50", "kl_p99", "dppl")}
    for _ in range(samples):
        picks = torch.randint(len(counts), (len(counts),), generator=g)
        pos = order[torch.cat([torch.arange(starts[p], starts[p] + counts[p]) for p in picks.tolist()])]
        kl = rows["kl"][pos]
        p50, p99 = torch.quantile(kl, torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
        out["top1"].append(rows["agree"][pos].double().mean().item())
        out["kl_p50"].append(p50)
        out["kl_p99"].append(p99)
        out["dppl"].append(math.expm1(rows["dnll"][pos].mean().item()))
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in out.items()}


@dataclasses.dataclass(frozen=True)
class Budgets:
    reference: str            # the reference recording's sha256
    prompts: str
    prompts_sha256: str
    kl_p50_max: float
    kl_p99_max: float
    top1_min: float
    dppl_max: float           # one-sided: the candidate's perplexity over the reference's, less 1
    pairs: tuple[dict, ...]   # each calibration pair's numbers, source or check
    false_fail: float         # the share of the source pairs' bootstrap draws these budgets fail
    false_fail_by: dict       # ... by statistic (KL p50, KL p99, top-1, ΔPPL): ≈ 1% or less each by construction

    def failures(self, m: Metrics) -> list[str]:
        checks = [(m.kl_p50 <= self.kl_p50_max, f"KL p50 {m.kl_p50:.3e} > {self.kl_p50_max:.3e}"),
                  (m.kl_p99 <= self.kl_p99_max, f"KL p99 {m.kl_p99:.3e} > {self.kl_p99_max:.3e}"),
                  (m.top1 >= self.top1_min, f"top-1 {m.top1:.4%} < {self.top1_min:.4%}"),
                  (m.dppl <= self.dppl_max, f"ΔPPL {m.dppl:+.3%} > {self.dppl_max:+.3%} (one-sided)")]
        return [msg for ok, msg in checks if not ok]

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Budgets:
        d = json.loads(pathlib.Path(path).read_text())
        return cls(**d | {"pairs": tuple(d["pairs"])})


def _pair(name: str, a: Recording, b: Recording, source: bool) -> tuple[dict, dict[str, torch.Tensor]]:
    m = metrics(a, b)[0]
    rows = position_rows(a, b)
    draws = _draws(rows)
    q = {f"{k}_q995": torch.quantile(draws[k], KL_QUANTILE).item() for k in ("kl_p50", "kl_p99")}
    return ({"name": name, "source": source, "bitwise": a.same_tensors(b), "n": m.n, "top1": m.top1,
             "kl_mean": m.kl_mean, "kl_p50": m.kl_p50, "kl_p99": m.kl_p99, "dppl": m.dppl,
             "se_top1": draws["top1"].std().item(), "se_dppl": draws["dppl"].std().item()} | q, draws)


def calibrate(reference: str | pathlib.Path, sources: Sequence[tuple[str, Recording, Recording]],
              checks: Sequence[tuple[str, Recording, Recording]] = ()) -> Budgets:
    """The budgets from the worst source pair; `checks` are reported with them. A source pair must differ in bits:
    budgets calibrated on no noise would fail every correct run on another card."""
    ref = Recording.load(reference)
    if failed := oracle_failures(ref.meta):
        raise ValueError(f"{reference}: {'; '.join(failed)}")
    for name, a, b in [*sources, *checks]:
        if any(r.meta["prompts_sha256"] != ref.meta["prompts_sha256"] for r in (a, b)):
            raise ValueError(f"pair {name}: another prompt set than the reference's")
    judged = [_pair(n, a, b, True) for n, a, b in sources]
    if not judged or all(p["bitwise"] for p, _ in judged):
        raise ValueError("no source pair differs in bits: nothing to calibrate the budgets from")
    worst = {k: max(max(KL_FACTOR * p[k], p[f"{k}_q995"]) for p, _ in judged) for k in ("kl_p50", "kl_p99")}
    worst |= {
"top1": min(p["top1"] for p, _ in judged), "se_top1": max(p["se_top1"] for p, _ in judged),
              "se_dppl": max(p["se_dppl"] for p, _ in judged)}
    margin = DPPL_Z * math.sqrt(2)
    budgets = Budgets(sha256(reference), ref.meta["prompts"], ref.meta["prompts_sha256"],
                      worst["kl_p50"], worst["kl_p99"],
                      worst["top1"] - max(TOP1_DELTA, margin * worst["se_top1"]),
                      DPPL_DELTA + margin * worst["se_dppl"],
                      tuple(p for p, _ in judged) + tuple(_pair(n, a, b, False)[0] for n, a, b in checks), 0.0, {})
    by = {name: max(fail(d).double().mean().item() for _, d in judged) for name, fail in (
        ("kl_p50", lambda d: d["kl_p50"] > budgets.kl_p50_max), ("kl_p99", lambda d: d["kl_p99"] > budgets.kl_p99_max),
        ("top1", lambda d: d["top1"] < budgets.top1_min), ("dppl", lambda d: d["dppl"] > budgets.dppl_max))}
    fails = [((d["kl_p50"] > budgets.kl_p50_max) | (d["kl_p99"] > budgets.kl_p99_max) | (d["top1"] < budgets.top1_min)
              | (d["dppl"] > budgets.dppl_max)).double().mean().item() for _, d in judged]
    return dataclasses.replace(budgets, false_fail=max(fails), false_fail_by=by)


@dataclasses.dataclass(frozen=True)
class Result:
    reference: dict
    candidate: dict
    overall: Metrics
    categories: dict[str, Metrics]
    budgets: Budgets
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json(self) -> dict:
        return dataclasses.asdict(self) | {"passed": self.passed}

    def table(self) -> str:
        b = self.budgets
        rows = [f"{'':10s} {'n':>6s} {'top-1':>8s} {'top-1 conf':>10s} {'KL p50':>9s} {'KL p99':>9s} {'KL max':>9s} "
                f"{'ΔPPL ± SE':>17s} {'missing':>7s}", _row("overall", self.overall),
                f"{'budget':10s} {'':6s} {'≥' + format(b.top1_min, '.3%'):>8s} {'':10s} "
                f"{'≤' + format(b.kl_p50_max, '.2e'):>9s} {'≤' + format(b.kl_p99_max, '.2e'):>9s} {'':9s} "
                f"{'≤' + format(b.dppl_max, '+.3%'):>17s}",
                f"KL mean {self.overall.kl_mean:.3e} (reported); |ΔPPL| {abs(self.overall.dppl):.3%} (two-sided, "
                f"reported); the budgets' false-fail rate {b.false_fail:.2%} ("
                + ", ".join(f"{k} {v:.2%}" for k, v in b.false_fail_by.items()) + ")"]
        rows += [_row(name, m) for name, m in self.categories.items()]
        where = (f"{self.candidate.get('model')} on {self.candidate.get('gpu_class') or '-'} vs the reference on "
                 f"{self.reference.get('gpu_class') or '-'}")
        verdict = "PASS" if self.passed else "FAIL: " + "; ".join(self.failures)
        return "\n".join([f"L2 (reference form) {where}, prompts {b.prompts}", *rows, verdict])


def judge(reference: Recording, cand: Recording, budgets: Budgets, reference_sha256: str | None = None) -> Result:
    """`cand` against `reference` under its committed `budgets` (D15a)."""
    failures = oracle_failures(reference.meta)
    if reference_sha256 is not None and reference_sha256 != budgets.reference:
        failures.append("the budgets were calibrated for another reference recording")
    if budgets.prompts_sha256 != reference.meta["prompts_sha256"]:
        failures.append("the budgets are another prompt set's")
    overall, categories = metrics(reference, cand)
    failures += _setting_failures(reference.meta, cand.meta) + budgets.failures(overall)
    return Result(_provenance(reference) | {"kind": "reference"}, _provenance(cand), overall, categories, budgets,
                  tuple(failures))
