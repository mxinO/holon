"""L2 comparator prototype: candidate teacher-forced top-k logprobs vs an fp32 golden → metrics + pass/fail.

KL(golden ‖ cand) is computed on k+1 buckets: the golden's top-k ids, plus one tail bucket holding the remaining
mass of each distribution. Merging outcomes never increases KL (log-sum inequality), so this is a lower bound on the
full-vocab KL; the gap is the within-tail KL weighted by the golden tail mass, which is small because the golden
top-20 holds almost all of the mass (Qwen3-8B: median tail 1e-6). `kl_validation` in results/noise_floor_*.json
measures the gap against full-vocab logprobs.

A golden id that is missing from the candidate's own top-k gets q = min(candidate k-th prob, leftover / (m+1)),
where leftover is the candidate mass outside its top-k and m the number of missing ids: a feasible value that keeps
the tail bucket non-empty. Engines that can return logprobs at requested ids avoid the imputation entirely.
"""

from dataclasses import asdict, dataclass

import torch

from golden import Recording


@dataclass(frozen=True)
class Budgets:
    top1_min: float     # top-1 agreement ≥
    kl_p50_max: float   # nats
    kl_p99_max: float   # nats
    dppl_max: float     # |relative ΔPPL| ≤ (0.01 = 1 %)

    @classmethod
    def from_floor(cls, floor: "Report", delta: "Budgets") -> "Budgets":
        """Pass rule of design §10.3: agreement ≥ floor − δ; KL p50/p99, ΔPPL ≤ floor + δ."""
        return cls(floor.top1 - delta.top1_min, floor.kl_p50 + delta.kl_p50_max, floor.kl_p99 + delta.kl_p99_max,
                   abs(floor.dppl) + delta.dppl_max)


def proposed_budgets(floor: "Report") -> Budgets:
    """S5 proposal for bf16 engines vs the fp32 golden, floor = HF bf16 (sdpa) vs fp32 on the same prompt set:
    δ_top1 = 0.25 pp, δ_KL = the floor itself (budget = 2× floor), δ_ΔPPL = 0.25 %."""
    return Budgets.from_floor(floor, Budgets(0.0025, floor.kl_p50, floor.kl_p99, 0.0025))


@dataclass
class Report:
    n: int
    top1: float          # argmax agreement
    top1_confident: float  # argmax agreement where the golden top-2 gap ≥ 0.1 nat (near-ties excluded)
    kl_mean: float
    kl_p50: float
    kl_p99: float
    kl_max: float
    dnll: float          # mean NLL(cand) − NLL(golden), nats/token
    dnll_se: float       # sequence-clustered standard error
    dppl: float          # relative ΔPPL = exp(dnll) − 1
    dppl_se: float
    ppl_golden: float
    missing: float       # share of golden top-k ids absent from the candidate's top-k (imputed)
    passed: bool | None = None
    failures: tuple[str, ...] = ()

    def dict(self) -> dict:
        return asdict(self)


def kl_topk(g_ids, g_lp, c_ids, c_lp) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position bucketed KL(golden ‖ cand) and the per-position count of imputed golden ids."""
    g_lp, c_lp = g_lp.double(), c_lp.double()
    match = g_ids[:, :, None] == c_ids[:, None, :]                      # [N, k, kc]
    found = match.any(-1)
    q_found = c_lp.gather(1, match.int().argmax(-1))
    leftover = (1 - c_lp.exp().sum(-1, keepdim=True)).clamp_min(0)
    n_missing = (~found).sum(-1, keepdim=True)
    q_missing = torch.minimum(c_lp[:, -1:].exp(), leftover / (n_missing + 1)).clamp_min(1e-30).log()
    q_lp = torch.where(found, q_found, q_missing)
    p = g_lp.exp()
    p_rest = (1 - p.sum(-1)).clamp_min(0)
    q_rest = (1 - q_lp.exp().sum(-1)).clamp_min(1e-30)
    tail = torch.where(p_rest > 0, p_rest * (p_rest.clamp_min(1e-30).log() - q_rest.log()), 0.0)
    return ((p * (g_lp - q_lp)).sum(-1) + tail).clamp_min(0).float(), n_missing[:, 0]


def kl_full(g_full: torch.Tensor, c_full: torch.Tensor) -> torch.Tensor:
    g, c = g_full.double(), c_full.double()
    return (g.exp() * (g - c)).sum(-1).float()


def position_metrics(golden: Recording, cand: Recording) -> dict[str, torch.Tensor]:
    assert golden.meta["prompts_sha256"] == cand.meta["prompts_sha256"], "different prompt sets"
    assert torch.equal(golden.offsets, cand.offsets)
    kl, missing = kl_topk(golden.topk_ids, golden.topk_lp, cand.topk_ids, cand.topk_lp)
    seq = torch.repeat_interleave(torch.arange(len(golden.offsets) - 1), golden.offsets.diff())
    return dict(agree=golden.topk_ids[:, 0] == cand.topk_ids[:, 0],
                gap=golden.topk_lp[:, 0] - golden.topk_lp[:, 1],
                kl=kl, missing=missing,
                dnll=(golden.target_lp.double() - cand.target_lp.double()),
                nll=-golden.target_lp.double(), seq=seq)


def summarize(m: dict[str, torch.Tensor], k: int, mask: torch.Tensor | None = None) -> Report:
    if mask is not None:
        m = {name: v[mask] for name, v in m.items()}
    n = len(m["kl"])
    confident = m["gap"] >= 0.1
    # Positions within a sequence are correlated: SE of the ratio estimator with sequences as clusters.
    seqs, inv = m["seq"].unique(return_inverse=True)
    d_s = torch.zeros(len(seqs), dtype=torch.float64).index_add_(0, inv, m["dnll"])
    n_s = torch.zeros(len(seqs), dtype=torch.float64).index_add_(0, inv, torch.ones(n, dtype=torch.float64))
    dnll = (d_s.sum() / n_s.sum()).item()
    s = len(seqs)
    se = ((s / max(s - 1, 1)) * ((d_s - dnll * n_s) ** 2).sum()).sqrt().item() / n_s.sum().item()
    dppl = torch.tensor(dnll).exp().item() - 1
    q = torch.quantile(m["kl"].double(), torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
    return Report(n=n, top1=m["agree"].double().mean().item(),
                  top1_confident=m["agree"][confident].double().mean().item(),
                  kl_mean=m["kl"].double().mean().item(), kl_p50=q[0], kl_p99=q[1], kl_max=m["kl"].max().item(),
                  dnll=dnll, dnll_se=se, dppl=dppl, dppl_se=se * (1 + dppl),
                  ppl_golden=m["nll"].mean().exp().item(), missing=m["missing"].double().sum().item() / (n * k))


def compare(golden: Recording, cand: Recording, budgets: Budgets | None = None) -> Report:
    """The L2 check: metrics over all positions, plus pass/fail if budgets are given."""
    r = summarize(position_metrics(golden, cand), golden.meta["k"])
    if budgets is not None:
        checks = [("top1", r.top1 >= budgets.top1_min, f"top1 {r.top1:.4f} < {budgets.top1_min:.4f}"),
                  ("kl_p50", r.kl_p50 <= budgets.kl_p50_max, f"KL p50 {r.kl_p50:.2e} > {budgets.kl_p50_max:.2e}"),
                  ("kl_p99", r.kl_p99 <= budgets.kl_p99_max, f"KL p99 {r.kl_p99:.2e} > {budgets.kl_p99_max:.2e}"),
                  ("dppl", abs(r.dppl) <= budgets.dppl_max, f"|ΔPPL| {abs(r.dppl):.2%} > {budgets.dppl_max:.2%}")]
        r.failures = tuple(msg for _, ok, msg in checks if not ok)
        r.passed = not r.failures
    return r
