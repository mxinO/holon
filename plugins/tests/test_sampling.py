"""L0-dist: the default sampler draws from exactly the oracle's distribution, reproducibly per (seed, position)."""
import dataclasses

import numpy as np
import pytest
import torch
from holon.api import SamplingParams
from holon.core.plugins import PluginSet, discover

V = 40


@dataclasses.dataclass
class _Plan:
    slots: np.ndarray
    n_logits: np.ndarray
    start_upper: np.ndarray
    n_query: np.ndarray


@dataclasses.dataclass
class _Ctx:
    batch: object


@dataclasses.dataclass
class _Dev:
    slots: torch.Tensor
    positions: torch.Tensor
    logits_index: torch.Tensor
    logits_rows: torch.Tensor


def _ctx(slots, positions):
    """One sampled decode row per entry: the token drawn at `positions[i]` for slot `slots[i]`."""
    n = len(slots)
    plan = _Plan(np.asarray(slots, np.int32), np.ones(n, np.int32), np.asarray(positions, np.int64) - 1,
                 np.ones(n, np.int32))
    dev = _Dev(torch.as_tensor(plan.slots), torch.as_tensor(plan.start_upper), torch.arange(n), torch.arange(n))
    return _Ctx(dataclasses.make_dataclass("B", ["plan", "dev"])(plan, dev))


@pytest.fixture(scope="module")
def std():
    plugins = PluginSet(discover())
    plugin = plugins.load("std.sampling")
    import importlib
    return plugin, importlib.import_module("holon_std.sampling.reference")


PARAMS = [SamplingParams(temperature=0.7), SamplingParams(top_k=5), SamplingParams(top_p=0.8),
          SamplingParams(temperature=1.3, min_p=0.1), SamplingParams(temperature=0.9, top_k=12, top_p=0.9, min_p=0.02)]


@pytest.mark.parametrize("params", PARAMS, ids=lambda p: f"T{p.temperature}k{p.top_k}p{p.top_p}m{p.min_p}")
def test_draws_follow_the_oracle(std, params):
    plugin, ref = std
    logits = torch.randn(V, generator=torch.Generator().manual_seed(0)) * 2
    sampler = plugin.sampler(1, torch.device("cpu"))
    sampler.admit(0, dataclasses.replace(params, seed=1234))
    n = 100_000
    tokens = sampler.sample(logits.expand(n, V).contiguous(), _ctx([0] * n, np.arange(n))).tokens
    freq = torch.bincount(tokens, minlength=V).double() / n
    p = ref.probs(logits, params)
    assert (freq[p == 0] == 0).all()                       # filtered tokens are never drawn
    assert 0.5 * (freq - p).abs().sum() < 0.02             # total variation (expected ≈ 0.01 at this n)


def test_draws_depend_only_on_seed_and_position(std):
    plugin, _ = std
    logits = torch.randn(6, V, generator=torch.Generator().manual_seed(1))
    a = plugin.sampler(4, torch.device("cpu"))
    for slot in range(4):
        a.admit(slot, SamplingParams(temperature=1.0, seed=100 + slot))
    together = a.sample(logits, _ctx([0, 1, 2, 3, 0, 1], [5, 5, 5, 5, 9, 9])).tokens
    b = plugin.sampler(8, torch.device("cpu"))                       # other slots, other batch composition
    b.admit(7, SamplingParams(temperature=1.0, seed=101))
    alone = b.sample(logits[[1, 5]], _ctx([7, 7], [5, 9])).tokens
    assert together[[1, 5]].tolist() == alone.tolist()


def test_greedy_and_logprobs(std):
    plugin, _ = std
    logits = torch.randn(3, V, generator=torch.Generator().manual_seed(2))
    s = plugin.sampler(3, torch.device("cpu"))
    s.admit(0, SamplingParams(temperature=0.0, seed=1, logprobs=3))
    s.admit(1, SamplingParams(temperature=1.0, seed=2))
    s.admit(2, SamplingParams(temperature=0.0, seed=3, logprobs=0))
    out = s.sample(logits, _ctx([0, 1, 2], [1, 1, 1]))
    assert out.tokens[[0, 2]].tolist() == logits[[0, 2]].argmax(-1).tolist()
    lp = logits.log_softmax(-1)
    torch.testing.assert_close(out.logprobs, lp.gather(-1, out.tokens[:, None])[:, 0])
    assert out.top_ids[0].tolist() == lp[0].topk(3).indices.tolist()


def test_inverse_cdf_matches_the_oracle_cdf_at_full_vocabulary():
    # Deterministic: for a grid of uniforms, the drawn token is the one whose float64 CDF interval contains u.
    import importlib
    sampler = importlib.import_module("holon_std.sampling.sampler")
    v = 151_936
    q = (torch.randn(1, v, generator=torch.Generator().manual_seed(3)) * 3).softmax(-1)
    cdf = q[0].double().cumsum(0)
    for u in torch.linspace(1e-4, 1 - 1e-4, 257):
        t = int(sampler.draw(q, u.reshape(1).float())[0])
        lo, hi = (cdf[t - 1] if t else 0.0), cdf[t]
        assert lo - 1e-6 <= u.item() * cdf[-1] < hi + 1e-6
