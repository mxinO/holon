"""L0 for the offload store's in-graph contracts (docs/v1-offload.md §5, D3, D4): `moe_resolve@1`, the placement
plugins' variants `moe_resolve.lfu@1` (F11) and `moe_resolve.lru@1` (F4) with their `lend`, and `expert_fetch@1` —
from kernels.expert_io (CUDA) and the CPU providers (testing.ref_kernels; each placement plugin's own) against their
oracles, exactly.

Resolve at E = 128, k = 8, two roles, 1 to 2049 tokens: decode, prefill and padding rows mixed, dropped routes
(negative ids), experts resident in one role and missing in the other, the null slots shared by every padding row and
dropped route of a rank; past its staging slots (the miss count raised, overflowing misses on the last staging
slot); corrupt map entries (null slots read, the RANGE bit set). The variants besides: whole experts resident or
missing, states full of ties, calls without decode rows (nothing changes), a full delta log (OVERFLOW); every output,
the map, the state and the log compared. Fetch: units gathered from a mapped host pool's device view (zero-copy) and
from device memory into the slots the list names, only the first n of the list, nothing else written."""
import importlib

import numpy as np
import pytest
import torch
from holon.core import platform
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_std.moe import reference
from holon_std.moe.kernels import DECODE, EXPERT_FETCH, MOE_RESOLVE, PADDING, PREFILL, RANGE

E, K, ROLES = 128, 8, 2
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]
VARIANTS = {"lfu": ("score",), "lru": ("last", "clock")}


@pytest.fixture(scope="module")
def plugins():
    return PluginSet(discover())


def kernel(plugins, device: str, op: str, provider: str, **facts):
    ctx = BuildContext(plugins, torch.device(device))
    spec = ctx.select_kernel(op, site="test", **facts)
    assert spec.provider == provider
    return spec.impl(ctx, **facts)


def routing(tokens: int, seed: int, whole: bool = False) -> dict[str, torch.Tensor]:
    """Unique ids per row (some dropped), row kinds of every sort, a map with about half the pieces resident (in
    distinct slots past the null and staging slots, per role; `whole`: both roles of an expert alike, but for a few)
    and host units a permutation."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(tokens)]).int()
    ids[torch.rand(tokens, K, generator=g) < 0.05] = -1                      # dropped routes
    kinds = torch.tensor([DECODE, PREFILL, PADDING], dtype=torch.uint8)[torch.randint(0, 3, (tokens,), generator=g)]
    residency = torch.full((E, ROLES), -1, dtype=torch.int32)
    resident = torch.rand(E, generator=g) < 0.5
    for r in range(ROLES):
        here = resident ^ (torch.rand(E, generator=g) < 0.04) if whole else torch.rand(E, generator=g) < 0.5
        residency[here, r] = (K + 200 + torch.randperm(1000, generator=g)[:int(here.sum())]).int()
    host = torch.stack([torch.randperm(E, generator=g) for _ in range(ROLES)], 1).int()
    counts = torch.randint(0, 50, (E,), generator=g).int()
    limits = torch.full((ROLES,), K + 1200, dtype=torch.int32)
    return {"ids": ids, "kinds": kinds, "residency": residency, "host": host, "limits": limits, "counts": counts}


def _log(capacity: int, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.full((capacity, 3), -9, dtype=torch.int32, device=device),
            torch.full((1,), n, dtype=torch.int32, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tokens, staging", [(1, 8), (5, 40), (64, E), (513, E), (2049, E), (64, 16)])
def test_resolve_matches_the_oracle(plugins, device, tokens, staging):
    """At 64 tokens two resident entries are corrupt (in the staging slots; past the arena): their entries read
    null slots and the RANGE bit is set."""
    provider = "testing.ref_kernels" if device == "cpu" else "kernels.expert_io"
    resolve = kernel(plugins, device, MOE_RESOLVE, provider, experts=E, top_k=K, roles=ROLES, staging=staging)
    case = routing(tokens, seed=tokens + staging)
    corrupt = tokens == 64
    if corrupt:                                    # two routed experts' slots: inside staging, past the arena
        live = case["ids"][case["kinds"] != PADDING]
        e1, e2 = live[live >= 0][:2].tolist()
        case["residency"][e1, 0], case["residency"][e2, 1] = K + 1, K + 1200
    slots, fetch, n_fetch, increments, seen, bits = reference.moe_resolve(
        case["ids"], case["kinds"], case["residency"], case["host"], case["limits"], staging)
    dev = {name: t.clone().to(device) for name, t in case.items()}
    misses = torch.full((1,), 3, dtype=torch.int32, device=device)
    errors = torch.zeros(1, dtype=torch.int32, device=device)
    delta, n_delta = _log(4, 2, device)
    routed = torch.zeros(E, dtype=torch.int32, device=device)
    routed[0] = 1                                                            # the step's earlier calls' needs stay
    got = resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"], dev["counts"], routed,
                  misses, errors, delta, n_delta, 5)
    assert int(errors[0]) == bits == (RANGE if corrupt else 0)
    assert torch.equal(got[0].cpu(), slots) and torch.equal(got[2].cpu(), n_fetch)
    for r in range(ROLES):
        n = min(int(n_fetch[r]), staging)
        assert torch.equal(got[1][r, :, :n].cpu(), fetch[r, :, :n])
    assert torch.equal(dev["counts"].cpu(), case["counts"] + increments)
    assert torch.equal(routed.cpu(), seen | (torch.arange(E) == 0).int())
    assert int(misses[0]) == max(3, int(n_fetch.max()))
    assert torch.equal(dev["residency"].cpu(), case["residency"]) and int(n_delta[0]) == 2 and (delta == -9).all()
    # the contract's own terms, on the oracle: null slot j for padding rows and dropped routes, staging past them
    null = (case["kinds"][:, None] == PADDING) | (case["ids"] < 0)
    assert (slots[:, null] == torch.arange(K).expand(tokens, K)[null]).all()
    if not corrupt:
        assert ((slots[:, ~null] >= K) & ((slots[:, ~null] < K + staging) | (slots[:, ~null] >= K + 200))).all()
    assert (n_fetch > staging).any() == (staging == 16)            # the overflow case overflows


def _state(policy: str, seed: int) -> dict[str, torch.Tensor]:
    """Scores on a coarse grid (ties everywhere, and equal after a decay), or stamps with ties and a clock."""
    g = torch.Generator().manual_seed(seed)
    if policy == "lfu":
        return {"score": (torch.randint(0, 6, (E,), generator=g) * 0.5).float()}
    return {"last": torch.randint(0, 6, (E,), generator=g).int(), "clock": torch.tensor([7], dtype=torch.int32)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", sorted(VARIANTS))
@pytest.mark.parametrize("tokens, kinds", [(1, "decode"), (6, "decode"), (12, "mixed"), (300, "mixed"),
                                           (64, "prefill"), (6, "overflow")])
def test_resolve_variants_match_their_oracles(plugins, device, policy, tokens, kinds):
    """A variant's every output and write: slots, fetch list (admitted misses into their victims' slots), counts,
    misses, the map, the state, the delta log after the records already in it — and the overflow case, whose log
    holds one record more than it has room for (the records past it dropped, OVERFLOW raised). 300 tokens route to
    every expert: no victim, nothing admitted; a prefill chunk counts nothing."""
    op = f"moe_resolve.{policy}@1"
    provider = f"placement.{policy}" if device == "cpu" else "kernels.expert_io"
    resolve = kernel(plugins, device, op, provider, experts=E, top_k=K, roles=ROLES, staging=E)
    oracle = getattr(importlib.import_module(f"holon_plugins.placement.{policy}.reference"), f"moe_resolve_{policy}")
    case, state = routing(tokens, seed=tokens + len(kinds), whole=True), _state(policy, tokens)
    if kinds == "decode":
        case["kinds"][:] = DECODE
    elif kinds in ("prefill", "overflow"):
        case["kinds"][:] = DECODE if kinds == "overflow" else PREFILL
    want = oracle(case["ids"], case["kinds"], case["residency"], case["host"], case["limits"], E, *state.values(), 3)
    capacity = 2 + (len(want["records"]) - 1 if kinds == "overflow" else E)
    delta, n_delta = _log(capacity, 2, device)
    dev = {name: t.clone().to(device) for name, t in (case | state).items()}
    misses, errors = torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device)
    routed = torch.zeros(E, dtype=torch.int32, device=device)
    slots, fetch, n_fetch = resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"],
                                    dev["counts"], routed, misses, errors, delta, n_delta, 3,
                                    **{n: dev[n] for n in state})
    assert torch.equal(slots.cpu(), want["slots"]) and torch.equal(n_fetch.cpu(), want["n_fetch"])
    for r in range(ROLES):
        assert torch.equal(fetch[r, :, :int(n_fetch[r])].cpu(), want["fetch"][r, :, :int(n_fetch[r])])
    assert torch.equal(dev["counts"].cpu(), case["counts"] + want["counts"]) and int(misses[0]) == n_fetch.max()
    assert torch.equal(routed.cpu(), want["routed"])
    assert torch.equal(dev["residency"].cpu(), want["residency"])
    for name in state:
        assert torch.equal(dev[name].cpu(), want[name].to(dev[name].dtype)), name
    records = torch.tensor(want["records"], dtype=torch.int32).view(-1, 3)
    assert int(n_delta[0]) == 2 + len(records) and torch.equal(delta[2:2 + len(records)].cpu(), records[:capacity - 2])
    assert int(errors[0]) == (2 if kinds == "overflow" else 0)          # OVERFLOW
    # the contract's own terms, on the oracle
    counted = kinds != "prefill"
    assert bool(len(records)) == (tokens < 300 and counted) and bool(want["counts"].sum() > 0) == counted
    admitted = {a for _, a, _ in want["records"]}
    for _, a, b in want["records"]:          # whole experts: a took b's slot in every role, b now in the RAM tier
        assert torch.equal(want["residency"][a], case["residency"][b]) and (want["residency"][b] == -1).all()
        assert not ((case["ids"] == b) & (case["kinds"][:, None] != PADDING)).any()      # b routed by no live entry
        for r in range(ROLES):
            row = (want["fetch"][r, 0, :int(want["n_fetch"][r])] == case["host"][a, r]).nonzero()
            assert want["fetch"][r, 1, row].item() == case["residency"][b, r]   # gathered straight into its slot
    staged = want["fetch"][:, 1][want["fetch"][:, 1] >= 0]
    assert (staged[staged < K + E] >= K).all() and len(admitted) == len(records)


def test_lfu_decays_then_counts_in_f32():
    """D4's order on one expert: × 0.915, then + 1.0 per decode entry (rounded each time in f32); a prefill row's
    entry neither decays nor counts; equal scores pair by expert id; a candidate only strictly hotter is admitted."""
    from holon_plugins.placement.lfu import reference as lfu

    residency = torch.full((E, ROLES), -1, dtype=torch.int32)
    residency[[4, 5, 6]] = torch.tensor([[K + 1, K + 2], [K + 3, K + 4], [K + 5, K + 6]], dtype=torch.int32)
    limits, host = torch.full((ROLES,), K + 9, dtype=torch.int32), torch.zeros(E, ROLES, dtype=torch.int32)
    score = torch.zeros(E)
    score[[1, 2, 4, 5, 6]] = torch.tensor([1.0, 0.0, 0.5, 1.915, 0.5])
    ids = torch.tensor([[1, 2], [1, 3], [7, 2]], dtype=torch.int32)
    kinds = torch.tensor([DECODE, DECODE, PREFILL], dtype=torch.uint8)
    out = lfu.moe_resolve_lfu(ids, kinds, residency, host, limits, 1, score, 0)
    f = np.float32
    assert out["score"][1] == f(f(f(1.0) * f(0.915)) + f(1.0)) + f(1.0) and out["score"][7] == 0
    # candidates 1 (2.915), 2, 3 (1.0, a tie: 2 first) against victims 4, 6 (0.4575, a tie: 4 first), 5 (1.752…)
    assert out["records"] == [(0, 1, 4), (0, 2, 6)]                 # 3 (1.0) is not hotter than 5: the pairing stops
    assert out["residency"][1].tolist() == [K + 1, K + 2] and out["residency"][2].tolist() == [K + 5, K + 6]
    assert out["residency"][4].tolist() == out["residency"][6].tolist() == [-1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("policy", sorted(VARIANTS))
@pytest.mark.parametrize("n", [0, 1, 7, 10_000])
def test_lend_matches_the_oracle(plugins, policy, n):
    """`lend` picks whole experts only, coldest first (ties by layer, then expert id) until every class has min(n,
    its resident slots) lent: 6 layers over 3 classes (the second role's class differs by layer), states with ties,
    and some experts resident in one role only (never lent)."""
    resolve = kernel(plugins, "cuda", f"moe_resolve.{policy}@1", "kernels.expert_io", experts=E, top_k=K, roles=ROLES,
                     staging=E)
    oracle = importlib.import_module(f"holon_plugins.placement.{policy}.reference").lend
    g = torch.Generator().manual_seed(n)
    residency = torch.where(torch.rand(6, E, 1, generator=g) < 0.4, 50, -1).int().expand(6, E, ROLES).clone()
    residency[torch.rand(6, E, ROLES, generator=g) < 0.03] = -1
    classes = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 1], [0, 2], [0, 2]])
    state = {n_: torch.stack([_state(policy, 6 * n + layer)[n_] for layer in range(6)]) for n_ in VARIANTS[policy]}
    want = oracle(n, residency, classes, *state.values())
    got = resolve.lend(n, residency.cuda(), classes.cuda(), **{k: v.cuda() for k, v in state.items()})
    assert torch.equal(got.cpu(), want)
    whole = (residency >= 0).all(-1)
    assert not (want & ~whole).any()
    lent = torch.bincount(classes[:, None, :].expand(6, E, ROLES)[want].reshape(-1), minlength=3)
    held = torch.bincount(classes[:, None, :].expand(6, E, ROLES)[whole].reshape(-1), minlength=3)
    assert (lent >= held.clamp(max=n)).all() and (n or not want.any())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mapped", [True, False])
def test_fetch_gathers_the_listed_units(plugins, device, mapped):
    if mapped and device == "cpu":
        pytest.skip("a mapped host pool is the CUDA case")
    provider = "testing.ref_kernels" if device == "cpu" else "kernels.expert_io"
    unit, staging, n = 4096 + 48, 6, 4
    fetch_ = kernel(plugins, device, EXPERT_FETCH, provider, unit_bytes=unit, staging=staging)
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (10, unit), dtype=torch.uint8, generator=g)
    if mapped:
        host = platform.host_memory(data.numel(), torch.device(device), mapped=True)
        host.view(10, unit).copy_(data)
        src = platform.device_alias(host, torch.device(device)).view(10, unit)
    else:
        src = data.to(device)
    dst = torch.randint(0, 256, (K + staging + 30, unit), dtype=torch.uint8, generator=g).to(device)
    units = torch.tensor([[7, 2, 9, 2, 0, 5], [K + 1, K + 29, K, K + 3, K + 4, K + 5]], dtype=torch.int32)
    want = reference.expert_fetch(units, n, data, dst.cpu())
    fetch_(units.to(device), torch.tensor([n], dtype=torch.int32, device=device), src, dst)
    assert torch.equal(dst.cpu(), want)
    assert torch.equal(want[K + 29], data[2]) and not torch.equal(want[K + 4], data[0])
