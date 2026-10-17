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
from holon_std.moe.kernels import DECODE, EXPERT_FETCH, EXPERT_PREFETCH, MOE_RESOLVE, PADDING, PREFILL, RANGE

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


def routing(tokens: int, seed: int, whole: bool = False, ring: bool = False) -> dict[str, torch.Tensor]:
    """Unique ids per row (some dropped), row kinds of every sort, a map with about half the pieces resident (in
    distinct slots past the null and staging slots, per role; `whole`: both roles of an expert alike, but for a few)
    and host units a permutation. `ring`: a third of the missing pieces streamed into ring slots, misses staged into a
    lent list of slots, and the layer's empty slots, some of them still free; else none, and the staging slots."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(tokens)]).int()
    ids[torch.rand(tokens, K, generator=g) < 0.05] = -1                      # dropped routes
    kinds = torch.tensor([DECODE, PREFILL, PADDING], dtype=torch.uint8)[torch.randint(0, 3, (tokens,), generator=g)]
    residency = torch.full((E, ROLES), -1, dtype=torch.int32)
    resident = torch.rand(E, generator=g) < 0.5
    for r in range(ROLES):
        here = resident ^ (torch.rand(E, generator=g) < 0.04) if whole else torch.rand(E, generator=g) < 0.5
        residency[here, r] = (K + 200 + torch.randperm(1000, generator=g)[:int(here.sum())]).int()
    streamed = (residency < 0) & (torch.rand(E, ROLES, generator=g) < 0.33) & ring
    case = {"ids": ids, "kinds": kinds, "residency": residency,
            "host": torch.stack([torch.randperm(E, generator=g) for _ in range(ROLES)], 1).int(),
            "limits": torch.full((ROLES,), K + 2000, dtype=torch.int32),
            "counts": torch.randint(0, 50, (E,), generator=g).int(),
            "ring": torch.where(streamed, K + 1300 + torch.randperm(E * ROLES, generator=g).view(E, ROLES), -1).int(),
            "empty": (K + 1600 + torch.randperm(40, generator=g)[:8]).expand(ROLES, 8).int().clone(),
            "n_empty": torch.tensor([3 if ring else 0], dtype=torch.int32)}
    return case


def _stage(case: dict, staging: int, ring: bool) -> torch.Tensor:
    """The slots staged misses go to: lent slots (a shuffled run past the ring's) or the staging slots."""
    if not ring:
        return (K + torch.arange(staging)).expand(ROLES, staging).int().contiguous()
    g = torch.Generator().manual_seed(staging)
    return (K + 1700 + torch.stack([torch.randperm(E, generator=g) for _ in range(ROLES)])).int()


def _log(capacity: int, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.full((capacity, 3), -9, dtype=torch.int32, device=device),
            torch.full((1,), n, dtype=torch.int32, device=device))


SOURCES = ("staging", "ring")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sources", SOURCES)
@pytest.mark.parametrize("tokens, staging", [(1, 8), (5, 40), (64, E), (513, E), (2049, E), (64, 16)])
def test_resolve_matches_the_oracle(plugins, device, sources, tokens, staging):
    """At 64 tokens two resident entries are corrupt (in the staging slots; past the arena): their entries read
    null slots and the RANGE bit is set. With a ring, its pieces are read in place (no fetch), and the misses go to the
    lent list."""
    provider = "testing.ref_kernels" if device == "cpu" else "kernels.expert_io"
    resolve = kernel(plugins, device, MOE_RESOLVE, provider, experts=E, top_k=K, roles=ROLES, staging=staging)
    ring = sources == "ring"
    case = routing(tokens, seed=tokens + staging, ring=ring)
    stage = _stage(case, staging, ring)
    corrupt = tokens == 64
    if corrupt:                                    # two routed experts' slots: inside staging, past the arena
        live = case["ids"][case["kinds"] != PADDING]
        e1, e2 = live[live >= 0][:2].tolist()
        case["residency"][e1, 0], case["residency"][e2, 1] = K + 1, K + 2000
    slots, fetch, n_fetch, increments, seen, bits = reference.moe_resolve(
        case["ids"], case["kinds"], case["residency"], case["host"], case["limits"], staging, case["ring"], stage)
    dev = {name: t.clone().to(device) for name, t in (case | {"stage": stage}).items()}
    misses = torch.full((1,), 3, dtype=torch.int32, device=device)
    errors = torch.zeros(1, dtype=torch.int32, device=device)
    delta, n_delta = _log(4, 2, device)
    routed = torch.zeros(E, dtype=torch.int32, device=device)
    routed[0] = 1                                                            # the step's earlier calls' needs stay
    got = resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"], dev["counts"], routed,
                  dev["ring"], dev["stage"], dev["empty"], dev["n_empty"], misses, errors, delta, n_delta, 5)
    assert int(errors[0]) == bits == (RANGE if corrupt else 0)
    assert torch.equal(got[0].cpu(), slots) and torch.equal(got[2].cpu(), n_fetch)
    for r in range(ROLES):
        n = min(int(n_fetch[r]), stage.shape[1])
        assert torch.equal(got[1][r, :, :n].cpu(), fetch[r, :, :n])
    assert torch.equal(dev["counts"].cpu(), case["counts"] + increments)
    assert torch.equal(routed.cpu(), seen | (torch.arange(E) == 0).int())
    assert int(misses[0]) == max(3, int(n_fetch.max()))
    assert torch.equal(dev["residency"].cpu(), case["residency"]) and int(n_delta[0]) == 2 and (delta == -9).all()
    # the contract's own terms, on the oracle: null slot j for padding rows and dropped routes, staging past them
    null = (case["kinds"][:, None] == PADDING) | (case["ids"] < 0)
    assert (slots[:, null] == torch.arange(K).expand(tokens, K)[null]).all()
    if not corrupt:
        lo, live = K + 1700 if ring else K, slots[:, ~null]
        assert (((live >= lo) & (live < lo + stage.shape[1])) | (live >= K + 200)).all()
    assert (n_fetch > stage.shape[1]).any() == (staging == 16 and not ring)       # the overflow case overflows
    if ring and tokens > 1:
        assert (n_fetch < torch.tensor([len({e for e in case["ids"][~null].tolist() if case["residency"][e, r] < 0})
                                        for r in range(ROLES)])).all()          # the ring's pieces are not fetched


def _state(policy: str, seed: int) -> dict[str, torch.Tensor]:
    """Scores on a coarse grid (ties everywhere, and equal after a decay), or stamps with ties and a clock."""
    g = torch.Generator().manual_seed(seed)
    if policy == "lfu":
        return {"score": (torch.randint(0, 6, (E,), generator=g) * 0.5).float()}
    return {"last": torch.randint(0, 6, (E,), generator=g).int(), "clock": torch.tensor([7], dtype=torch.int32)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", sorted(VARIANTS))
@pytest.mark.parametrize("sources", SOURCES)
@pytest.mark.parametrize("tokens, kinds", [(1, "decode"), (6, "decode"), (12, "mixed"), (300, "mixed"),
                                           (64, "prefill"), (6, "overflow")])
def test_resolve_variants_match_their_oracles(plugins, device, policy, sources, tokens, kinds):
    """A variant's every output and write: slots, fetch list (admitted misses into their victims' or the layer's empty
    slots), counts, misses, the map, the empty slots left, the state, the delta log after the records already in it —
    and the overflow case, whose log holds one record more than it has room for (the records past it dropped,
    OVERFLOW raised). 300 tokens route to every expert: no victim (with a ring, only empty slots), a prefill chunk
    counts nothing. With a ring, a candidate streamed into it is still admitted (and fetched into its new slot)."""
    op = f"moe_resolve.{policy}@1"
    provider = f"placement.{policy}" if device == "cpu" else "kernels.expert_io"
    resolve = kernel(plugins, device, op, provider, experts=E, top_k=K, roles=ROLES, staging=E)
    oracle = getattr(importlib.import_module(f"holon_plugins.placement.{policy}.reference"), f"moe_resolve_{policy}")
    ring = sources == "ring"
    case, state = routing(tokens, seed=tokens + len(kinds), whole=True, ring=ring), _state(policy, tokens)
    stage = _stage(case, E, ring)
    if kinds == "decode":
        case["kinds"][:] = DECODE
    elif kinds in ("prefill", "overflow"):
        case["kinds"][:] = DECODE if kinds == "overflow" else PREFILL
    want = oracle(case["ids"], case["kinds"], case["residency"], case["host"], case["limits"], E, *state.values(), 3,
                  case["ring"], stage, case["empty"], int(case["n_empty"][0]))
    capacity = 2 + (len(want["records"]) - 1 if kinds == "overflow" else E)
    delta, n_delta = _log(capacity, 2, device)
    dev = {name: t.clone().to(device) for name, t in (case | state | {"stage": stage}).items()}
    misses, errors = torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device)
    routed = torch.zeros(E, dtype=torch.int32, device=device)
    routed[:3] = 3                                     # an earlier call's marks (scope included) stay
    slots, fetch, n_fetch = resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"],
                                    dev["counts"], routed, dev["ring"], dev["stage"], dev["empty"], dev["n_empty"],
                                    misses, errors, delta, n_delta, 3, **{n: dev[n] for n in state})
    assert torch.equal(slots.cpu(), want["slots"]) and torch.equal(n_fetch.cpu(), want["n_fetch"])
    for r in range(ROLES):
        assert torch.equal(fetch[r, :, :int(n_fetch[r])].cpu(), want["fetch"][r, :, :int(n_fetch[r])])
    assert torch.equal(dev["counts"].cpu(), case["counts"] + want["counts"]) and int(misses[0]) == n_fetch.max()
    earlier = torch.zeros(E, dtype=torch.int32)
    earlier[:3] = 3
    assert torch.equal(routed.cpu(), want["routed"] | earlier) and int(dev["n_empty"][0]) == want["n_empty"]
    assert torch.equal(dev["residency"].cpu(), want["residency"])
    for name in state:
        assert torch.equal(dev[name].cpu(), want[name].to(dev[name].dtype)), name
    records = torch.tensor(want["records"], dtype=torch.int32).view(-1, 3)
    assert int(n_delta[0]) == 2 + len(records) and torch.equal(delta[2:2 + len(records)].cpu(), records[:capacity - 2])
    assert int(errors[0]) == (2 if kinds == "overflow" else 0)          # OVERFLOW
    # the contract's own terms, on the oracle
    counted = kinds != "prefill"
    assert bool(len(records)) == ((tokens < 300 or ring) and counted) and bool(want["counts"].sum() > 0) == counted
    assert not (ring and counted) or any(b < -1 for _, _, b in want["records"])       # empty slots taken first
    admitted = {a for _, a, _ in want["records"]}
    for _, a, b in want["records"]:          # whole experts: a took b's slot in every role, b now in the RAM tier
        source = case["empty"][:, -2 - b] if b < 0 else case["residency"][b]
        assert torch.equal(want["residency"][a], source) and (b < 0 or (want["residency"][b] == -1).all())
        assert b < 0 or not ((case["ids"] == b) & (case["kinds"][:, None] != PADDING)).any()   # b routed by no row
        for r in range(ROLES):
            row = (want["fetch"][r, 0, :int(want["n_fetch"][r])] == case["host"][a, r]).nonzero()
            assert want["fetch"][r, 1, row].item() == source[r]                  # gathered straight into its slot
    assert len(admitted) == len(records)


def _lfu_call(plugins, device: str, ids, kinds, resident: dict[int, int], score: dict[int, float]) -> dict:
    """One `moe_resolve.lfu@1` call at top-2 through the device's provider: the experts `resident` maps to their first
    slot (both roles), the rest missing; the map, scores and delta records after it."""
    provider = "placement.lfu" if device == "cpu" else "kernels.expert_io"
    resolve = kernel(plugins, device, "moe_resolve.lfu@1", provider, experts=E, top_k=2, roles=ROLES, staging=1)
    residency = torch.full((E, ROLES), -1, dtype=torch.int32)
    for e, slot in resident.items():
        residency[e] = torch.tensor([slot, slot + 1])
    scores = torch.zeros(E)
    for e, v in score.items():
        scores[e] = v
    dev = {n: t.to(device) for n, t in {"ids": torch.tensor(ids, dtype=torch.int32), "residency": residency,
                                        "kinds": torch.tensor(kinds, dtype=torch.uint8), "score": scores}.items()}
    words = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(3)]
    zeros = {n: torch.zeros(E, *shape, dtype=torch.int32, device=device) for n, shape in
             (("host", (ROLES,)), ("counts", ()), ("routed", ()), ("delta", (3,)))}
    ring = torch.full((E, ROLES), -1, dtype=torch.int32, device=device)
    stage = torch.full((ROLES, 1), 2, dtype=torch.int32, device=device)
    empty = torch.zeros(ROLES, 1, dtype=torch.int32, device=device)
    resolve(dev["ids"], dev["kinds"], dev["residency"], zeros["host"], torch.full((ROLES,), 40, dtype=torch.int32,
            device=device), zeros["counts"], zeros["routed"], ring, stage, empty, words[0].clone(), words[0],
            words[1], zeros["delta"], words[2], 0, score=dev["score"])
    delta = zeros["delta"]
    return {"residency": dev["residency"].cpu(), "score": dev["score"].cpu().numpy(),
            "records": [tuple(r) for r in delta[:int(words[2][0])].cpu().tolist()]}


@pytest.mark.parametrize("device", DEVICES)
def test_lfu_decays_then_counts_in_f32(plugins, device):
    """D4's order, through both providers: × 0.915, then + 1.0 per decode entry, each rounded in f32 (an expert at 1.5
    with 7 decode entries ends at 8.372499, where one add of 7 would give 8.3725); a prefill row's entry neither decays
    nor counts; equal scores pair by expert id; a candidate is admitted only when strictly hotter than its victim (a
    candidate at 0.5 with one entry reaches 1.4575, exactly what a victim at 1.5928961 decays to: not admitted)."""
    f = np.float32
    out = _lfu_call(plugins, device, [[1, 2], [1, 3], [7, 2]], [DECODE, DECODE, PREFILL], {4: 11, 5: 13, 6: 15},
                    {1: 1.0, 2: 0.0, 4: 0.5, 5: 1.915, 6: 0.5})
    assert out["score"][1] == f(f(f(1.0) * f(0.915)) + f(1.0)) + f(1.0) and out["score"][7] == 0
    # candidates 1 (2.915), 2, 3 (1.0, a tie: 2 first) against victims 4, 6 (0.4575, a tie: 4 first), 5 (1.752…)
    assert out["records"] == [(0, 1, 4), (0, 2, 6)]                 # 3 (1.0) is not hotter than 5: the pairing stops
    assert out["residency"][1].tolist() == [11, 12] and out["residency"][2].tolist() == [15, 16]
    assert out["residency"][4].tolist() == out["residency"][6].tolist() == [-1, -1]
    out = _lfu_call(plugins, device, [[9, 1]] * 7, [DECODE] * 7, {}, {9: 1.5})
    assert out["score"][9] == f(8.372499) != f(f(f(1.5) * f(0.915)) + f(7))
    out = _lfu_call(plugins, device, [[2, 3]], [DECODE], {4: 11}, {2: 0.5, 4: 1.5928961})
    assert out["score"][2] == out["score"][4] == f(1.4575) and out["records"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("policy", sorted(VARIANTS))
@pytest.mark.parametrize("need", [(0, 0, 0), (1, 0, 0), (0, 7, 3), (5, 2, 9), (10_000,) * 3])
def test_lend_matches_the_oracle(plugins, policy, need):
    """`lend` picks whole experts only, coldest first (ties by expert id, then layer), for each class c the first
    need[c] holding a slot of c: 6 layers over 3 classes (the second role's class differs by layer), states with ties,
    and some experts resident in one role only (never lent). On the device without a host sync."""
    resolve = kernel(plugins, "cuda", f"moe_resolve.{policy}@1", "kernels.expert_io", experts=E, top_k=K, roles=ROLES,
                     staging=E)
    oracle = importlib.import_module(f"holon_plugins.placement.{policy}.reference").lend
    g = torch.Generator().manual_seed(sum(need))
    residency = torch.where(torch.rand(6, E, 1, generator=g) < 0.4, 50, -1).int().expand(6, E, ROLES).clone()
    residency[torch.rand(6, E, ROLES, generator=g) < 0.03] = -1
    classes = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 1], [0, 2], [0, 2]])
    state = {n_: torch.stack([_state(policy, 6 * sum(need) + layer)[n_] for layer in range(6)])
             for n_ in VARIANTS[policy]}
    want = oracle(np.array(need), residency, classes, *state.values())
    args = torch.tensor(need).cuda(), residency.cuda(), classes.cuda()
    dev = {k: v.cuda() for k, v in state.items()}
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = resolve.lend(*args, **dev)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(got.cpu(), want)
    whole = (residency >= 0).all(-1)
    assert not (want & ~whole).any()
    holds = torch.zeros(6, 3, dtype=torch.bool)
    holds[torch.arange(6)[:, None], classes] = True
    for c in range(3):                          # each class's need met, or every holder lent
        holders = whole & holds[:, c, None]
        assert int((want & holders).sum()) >= min(need[c], int(holders.sum()))


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


def test_prefetch_is_selected_for_its_copy_mode():
    """`expert_prefetch@1`'s `copy` fact reaches selection: a provider declaring one mode only (here a toy above the
    CPU provider's priority, `sm` only) serves that mode and is never selected for the other."""
    from holon.api import Constraint, KernelSpec

    plugins = PluginSet(discover())
    ctx = BuildContext(plugins, torch.device("cpu"))
    ctx.candidates(EXPERT_PREFETCH, {"unit_bytes": 16, "copy": "sm"})          # loads the installed providers
    plugins.regs.kernels.add("expert_prefetch", KernelSpec(
        op="expert_prefetch", provider="tests.sm_only", impl=object, requires=Constraint(devices=frozenset({"cpu"})),
        priority=100, supports=lambda f: f["copy"] == "sm"))
    picked = {copy: ctx.select_kernel(EXPERT_PREFETCH, site=copy, unit_bytes=16, copy=copy).provider
              for copy in ("sm", "dma")}
    assert picked == {"sm": "tests.sm_only", "dma": "testing.ref_kernels"}
    assert ctx.kernels["dma/expert_prefetch@1"]["facts"]["copy"] == "dma"             # the lock records the mode


@pytest.mark.parametrize("device", DEVICES)
def test_prefetch_copies_the_listed_units_before_its_wait(plugins, device):
    """`expert_prefetch@1` (D8's ring): `expert_fetch@1`'s copy from a mapped host pool (the store's allocation), its
    table and count on the device (SM copies) or on the host (copy engines), landed once the current stream has waited
    for the key; nothing else written."""
    provider = "testing.ref_kernels" if device == "cpu" else "kernels.expert_io"
    unit = 4096 + 48
    prefetch, engines = (kernel(plugins, device, EXPERT_PREFETCH, provider, unit_bytes=unit, copy=copy)
                         for copy in ("sm", "dma"))
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, 256, (10, unit), dtype=torch.uint8, generator=g)
    host = platform.host_memory(data.numel(), torch.device(device), mapped=True)
    host.view(10, unit).copy_(data)
    src = platform.device_alias(host, torch.device(device)).view(10, unit)
    dst = torch.randint(0, 256, (30, unit), dtype=torch.uint8, generator=g).to(device)
    fetch = torch.tensor([[7, 8, 9, 2, 0], [3, 4, 5, 20, 11]], dtype=torch.int32)
    want = reference.expert_fetch(fetch, 4, data, dst.cpu())
    dma = dst.clone()
    prefetch.start(5, fetch.to(device), torch.tensor([4], dtype=torch.int32, device=device), src, dst)
    engines.start(6, fetch, torch.tensor([4], dtype=torch.int32), host.view(10, unit), dma)    # the host's list
    prefetch.wait(5)
    engines.wait(6)
    assert torch.equal(dst.cpu(), want) and torch.equal(want[4], data[8]) and not torch.equal(want[11], data[0])
    assert torch.equal(dma.cpu(), want)
