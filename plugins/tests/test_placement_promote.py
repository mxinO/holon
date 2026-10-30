"""placement.promote (F10, docs/dev/v1-offload.md D1, D3, §13: promote-on-fetch into 3 device-owned spare slots per
layer):
its resolve contract `moe_resolve.promote@1` from both providers — the plugin's CPU provider and its own kernel plugin's
CUDA provider (kernels.promote, around kernels.expert_io's `moe_resolve@1`) — against the oracle, exactly, and its
`lend`; then end to end on the CPU and the GPU: bitwise the resident store, the residency the replay oracle's at every
step (the M6 replay's `promote_on_fetch` in the device's terms), the host mirror the device's map but for the spares'
owners."""
import contextlib
from unittest import mock

import numpy as np
import pytest
import torch
from holon_plugins.placement.promote import reference
from holon_std.moe.kernels import DECODE, PADDING, PREFILL

from .test_expert_io import DEVICES, ROLES, E, K, _jobs, _log, _share, _stage, kernel, plugins, routing  # noqa: F401
from .test_moe_engine import KEY, Q, _engine, _gguf_and_dense, _gpu_engine, _served, gpu_moe, tiny_moe_deep  # noqa: F401

PREBUILT = ("expert_io",)  # CUDA extensions built if missing (conftest)


OP = "moe_resolve.promote@1"
PROVIDERS = {"cpu": "placement.promote", "cuda": "kernels.promote"}


def _state(case: dict, steady: bool, seed: int) -> dict[str, torch.Tensor]:
    """Stamps with ties and a clock; `steady`: two spares beside an absent one — the first owned by a missing expert a
    decode row routes (it takes its spare back if promoted), the last by one no row routes where there is one — else
    every spare absent (a seed's)."""
    g = torch.Generator().manual_seed(seed)
    state = {"last": torch.randint(0, 6, (E,), generator=g).int(), "clock": torch.tensor([7], dtype=torch.int32),
             "spare": torch.zeros(reference.SPARES, 1 + ROLES, dtype=torch.int32),
             "exclusive": torch.zeros(1, dtype=torch.int32)}
    if steady:
        live = case["ids"][(case["kinds"] != PADDING)[:, None] & (case["ids"] >= 0)]
        routed = case["ids"][(case["kinds"] == DECODE)[:, None] & (case["ids"] >= 0)].unique().tolist() or \
            live.unique().tolist()                                                      # (a prefill chunk)
        idle = sorted(set(range(E)) - set(live.tolist()))
        owners = [routed[len(routed) // 2], idle[0] if idle else routed[-1]]          # (300 tokens route all)
        for s, o in zip((0, 2), owners, strict=True):
            case["residency"][o] = -1
            state["spare"][s] = torch.tensor([o + 1, *[K + 1250 + s] * ROLES])
    return state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("steady", [False, True])
@pytest.mark.parametrize("sources", ["staging", "ring"])
@pytest.mark.parametrize("tokens, kinds", [(1, "decode"), (6, "decode"), (12, "mixed"), (300, "mixed"),
                                           (64, "prefill"), (6, "overflow"), (64, "narrow"), (6, "exclusive"),
                                           (12, "mixed exclusive"), (6, "exclusive budget"),
                                           (12, "mixed exclusive budget"), (6, "exclusive budget empty")])
def test_resolve_matches_the_oracle(plugins, device, steady, sources, tokens, kinds):  # noqa: F811
    """Every output and write: slots, the fetch list (staged misses by id, then the promoted into their slots), counts,
    misses, the map, the empty slots left, the state (stamps, clock, spares), the delta log after the records already
    in it — and the overflow case, one record short; the narrow case's staged and promoted misses pass its 16 fetch
    entries (all counted, only the first 16 written). A promoted owner takes its spare back unlogged, the others
    take the empty slots (with a ring), then the spares, logged; each absent spare (all of them at a seed) is refilled
    from the least recently routed resident; a prefill chunk changes nothing but @1's; the CPU's share (F5b) comes from
    the misses left after promoting. Under exclusive tiers (the policy's `exclusive` flag, docs/dev/v1-glm.md D8a) a
    refilled spare's owner stays mapped and is logged `(group, SPARE, b)`, an owner taking its spare back
    `(group, OWN, b)`: `moe_tiers@2` places them. With the `budget` fact 1 (docs/dev/v1-glm.md D8t; a third of the
    experts SSD-only here): the spares' owners, then every SSD miss by first entry, then the one RAM miss used most
    recently before the call's stamp (two or more with different stamps), within the empty slots and spares — a RAM
    miss filling an empty slot displaces nothing and is not counted (`empty`: the 8 empty slots a lending step leaves
    when its loan comes back, RAM misses filling them past the budget)."""
    budget = 1 if "budget" in kinds else None
    resolve = kernel(plugins, device, OP, PROVIDERS[device], experts=E, top_k=K, roles=ROLES, staging=E,
                     **({} if budget is None else {"budget": budget}))
    ring = sources == "ring"
    case = routing(tokens, seed=tokens + len(kinds) + steady, whole=True, ring=ring)
    exclusive = "exclusive" in kinds
    if kinds in ("decode", "overflow", "narrow", "exclusive", "exclusive budget", "exclusive budget empty"):
        case["kinds"][:] = DECODE
    elif kinds == "prefill":
        case["kinds"][:] = PREFILL
    if budget is not None:                             # SSD-only experts: no host unit in any role
        case["host"][torch.arange(E) % 3 == 0] = -1
    if "empty" in kinds:                               # after a lending step: its slots back, empty
        case["n_empty"][0] = case["empty"].shape[1]
    state, stage = _state(case, steady, tokens), _stage(case, E, ring)
    state["exclusive"][0] = int(exclusive)
    if kinds == "narrow":
        stage = stage[:, :16].contiguous()
    plan, jobs = _share(tokens, device, kinds == "mixed")
    want = reference.moe_resolve_promote(case["ids"], case["kinds"], case["residency"], case["host"], case["limits"],
                                         E, state["last"], state["clock"], state["spare"], 3, case["ring"], stage,
                                         case["empty"], int(case["n_empty"][0]), plan.cpu(), exclusive, budget=budget)
    overflow = kinds == "overflow" and bool(want["records"])
    capacity = 2 + (len(want["records"]) - 1 if overflow else E)
    delta, n_delta = _log(capacity, 2, device)
    dev = {name: t.clone().to(device) for name, t in (case | state | {"stage": stage}).items()}
    misses, errors = torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device)
    routed = torch.zeros(E, dtype=torch.int32, device=device)
    routed[:3] = 3                                     # an earlier call's marks (scope included) stay
    slots, fetch, n_fetch, cpu = resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"],
                                         dev["counts"], routed, dev["ring"], dev["stage"], dev["empty"],
                                         dev["n_empty"], misses, errors, delta, n_delta, 3, plan, jobs,
                                         **{n: dev[n] for n in state})
    assert torch.equal(slots.cpu(), want["slots"]) and torch.equal(n_fetch.cpu(), want["n_fetch"])
    assert torch.equal(cpu.cpu(), want["cpu"])
    _jobs(jobs, want["jobs"])
    for r in range(ROLES):
        assert torch.equal(fetch[r, :, :int(n_fetch[r])].cpu(), want["fetch"][r, :, :int(n_fetch[r])])
    assert torch.equal(dev["counts"].cpu(), case["counts"] + want["counts"]) and int(misses[0]) == n_fetch.max()
    earlier = torch.zeros(E, dtype=torch.int32)
    earlier[:3] = 3
    assert torch.equal(routed.cpu(), want["routed"] | earlier) and int(dev["n_empty"][0]) == want["n_empty"]
    assert torch.equal(dev["residency"].cpu(), want["residency"])
    for name in ("last", "clock", "spare"):
        assert torch.equal(dev[name].cpu(), want[name].to(dev[name].dtype)), name
    records = torch.tensor(want["records"], dtype=torch.int32).view(-1, 3)
    assert int(n_delta[0]) == 2 + len(records) and torch.equal(delta[2:2 + len(records)].cpu(), records[:capacity - 2])
    assert int(errors[0]) == (2 if overflow else 0)                       # OVERFLOW
    # the contract's own terms, on the oracle
    if budget is not None:
        promoted = [a for _, a, _ in want["records"] if a >= 0]
        before = {int(o) - 1 for o in state["spare"][:, 0] if o > 0}
        missing = (case["residency"] < 0).all(-1)
        ram = [a for a in promoted if (case["host"][a] >= 0).all() and a not in before]
        routed = set(case["ids"][(case["kinds"] == DECODE)[:, None] & (case["ids"] >= 0)].tolist())
        cands = [e for e in routed if missing[e] and (case["host"][e] >= 0).all() and e not in before]
        n_ssd = sum(1 for e in routed if missing[e] and not (case["host"][e] >= 0).all() and e not in before)
        fill = max(0, int(case["n_empty"][0]) - n_ssd)                     # empty slots left to RAM misses: exempt
        assert len(ram) <= fill + budget                                    # one RAM promotion over a resident
        room = int(case["n_empty"][0]) + len(before) - len(set(promoted) & before) - n_ssd   # slots left to them
        assert "empty" not in kinds or len(ram) == min(len(cands), fill + budget, room) > budget
        assert tokens < 12 or len({int(state["last"][e]) for e in cands}) > 1   # (recency decides at 12 tokens)
        if ram:
            assert ram[0] == min(cands, key=lambda e: (-int(state["last"][e]), e))
        ssd = [a for a in promoted if not (case["host"][a] >= 0).all()]
        assert promoted[:len(ssd)] == ssd or set(promoted) & before          # SSD misses before the RAM one
    decode = kinds != "prefill"
    assert kinds != "narrow" or (want["n_fetch"] > 16).all()
    if not decode:
        assert torch.equal(want["residency"], case["residency"]) and not records.numel()
        return
    spare = want["spare"]
    owners = spare[:, 0][spare[:, 0] > 0] - 1
    assert len(owners) == reference.SPARES                         # refilled: unmapped, or (exclusive) still mapped
    assert (want["residency"][owners] == -1).all() if not exclusive else \
        {b for _, a, b in want["records"] if a == reference.SPARE} <= set(owners.tolist())
    resident = (want["residency"] >= 0).all(-1)        # refilled by the least recently routed
    assert want["last"][resident].min() >= want["last"][int(spare[1, 0]) - 1]
    before = {int(o) - 1 for o in state["spare"][:, 0] if o > 0}
    assert not {a for _, a, _ in want["records"]} & before       # an owner takes its spare back, unlogged
    if steady:
        o = int(state["spare"][0, 0]) - 1
        assert not (want["residency"][o] >= 0).all() or (want["residency"][o] == K + 1250).all()
    for _, a, _ in (x for x in want["records"] if x[1] >= 0):   # each promoted expert fetched straight into its
        held = want["residency"][a]           # slots (where it is, or, refilled at once — every expert routed —,
        #                                       the spare it owns)
        held = held if (held >= 0).all() else spare[spare[:, 0] == a + 1][0, 1:]
        for r in range(ROLES):
            src = int(case["host"][a, r]) if case["host"][a, r] >= 0 else -2 - a       # (SSD-only: −2 − a)
            row = (want["fetch"][r, 0, :int(want["n_fetch"][r])] == src).nonzero()
            assert want["fetch"][r, 1, row].item() == held[r] if len(row) else want["n_fetch"][r] > len(stage[r])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("need", [(0, 0, 0), (5, 2, 9), (10_000,) * 3])
def test_lend_matches_the_oracle(plugins, need):  # noqa: F811
    """`lend` picks whole experts least recently routed first (ties by expert id, then layer), for each class c the
    first need[c] holding a slot of c, never a spare's owner (unmapped); on the device without a host sync."""
    resolve = kernel(plugins, "cuda", OP, "kernels.promote", experts=E, top_k=K, roles=ROLES, staging=E)
    g = torch.Generator().manual_seed(sum(need))
    residency = torch.where(torch.rand(6, E, 1, generator=g) < 0.4, 50, -1).int().expand(6, E, ROLES).clone()
    residency[torch.rand(6, E, ROLES, generator=g) < 0.03] = -1
    classes = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 1], [0, 2], [0, 2]])
    last = torch.randint(0, 6, (6, E), generator=g).int()
    state = {"clock": torch.zeros(6, 1, dtype=torch.int32), "spare": torch.zeros(6, reference.SPARES, 1 + ROLES,
                                                                                  dtype=torch.int32)}
    want = reference.lend(np.array(need), residency, classes, last, *state.values())
    args, dev = (torch.tensor(need).cuda(), residency.cuda(), classes.cuda(), last.cuda()), \
        {n: t.cuda() for n, t in state.items()}
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = resolve.lend(*args, **dev)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(got.cpu(), want) and not (want & ~(residency >= 0).all(-1)).any()


@pytest.mark.parametrize("device", DEVICES)
def test_lend_tail_releases_the_lending_layers_spares(plugins, device):  # noqa: F811
    """`lend_tail` (docs/dev/v1-glm.md D8a; step 1's review S1-M1): every spare of a layer with a role of the tail's
    class is released — absent, its owner mapped back to its slots and picked with the range's holders, so the caller
    lends it and returns its slots at the step's end (none leaks, none held from the step's staging) —, the other
    layers' kept; each provider as the oracle (kernels.promote's own device ops, no host sync)."""
    resolve = kernel(plugins, device, OP, PROVIDERS[device], experts=E, top_k=K, roles=ROLES, staging=E)
    g = torch.Generator().manual_seed(4)
    limits = torch.full((6, ROLES), K + E + 200, dtype=torch.int32)
    residency = (K + E + torch.randperm(190, generator=g)[:E]).int()[None, :, None].expand(6, E, ROLES).clone()
    residency[:, :reference.SPARES] = -1                             # the spares' owners: unmapped
    classes = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 1], [0, 2], [0, 2]])
    spare = torch.zeros(6, reference.SPARES, 1 + ROLES, dtype=torch.int32)
    spare[:, :, 0] = torch.arange(1, 1 + reference.SPARES)
    spare[:, :, 1:] = torch.tensor([K + E + 195, K + E + 190, K + E + 199])[None, :, None]   # in, out, in the tail
    want_res, want_spare, held = residency.clone(), spare.clone(), spare[..., 1].clone()
    want = reference.release_tail(1, 8, want_res, classes, limits, want_spare)
    dev = {n: t.to(device) for n, t in (("residency", residency), ("classes", classes), ("limits", limits),
                                        ("spare", spare))}
    if device == "cuda":
        torch.cuda.set_sync_debug_mode("error")
    try:
        got = resolve.lend_tail(1, 8, dev["residency"], dev["classes"], dev["limits"], spare=dev["spare"],
                                last=torch.zeros(6, E, dtype=torch.int32, device=device))
    finally:
        if device == "cuda":
            torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(got.cpu(), want) and torch.equal(dev["spare"].cpu(), want_spare)
    assert torch.equal(dev["residency"].cpu(), want_res)
    lending = torch.tensor([True, True, False, True, False, False])
    assert (want_spare[lending, :, 0] == 0).all() and (want_spare[~lending, :, 0] > 0).all()
    assert want[lending, :reference.SPARES].all() and not want[~lending, :reference.SPARES].any()
    assert torch.equal(want_res[lending, :reference.SPARES, 0], held[lending])     # owners on their slots
    tail = ((classes[:, None] == 1) & (want_res >= (limits - 8)[:, None])).any(-1)
    assert torch.equal(want, tail | (lending[:, None] & (torch.arange(E) < reference.SPARES)))


@contextlib.contextmanager
def _routes(engine):
    """Per committed step: each layer's decode routes in the order of their first entry (read at its resolves: eager
    steps only), the step's `lend` needs, and the device's residency (whole experts) and empty slots after it (the
    synchronous path only: under overlap the next step has launched by a commit)."""
    from holon_std.tiered.resolver import DeviceResolver

    store = engine.ctx.shared_objects[KEY]
    trace = {"seed": (store.residency.host >= 0).all(-1), "steps": []}
    calls, call = [[] for _ in store.layers], DeviceResolver.__call__

    def resolve(self, ids, first, *args, **kw):
        decode = ids.cpu()[store.kinds[first:first + len(ids)].cpu() == DECODE]       # token, then rank
        calls[self.group] += [e for e in decode.view(-1).tolist() if e >= 0]
        return call(self, ids, first, *args, **kw)

    def tap(step, record) -> None:
        trace["steps"].append({"order": [list(dict.fromkeys(c)) for c in calls], "lends": record["lends"],
                               "resident": (store.residency.device >= 0).all(-1).cpu().numpy(),
                               "empty": store.n_empty.cpu().numpy().copy(),
                               "promoted": int((record["records"][:, 1] >= 0).sum()),
                               "into_empty": int(((record["records"][:, 1] >= 0) &
                                                  (record["records"][:, 2] <= -2)).sum())})
        for c in calls:
            c.clear()
    store.taps.append(tap)
    try:
        with mock.patch.object(DeviceResolver, "__call__", resolve):
            yield trace
    finally:
        store.taps.remove(tap)


def _parity(store, trace) -> int | None:
    """The first step whose residency or empty slots differ from the replay oracle's (None: none); and the host mirror
    is the device's map but for the spares' owners, which it maps at their spares' slots."""
    want = reference.replay(trace["seed"], trace["steps"], store.class_table,
                            budget=store.policy.resolve_facts.get("budget"))
    device, host = store.residency.device.cpu().numpy(), store.residency.host.copy()
    for g, rows in enumerate(store.state["spare"].cpu().numpy()):
        for o, *slots in rows[rows[:, 0] > 0]:
            assert (host[g, o - 1] == slots).all()
            host[g, o - 1] = -1
    assert np.array_equal(np.maximum(device, -1), np.maximum(host, -1))
    return next((n for n, (w, s) in enumerate(zip(want, trace["steps"], strict=True))
                 if not (np.array_equal(w["resident"], s["resident"]) and np.array_equal(w["empty"], s["empty"]))),
                None)


def test_promote_serves_the_resident_stores_outputs(tiny_moe_deep, tmp_path):  # noqa: F811
    """F10 end to end on the CPU providers, three quarters of the experts cached (each layer: 3 spares once set up),
    prompts streamed through a ring of lent slots and staged in lent slots (mixed steps too, `chunk_with_decode`):
    outputs bitwise the resident store's; the residency and empty slots the replay oracle's at every step; promotions
    into spares and into the empty slots lending returned. The judge's `route_dump` records each step's residency
    from the device's map, as `_routes` does: the host mirror would count the spares' owners resident."""
    from holon_judge.placement import route_dump

    checkpoint, prompts = tiny_moe_deep
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    want = _served(_engine(file), prompts)
    engine = _engine(file, host_memory=1 << 24, chunk_with_decode=4, overlap=False, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.75", "moe.placement": "placement.promote@1",
        "moe.ring_tokens": 10})            # (three quarters: a layout lends only past the spares, step 7)
    store = engine.ctx.shared_objects[KEY]
    with _routes(engine) as trace, route_dump(engine) as dump:
        assert _served(engine, prompts) == want
    assert store.borrowed["ring"] and store.borrowed["stage"], store.borrowed
    assert all(np.array_equal(a["resident"], b["resident"]) for a, b in zip(trace["steps"], dump.steps, strict=True))
    assert not dump.mirrored and np.array_equal(dump.seed, trace["seed"])
    assert sum(s["into_empty"] for s in trace["steps"]) > 0 and (store.state["spare"][:, :, 0] > 0).all()
    assert sum(s["promoted"] for s in trace["steps"]) > sum(s["into_empty"] for s in trace["steps"])
    assert _parity(store, trace) is None


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_promote_on_the_gpu_is_bitwise_the_resident_store(gpu_moe):  # noqa: F811
    """F10 on the native path (kernels.promote around kernels.expert_io's resolve, the zero-copy gather into spares,
    decode steps replayed from CUDA graphs, prefill chunked past the mmvq limit): the streams and prompt logprobs are
    the resident store's, bit for bit, at a quarter and half of the experts cached; on the synchronous eager path the
    residency is the replay oracle's at every step."""
    file, prompts, want = gpu_moe
    options = {"moe.store": "std.moe.offload@1", "moe.placement": "placement.promote@1"}
    for cache in ("*=0.25", "*=0.5"):
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=options | {"moe.cache": cache})
        store = engine.ctx.shared_objects[KEY]
        promoted = []
        store.taps.append(lambda step, record, n=promoted: n.append(int((record["records"][:, 1] >= 0).sum())))
        assert _served(engine, prompts) == want, cache
        assert engine.stats()["graphs"] > 0 and sum(promoted) > 0
        assert {(k.rpartition("/")[2], v["provider"]) for k, v in engine.ctx.kernels.items()
                if "moe_resolve" in k} == {(OP, "kernels.promote"), ("moe_resolve@1", "kernels.expert_io")}
        engine.close()
    engine = _gpu_engine(file, host_memory=1 << 27, cuda_graphs=False, overlap=False,
                         plugin_options=options | {"moe.cache": "*=0.5"})
    store = engine.ctx.shared_objects[KEY]
    with _routes(engine) as trace:
        assert _served(engine, prompts) == want
    assert _parity(store, trace) is None and sum(s["promoted"] for s in trace["steps"]) > 0
    engine.close()
