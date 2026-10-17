"""placement.promote (F10, docs/v1-offload.md D1, D3, §13: promote-on-fetch into 3 device-owned spare slots per layer):
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

OP = "moe_resolve.promote@1"
PROVIDERS = {"cpu": "placement.promote", "cuda": "kernels.promote"}


def _state(case: dict, steady: bool, seed: int) -> dict[str, torch.Tensor]:
    """Stamps with ties and a clock; `steady`: two spares beside an absent one — the first owned by a missing expert a
    decode row routes (it takes its spare back if promoted), the last by one no row routes where there is one — else
    every spare absent (a seed's)."""
    g = torch.Generator().manual_seed(seed)
    state = {"last": torch.randint(0, 6, (E,), generator=g).int(), "clock": torch.tensor([7], dtype=torch.int32),
             "spare": torch.zeros(reference.SPARES, 1 + ROLES, dtype=torch.int32)}
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
                                           (64, "prefill"), (6, "overflow")])
def test_resolve_matches_the_oracle(plugins, device, steady, sources, tokens, kinds):  # noqa: F811
    """Every output and write: slots, the fetch list (staged misses by id, then the promoted into their slots), counts,
    misses, the map, the empty slots left, the state (stamps, clock, spares, set-up), the delta log after the records
    already in it — and the overflow case, one record short. A promoted owner takes its spare back unlogged, the others
    take the empty slots (with a ring), then the spares, logged; each absent spare (all of them at a seed) is refilled
    from the least recently routed resident; a prefill chunk changes nothing but @1's; the CPU's share (F5b) comes from
    the misses left after promoting."""
    resolve = kernel(plugins, device, OP, PROVIDERS[device], experts=E, top_k=K, roles=ROLES, staging=E)
    ring = sources == "ring"
    case = routing(tokens, seed=tokens + len(kinds) + steady, whole=True, ring=ring)
    if kinds in ("decode", "overflow"):
        case["kinds"][:] = DECODE
    elif kinds == "prefill":
        case["kinds"][:] = PREFILL
    state, stage = _state(case, steady, tokens), _stage(case, E, ring)
    plan, jobs = _share(tokens, device, kinds == "mixed")
    want = reference.moe_resolve_promote(case["ids"], case["kinds"], case["residency"], case["host"], case["limits"],
                                         E, *state.values(), 3, case["ring"], stage, case["empty"],
                                         int(case["n_empty"][0]), plan.cpu())
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
    for name in state:
        assert torch.equal(dev[name].cpu(), want[name].to(dev[name].dtype)), name
    records = torch.tensor(want["records"], dtype=torch.int32).view(-1, 3)
    assert int(n_delta[0]) == 2 + len(records) and torch.equal(delta[2:2 + len(records)].cpu(), records[:capacity - 2])
    assert int(errors[0]) == (2 if overflow else 0)                       # OVERFLOW
    # the contract's own terms, on the oracle
    decode = kinds != "prefill"
    if not decode:
        assert torch.equal(want["residency"], case["residency"]) and not records.numel()
        return
    spare = want["spare"]
    owners = spare[:, 0][spare[:, 0] > 0] - 1
    assert len(owners) == reference.SPARES and (want["residency"][owners] == -1).all()   # refilled, unmapped
    resident = (want["residency"] >= 0).all(-1)        # refilled by the least recently routed
    assert want["last"][resident].min() >= want["last"][int(spare[1, 0]) - 1]
    before = {int(o) - 1 for o in state["spare"][:, 0] if o > 0}
    assert not {a for _, a, _ in want["records"]} & before       # an owner takes its spare back, unlogged
    if steady:
        o = int(state["spare"][0, 0]) - 1
        assert not (want["residency"][o] >= 0).all() or (want["residency"][o] == K + 1250).all()
    for _, a, _ in want["records"]:          # each promoted expert fetched straight into its slots (where it is,
        held = want["residency"][a]           # or, refilled at once — every expert routed —, the spare it owns)
        held = held if (held >= 0).all() else spare[spare[:, 0] == a + 1][0, 1:]
        for r in range(ROLES):
            row = (want["fetch"][r, 0, :int(want["n_fetch"][r])] == case["host"][a, r]).nonzero()
            assert want["fetch"][r, 1, row].item() == held[r]


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
                               "promoted": int((record["records"][:, 1] >= 0).sum())})
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
    want = reference.replay(trace["seed"], trace["steps"], store.class_table)
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
    """F10 end to end on the CPU providers, half the experts cached (each layer: 3 spares, 1 resident once set up),
    prompts streamed through a ring of lent slots and staged in lent slots (mixed steps too, `chunk_with_decode`):
    outputs bitwise the resident store's; the residency and empty slots the replay oracle's at every step; promotions
    into spares and into the empty slots lending returned."""
    checkpoint, prompts = tiny_moe_deep
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    want = _served(_engine(file), prompts)
    engine = _engine(file, host_memory=1 << 24, chunk_with_decode=4, overlap=False, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.placement": "placement.promote@1",
        "moe.ring_tokens": 10})
    store = engine.ctx.shared_objects[KEY]
    with _routes(engine) as trace:
        assert _served(engine, prompts) == want
    assert store.borrowed["ring"] and store.borrowed["stage"], store.borrowed
    assert sum(s["promoted"] for s in trace["steps"]) > 0 and (store.state["spare"][:, :, 0] > 0).all()
    assert _parity(store, trace) is None


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
