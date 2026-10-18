"""Routes and residency of the offload store, step by step, and the L1 policy-parity check (docs/v1-offload.md §6).

`route_dump(engine)` taps the engine's offload store (`std.moe.offload@1`'s `std.tiered.store@1`, through its route
taps) while it runs: per committed step, its routing counts [layers, experts] of decode and verify rows, the experts
the call holding its decode rows routed (the resolves' victims' scope, graphed steps included), the delta records the
device logged, and the residency (whole experts) the device's map holds as the step left it (`device_maps`: one copy
per step, judge runs only) — not the host mirror, which knows only what the delta log says (F10's spares are unmapped
without a record, so the mirror would count their owners resident).

At the end, with every launched step committed, the device's map must be the mirror (`Trace.mirrored`): a kernel
that changed the map without logging it shows there.

`parity(trace)` replays a trace recorded under `placement.lfu@1` (F11) with the oracle extended from the M6 replay's
`lfu` to D4's spec (placement.lfu's `reference.replay`) from the residency, slots and empty slots the trace starts
with (a tail step's lending, docs/v1-moe-prefill.md D11, reads the slots), and compares every step's residency and
empty slots (their stacks, as the delta log orders them) with the device's, exactly. A policy bug shows only as lost
speed, since every placement is exact: this is the check that sees it.

`cpu_split(engine)` is F5b's frozen split (§6): the split table and the static placement's experts per layer, what a
path-aware floor needs to know which experts the CPU computed (`holon_judge golden --cpu-split`)."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterator
from unittest import mock

import numpy as np


@contextlib.contextmanager
def device_maps() -> Iterator[Callable[[object, int], np.ndarray]]:
    """`resident(store, step)`: the experts (whole, [groups, items]) a tiered store's device map holds as step `step`
    left it, for a tap at that step's commit. Under overlap the next step has launched by then (on the CPU it has even
    run), so the map is copied on the device at that launch — the store asks its map for the launch's edits before the
    step is enqueued —; a step committed before the next launch reads the map at its commit. One copy per step."""
    from holon_std.tiered.residency import ResidencyMap

    writes, last, left = ResidencyMap.writes, {}, {}

    def copied(self, launch):
        if (step := last.get(id(self))) is not None:          # launched, not committed: the map it left
            left[id(self), step] = self.device.clone()
        last[id(self)] = launch
        return writes(self, launch)

    def resident(store, step: int) -> np.ndarray:
        m = store.residency
        if last.get(id(m)) == step:                          # nothing launched after it
            last[id(m)] = None
            device = m.device
        else:
            device = left.pop((id(m), step), m.device)       # (a step launched before the dump began: as it is now)
        return (device >= 0).all(-1).cpu().numpy()
    with mock.patch.object(ResidencyMap, "writes", copied):
        yield resident


@dataclasses.dataclass
class Trace:
    seed: np.ndarray                                   # [layers, experts] resident at the seed
    steps: list[dict] = dataclasses.field(default_factory=list)   # step, counts, needed, records, resident
    mirrored: bool | None = None                       # at the end: the device's map is the host mirror's
    score: np.ndarray | None = None                    # F11's scores at the start
    classes: np.ndarray | None = None                  # [layers, roles] -> the slot class (lending counts per class)
    slots: np.ndarray | None = None                    # [layers, experts, roles] the arena slots at the start, and
    stacks: list | None = None                         # each layer's empty slots ([roles] each, bottom first), and
    sizes: np.ndarray | None = None                    # the classes' arena slots: what a tail's replay reads (D11)

    @property
    def admissions(self) -> int:
        return sum(int((s["records"][:, 1] >= 0).sum()) for s in self.steps)


@contextlib.contextmanager
def route_dump(engine) -> Iterator[Trace]:
    from holon_std.moe.offload import KEY

    store = engine.ctx.shared_objects[KEY]
    device, n_empty = store.residency.device.cpu().numpy(), store.n_empty.cpu().tolist()
    trace = Trace((device >= 0).all(-1), classes=store.class_table, slots=np.maximum(device, -1),
                  stacks=[list(store.empty[g, :, :n].t().cpu().numpy()) for g, n in enumerate(n_empty)],
                  sizes=np.array([c.size for c in store.classes.values()]))
    if "score" in store.state:
        trace.score = store.state["score"].cpu().numpy().copy()

    def tap(step, record) -> None:
        trace.steps.append({"step": step.step, "counts": record["counts"].copy(), "needed": record["scope"].copy(),
                            "cpu": record["cpu"].copy(),
                            "records": record["records"].copy(), "lends": record["lends"],
                            "resident": resident(store, step.step),
                            "empty": np.array([len(e) for e in store.residency.empty]),
                            "stacks": [np.array(e).reshape(-1, store.roles) for e in store.residency.empty]})
    with device_maps() as resident:
        store.taps.append(tap)
        try:
            yield trace
            if not engine.has_work():                  # every launched step committed: the log told the host all
                device = store.residency.device.cpu().numpy()   # (VRAM slots; tier codes collapsed: the host mirror
                trace.mirrored = np.array_equal(np.maximum(device, -1), np.maximum(store.residency.host, -1))  # none)
        finally:
            store.taps.remove(tap)


def cpu_split(engine) -> dict | None:
    """The engine's F5b split, `{"plan": plan[0 … E], "tokens": the most tokens of a pass with a CPU share, "layers":
    {"<layer>": [resident experts]}}`; None without F5b. Only a frozen placement (`std.tiered.static@1`) has one."""
    from holon_std.moe.offload import CPU, KEY

    share, store = engine.ctx.shared_objects.get(CPU), engine.ctx.shared_objects.get(KEY)
    if share is None:
        return None
    if engine.cfg.max_step_tokens != 1:          # the floor applies the rule per token: one token per pass only
        raise ValueError(f"F5b's floor needs one token per step, not max_step_tokens {engine.cfg.max_step_tokens}")
    if store.policy_ref != "std.tiered.static@1":
        raise ValueError(f"F5b under {store.policy_ref}: a path-aware floor needs a frozen placement (static)")
    resident = (store.residency.host >= 0).all(-1)
    return {"plan": share.plan.tolist(), "tokens": share.tokens, "max_step_tokens": 1,
            "layers": {str(layer): np.flatnonzero(resident[g]).tolist() for g, layer in enumerate(store.layers)}}


def split_digest(split: dict) -> str:
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()[:16]


def parity(trace: Trace) -> dict:
    """The replay against the device, step by step: the first step that differs (None: none), and what was compared."""
    from holon_plugins.placement.lfu.reference import replay

    want = replay(trace.seed, trace.steps, trace.score, trace.classes, trace.slots, trace.sizes, trace.stacks)
    differs = next((s["step"] for w, s in zip(want, trace.steps, strict=True)
                    if not (np.array_equal(w["resident"], s["resident"]) and np.array_equal(w["empty"], s["empty"])
                            and all(map(np.array_equal, w["stacks"], s["stacks"])))), None)
    return {"steps": len(trace.steps), "decode_steps": sum(bool(s["counts"].any()) for s in trace.steps),
            "borrowing_steps": sum(s["lends"] is not None for s in trace.steps),
            "lends": sum(int((np.asarray(s["lends"]) > 0).any(1).sum()) for s in trace.steps if s["lends"] is not None),
            "admissions": trace.admissions, "first_differing_step": differs, "device_is_mirror": trace.mirrored}
