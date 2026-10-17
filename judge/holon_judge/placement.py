"""Routes and residency of the offload store, step by step, and the L1 policy-parity check (docs/v1-offload.md §6).

`route_dump(engine)` taps the engine's offload store (`std.moe.offload@1`'s `std.tiered.store@1`, through its route
taps) while it runs: per committed step, its routing counts [layers, experts] of decode and verify rows, the experts
the call holding its decode rows routed (the resolves' victims' scope, graphed steps included), the delta records the
device logged, and the residency (whole experts) the store's host mirror holds after the step — the device's map,
from its delta log.

At the end, with every launched step committed, the device's map must be the mirror (`Trace.mirrored`): a kernel
that changed the map without logging it shows there.

`parity(trace)` replays a trace recorded under `placement.lfu@1` (F11) with the oracle extended from the M6 replay's
`lfu` to D4's spec (placement.lfu's `reference.replay`) from the seeded residency, and compares every step's
residency with the device's, exactly. A policy bug shows only as lost speed, since every placement is exact: this is
the check that sees it."""
from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator

import numpy as np


@dataclasses.dataclass
class Trace:
    seed: np.ndarray                                   # [layers, experts] resident at the seed
    steps: list[dict] = dataclasses.field(default_factory=list)   # step, counts, needed, records, resident
    mirrored: bool | None = None                       # at the end: the device's map is the host mirror's
    score: np.ndarray | None = None                    # F11's scores at the start
    classes: np.ndarray | None = None                  # [layers, roles] -> the slot class (lending counts per class)

    @property
    def admissions(self) -> int:
        return sum(int((s["records"][:, 1] >= 0).sum()) for s in self.steps)


@contextlib.contextmanager
def route_dump(engine) -> Iterator[Trace]:
    from holon_std.moe.offload import KEY

    store = engine.ctx.shared_objects[KEY]
    trace = Trace((store.residency.host >= 0).all(-1).copy(), classes=store.class_table)
    if "score" in store.state:
        trace.score = store.state["score"].cpu().numpy().copy()

    def tap(step, record) -> None:
        trace.steps.append({"step": step.step, "counts": record["counts"].copy(), "needed": record["scope"].copy(),
                            "cpu": record["cpu"].copy(),
                            "records": record["records"].copy(), "lends": record["lends"],
                            "resident": (store.residency.host >= 0).all(-1),
                            "empty": np.array([len(e) for e in store.residency.empty])})
    store.taps.append(tap)
    try:
        yield trace
        if not engine.has_work():                      # every launched step committed: the log told the host all
            trace.mirrored = np.array_equal(store.residency.device.cpu().numpy(), store.residency.host)
    finally:
        store.taps.remove(tap)


def parity(trace: Trace) -> dict:
    """The replay against the device, step by step: the first step that differs (None: none), and what was compared."""
    from holon_plugins.placement.lfu.reference import replay

    want = replay(trace.seed, trace.steps, trace.score, trace.classes)
    differs = next((s["step"] for w, s in zip(want, trace.steps, strict=True)
                    if not (np.array_equal(w["resident"], s["resident"]) and np.array_equal(w["empty"], s["empty"]))),
                   None)
    return {"steps": len(trace.steps), "decode_steps": sum(bool(s["counts"].any()) for s in trace.steps),
            "borrowing_steps": sum(s["lends"] is not None for s in trace.steps),
            "lends": sum(int((np.asarray(s["lends"]) > 0).any(1).sum()) for s in trace.steps if s["lends"] is not None),
            "admissions": trace.admissions, "first_differing_step": differs, "device_is_mirror": trace.mirrored}
