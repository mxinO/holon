"""Routes and residency of the offload store, step by step, and the L1 policy-parity check (docs/v1-offload.md §6).

`route_dump(engine)` taps the engine's offload store (`std.moe.offload@1`'s `std.tiered.store@1`, through its route
taps) while it runs: per committed step, its routing counts [layers, experts] of decode and verify rows (the store's
`after_commit` telemetry), the experts any of its rows routed (read from the resolves of eagerly run steps; a graphed
step holds decode and verify rows only, whose routes the counts give), the delta records the device logged, and the
residency (whole experts) the store's host mirror holds after the step — the device's map, from its delta log.

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
from unittest import mock

import numpy as np

KEY = "std.moe.offload"           # the offload store's shared-object key


@dataclasses.dataclass
class Trace:
    seed: np.ndarray                                   # [layers, experts] resident at the seed
    steps: list[dict] = dataclasses.field(default_factory=list)   # step, counts, needed, records, resident
    mirrored: bool | None = None                       # at the end: the device's map is the host mirror's

    @property
    def admissions(self) -> int:
        return sum(int((s["records"][:, 1] >= 0).sum()) for s in self.steps)


@contextlib.contextmanager
def route_dump(engine) -> Iterator[Trace]:
    from holon_std.tiered.resolver import DeviceResolver
    from holon_std.tiered.store import PADDING

    store = engine.ctx.shared_objects[KEY]
    trace = Trace((store.residency.host >= 0).all(-1).copy())
    needed: dict[int, np.ndarray] = {}                 # step -> experts its eager resolves saw routed
    call = DeviceResolver.__call__

    def resolve(self, ids, first):
        if self.store is store:
            live = ids[((store.kinds[first:first + len(ids)] != PADDING)[:, None] & (ids >= 0))].long().cpu().numpy()
            needed.setdefault(store.step, np.zeros(trace.seed.shape, dtype=bool))[self.group, live] = True
        return call(self, ids, first)

    def tap(step, telemetry) -> None:
        counts = telemetry["counts"].numpy().copy()
        trace.steps.append({"step": step.step, "counts": counts, "needed": needed.pop(step.step, counts > 0),
                            "records": telemetry["delta"][:int(telemetry["n_delta"][0])].numpy().copy(),
                            "resident": (store.residency.host >= 0).all(-1)})
    store.taps.append(tap)
    try:
        with mock.patch.object(DeviceResolver, "__call__", resolve):
            yield trace
        if not engine.has_work():                      # every launched step committed: the log told the host all
            trace.mirrored = np.array_equal(store.residency.device.cpu().numpy(), store.residency.host)
    finally:
        store.taps.remove(tap)


def parity(trace: Trace) -> dict:
    """The replay against the device, step by step: the first step that differs (None: none), and what was compared."""
    from holon_plugins.placement.lfu.reference import replay

    want = replay(trace.seed, [s["counts"] for s in trace.steps], [s["needed"] for s in trace.steps])
    differs = next((s["step"] for w, s in zip(want, trace.steps, strict=True) if not np.array_equal(w, s["resident"])),
                   None)
    return {"steps": len(trace.steps), "decode_steps": sum(bool(s["counts"].any()) for s in trace.steps),
            "admissions": trace.admissions, "first_differing_step": differs, "device_is_mirror": trace.mirrored}
