"""std.tiered.pinned@1: the pinned RAM tier — a class's pieces in a mapped host pool kernels read in place
(docs/v1-offload.md D10). It holds every piece (inclusive: filled from the checkpoint's extents when the store prepares
it), unless its `ram` setting caps it — then RAM is exclusive of VRAM, a slower tier (std.tiered.gguf_direct@1) holds
the rest, and the store places pieces in it in the graph (`moe_tiers@1`).

Setting (the store's, forwarded: the MoE store's `moe.ram` option): `ram` — units per class, pinned:
`<class>=<slots or fraction>`, `*` for every class not named (as the store's `cache`); empty: every piece."""
from __future__ import annotations

import torch

from . import files
from .store import _parse_cache


class Pinned:
    device_readable = True

    def __init__(self, ctx, *, ram: str = "", **site) -> None:
        self._cap = _parse_cache(ram)
        self.pools: dict[str, object] = {}

    def holds(self, cls: str, pieces: int) -> int:
        cap = self._cap.get(cls, self._cap.get("*"))
        held = pieces if cap is None else cap if isinstance(cap, int) else round(cap * pieces)
        if not 0 <= held <= pieces:
            raise ValueError(f"std.tiered.pinned@1: ram {cls}: {held} units of {pieces} pieces")
        return held

    def prepare(self, ctx, cls: str, unit_bytes: int, extents, staging: int = 0, granule: int = 0) -> None:
        granule = granule or unit_bytes
        held, per = self.holds(cls, len(extents)), unit_bytes // granule
        pool = self.pools[cls] = ctx.pool(f"tiered.pinned.{cls}", unit_bytes=granule, min_units=(held + staging) * per,
                                          pref_units=(held + staging) * per, device="host", mapped=True)
        if held == len(extents):
            view, jobs = pool.view(torch.uint8, (held + staging, unit_bytes)).numpy(), []
            for piece, parts in enumerate(extents):
                at = 0
                for e in parts:
                    jobs.append((e, view[piece, at:at + e.nbytes]))
                    at += e.nbytes
            files.read(jobs)

    def units(self, cls: str):
        return self.pools[cls]

    def read(self, cls: str, piece: int) -> tuple:
        return "host_slot", piece
