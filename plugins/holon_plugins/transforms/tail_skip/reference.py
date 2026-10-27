"""The oracle of transforms.tail_skip (A1, docs/dev/v1-offload.md D1, D3, §6: Strata's route tail skip, `approx`).

`moe_route_transform.tail_skip@1` is `moe_route_transform@1` (std.moe.kernels: `(ids [T, k] int32, weights [T, k] f32,
residency [E, R] int32) -> (ids, weights)`, between the top-k and the resolve, with the layer's map as the resolve
reads it): an entry (t, j) routing expert e ≥ 0 is dropped — its id becomes −1, the rank's null slot: no fetch, no CPU
job, no count, a zero contribution — when e is missed (missing in some role of the map: it would be fetched; a ring
piece is present) and e's best routing rank over the call's rows (the least j' of the entries routing it) is at least
`rank` (default 7, Strata's: the last rank of a top-8 router). Every entry of such an expert is dropped; resident
experts, and missed ones some row routes better, are untouched. The weights are returned as they are: never
renormalized. The call's rows are the expert pass's: other requests' rows count (the transform sees no row kinds), graph
padding does not (the store hands its rows in dropped, −1)."""
from __future__ import annotations

import torch

RANK = 7                         # Strata's default


def tail_skip(ids: torch.Tensor, weights: torch.Tensor, residency: torch.Tensor,
              rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor]:
    t, k = ids.shape
    live = ids >= 0
    e = ids.long().clamp_min(0)
    best = torch.full((len(residency),), k, dtype=torch.long, device=ids.device)
    best.scatter_reduce_(0, e[live], torch.arange(k, device=ids.device).expand(t, k)[live], "amin")
    drop = live & (residency < 0).any(-1)[e] & (best[e] >= rank)
    return torch.where(drop, -1, ids), weights
