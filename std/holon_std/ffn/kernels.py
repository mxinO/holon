"""Op contract std.ffn owns. Its oracle is in `reference.py`; kernels come from kernel plugins.

- `silu_mul@1`: instance `(gate_up [T, 2*inter]) -> silu(gate) * up [T, inter]`, gate first; with the fact
  `glu_limit` L > 0, `silu(min(gate, L)) · clamp(up, −L, L)` (docs/dev/v1-glm.md D6; given only when set). Facts: inter,
  dtype[, glu_limit].
"""
from __future__ import annotations

SILU_MUL = "silu_mul@1"
