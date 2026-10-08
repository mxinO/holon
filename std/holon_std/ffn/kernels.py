"""Op contract std.ffn owns. Its oracle is in `reference.py`; kernels come from kernel plugins.

- `silu_mul@1`: instance `(gate_up [T, 2*inter]) -> silu(gate) * up [T, inter]`, gate first. Facts: inter, dtype.
"""
from __future__ import annotations

SILU_MUL = "silu_mul@1"
