"""`linear_experts@2`'s `pass_rows` promise (std.moe.kernels; docs/v1-moe-prefill.md D0, §4), checked on a bound
instance: every provider that declares `pass_rows` runs it in its L0, over the (SM, type) pairs it declares."""
from __future__ import annotations

import torch
from holon_std.moe.kernels import CHUNK


def check_pass_rows(kernel, parts: dict[str, torch.Tensor], x: torch.Tensor, ids: torch.Tensor, what: str = "") -> None:
    """A call of the first n·CHUNK rows of `x` [T, R, in] / `ids` [T, k] (T a multiple of CHUNK) gives each row the bits
    the CHUNK-row call of that row's own chunk gives it, for n = 1, 2, 3 and the most whole CHUNKs `kernel.pass_rows`
    holds (within T), and so does the call of all T rows (past `pass_rows` the instance splits it)."""
    rows, n = kernel.pass_rows, len(ids) // CHUNK
    assert rows >= CHUNK and rows % CHUNK == 0 and n * CHUNK == len(ids), (rows, len(ids))
    ref = torch.cat([kernel(x[a:a + CHUNK], parts, ids[a:a + CHUNK]) for a in range(0, len(ids), CHUNK)])
    for m in sorted({m for m in (1, 2, 3, rows // CHUNK, n) if m <= n}):
        got = kernel(x[:m * CHUNK], parts, ids[:m * CHUNK])
        assert torch.equal(got, ref[:m * CHUNK]), f"{what}: a call of {m * CHUNK} rows (pass_rows {rows})"
