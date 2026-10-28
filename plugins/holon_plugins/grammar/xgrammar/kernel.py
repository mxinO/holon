"""Applying a compact block of token bitmasks to the logits (docs/dev/m05-grammar.md D4): `mask[i]` (int32 words, bit t
of
word w allows token 32·w + t — xgrammar's layout) masks logits row `rows[i]` to -inf where a bit is 0; rows at or past
the end of `logits` are padding and untouched. On the GPU a Triton kernel: one program per (mask row, block of
32·WORDS tokens) loads that block's words once and stores -inf only where a bit is 0, with no unpacked [R, V, 32]
intermediate, and a fixed grid for a fixed block size, so it is graph-capturable (its row count is the graph's). Its
reference (L0-mask) is xgrammar's CPU `apply_token_bitmask_inplace`, which serves CPU engines.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import xgrammar as xgr

WORDS = 32                                      # mask words per program: 1024 tokens


@triton.jit
def _apply(logits, mask, rows, n_rows, vocab, stride, width, WORDS: tl.constexpr):
    i, block = tl.program_id(0), tl.program_id(1)
    row = tl.load(rows + i)
    if row < n_rows:
        w = block * WORDS + tl.arange(0, WORDS)
        words = tl.load(mask + i * width + w, mask=w < width, other=-1)
        bit = tl.arange(0, 32)
        token = w[:, None] * 32 + bit[None, :]
        refused = ((words[:, None] >> bit[None, :]) & 1) == 0
        tl.store(logits + row * stride + token, float("-inf"), mask=refused & (token < vocab))


def apply(logits: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor) -> None:
    """In place: logits [R, V] (any float dtype, unit stride along V), mask [M, ⌈V/32⌉] int32, rows [M] int64."""
    r, v = logits.shape
    if logits.device.type == "cpu":
        keep = rows < r
        if keep.any():
            full = torch.full((r, mask.shape[1]), -1, dtype=torch.int32)
            full[rows[keep]] = mask[keep]
            xgr.apply_token_bitmask_inplace(logits, full, indices=rows[keep].tolist())
        return
    m, width = mask.shape
    if m:
        _apply[(m, triton.cdiv(width, WORDS))](logits, mask, rows, r, v, logits.stride(0), width, WORDS=WORDS)
