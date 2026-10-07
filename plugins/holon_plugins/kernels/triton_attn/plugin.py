"""Our Triton paged attention as an `attention.paged_decode@1` kernel (op contract: std.mixers.gqa)."""
from __future__ import annotations

import dataclasses

import torch
from holon.api import Constraint, KernelSpec

from .kernel import Config, paged_attention, splits_for

OP = "attention.paged_decode"


@dataclasses.dataclass(frozen=True)
class _Handle:
    table: torch.Tensor
    rows: torch.Tensor
    lens: torch.Tensor
    splits: int
    source_lens: torch.Tensor


class Decode:
    nq = 1

    """One query row per sequence. The block table and lengths are read on the device from buffers this instance
    owns (stable addresses for graph capture); the only host decision is the split count, taken from the batch size
    and the table's capacity, so one graph per batch bucket serves every length (splits past a sequence's end are
    empty)."""

    def __init__(self, ctx, *, heads: int, kv_heads: int, head_dim: int, page: int, q_dtype: str, kv_dtype: str,
                 layout: str, max_seqs: int) -> None:
        fp = ctx.fingerprint
        self.kv_heads, self.num_sms = kv_heads, fp.num_sms
        self.cfg = Config.pinned(fp.gpu, fp.sm, head_dim)
        self.rows = torch.zeros(max_seqs, dtype=torch.int32, device=ctx.device)
        self.lens = torch.zeros(max_seqs, dtype=torch.int32, device=ctx.device)

    def plan(self, batch, pages) -> _Handle:
        b = len(batch.plan.slots)
        self.rows[:b].copy_(batch.dev.slots, non_blocking=True)
        self.lens[b:].zero_()                    # rows the instance holds beyond this batch: no stale lengths
        splits = splits_for(b, self.kv_heads, pages.table.shape[1] * pages.page, self.num_sms)
        return _Handle(pages.table, self.rows[:b], self.lens[:b], splits, pages.lens)

    def prepare(self, h: _Handle) -> None:
        h.lens.copy_(h.source_lens)             # exact device lengths of this step

    def graph_key(self, h: _Handle) -> tuple:
        return ("triton_attn.decode", len(h.rows), h.splits)

    def run(self, q: torch.Tensor, kv: torch.Tensor, h: _Handle) -> torch.Tensor:
        return paged_attention(q, kv, h.table, h.rows, h.lens, nq=len(q) // len(h.rows), splits=h.splits, cfg=self.cfg)


class Verify(Decode):
    """1 + k query rows per sequence (the same k for all), causal among themselves; lengths from the device."""

    def graph_key(self, h: _Handle) -> tuple:
        return ("triton_attn.verify", len(h.rows), h.splits)


def _supports(f: dict) -> bool:
    return (f["layout"] == "kv.hnd" and f["q_dtype"] == f["kv_dtype"] and f["q_dtype"] in ("bfloat16", "float16")
            and f["head_dim"] in (64, 128, 256) and f["heads"] % f["kv_heads"] == 0)


class Plugin:
    def register(self, regs) -> None:
        for op, impl in ((OP, Decode), ("attention.paged_verify", Verify)):
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.triton_attn", impl=impl, in_layouts={"kv": "kv.hnd"},
                requires=Constraint(sm_min=80, sm_max=129), supports=_supports,
                props=frozenset({"graph_safe", "deterministic"}), priority=10))
