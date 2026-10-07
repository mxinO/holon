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


class Decode:
    """One query row per sequence. The block table and lengths are read on the device from buffers this instance
    owns (stable addresses for graph capture); the only host decision is the split count, taken from a
    power-of-two bound on the lengths so it stays constant within a graph bucket."""

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
        self.lens[:b].copy_(pages.lens, non_blocking=True)
        self.lens[b:].zero_()                    # rows past the batch (graph padding) attend to nothing
        bound = int((batch.plan.start_upper + batch.plan.n_query).max())
        splits = splits_for(b, self.kv_heads, 1 << (bound - 1).bit_length(), self.num_sms)
        return _Handle(pages.table, self.rows[:b], self.lens[:b], splits)

    def run(self, q: torch.Tensor, kv: torch.Tensor, h: _Handle) -> torch.Tensor:
        return paged_attention(q, kv, h.table, h.rows, h.lens, nq=1, splits=h.splits, cfg=self.cfg)


def _supports(f: dict) -> bool:
    return (f["layout"] == "kv.hnd" and f["q_dtype"] == f["kv_dtype"] and f["q_dtype"] in ("bfloat16", "float16")
            and f["head_dim"] in (64, 128, 256) and f["heads"] % f["kv_heads"] == 0)


class Plugin:
    def register(self, regs) -> None:
        regs.kernels.add(OP, KernelSpec(
            op=OP, provider="kernels.triton_attn", impl=Decode, in_layouts={"kv": "kv.hnd"},
            requires=Constraint(sm_min=80, sm_max=129), supports=_supports,
            props=frozenset({"graph_safe", "deterministic"}), priority=10))
