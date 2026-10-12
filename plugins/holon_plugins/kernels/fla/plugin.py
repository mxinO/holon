"""flash-linear-attention's Gated DeltaNet kernels (vLLM's copy, vendored at a pinned commit with `vendor.patch`; the
subset spike S3 measured) as the `gated_delta_rule@1` kernel (op contract: std.mixers.gdn; M0.3 design D3).

A step whose rows all have one token runs the fused recurrent kernel: it computes g and β itself and reads and writes
each row's state unit in place by slot (patched: slot 0 is a real sequence, and a row without has_init reads nothing).
Other steps run the chunked kernel over the rows that have tokens: it takes no state indices, so their states are
gathered into a scratch (zeros without has_init) and the final states scattered back; a row's in-step checkpoint
(`snapshots`) is the chunk kernel's own f32 state at the chunk where it falls (patched: stored into the payload
unit). `plan` builds the chunk indices, offsets and snapshot rows on the host, so no call syncs; a decode handle
depends on the row count only and its tensors belong to the instance, so decode calls capture into CUDA graphs and
replay for any inputs of that shape."""
from __future__ import annotations

import dataclasses
import itertools

import torch
import torch.nn.functional as F
from holon.api import Constraint, KernelSpec

from .vendor import chunk_gated_delta_rule, fused_sigmoid_gating_delta_rule_update
from .vendor.utils import FLA_CHUNK_SIZE

OP = "gated_delta_rule"


@dataclasses.dataclass(frozen=True)
class _Handle:
    decode: bool                          # every row has one token: the fused recurrent kernel
    tokens: int
    cu_seqlens: torch.Tensor | None       # [n + 1] int32: token offsets of the rows with tokens
    chunk_indices: torch.Tensor | None    # chunked: (row, chunk) per 64-token chunk
    chunk_offsets: torch.Tensor | None    # chunked: each row's first chunk
    live: torch.Tensor | None             # chunked: the rows with tokens, when some row has none
    snap: tuple[torch.Tensor, torch.Tensor] | None = None   # chunked: per live row, payload unit and chunk (-1: none)


def _to_device(values: list, device: torch.device) -> torch.Tensor:
    """Host ints on the device through pinned memory: an asynchronous copy, no host sync."""
    return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)


class GatedDeltaRule:
    def __init__(self, ctx, *, heads_k: int, heads_v: int, head_k_dim: int, head_v_dim: int, v_head_order: str,
                 dtype: str) -> None:
        self.device, self.tiled = ctx.device, v_head_order == "tiled"
        self._offsets: dict[int, torch.Tensor] = {}       # decode cu_seqlens per row count, kept for graph replay

    def plan(self, lens: list[int], snapshots=()) -> _Handle:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() runs on the host before a step, never inside a graph capture")
        n = len(lens)
        for row, offset, _ in snapshots:
            if offset % FLA_CHUNK_SIZE or not 0 < offset // FLA_CHUNK_SIZE < -(-lens[row] // FLA_CHUNK_SIZE):
                raise ValueError(f"snapshot of row {row} at {offset}: a chunk boundary inside its {lens[row]} tokens")
        if n and all(t == 1 for t in lens):
            if n not in self._offsets:
                self._offsets[n] = torch.arange(n + 1, dtype=torch.int32, device=self.device)
            return _Handle(True, n, self._offsets[n], None, None, None)
        rows = [r for r, t in enumerate(lens) if t > 0]
        if not rows:
            return _Handle(False, 0, None, None, None, None)
        live = [lens[r] for r in rows]
        chunks = [-(-t // FLA_CHUNK_SIZE) for t in live]
        indices = [(i, c) for i, nc in enumerate(chunks) for c in range(nc)]
        snap = None
        if snapshots:                              # per live row (zero-token rows are dropped): unit, chunk
            at = {row: (unit, offset // FLA_CHUNK_SIZE) for row, offset, unit in snapshots}
            per = [at.get(r, (-1, -1)) for r in rows]
            snap = (_to_device([u for u, _ in per], self.device), _to_device([c for _, c in per], self.device))
        return _Handle(False, sum(live), _to_device([0, *itertools.accumulate(live)], self.device),
                       _to_device(indices, self.device), _to_device([0, *itertools.accumulate(chunks)], self.device),
                       None if len(rows) == n else _to_device(rows, self.device), snap)

    @torch.no_grad()
    def __call__(self, h: _Handle, q, k, v, a, b, a_log, dt_bias, state, slots, has_init, snapshot=None
                 ) -> torch.Tensor:
        if q.shape[0] != h.tokens:
            raise ValueError(f"{q.shape[0]} tokens for a plan of {h.tokens}")
        if state.stride()[1:] != (state.shape[2] * state.shape[3], state.shape[3], 1):
            raise ValueError(f"each state unit must be contiguous (strides {state.stride()})")
        if h.tokens == 0:
            return torch.empty(v.shape, dtype=v.dtype, device=v.device)
        if h.decode:
            o, _ = fused_sigmoid_gating_delta_rule_update(
                a_log, a, b, dt_bias, q[None], k[None], v[None], initial_state=state, inplace_final_state=True,
                cu_seqlens=h.cu_seqlens, ssm_state_indices=slots, use_qk_l2norm_in_kernel=True,
                has_initial_state=has_init, v_head_tiled=self.tiled)
            return o[0]
        if h.live is not None:
            slots, has_init = slots.index_select(0, h.live), has_init.index_select(0, h.live)
        units = slots.long()
        h0 = state.index_select(0, units).masked_fill_(~has_init.bool().view(-1, 1, 1, 1), 0.0)
        g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        beta = torch.sigmoid(b.float()).to(v.dtype)      # the chunk kernels' operand dtype (as vLLM and HF feed it)
        o = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        snap = None
        if h.snap is not None:
            if snapshot is None or not snapshot.is_contiguous() or snapshot.dtype != torch.float32:
                raise ValueError("a plan with snapshots needs a contiguous f32 payload view to store them in")
            snap = (snapshot, *h.snap)
        _, final = chunk_gated_delta_rule(
            q[None], k[None], v[None], g[None], beta[None], initial_state=h0, output_final_state=True,
            cu_seqlens=h.cu_seqlens, chunk_indices=h.chunk_indices, chunk_offsets=h.chunk_offsets,
            use_qk_l2norm_in_kernel=True, core_attn_out=o[None], v_head_tiled=self.tiled, snapshot=snap)
        state.index_copy_(0, units, final)
        return o


def _supports(f: dict) -> bool:
    return (f["dtype"] == "bfloat16" and f["heads_v"] % f["heads_k"] == 0
            and f["head_k_dim"] in (64, 128) and f["head_v_dim"] in (64, 128)
            and f["v_head_order"] in ("grouped", "tiled"))


class Plugin:
    def register(self, regs) -> None:
        regs.kernels.add(OP, KernelSpec(op=OP, provider="kernels.fla", impl=GatedDeltaRule,
                                        requires=Constraint(sm_min=80), supports=_supports,
                                        props=frozenset({"graph_safe", "deterministic"})))
