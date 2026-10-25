"""flash-linear-attention's Gated DeltaNet kernels (vLLM's copy, vendored at a pinned commit with `vendor.patch`; the
subset spike S3 measured) as the `gated_delta_rule@1` kernel (op contract: std.mixers.gdn; M0.3 design D3).

A step whose rows all have one token, or planned `per_token` (verify rows and their replays, M0.4 D6), runs the fused
recurrent kernel: it computes g and β itself, token by token, and reads each row's state unit by slot (patched: slot 0
is a real sequence, and a row without has_init reads nothing); one-token rows write it back in place, `per_token` rows
once after their last token, to `out_slots` (patched).
Other steps run the chunked kernel over the rows that have tokens: it takes no state indices, so their states are
gathered into a scratch (zeros without has_init) and the final states scattered back; a row's in-step checkpoint
(`snapshots`) is the chunk kernel's own f32 state at the chunk where it falls (patched: stored into the payload
unit). `plan` builds the chunk indices, offsets and snapshot rows on the host, so no call syncs; a decode handle
depends on the row count only and its tensors belong to the instance, so decode calls capture into CUDA graphs and
replay for any inputs of that shape.
A step of more than `SUB_TOKENS` tokens runs the chunked kernel in calls of at most `SUB_TOKENS` tokens and
`SUB_TOKENS / 64` chunks: consecutive whole rows packed into one call, a longer row in sub-chunks (whole 64-token
chunks), each from the state the one before left — the same chunks in the same order, so the same bits (tested), at
a quarter of the temporaries, the largest part of a prompt step's activation peak (FN §23.3), whatever the step's rows
(the startup profile measures a one-row step; two prompts' chunks in one step must not need more, §22.9).

`kda@1` (GLM-5.3-Flash's Kimi delta attention, docs/v1-glm.md D4) is the same rule with a per-channel gate: the same
plans, rows, state units, snapshots and calls; its recurrent rows through the fused kernel's `IS_KDA` branch (patched:
the bounded gate `lower_bound · σ(exp(A_log) · (f + dt_bias))`), its chunks through vLLM's `chunk_kda` with the fused
gate and chunk-local cumsum (vendored `kda.py`; FLA's 16-row sub-chunks keep the in-chunk decay exponents finite at
the −5 bound).

Also `gdn_rewind@1` (`rewind.py`, ours): a verify step's rewind glue for every GDN layer in one launch (D5)."""
from __future__ import annotations

import dataclasses
import itertools
import types

import torch
import torch.nn.functional as F
from holon.api import Constraint, KernelSpec

from . import rewind
from .vendor import chunk_gated_delta_rule, fused_sigmoid_gating_delta_rule_update
from .vendor.kda import chunk_kda_with_fused_gate
from .vendor.utils import FLA_CHUNK_SIZE

OP, KDA_OP, REWIND = "gated_delta_rule", "kda", "gdn_rewind"
SUB_TOKENS = 1024                     # a one-row step's sub-chunk (a multiple of FLA_CHUNK_SIZE)


@dataclasses.dataclass(frozen=True)
class _Handle:
    decode: bool                          # one-token or per-token rows: the fused recurrent kernel
    tokens: int
    cu_seqlens: torch.Tensor | None       # [n + 1] int32: token offsets of the rows with tokens
    chunk_indices: torch.Tensor | None    # chunked: (row, chunk) per 64-token chunk
    chunk_offsets: torch.Tensor | None    # chunked: each row's first chunk
    live: torch.Tensor | None             # chunked: the rows with tokens, when some row has none
    snap: tuple[torch.Tensor, torch.Tensor] | None = None   # chunked: per live row, payload unit and chunk (-1: none)
    subs: tuple | None = None             # a step past SUB_TOKENS: per call (its live rows [i0, i1), first token,
    #                                       tokens, handle, a snapshot's unit at its start, starts / ends its rows)


def _to_device(values: list, device: torch.device) -> torch.Tensor:
    """Host ints on the device through pinned memory: an asynchronous copy, no host sync."""
    return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)


class GatedDeltaRule:
    strided = True      # q, k, v, a and b may be column views of wider rows (read in place by the recurrent kernel)

    def __init__(self, ctx, *, heads_k: int, heads_v: int, head_k_dim: int, head_v_dim: int, v_head_order: str,
                 dtype: str, out_slots: bool = False) -> None:
        self.device, self.tiled = ctx.device, v_head_order == "tiled"
        self._offsets: dict[tuple[int, ...], torch.Tensor] = {}   # recurrent cu_seqlens per row lengths (graphs)

    def plan(self, lens: list[int], snapshots=(), per_token: bool = False, sub: bool = False) -> _Handle:
        """`sub`: a sub-chunk's plan (chunked, whatever its tokens; never split again)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() runs on the host before a step, never inside a graph capture")
        n = len(lens)
        for row, offset, _ in snapshots:
            if offset % FLA_CHUNK_SIZE or not 0 < offset // FLA_CHUNK_SIZE < -(-lens[row] // FLA_CHUNK_SIZE):
                raise ValueError(f"snapshot of row {row} at {offset}: a chunk boundary inside its {lens[row]} tokens")
        if per_token and snapshots:
            raise ValueError("per-token rows take no snapshots")
        if not sub and n and (per_token or all(t == 1 for t in lens)):
            key = tuple(lens)
            if key not in self._offsets:
                self._offsets[key] = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32,
                                                  device=self.device)
            return _Handle(True, sum(lens), self._offsets[key], None, None, None)
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
        subs = None
        if not sub and sum(live) > SUB_TOKENS:             # in calls of at most SUB_TOKENS tokens
            at = {row: (offset, unit) for row, offset, unit in snapshots}
            subs, first, i, most = [], 0, 0, SUB_TOKENS // FLA_CHUNK_SIZE
            while i < len(rows):
                t = live[i]
                if t > SUB_TOKENS:                         # a long row: in sub-chunks, its state carried
                    offset, unit = at.get(rows[i], (-1, -1))
                    chunk = offset // FLA_CHUNK_SIZE if offset >= 0 else -1
                    for s0 in range(0, t, SUB_TOKENS):
                        tokens, c0 = min(t, s0 + SUB_TOKENS) - s0, s0 // FLA_CHUNK_SIZE
                        inner = [(0, (chunk - c0) * FLA_CHUNK_SIZE, unit)] \
                            if 0 < chunk - c0 < -(-tokens // FLA_CHUNK_SIZE) else []   # its snapshot inside it, or …
                        subs.append((i, i + 1, first + s0, tokens, self.plan([tokens], inner, sub=True),
                                     unit if chunk == c0 else None, s0 == 0, s0 + tokens == t))   # … at its start
                    first, i = first + t, i + 1
                    continue
                j, packed, calls = i, 0, 0                # whole rows packed while the call stays that small
                while j < len(rows) and live[j] <= SUB_TOKENS and packed + live[j] <= SUB_TOKENS and \
                        calls + -(-live[j] // FLA_CHUNK_SIZE) <= most:
                    packed, calls, j = packed + live[j], calls + -(-live[j] // FLA_CHUNK_SIZE), j + 1
                inner = [(r - i, *at[rows[r]]) for r in range(i, j) if rows[r] in at]
                subs.append((i, j, first, packed, self.plan(live[i:j], inner, sub=True), None, True, True))
                first, i = first + packed, j
            subs = tuple(subs)
        return _Handle(False, sum(live), _to_device([0, *itertools.accumulate(live)], self.device),
                       _to_device(indices, self.device), _to_device([0, *itertools.accumulate(chunks)], self.device),
                       None if len(rows) == n else _to_device(rows, self.device), snap, subs)

    @torch.no_grad()
    def __call__(self, h: _Handle, q, k, v, a, b, a_log, dt_bias, state, slots, has_init, snapshot=None,
                 out_slots=None) -> torch.Tensor:
        if q.shape[0] != h.tokens:
            raise ValueError(f"{q.shape[0]} tokens for a plan of {h.tokens}")
        if state.stride()[1:] != (state.shape[2] * state.shape[3], state.shape[3], 1):
            raise ValueError(f"each state unit must be contiguous (strides {state.stride()})")
        if h.tokens == 0:
            return torch.empty(v.shape, dtype=v.dtype, device=v.device)
        if h.decode:
            # One-token rows store in place in the loop (unchanged); multi-token rows, or another destination, store
            # once after the last token.
            once = None if out_slots is None and h.tokens == len(slots) else (slots if out_slots is None else out_slots)
            o, _ = fused_sigmoid_gating_delta_rule_update(
                a_log, self._gate_rows(a), b, dt_bias, q[None], k[None], v[None], initial_state=state,
                inplace_final_state=True, cu_seqlens=h.cu_seqlens, ssm_state_indices=slots,
                use_qk_l2norm_in_kernel=True, has_initial_state=has_init, v_head_tiled=self.tiled, out_indices=once,
                **self._recurrent)
            return o[0]
        if out_slots is not None:
            raise ValueError("out_slots needs a per-token plan (the chunked path writes the rows' own units)")
        if h.live is not None:
            slots, has_init = slots.index_select(0, h.live), has_init.index_select(0, h.live)
        units, init = slots.long(), ~has_init.bool().view(-1, 1, 1, 1)
        g = self._chunk_gates(a, a_log, dt_bias)
        beta = self._beta(b, v)
        o = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        if h.snap is not None and (snapshot is None or not snapshot.is_contiguous() or snapshot.dtype != torch.float32):
            raise ValueError("a plan with snapshots needs a contiguous f32 payload view to store them in")
        if h.subs is None:                             # one call over every row
            final = self._chunk(h, slice(None), q, k, v, g, beta, a_log, dt_bias,
                                state.index_select(0, units).masked_fill_(init, 0.0), o, snapshot)
            state.index_copy_(0, units, final)
            return o
        cur = None                                     # call by call: whole rows, or one row's sub-chunks in turn
        for i0, i1, s0, tokens, sub, at_start, starts, ends in h.subs:
            if starts:                                 # its rows' states (a long row's: carried between its calls)
                cur = state.index_select(0, units[i0:i1]).masked_fill_(init[i0:i1], 0.0)
            if at_start is not None:                   # its snapshot at this sub-chunk's start: the state carried
                snapshot[at_start].copy_(cur[0])
            cur = self._chunk(sub, slice(s0, s0 + tokens), q, k, v, g, beta, a_log, dt_bias, cur, o, snapshot)
            if ends:
                state.index_copy_(0, units[i0:i1], cur)
        return o

    # The rule's gate, by path: GDN's scalar gate per v-head.
    _recurrent: types.MappingProxyType = types.MappingProxyType({})   # the fused recurrent kernel's gate arguments

    @staticmethod
    def _gate_rows(a: torch.Tensor) -> torch.Tensor:
        return a

    @staticmethod
    def _chunk_gates(a, a_log, dt_bias) -> torch.Tensor:
        return -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())

    @staticmethod
    def _beta(b, v) -> torch.Tensor:
        return torch.sigmoid(b.float()).to(v.dtype)     # the chunk kernels' operand dtype (as vLLM and HF feed it)

    def _chunk(self, h: _Handle, part: slice, q, k, v, g, beta, a_log, dt_bias, initial, o, snapshot) -> torch.Tensor:
        """One chunked call over tokens `part` from the rows' `initial` states: their final states."""
        _, final = chunk_gated_delta_rule(
            q[None, part], k[None, part], v[None, part], g[None, part], beta[None, part], initial_state=initial,
            output_final_state=True, cu_seqlens=h.cu_seqlens, chunk_indices=h.chunk_indices,
            chunk_offsets=h.chunk_offsets, use_qk_l2norm_in_kernel=True, core_attn_out=o[None, part],
            v_head_tiled=self.tiled, snapshot=None if h.snap is None else (snapshot, *h.snap))
        return final


class KDA(GatedDeltaRule):
    """`kda@1`: `gated_delta_rule`'s plans and calls with KDA's per-channel bounded gate (f [T, H·Dk], A_log [H],
    dt_bias [H·Dk]); no head grouping."""

    def __init__(self, ctx, *, heads: int, head_k_dim: int, head_v_dim: int, lower_bound: float, dtype: str,
                 out_slots: bool = False) -> None:
        super().__init__(ctx, heads_k=heads, heads_v=heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
                         v_head_order="grouped", dtype=dtype, out_slots=out_slots)
        self.heads, self.dk, self.lower_bound = heads, head_k_dim, float(lower_bound)
        self._recurrent = types.MappingProxyType({"is_kda": True, "lower_bound": self.lower_bound})

    def _gate_rows(self, f: torch.Tensor) -> torch.Tensor:
        return f.view(f.shape[0], self.heads, self.dk)            # per token, per head, per key channel

    def _chunk_gates(self, f, a_log, dt_bias) -> torch.Tensor:
        return self._gate_rows(f)                       # raw: the chunk call fuses the gate into its cumsum

    @staticmethod
    def _beta(b, v) -> torch.Tensor:
        return torch.sigmoid(b.float())                 # f32, as the recurrent path computes it (D4)

    def _chunk(self, h: _Handle, part: slice, q, k, v, f, beta, a_log, dt_bias, initial, o, snapshot) -> torch.Tensor:
        _, final = chunk_kda_with_fused_gate(
            q[None, part], k[None, part], v[None, part], f[None, part], beta[None, part], a_log, dt_bias,
            initial_state=initial, output_final_state=True, lower_bound=self.lower_bound,
            use_qk_l2norm_in_kernel=True, cu_seqlens=h.cu_seqlens, chunk_indices=h.chunk_indices,
            chunk_offsets=h.chunk_offsets, snapshot=None if h.snap is None else (snapshot, *h.snap),
            core_attn_out=o[None, part])
        return final


def _supports(f: dict) -> bool:
    return (f["dtype"] == "bfloat16" and f["heads_v"] % f["heads_k"] == 0         # any `out_slots` fact: supported
            and f["head_k_dim"] in (64, 128) and f["head_v_dim"] in (64, 128)
            and f["v_head_order"] in ("grouped", "tiled"))


def _supports_kda(f: dict) -> bool:
    return (f["dtype"] == "bfloat16" and f["head_k_dim"] in (64, 128) and f["head_v_dim"] in (64, 128)
            and f["lower_bound"] < 0)                           # any `out_slots` fact: supported


class Plugin:
    def register(self, regs) -> None:
        for op, impl, supports in ((OP, GatedDeltaRule, _supports), (KDA_OP, KDA, _supports_kda),
                                   (REWIND, rewind.GDNRewind, rewind.supports)):
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.fla", impl=impl, requires=Constraint(sm_min=80),
                                            supports=supports, props=frozenset({"graph_safe", "deterministic"})))
