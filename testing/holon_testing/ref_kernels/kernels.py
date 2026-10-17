"""The std oracles wrapped as kernel instances, for CPU tests (tiny models end to end) and as oracles in GPU tests.

Each class has the instance API of the op it serves (see the owning unit's `kernels.py`).
"""
from __future__ import annotations

import gguf
import numpy as np
import torch
from holon_std.ffn import reference as ffn
from holon_std.mixers.gdn import reference as gdn
from holon_std.mixers.gqa import reference as gqa
from holon_std.moe import reference as moe
from holon_std.ops import reference as ops
from holon_std.weights.hf import reference as hf


class RMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, offset: float, dtype: str) -> None:
        self.eps, self.offset = eps, offset

    def __call__(self, x, weight):
        return ops.rmsnorm(x, weight, self.eps, self.offset)


class AddRMSNorm(RMSNorm):
    def __call__(self, x, residual, weight):
        return ops.add_rmsnorm(x, residual, weight, self.eps, self.offset)


class Rope:
    def __init__(self, ctx, *, head_dim: int, rotary_dim: int, theta: float, max_positions: int, dtype: str) -> None:
        self.theta, self.rotary_dim = theta, rotary_dim

    def __call__(self, q, k, positions):
        return (ops.rope_neox(q, positions, self.theta, self.rotary_dim),
                ops.rope_neox(k, positions, self.theta, self.rotary_dim))


class SiluMul:
    def __init__(self, ctx, *, inter: int, dtype: str) -> None:
        self.inter = inter

    def __call__(self, gate_up):
        return ffn.silu_mul(gate_up[..., : self.inter], gate_up[..., self.inter:])


class PagedAttention:
    """`attention.paged_prefill/decode/verify@1` (any dtype; the `causal` and `window` facts too)."""

    def __init__(self, ctx, *, causal: bool = True, window: int = 0, **facts) -> None:
        self.causal, self.window = causal, window

    def plan(self, batch, pages):
        return pages.table, batch.dev.slots, pages.lens, batch.dev.query_start

    def run(self, q, kv, handle):
        return gqa.paged_attention(q, kv, *handle, causal=self.causal, window=self.window)


class KvWrite:
    def __init__(self, ctx, **facts) -> None:
        pass

    def __call__(self, k, v, kv, unit, off) -> None:
        gqa.kv_write(k, v, kv, unit, off)


class CausalAttention:
    def __init__(self, ctx, *, causal: bool = True, window: int = 0, **facts) -> None:
        self.causal, self.window = causal, window

    def __call__(self, q, k, v, query_start):
        return gqa.causal_attention(q, k, v, query_start, self.causal, self.window)


class _Dequantized:
    """Quantized-op references: ggml layouts (`ggml.<type>/gguf.native`) dequantized by gguf-py, the oracle the judge
    uses; HF layouts (`<elem>/hf`, docs/m06-lowp.md D2) by std.weights.hf's reference."""

    def __init__(self, ctx, *, layout: str, **facts) -> None:
        self.elem, name = layout.split("/")
        self.ggml = gguf.GGMLQuantizationType[self.elem.removeprefix("ggml.").upper()] if name == "gguf.native" \
            else None
        self.facts, self._cache = facts, {}

    def dequantized(self, parts: dict[str, torch.Tensor], make=None) -> torch.Tensor:
        """`make(parts)` (default: the whole weight dequantized), once per weight, keyed by its primary part."""
        first = next(iter(parts.values()))
        key = (first.data_ptr(), first.shape)
        if key not in self._cache:
            self._cache[key] = (make or self._dequantize)(parts)
        return self._cache[key]

    def _dequantize(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.ggml == gguf.GGMLQuantizationType.MXFP4:
            return ggml_mxfp4(parts["blocks"].cpu())
        if self.ggml is not None:
            return torch.from_numpy(gguf.quants.dequantize(np.asarray(parts["blocks"].cpu()), self.ggml))
        return hf.dequantize(self.elem, parts)


def ggml_mxfp4(blocks: torch.Tensor) -> torch.Tensor:
    """gguf-py's MXFP4 dequantization, vectorized in torch (gguf-py's runs on one core: ≈ 18 s for a 30B layer's expert
    stacks, ≈ 1 s here): per 32 values an e8m0 byte, then 16 bytes holding element j low and j + 16
    high; `e2m1 · 2^(e − 127)` equals gguf-py's doubled table times half the scale by value."""
    b = blocks.reshape(-1, 17)
    codes = torch.cat((b[:, 1:] & 0x0F, b[:, 1:] >> 4), dim=-1)
    return (hf.E2M1[codes.long()] * hf.e8m0(b[:, :1])).reshape(*blocks.shape[:-1], -1)


class QLinear(_Dequantized):
    def __call__(self, x, parts):
        return ops.qlinear(x, self.dequantized(parts)).to(getattr(torch, self.facts["out_dtype"]))


class QLinearGlu(_Dequantized):
    """`qlinear_glu@1` at every row count: silu(gate) · up of the merged weight's f32 product."""

    def __call__(self, x, parts):
        return ops.qlinear_glu(x, self.dequantized(parts)).to(getattr(torch, self.facts["out_dtype"]))


class QEmbedding(_Dequantized):
    def __call__(self, ids, blocks):
        return ops.qembedding(ids, self.dequantized({"blocks": blocks})).to(getattr(torch, self.facts["dtype"]))


# A stack's f32 decode is cached up to this size (every test model's, and a small model's: ≤ 12 GiB over 96 stacks);
# a larger one (a 30B's 0.8–1.6 GB, ≈ 122 GB in all) is decoded on every call, only the experts the call routes to:
# Tier A fits a 64 GB budget, at the cost of a decode per layer and step (MXFP4 in torch: ≈ 1 s a layer; gguf-py's
# k-quants and i-quants, on one core: ≈ 10–20 s a layer, so a 30B GGUF Tier A of those types takes hours).
CACHE_BYTES = 128 << 20


class LinearExperts:
    """Expert stacks: ggml layouts dequantized per expert (gguf-py, MXFP4 in torch), HF layouts per expert and unit by
    std.weights.hf's reference — cached when small (`CACHE_BYTES`), else per call; dense ones as stored, the
    activation rounded to their dtype (the contract). A cached decode is checked against the stack's bytes: an offload
    store's arena (up to `experts` slots) changes between calls."""

    def __init__(self, ctx, *, layout: str, experts: int, out: int, glu: str, out_dtype: str, **facts) -> None:
        self.quantized = None if layout.startswith("dense/") else _Dequantized(ctx, layout=layout)
        self.experts, self.out, self.glu, self.out_dtype = experts, out, glu, getattr(torch, out_dtype)
        self._stacks: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}   # stack -> (its bytes, their decode)
        self.errors = torch.zeros(1, dtype=torch.int32)        # `linear_experts_arena@1`'s word: never set here

    def __call__(self, x, parts, ids):
        if self.quantized is None:
            weight = parts["weight"]
            y = moe.linear_experts(x, weight, ids, self.glu, operand_dtype=weight.dtype)
        else:
            first = next(iter(parts.values()))
            if len(first) * self.out * x.shape[-1] * 4 <= CACHE_BYTES:
                key = (first.data_ptr(), first.shape)          # the decode is kept while the bytes stay the same
                if key not in self._stacks or not torch.equal(self._stacks[key][0], first):
                    self._stacks[key] = first.clone(), self._decode(parts, range(len(first)))
                stack = self._stacks[key][1]
            else:
                stack = self._decode(parts, ids.unique().tolist())
            y = moe.linear_experts(x, stack, ids, self.glu)
        return y.to(self.out_dtype)

    def _decode(self, parts, experts) -> torch.Tensor:
        """The f32 stack [E, out, in] with `experts` decoded (the others zero)."""
        q, n = self.quantized, len(next(iter(parts.values())))
        if q.ggml is not None:
            rows = [q._dequantize({"blocks": parts["blocks"][e]}) for e in experts]
        else:
            units = moe.expert_parts(parts, n, merged=self.glu == "swiglu")
            rows = [torch.cat([hf.dequantize(q.elem, u) for u in units[e]]) for e in experts]
        stack = rows[0].new_zeros(n, *rows[0].shape)
        stack[list(experts)] = torch.stack(rows)
        return stack


class MoeTopk:
    def __init__(self, ctx, *, experts: int, top_k: int, renormalize: bool) -> None:
        self.top_k, self.renormalize = top_k, renormalize

    def __call__(self, logits):
        return moe.moe_topk(logits, self.top_k, self.renormalize)


class MoeCombine:
    def __init__(self, ctx, *, dtype: str, **facts) -> None:
        self.dtype = getattr(torch, dtype)

    def __call__(self, y, weights, out):
        out.copy_(moe.moe_combine(y, weights, self.dtype))


class MoeResolve:
    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.staging = staging

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs):
        slots, fetch, n_fetch, increments, seen, bits, cpu, rows = moe.moe_resolve(ids, kinds, residency, host, limits,
                                                                                  self.staging, ring, stage, plan)
        counts += increments
        routed |= seen
        misses.clamp_(min=int(n_fetch.max()))
        errors |= bits
        jobs[0, 0] = len(rows)
        for i, row in enumerate(rows):
            jobs[1 + i, :len(row)] = torch.tensor(row)
        return slots, fetch, n_fetch, cpu


class ExpertFetch:
    def __init__(self, ctx, *, unit_bytes: int, staging: int) -> None:
        pass

    def __call__(self, fetch, n, src, dst):
        dst.copy_(moe.expert_fetch(fetch, int(n[0]), src, dst))


class ExpertPrefetch:
    def __init__(self, ctx, *, unit_bytes: int, copy: str) -> None:      # either mode: a CPU step copies at once
        pass

    def start(self, key, fetch, n, src, dst):       # at once: a CPU step has nothing to overlap
        dst.copy_(moe.expert_fetch(fetch, int(n[0]), src, dst))

    def wait(self, key):
        pass


class CpuTask:
    """`cpu_task@1` on a CPU engine: a slot's task runs at its signal (a CPU step has nothing to overlap)."""

    def __init__(self, ctx, *, pool: str) -> None:
        self.tasks, self.errors = [], ctx.static(f"cpu_task.{pool}.errors", (1,), torch.int32)

    def bind(self, task) -> int:
        self.tasks.append(task)
        return len(self.tasks) - 1

    def signal(self, slot: int) -> None:
        moe.cpu_task(self.tasks[slot])

    def wait(self, slot: int) -> None:
        pass

    def copy(self, roles):
        return lambda: moe.copy_task(roles)


class CpuExperts:
    """`cpu_experts@1`'s oracle; a task over fixed buffers runs it (synchronous only: no `native`)."""

    def __init__(self, ctx, *, gate_up: str, down: str, inter: int, top_k: int, **facts) -> None:
        self.gate_up, self.down, self.inter, self.top_k = gate_up, down, inter, top_k

    def __call__(self, jobs, x, gate_up, down, rows, threads: int = 1) -> None:
        rows.copy_(moe.cpu_experts(jobs, x, gate_up, down, rows, self.gate_up, self.down, self.inter, self.top_k))

    def task(self, jobs, x, gate_up, down, rows):
        return lambda: self(jobs, x, gate_up, down, rows)


class MoeTiers:
    """`moe_tiers@1`'s oracle, writing in place what the contract writes."""

    def __init__(self, ctx, **facts) -> None:
        pass

    def __call__(self, ids, kinds, group, fetch, n_fetch, stage, delta, start, n_delta, residency, host, piece, classes,
                 owner, capacity, heat, io, moves, n_moves) -> None:
        records = [tuple(r) for r in delta[int(start[0]):int(n_delta[0])].tolist()]
        out = moe.moe_tiers(ids, kinds, group, fetch, n_fetch, stage, records, residency, host, piece, classes, owner,
                            capacity, heat)
        for name, t in (("fetch", fetch), ("residency", residency), ("host", host), ("owner", owner),
                        ("moves", moves), ("n_moves", n_moves)):
            t.copy_(out[name])
        for r, rows in enumerate(out["io"]):
            io[r, 0, 0] = len(rows)
            for i, row in enumerate(rows):
                io[r, 1 + i] = torch.tensor(row)


class MoePatchRows:
    def __init__(self, ctx, **facts) -> None:
        pass

    def publish(self, x, cpu, out) -> None:
        out.copy_(moe.moe_publish_rows(x, cpu, out))

    def __call__(self, y, rows, cpu) -> None:
        y.copy_(moe.moe_patch_rows(y, rows, cpu))


class CausalConv1d:
    def __init__(self, ctx, *, channels: int, kernel_size: int, dtype: str) -> None:
        pass

    def __call__(self, x, weight, state, slots, query_start, has_init):
        return gdn.causal_conv1d(x, weight, state, slots, query_start, has_init)


class GatedDeltaRule:
    def __init__(self, ctx, *, heads_k: int, heads_v: int, head_k_dim: int, head_v_dim: int, v_head_order: str,
                 dtype: str, out_slots: bool = False) -> None:
        self.order = v_head_order

    def plan(self, lens: list[int], snapshots=(), per_token: bool = False):
        if per_token and snapshots:
            raise ValueError("per-token rows take no snapshots")
        for row, offset, _ in snapshots:
            if offset % 64 or not 0 < offset < lens[row]:
                raise ValueError(f"snapshot of row {row} at {offset}: a multiple of 64 inside its {lens[row]} tokens")
        return (torch.tensor([0, *np.cumsum(lens).tolist()], dtype=torch.int32),
                {row: (offset, unit) for row, offset, unit in snapshots})

    def __call__(self, handle, q, k, v, a, b, a_log, dt_bias, state, slots, has_init, snapshot=None, out_slots=None):
        query_start, snaps = handle
        return gdn.gated_delta_rule(q, k, v, a, b, a_log, dt_bias, state, slots, query_start, has_init, self.order,
                                    snapshot, snaps, out_slots)


class GatedRMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, dtype: str) -> None:
        self.eps = eps

    def __call__(self, o, z, weight):
        return gdn.gated_rmsnorm(o, z, weight, self.eps)


class GDNRewind:
    def __init__(self, ctx, *, channels: int, kernel_size: int, heads_v: int, dtype: str) -> None:
        pass

    def __call__(self, state, window, x, a, b, slots, accepted, width):
        gdn.gdn_rewind(state, window, x, a, b, slots, accepted, width)


BY_OP = {"rmsnorm": RMSNorm, "add_rmsnorm": AddRMSNorm, "rope": Rope, "silu_mul": SiluMul,
         "attention.paged_prefill": PagedAttention, "attention.paged_decode": PagedAttention,
         "attention.paged_verify": PagedAttention, "attention.causal": CausalAttention,
         "qlinear": QLinear, "qlinear_glu": QLinearGlu, "qembedding": QEmbedding, "linear_experts": LinearExperts,
         "linear_experts_arena": LinearExperts,
         "moe_topk": MoeTopk, "moe_combine": MoeCombine, "moe_resolve": MoeResolve, "expert_fetch": ExpertFetch,
         "expert_prefetch": ExpertPrefetch, "cpu_task": CpuTask, "cpu_experts": CpuExperts,
         "moe_patch_rows": MoePatchRows, "moe_tiers": MoeTiers,
         "causal_conv1d": CausalConv1d, "gated_delta_rule": GatedDeltaRule, "gated_rmsnorm": GatedRMSNorm,
         "gdn_rewind": GDNRewind, "kv_write": KvWrite}
