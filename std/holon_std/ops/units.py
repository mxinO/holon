"""std.ops units: small modules that own their weights and call the kernel the core selected for them."""
from __future__ import annotations

import torch
from holon.api import QTensor
from torch import nn

from . import kernels


def _kernel(ctx, op: str, site: str, **facts):
    """This op's kernel for these facts, chosen by the core among the installed kernel plugins."""
    return ctx.select_kernel(op, site=site, **facts).impl(ctx, **facts)


class RMSNorm(nn.Module):
    def __init__(self, ctx, *, dim: int, eps: float, offset: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=ctx.dtype), requires_grad=False)
        facts = {"dim": dim, "eps": eps, "offset": offset, "dtype": str(ctx.dtype).removeprefix("torch.")}
        site = f"std.ops.rmsnorm[dim={dim}]"
        self._norm = _kernel(ctx, kernels.RMSNORM, site, **facts)
        self._add_norm = _kernel(ctx, kernels.ADD_RMSNORM, site, **facts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x, self.weight)

    def add(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(norm(residual + x), residual + x), fused where the kernel allows; `x` and `residual` may be reused."""
        return self._add_norm(x, residual, self.weight)


class Rope(nn.Module):
    """Rotates queries and keys together (neox layout)."""

    def __init__(self, ctx, *, theta: float, style: str = "neox", rotary_dim: int | None = None,
                 head_dim: int | None = None) -> None:
        super().__init__()
        if style != "neox":
            raise ValueError(f"unsupported rope style {style!r}")
        if head_dim is None:
            raise ValueError("rope needs the head_dim of the site that uses it")
        self._rope = _kernel(ctx, kernels.ROPE, f"std.ops.rope[head_dim={head_dim}]", head_dim=head_dim,
                             rotary_dim=rotary_dim or head_dim, theta=theta,
                             max_positions=ctx.options.get("max_model_len", 8192),
                             dtype=str(ctx.dtype).removeprefix("torch."))

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rope(q, k, positions)


class _Quantized:
    """A unit's quantized weight (adopted at load, design §5.5): its layout and the op that reads it. The kernel is
    bound by `bind`, after merges, so a unit merged away binds none and leaves no site in the lock."""

    def __init__(self, ctx, layout, op: str, **facts) -> None:
        self.ctx, self.layout, self.op = ctx, layout, op
        self.facts = {"layout": f"{layout.scheme.elem}/{layout.name}", **facts}
        self._kernel = None

    def bind(self, parts: dict[str, torch.Tensor], shape: tuple[int, ...],
             param: nn.Parameter | None) -> dict[str, torch.Tensor]:
        """Select the kernel; returns the parts it reads: `parts`, or what a `qlinear@2` kernel's `prepare` makes of
        them (docs/m06-lowp.md D4), the primary part swapped into `param` so the source parts are released (a tied
        head has no `param`). `qembedding@1` has no `prepare`: a table is read as adopted."""
        site = f"std.ops.{self.op.split('@')[0]}[" + ",".join(f"{k}={v}" for k, v in self.facts.items()) + "]"
        spec = self.ctx.select_kernel(self.op, site=site, **self.facts)
        _one_act_mode(self.ctx, spec, site)
        self._kernel = spec.impl(self.ctx, **self.facts)
        if self.op != kernels.QLINEAR or not hasattr(self._kernel, "prepare"):
            return parts
        q = QTensor(self.layout, shape, parts)
        if (prepared := self._kernel.prepare(q)) is q:
            return parts
        self.layout = prepared.layout                    # what the parts now are (the site's facts keep the adopted)
        return dict(prepared.parts) if param is None else _hold(param, prepared.parts, self.ctx.device)

    def __call__(self, x: torch.Tensor, weight) -> torch.Tensor:
        return self._kernel(x, weight)


def _hold(param: nn.Parameter, parts: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """`parts` on the device by role, the first (the primary part) held by `param`: what the parameter held (meta,
    or a part `prepare` replaced) is swapped out, the Parameter object kept."""
    (role, primary), *rest = parts.items()
    torch.utils.swap_tensors(param, nn.Parameter(primary.to(device), requires_grad=False))
    return {role: param} | {r: t.to(device) for r, t in rest}


def _act_quant(ctx) -> str:
    """The deployment's `quant.act` plugin option for quantized weights (default `a8`, docs/m06-lowp.md D3)."""
    act = ctx.options.get("plugins", {}).get("quant.act", "a8")
    if act not in kernels.ACT_QUANT:
        raise ValueError(f"quant.act {act!r}: one of {list(kernels.ACT_QUANT)} for quantized weights")
    return act


def _one_act_mode(ctx, spec, site: str) -> None:
    """Mixed steps (`chunk_with_decode` > 0, docs/m07-perf.md §3.1) run decode rows and a prefill chunk in one call, on
    the path the call's row count picks: a kernel whose paths compute in different activation modes (its spec's
    `config["act"]`, path -> mode) would then run a decode row at a precision its unmixed step does not use."""
    modes = spec.config.get("act", {})
    if ctx.options.get("chunk_with_decode", 0) > 0 and len(set(modes.values())) > 1:
        raise ValueError(f"{site}: chunk_with_decode > 0 needs one activation mode on every path; {spec.provider}'s "
                         f"kernel computes {modes}")


def _dtype(t: torch.dtype) -> str:
    return str(t).removeprefix("torch.")


def _by_rows(parts: list[torch.Tensor], outs: list[int]) -> bool:
    """Every unit's part has `out / g` rows of its unit's `out`, for one `g`; a part of rank ≤ 1 and one element ([]
    or [1]) in every unit is a per-tensor value instead (a per-row part is 2-D: an FP8 block scale [1, 1] is a row)."""
    if all(p.dim() <= 1 and p.numel() == 1 for p in parts):
        return False
    gs = {out // p.shape[0] if p.dim() and p.shape[0] and out % p.shape[0] == 0 else 0
          for p, out in zip(parts, outs, strict=True)}
    return len(gs) == 1 and 0 not in gs


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


def _merged_parts(units: tuple[Linear, ...]) -> dict[str, torch.Tensor] | None:
    """The units' quantized parts as one weight's (docs/m06-lowp.md D5): a part with `out / g` rows in every unit is
    concatenated by rows; any other (a per-tensor scale, [] or [1]) must be bit-equal in every unit, else None."""
    if len({tuple(u.parts) for u in units}) > 1:
        return None
    merged = {}
    for role, first in units[0].parts.items():
        parts = [u.parts[role] for u in units]
        if _by_rows(parts, [u.out_features for u in units]):
            merged[role] = torch.cat(parts)
        elif all(_same_bits(p, first) for p in parts):
            merged[role] = first
        else:
            return None
    return merged


class Linear(nn.Module):
    """`x @ weight.T (+ bias)`. Dense, or a quantized weight adopted at load: its parts by role, the first (the primary
    part) held by `weight`, read by the `qlinear@2` kernel bound after merges."""

    def __init__(self, ctx, *, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features, self.out_features, self._ctx = in_features, out_features, ctx
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=ctx.dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=ctx.dtype), requires_grad=False) if bias else None
        self._q: _Quantized | None = None
        self.parts: dict[str, torch.Tensor] | None = None            # a quantized weight's, by role

    def _quantized(self, layout) -> _Quantized:
        dtype = _dtype(self._ctx.dtype)
        return _Quantized(self._ctx, layout, kernels.QLINEAR, **{"in": self.in_features, "out": self.out_features},
                          act_dtype=dtype, out_dtype=dtype, act_quant=_act_quant(self._ctx))

    def adopt_weights(self, ctx, name: str, q, index: int | None = None) -> bool:
        if name != "weight" or index is not None or tuple(q.logical_shape) != (self.out_features, self.in_features):
            return False
        self._q, self.parts = self._quantized(q.layout), _hold(self.weight, q.parts, ctx.device)
        return True

    def bind_kernels(self, ctx) -> None:
        if self._q is not None and self.weight is not None:          # not merged away
            self.parts = self._q.bind(self.parts, (self.out_features, self.in_features), self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._q is None:
            return nn.functional.linear(x, self.weight, self.bias)
        y = self._q(x, self.parts)
        return y if self.bias is None else y + self.bias

    @staticmethod
    def merge(*units: Linear) -> Linear | None:
        """One unit for several projections of the same input — one GEMM, or one quantized matmul, instead of
        several — whose output rows are the units' outputs in order; it owns the weight and the units give theirs up
        (they are not called afterwards). None if they cannot share one: biases, different dtypes, layouts or input
        sizes, or quantized parts that merge neither by rows nor as equal values (`_merged_parts`; ggml rows are
        independent block runs, so weights of one type concatenate by rows).

        Optional capability of every `unit.std.ops.linear@1` provider, used duck-typed by composites (attention q/k/v,
        gate/up) in `finalize_weights`."""
        first = units[0]
        kinds = {(u.weight.dtype, u._q and u._q.facts["layout"], u.in_features) for u in units}
        if any(u.bias is not None for u in units) or len(kinds) > 1:
            return None
        parts = {"weight": torch.cat([u.weight.data for u in units])} if first._q is None else _merged_parts(units)
        if parts is None:
            return None
        with torch.device("meta"):
            merged = Linear(first._ctx, in_features=first.in_features, out_features=sum(u.out_features for u in units))
        held = _hold(merged.weight, parts, first._ctx.device)
        if first._q is not None:
            merged._q, merged.parts = merged._quantized(first._q.layout), held
        for u in units:
            u.weight = u.parts = None
        return merged


class Embedding(nn.Module):
    """The token table: dense, or a quantized one adopted at load (its parts as `Linear`'s), read by `qembedding@1`
    from `weight`, its primary part; a tied head reads all its parts."""

    def __init__(self, ctx, *, vocab: int, hidden: int) -> None:
        super().__init__()
        self.vocab, self.hidden, self.dtype = vocab, hidden, ctx.dtype
        self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)
        self._q: _Quantized | None = None
        self.parts: dict[str, torch.Tensor] | None = None

    def adopt_weights(self, ctx, name: str, q, index: int | None = None) -> bool:
        if name != "weight" or index is not None or tuple(q.logical_shape) != (self.vocab, self.hidden):
            return False
        self._q = _Quantized(ctx, q.layout, kernels.QEMBEDDING, vocab=self.vocab, hidden=self.hidden,
                             dtype=_dtype(self.dtype))
        self.parts = _hold(self.weight, q.parts, ctx.device)
        return True

    def bind_kernels(self, ctx) -> None:
        if self._q is not None:
            self.parts = self._q.bind(self.parts, (self.vocab, self.hidden), self.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(ids, self.weight) if self._q is None else self._q(ids, self.weight)


class LMHead(nn.Module):
    """Output projection; `tied=True` reuses the embedding table (passed in as `embed`). Logits are float32."""

    def __init__(self, ctx, *, vocab: int, hidden: int, tied: bool, embed: Embedding | None = None) -> None:
        super().__init__()
        self.vocab, self.hidden, self.dtype = vocab, hidden, ctx.dtype
        if tied:
            if embed is None:
                raise ValueError("tied lm_head needs the embedding unit")
            # Held in a list, not as a submodule: the shared parameter must keep a single owner.
            self._embed = [embed]
            self.weight = None
        else:
            self._embed = []
            self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)
        self._q: _Quantized | None = None
        self.parts: dict[str, torch.Tensor] | None = None

    def _quantized(self, ctx, layout) -> _Quantized:
        return _Quantized(ctx, layout, kernels.QLINEAR, **{"in": self.hidden, "out": self.vocab},
                          act_dtype=_dtype(self.dtype), out_dtype="float32", act_quant=_act_quant(ctx))

    def adopt_weights(self, ctx, name: str, q, index: int | None = None) -> bool:
        if self._embed or name != "weight" or index is not None or tuple(q.logical_shape) != (self.vocab, self.hidden):
            return False
        self._q, self.parts = self._quantized(ctx, q.layout), _hold(self.weight, q.parts, ctx.device)
        return True

    def finalize_weights(self, ctx) -> None:
        if self._embed and self._embed[0]._q is not None:          # tied to a quantized table: read its parts
            self._q = self._quantized(ctx, self._embed[0]._q.layout)

    def bind_kernels(self, ctx) -> None:
        if self._q is not None:                                    # after the embedding's (model order)
            # tied: the embedding's parts as adopted (qembedding@1 has no `prepare`); this head prepares its own copy
            parts = self._embed[0].parts if self._embed else self.parts
            self.parts = self._q.bind(parts, (self.vocab, self.hidden), self.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self._q is not None:
            return self._q(h, self.parts)
        return nn.functional.linear(h, self._embed[0].weight if self._embed else self.weight).float()
