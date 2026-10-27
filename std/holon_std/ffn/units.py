from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

from . import kernels

LINEAR = "std.ops.linear@1"


class SwiGLU(nn.Module):
    def __init__(self, ctx, *, hidden: int, inter: int, layer: int = 0) -> None:
        super().__init__()
        self.gate_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=inter))
        self.up_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=inter))
        self.down_proj = ctx.unit(Ref(LINEAR, in_features=inter, out_features=hidden))
        facts = {"inter": inter, "dtype": str(ctx.dtype).removeprefix("torch.")}
        spec = ctx.select_kernel(kernels.SILU_MUL, site=f"std.ffn.swiglu[inter={inter}]", **facts)
        self._act = spec.impl(ctx, **facts)
        self._gate_up: nn.Module | None = None

    def finalize_weights(self, ctx) -> None:
        merge = getattr(type(self.gate_proj), "merge", None)       # one matmul for gate and up where possible
        self._gate_up = merge(self.gate_proj, self.up_proj) if merge else None
        if self._gate_up is not None and hasattr(self._gate_up, "use_glu"):   # and the activation fused into it
            self._gate_up.use_glu("swiglu")                                    # where its kernel can (D6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gate_up is not None:
            fused = self._gate_up.glu(x) if hasattr(self._gate_up, "glu") else None
            return self.down_proj(self._act(self._gate_up(x)) if fused is None else fused)
        h = getattr(self.gate_proj, "prepare_input", lambda x: x)(x)    # one input, two projections (D6)
        return self.down_proj(self._act(torch.cat([self.gate_proj(h), self.up_proj(h)], dim=-1)))


class GatedSwiGLU(SwiGLU):
    """A shared expert gated per token (docs/dev/v1-flash-next.md D7): `swiglu(x) · σ(gate · x)`, `gate` [1, hidden]."""

    def __init__(self, ctx, *, hidden: int, inter: int, layer: int = 0) -> None:
        super().__init__(ctx, hidden=hidden, inter=inter, layer=layer)
        self.gate = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * torch.sigmoid(self.gate(x))


class Sum(nn.Module):
    """FFN parts summed (D7): each `parts` Ref built at this layer (e.g. routed experts, then a gated shared expert).
    The later parts are summed in order, then added to the first part's output — in the first part's own kernel when
    it has the ffn interface's `add_residual` (std.moe@1: its combine's `add`, one launch less), bitwise the plain add
    (for two parts, the same sum as before: IEEE addition commutes). On CUDA, where the first part takes a `join`
    (std.moe@1), the later parts run on a stream of their own beside it — the shared expert beside the routed
    experts' router, resolve and fetch — and join just before its combine reads their sum: the same kernels on the
    same inputs, so bitwise the same, in graphs too (the side stream forks from and joins the capture's). The plugin
    option `ffn.fork=off` keeps them on one stream."""

    def __init__(self, ctx, *, hidden: int, parts: tuple[Ref, ...], layer: int = 0) -> None:
        super().__init__()
        self.parts = nn.ModuleList(ctx.unit(p, hidden=hidden, layer=layer) for p in parts)
        cuda = torch.device(ctx.device).type == "cuda" and \
            str(ctx.options.get("plugins", {}).get("ffn.fork", "on")) != "off"
        # the later parts' stream: one per engine, shared by its sum units (built once, released with the engine)
        self._side = ctx.shared("std.ffn.side_stream", lambda root: torch.Stream(device=root.device)) if cuda else None

    def _rest(self, x: torch.Tensor) -> torch.Tensor:
        acc = self.parts[1](x)
        for part in self.parts[2:]:
            acc = acc + part(x)
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.parts[0]
        if len(self.parts) == 1:
            return first(x)
        add = getattr(first, "add_residual", None)
        if add is None:
            return first(x) + self._rest(x)
        if self._side is None or not getattr(first, "joins", False):
            return add(x, self._rest(x))
        main = torch.accelerator.current_stream(x.device)
        self._side.wait_stream(main)                         # x is ready
        with self._side:
            acc = self._rest(x)
        # `acc` goes back to the side stream's blocks when it is dropped, with no event: the side stream is shared by
        # the engine's sum units and waits for the main stream's work at every fork, so no later write of the side
        # stream can pass the main stream's read of it (an event would keep one per layer alive while the host runs
        # ahead: a prompt step's 11 shared-expert outputs, 20 MiB each at 4K tokens, at its activation peak)
        return add(x, acc, join=lambda: main.wait_stream(self._side))
