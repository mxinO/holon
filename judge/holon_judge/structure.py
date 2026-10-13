"""The structural test of an HF MoE load (docs/m02-moe.md §4.6; quantized stacks docs/m06-lowp.md §4.3): every
stacked expert slice the engine holds — the merged gate/up stack's halves, the down stack — equals the checkpoint's
tensor, and the router equals it bit for bit, streamed per layer. Dense slices are compared as stored; quantized ones
by value, the engine's parts dequantized by std.weights.hf's reference against the checkpoint's own tensors as the
judge reads them (`hf_quant`, its own name map). Exact and independent of any golden: a misplaced expert, or a part
of one, fails here whatever the routing touches."""
from __future__ import annotations

import pathlib

import torch
from holon_std.moe import reference as moe
from holon_std.weights.hf import reference

from . import hf_quant


def mismatches(engine, checkpoint: str | pathlib.Path) -> list[str]:
    """The (layer, tensor) pairs whose loaded slice differs from the checkpoint's tensor; empty when the load is
    exact."""
    bad = []
    with hf_quant.opened(checkpoint) as (get, names):
        for i, layer in enumerate(engine.executor.model.layers):
            moe, mlp = layer.ffn, f"model.layers.{i}.mlp"
            inter = moe.down.in_features
            if not torch.equal(moe.router.cpu(), get(f"{mlp}.gate.weight").float()):
                bad.append(f"{mlp}.gate.weight")
            gate_up, down = _experts(moe._gate_up), _experts(moe.down)
            for e in range(moe.down.experts):
                for got, proj in ((gate_up[e, :inter], "gate"), (gate_up[e, inter:], "up"), (down[e], "down")):
                    module = f"{mlp}.experts.{e}.{proj}_proj"
                    if not torch.equal(got, hf_quant.weight(get, names, module).to(got.dtype)):
                        bad.append(f"{module}.weight")
    return bad


def _experts(stack) -> torch.Tensor:
    """A stack's [E, out, in] weights: a dense one as held, a quantized one dequantized by the reference."""
    if stack.layout is None:
        return stack.weight.cpu()
    parts = {role: p.cpu() for role, p in stack.parts.items()}
    return torch.stack([torch.cat([reference.dequantize(stack.layout.scheme.elem, u) for u in units])
                        for units in moe.expert_parts(parts, stack.experts, merged=stack.glu == "swiglu")])
