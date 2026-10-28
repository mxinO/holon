"""The structural test of an HF MoE load (docs/dev/m02-moe.md §4.6; quantized stacks docs/dev/m06-lowp.md §4.3): every
stacked expert slice the engine holds — the merged gate/up stack's halves, the down stack — equals the checkpoint's
tensor, and the router equals it bit for bit, streamed per layer. Dense slices are compared as stored; quantized ones
by value, the engine's parts dequantized by std.weights.hf's reference against the checkpoint's own tensors as the
judge reads them (`hf_quant`, its own name map) — or, where the stack's kernel prepared them into ggml blocks
(kernels.ggml_cuda's FP4 repack, docs/dev/m06-lowp.md D8), by gguf-py × their `scale2` row, the view leaving out the
`fp4` spec's zeroed tail. Exact and independent of any golden: a misplaced expert, or a part of one, fails here
whatever the routing touches."""
from __future__ import annotations

import pathlib

import gguf
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
    """A stack's [E, out, in] weights: a dense one as held, a quantized one dequantized by the reference, a prepared
    one by gguf-py, each half (one, or gate's and up's) times its row of `scale2` ([E] or [2, E]) if it has one."""
    if stack.layout is None:
        return stack.weight.cpu()
    parts = {role: p.cpu() for role, p in stack.parts.items()}       # a view's bytes only: no tail
    if stack.layout.name == "gguf.native":
        blocks = parts["blocks"]
        qtype = gguf.GGMLQuantizationType[stack.layout.scheme.elem.removeprefix("ggml.").upper()]
        w = torch.from_numpy(gguf.quants.dequantize(blocks.flatten(0, 1).numpy(), qtype)).view(*blocks.shape[:2], -1)
        if "scale2" not in parts:
            return w
        rows = parts["scale2"].reshape(-1, stack.experts)
        return torch.cat([h * s[:, None, None] for h, s in zip(w.chunk(len(rows), dim=1), rows, strict=True)], dim=1)
    return torch.stack([torch.cat([reference.dequantize(stack.layout.scheme.elem, u) for u in units])
                        for units in moe.expert_parts(parts, stack.experts, merged=stack.glu == "swiglu")])
