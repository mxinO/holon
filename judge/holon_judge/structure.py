"""The structural test of an HF MoE load (docs/m02-moe.md §4.6): every stacked expert slice the engine holds equals the
checkpoint's tensor bit for bit — the merged gate/up stack's halves, the down stack, the router — streamed per layer.
Exact and independent of any golden: a misplaced expert fails here whatever the routing touches."""
from __future__ import annotations

import json
import pathlib

import torch
from safetensors import safe_open


def mismatches(engine, checkpoint: str | pathlib.Path) -> list[str]:
    """The (layer, tensor) pairs whose loaded slice differs from the checkpoint's tensor; empty when the load is
    exact."""
    path = pathlib.Path(checkpoint)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"] \
        if (path / "model.safetensors.index.json").is_file() else None
    handles: dict[str, object] = {}

    def tensor(name: str) -> torch.Tensor:
        file = path / (index[name] if index else "model.safetensors")
        if file not in handles:
            handles[file] = safe_open(file, "pt")
        return handles[file].get_tensor(name)

    bad = []
    for i, layer in enumerate(engine.executor.model.layers):
        moe, mlp = layer.ffn, f"model.layers.{i}.mlp"
        inter = moe.down.in_features
        if not torch.equal(moe.router.cpu(), tensor(f"{mlp}.gate.weight").float()):
            bad.append(f"{mlp}.gate.weight")
        for e in range(moe.down.experts):
            gate_up, down = moe._gate_up.weight[e].cpu(), moe.down.weight[e].cpu()
            for got, name in ((gate_up[:inter], "gate_proj"), (gate_up[inter:], "up_proj"), (down, "down_proj")):
                if not torch.equal(got, tensor(f"{mlp}.experts.{e}.{name}.weight").to(got.dtype)):
                    bad.append(f"{mlp}.experts.{e}.{name}.weight")
    return bad
