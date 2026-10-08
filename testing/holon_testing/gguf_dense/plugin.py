"""Tier A's weight source (M0.1 design §4.6): a GGUF file read by the production source — its configuration, names
and tokenizer — with every quantized tensor dequantized by gguf-py at load, so the engine runs its dense path and is
judged against the dequantized weights at M0 sensitivity. Never detected (no `detect` patterns): named with
`EngineConfig(source="gguf_dense")`. Its kind stays "gguf", so the recipe applies the GGUF weight map; plans refuse
test-only plugins, so it never reaches a lock."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from holon.api import QTensor
from holon_std.weights.gguf.source import GGUFSource


class GGUFDense(GGUFSource):
    def tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, t in super().tensors():
            if isinstance(t, QTensor):
                kind = GGMLQuantizationType[t.layout.scheme.elem.removeprefix("ggml.").upper()]
                x = dequantize(t.parts["blocks"].numpy(), kind).astype(np.float32, copy=False)
                t = torch.from_numpy(x).reshape(tuple(t.logical_shape))
            yield name, t


class Plugin:
    def register(self, regs) -> None:
        regs.weight_sources.add("gguf_dense", GGUFDense)
