"""Quantized HF checkpoints as the judge reads them (docs/m06-lowp.md §4.8, D10): its own map from a module's tensor
names and dtypes to the normalized parts (not the engine's dialects), std.weights.hf's `reference.py` for the
arithmetic, and the format owners' decoders that reference is cross-checked against (`owner_check`), compared by
value in f32 ulps (signed zeros equal).

Owners (D10): FP8 — transformers' `Fp8Dequantize` (exact; given the f32 scales the dialects normalize to, since with
BF16 scales it rounds its product to bf16) or, for compressed-tensors' block FP8, its `dequantize`; compressed-tensors
NVFP4 / MXFP4 — its own unpacking, scale decoding and `dequantize`, in f32 (≤ 2 ulp, exact); ModelOpt NVFP4 —
`NVFP4QTensor.dequantize` (≤ 1 ulp; test-only, when importable); Quark MXFP4 — amd-quark's `Pack_fp4.unpack` ×
2^(e − 127) (exact; when importable); and, given a llama.cpp checkout, gguf-py on its converter's repack of every FP4
tensor (`_nvfp4_pack` × scale2, `repack_mxfp4_blocks`; exact).
"""
from __future__ import annotations

import contextlib
import itertools
import json
import pathlib
import sys
import types
from collections.abc import Callable, Iterator

import torch
from holon_std.weights.hf import reference
from safetensors import safe_open

Get = Callable[[str], torch.Tensor]


@contextlib.contextmanager
def opened(path: str | pathlib.Path) -> Iterator[tuple[Get, set[str]]]:
    """Every shard of a checkpoint directory open: (tensor by name, the names)."""
    with contextlib.ExitStack() as stack:
        where = {}
        for file in sorted(pathlib.Path(path).glob("*.safetensors")):
            f = stack.enter_context(safe_open(file, "pt"))
            where |= dict.fromkeys(f.keys(), f)
        yield (lambda name: where[name].get_tensor(name)), set(where)


def parts(get: Get, names: set[str], module: str) -> tuple[str, dict[str, torch.Tensor]] | None:
    """A module's (elem, normalized parts) by its tensors' names and dtypes alone, or None if it is dense: an e4m3
    weight with a block scale (`weight_scale_inv` | `weight_scale`) is FP8; a packed U8 weight (`weight` |
    `weight_packed`) with an e4m3 scale is NVFP4 (`weight_scale_2`, or the reciprocal of `weight_global_scale`), with
    a U8 scale MXFP4."""
    def has(suffix: str) -> bool:
        return f"{module}.{suffix}" in names

    w = get(f"{module}.weight_packed" if has("weight_packed") else f"{module}.weight")
    if w.dtype == torch.float8_e4m3fn:
        s = get(f"{module}.weight_scale_inv" if has("weight_scale_inv") else f"{module}.weight_scale")
        return "fp8_e4m3", {"weight": w.view(torch.uint8), "scale": s.float()}
    if w.dtype != torch.uint8:
        return None
    s = get(f"{module}.weight_scale")
    if s.dtype == torch.uint8:
        return "mxfp4", {"weight": w, "scale": s}
    scale2 = get(f"{module}.weight_scale_2").float() if has("weight_scale_2") else \
        1.0 / get(f"{module}.weight_global_scale").float()
    return "nvfp4", {"weight": w, "scale": s.view(torch.uint8), "scale2": scale2.reshape(())}


def weight(get: Get, names: set[str], module: str) -> torch.Tensor:
    """A module's weight as the checkpoint means it, fp32: dense as stored, quantized by the reference."""
    q = parts(get, names, module)
    return get(f"{module}.weight").float() if q is None else reference.dequantize(*q)


def ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a − b| in f32 ulps, on the ordered integer line of the f32 bit patterns (+0 and −0 are one point; NaN is
    refused at load)."""
    def line(x: torch.Tensor) -> torch.Tensor:
        bits = x.float().contiguous().view(torch.int32).long()
        return torch.where(bits < 0, -(bits & 0x7FFFFFFF), bits)
    return (line(a) - line(b)).abs()


def quantization_config(path: str | pathlib.Path) -> dict:
    """config.json's `quantization_config`, else hf_quant_config.json's (ModelOpt's own file), as the method's."""
    path = pathlib.Path(path)
    if qc := json.loads((path / "config.json").read_text()).get("quantization_config"):
        return qc
    raw = json.loads((path / "hf_quant_config.json").read_text())
    return {"quant_method": raw["producer"]["name"], **raw["quantization"]}


def owners(qc: dict, llama_cpp: str | pathlib.Path | None = None) -> dict[str, tuple[Callable, int]]:
    """The owners' decoders of a checkpoint's format (its quantization config), name -> (decode(the module's raw
    tensors by suffix) -> f32, ulp bound); those whose package is missing are left out."""
    method, fmt = qc.get("quant_method"), qc.get("format")
    out: dict[str, tuple[Callable, int]] = {}
    if method == "fp8":
        out["transformers.Fp8Dequantize"] = (_transformers_fp8(qc["weight_block_size"]), 0)
    elif method == "compressed-tensors":
        out["compressed-tensors"] = (_compressed_tensors(fmt), 2 if fmt == "nvfp4-pack-quantized" else 0)
    elif method == "modelopt" and (modelopt := _modelopt()):
        out["modelopt.NVFP4QTensor"] = (modelopt, 1)
    elif method == "quark" and (quark := _quark()):
        out["quark.Pack_fp4"] = (quark, 0)
    if llama_cpp is not None and (method, fmt) != ("compressed-tensors", "float-quantized") and method != "fp8":
        out["gguf-py on llama.cpp's repack"] = (_llama_cpp(llama_cpp), 0)
    return out


def owner_check(path: str | pathlib.Path, llama_cpp: str | pathlib.Path | None = None,
                limit: int | None = None) -> dict:
    """Every quantized module the engine's weight source yields, dequantized by the reference, against each owner's
    decoding of the checkpoint's own tensors: per owner the modules checked, the largest difference in ulps, and
    the modules over the owner's bound. `limit`: the first modules only."""
    from holon.api import QTensor
    from holon_std.weights.hf.source import HFSafetensors

    decoders = owners(quantization_config(path), llama_cpp)
    report = {name: {"modules": 0, "max_ulps": 0, "over": []} for name in decoders}
    with opened(path) as (get, names):
        by_module: dict[str, list[str]] = {}
        for n in names:
            by_module.setdefault(n.rpartition(".")[0], []).append(n)
        quantized = ((n, t) for n, t in HFSafetensors(path).tensors() if isinstance(t, QTensor))
        for name, q in itertools.islice(quantized, limit):
            module = name.removesuffix(".weight")
            ours = reference.dequantize(q.layout.scheme.elem, q.parts)
            raw = {n.rpartition(".")[2]: get(n) for n in by_module[module]}
            for owner, (decode, bound) in decoders.items():
                d = int(ulps(ours, decode(raw)).max())
                r = report[owner]
                r["modules"] += 1
                r["max_ulps"] = max(r["max_ulps"], d)
                if d > bound:
                    r["over"].append(module)
    return report


def _transformers_fp8(block: list[int]) -> Callable:
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    op = Fp8Dequantize(types.SimpleNamespace(quantization_config=types.SimpleNamespace(weight_block_size=block)))
    return lambda raw: op.convert({"weight$": [raw["weight"]], "weight_scale_inv": [raw["weight_scale_inv"].float()]},
                                  full_layer_name="w")["w"]


def _compressed_tensors(fmt: str) -> Callable:
    from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    def decode(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        if fmt == "float-quantized":
            w, s = raw["weight"], raw["weight_scale"]
            block = [w.shape[0] // s.shape[0], w.shape[1] // s.shape[1]]
            args = QuantizationArgs(num_bits=8, type="float", strategy=QuantizationStrategy.BLOCK,
                                    block_structure=block)
            return dequantize(w.float(), s.float(), args=args, dtype=torch.float32)
        packed, s = raw["weight_packed"], raw["weight_scale"]
        codes = unpack_fp4_from_uint8(packed, packed.shape[0], 2 * packed.shape[1], dtype=torch.float32)
        args = QuantizationArgs(num_bits=4, type="float", strategy=QuantizationStrategy.GROUP,
                                group_size=codes.shape[1] // s.shape[1])
        if fmt == "mxfp4-pack-quantized":
            return dequantize(codes, decompress_mx_scale(s).float(), args=args, dtype=torch.float32)
        return dequantize(codes, s.float(), args=args, global_scale=raw["weight_global_scale"].float(),
                          dtype=torch.float32)
    return decode


def _modelopt() -> Callable | None:
    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    except ImportError:
        return None

    def decode(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        w = raw["weight"]
        shape = torch.Size((w.shape[0], 2 * w.shape[1]))
        q = NVFP4QTensor(shape, torch.float32, w)
        return q.dequantize(dtype=torch.float32, scale=raw["weight_scale"], double_scale=raw["weight_scale_2"].float(),
                            block_sizes={-1: 16})
    return decode


def _quark() -> Callable | None:
    try:
        from quark.torch.utils.pack import Pack_fp4
    except ImportError:
        return None
    pack = Pack_fp4(None, "fp4")

    def decode(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        values = pack.unpack(raw["weight"], reorder=True)
        e = raw["weight_scale"].double() - 127
        return (values.double() * torch.exp2(e).repeat_interleave(values.shape[1] // e.shape[1], 1)).float()
    return decode


def _llama_cpp(root: str | pathlib.Path) -> Callable:
    """gguf-py's dequantization of llama.cpp's own repack (its converter at `root`), × scale2 for NVFP4."""
    import gguf
    import numpy as np

    path = list(sys.path)                    # restored: the converter adds its gguf-py, whose `tests` shadows ours
    sys.path.insert(0, str(root))
    try:
        from conversion.base import ModelBase
    finally:
        sys.path[:] = path

    def decode(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        w, s = raw.get("weight_packed", raw.get("weight")), raw["weight_scale"]
        if s.dtype == torch.uint8:
            blocks = ModelBase.repack_mxfp4_blocks(w, s)
            return torch.from_numpy(gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.MXFP4).astype(np.float32))
        blocks, _ = ModelBase._nvfp4_pack(w, s)
        scale2 = raw["weight_scale_2"].float() if "weight_scale_2" in raw else 1.0 / raw["weight_global_scale"].float()
        values = torch.from_numpy(gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.NVFP4).astype(np.float32))
        return values * scale2.reshape(())
    return decode
