"""Quantized HF checkpoints as the judge reads them (docs/dev/m06-lowp.md §4.8, D10): its own map from a module's tensor
names and dtypes to the normalized parts (not the engine's dialects), std.weights.hf's `reference.py` for the
arithmetic, and the format owners' decoders that reference is cross-checked against (`owner_check`), compared by
value in f32 ulps (signed zeros equal).

Owners (D10): FP8 — transformers' `Fp8Dequantize` (exact; given the f32 scales the dialects normalize to, since with
BF16 scales it rounds its product to bf16) or, for compressed-tensors' block FP8, its `dequantize`; compressed-tensors
NVFP4 / MXFP4 — its own unpacking, scale decoding and `dequantize`, in f32 (≤ 2 ulp, exact); ModelOpt NVFP4 —
`NVFP4QTensor.dequantize` (≤ 1 ulp; test-only, when importable); Quark MXFP4 — amd-quark's `Pack_fp4.unpack` ×
2^(e − 127) (exact; when importable); and, given a llama.cpp checkout, gguf-py on its converter's repack of every FP4
tensor (`_nvfp4_pack` × scale2, `repack_mxfp4_blocks`; exact).

`repack_check` is acceptance 4's lossless repack (docs/dev/m06-lowp.md §1): kernels.ggml_cuda's own ggml blocks of every
FP4 tensor, dequantized by gguf-py × scale2, against the reference, and byte for byte against the converter's.

`state` is a checkpoint's model as the goldens and floors build it (§4.8), one layer at a time; `gate` holds what the
engine's weight source yields against the bf16 original, per tensor and per expert (§4.8's gates). The expert and
norm patterns are the Qwen family's HF names (the judge's targets).
"""
from __future__ import annotations

import contextlib
import importlib
import itertools
import json
import pathlib
import re
import sys
import types
from collections.abc import Callable, Iterator

import torch
from holon_std.weights.hf import reference
from safetensors import safe_open

Get = Callable[[str], torch.Tensor]


# Parts of a quantized module other than its weight: read through `parts`, or not at all (input and KV scales).
AUX = frozenset({"weight_packed", "weight_scale", "weight_scale_inv", "weight_scale_2", "weight_global_scale",
                 "input_scale", "input_global_scale", "k_scale", "v_scale"})
# NMSE vs bf16 per tensor: dense exact; a dense one folded with a norm (`gate`) within the products' f32 rounding
GATES = {"fp8_e4m3": 1e-2, "nvfp4": 0.1, "mxfp4": 0.1, None: 0.0, "folded": 1e-10}
# Qwen-family HF names (Qwen3, Qwen3-MoE; Qwen3.5's `model.language_model.` in `opened`): per-expert projections; the
# gate's linears reading a norm's output (q/k/v its input norm's; gate/up, experts' and the router its
# post-attention norm's), and those norms.
_EXPERT = re.compile(r"(.+\.mlp)\.experts\.(\d+)\.(gate|up|down)_proj")
_NORMED = re.compile(r"(model\.layers\.\d+)\.(self_attn|mlp)\."
                     r"(?:[qkv]_proj|(?:experts\.\d+\.)?(?:gate|up)_proj|gate)\.weight")
_NORM = "{layer}.{which}_layernorm.weight"
_HEADER_DTYPES = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16,
                  "F32": torch.float32}


@contextlib.contextmanager
def opened(path: str | pathlib.Path, meta: bool = False) -> Iterator[tuple[Get, set[str]]]:
    """Every shard of a checkpoint directory open: (tensor by name, the names), named as the text model's state dict
    names them (a multimodal checkpoint's `model.language_model.` is `model.`, as transformers loads it). `meta`:
    tensors on the meta device, from the headers alone (shapes and dtypes; a dtype `parts` does not tell apart read
    as F32)."""
    with contextlib.ExitStack() as stack:
        where = {}
        for file in sorted(pathlib.Path(path).glob("*.safetensors")):
            f = stack.enter_context(safe_open(file, "pt"))
            where |= {re.sub(r"^model\.language_model\.", "model.", n): (f, n) for n in f.keys()}

        def get(name: str) -> torch.Tensor:
            f, n = where[name]
            if not meta:
                return f.get_tensor(n)
            s = f.get_slice(n)
            return torch.empty(s.get_shape(), dtype=_HEADER_DTYPES.get(s.get_dtype(), torch.float32), device="meta")
        yield get, set(where)


def elems(path: str | pathlib.Path) -> dict[str, str]:
    """Each quantized module's elem by `parts`' rules, read from the shard headers alone."""
    with opened(path, meta=True) as (get, names):
        modules = sorted({n.rpartition(".")[0] for n in names if n.endswith((".weight", ".weight_packed"))})
        return {m: q[0] for m in modules if (q := parts(get, names, m)) is not None}


def expert(module: str) -> bool:
    """A per-expert projection (stacked by the engine), by its HF name."""
    return _EXPERT.fullmatch(module) is not None


def quantized(path: str | pathlib.Path) -> bool:
    path = pathlib.Path(path)
    return (path / "hf_quant_config.json").is_file() or \
        "quantization_config" in json.loads((path / "config.json").read_text())


def state(get: Get, names: set[str], select: Callable[[str], bool]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The fp32 state-dict entries of the checkpoint tensors `select` keeps (by name) — dense ones as stored, each
    quantized module's weight by the reference — with every layer's per-expert projections fused as transformers
    holds them (`mlp.experts.gate_up_proj` [E, 2I, H], gate's rows first; `down_proj` [E, H, I]); and the elem of each
    quantized linear and expert stack, by module (a stack as the floors' MoE block names it: `<mlp>.gate_up`,
    `<mlp>.down`)."""
    out, kinds, experts, stacked = {}, {}, {}, {}
    for name in sorted(filter(select, names)):
        module, _, suffix = name.rpartition(".")
        if suffix == "weight" and f"{module}.weight_packed" in names or suffix in AUX - {"weight_packed"}:
            continue
        if suffix not in ("weight", "weight_packed"):
            out[name] = get(name).float()
            continue
        q = parts(get, names, module)
        w = get(name).float() if q is None else reference.dequantize(*q)
        if (m := _EXPERT.fullmatch(module)) is None:
            out[f"{module}.weight"] = w
            if q is not None:
                kinds[module] = q[0]
            continue
        experts.setdefault(m[1], {})[m[3], int(m[2])] = w
        stacked.setdefault(f"{m[1]}.{'down' if m[3] == 'down' else 'gate_up'}", set()).add(q and q[0])
    for mlp, w in experts.items():
        n = len(w) // 3
        out[f"{mlp}.experts.gate_up_proj"] = torch.stack([torch.cat([w["gate", e], w["up", e]]) for e in range(n)])
        out[f"{mlp}.experts.down_proj"] = torch.stack([w["down", e] for e in range(n)])
    for stack, elems in stacked.items():
        if len(elems) != 1:
            raise ValueError(f"{stack}: experts of different layouts {sorted(map(str, elems))}")
        if (elem := elems.pop()) is not None:
            kinds[stack] = elem
    return out, kinds


def input_scale(get: Get, names: set[str], module: str) -> torch.Tensor:
    """A module's static activation scale as the checkpoint stores it: ModelOpt's `input_scale`, compressed-tensors'
    reciprocal `input_global_scale` (the W4A4-static floor's, §4.8)."""
    if f"{module}.input_scale" in names:
        return get(f"{module}.input_scale").float().reshape(())
    return 1.0 / get(f"{module}.input_global_scale").float().reshape(())


def gate(path: str | pathlib.Path, original: str | pathlib.Path) -> tuple[dict[str, tuple[str | None, float]],
                                                                         list[str]]:
    """§4.8's gates: every tensor the engine's weight source yields — a quantized module dequantized by the reference,
    so a dialect's misreading shows; a dense one cast to the original's dtype, so exact means bit-equal — against the
    bf16 original's tensor of that name: (name -> (elem, None if dense, "folded" if dense and folded; NMSE), the norms
    folded). Per expert, as
    the source yields them (before stacking). Where a norm differs from the original's — SmoothQuant (RedHatAI's
    recipe) moves scale from it into its consumers' columns — the linears reading it are compared as they apply to
    the normalized input, W · diag(γ), each side with its own norm, and the norm is not compared alone."""
    from holon.api import QTensor
    from holon_std.weights.hf.source import HFSafetensors

    out, folded = {}, {}
    with opened(original) as (ref, _), opened(path) as (own, _):
        def smoothed(norm: str) -> bool:
            if norm not in folded:
                r = ref(norm)
                folded[norm] = not torch.equal(own(norm).to(r.dtype), r)
            return folded[norm]

        for name, t in HFSafetensors(path).tensors():
            if name.endswith("_layernorm.weight") and smoothed(name):
                continue
            q, r = isinstance(t, QTensor), ref(name)
            w = reference.dequantize(t.layout.scheme.elem, t.parts) if q else t.to(r.dtype).float()
            r = r.float()
            elem = t.layout.scheme.elem if q else None
            if (m := _NORMED.fullmatch(name)) and smoothed(
                    norm := _NORM.format(layer=m[1], which="input" if m[2] == "self_attn" else "post_attention")):
                w, r, elem = w * own(norm).float(), r * ref(norm).float(), elem or "folded"
            out[name] = (elem, ((w - r).pow(2).sum() / r.pow(2).sum()).item())
    return out, sorted(n for n, f in folded.items() if f)


def parts(get: Get, names: set[str], module: str) -> tuple[str, dict[str, torch.Tensor]] | None:
    """A module's (elem, normalized parts) by its tensors' names and dtypes alone, or None if it is dense: an e4m3
    weight with a block scale (`weight_scale_inv` | `weight_scale`) is FP8; a packed U8 weight (`weight` |
    `weight_packed`) with an e4m3 scale is NVFP4 (`weight_scale_2`, or the reciprocal of `weight_global_scale`), with
    a U8 scale MXFP4, with any other scale an error."""
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
    if s.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{module}: a packed U8 weight with a {s.dtype} scale is neither NVFP4 nor MXFP4")
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


def converter(root: str | pathlib.Path) -> Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor]:
    """llama.cpp's converter repack (its checkout at `root`): (elem, packed codes, block scales) -> ggml blocks u8,
    `_nvfp4_pack` or `repack_mxfp4_blocks`."""
    path = list(sys.path)                    # restored: the converter adds its gguf-py, whose `tests` shadows ours
    sys.path.insert(0, str(root))
    try:                                     # a module of that checkout, not a dependency
        ModelBase = importlib.import_module("conversion.base").ModelBase
    finally:
        sys.path[:] = path

    def repack(elem: str, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        blocks = ModelBase.repack_mxfp4_blocks(w, s) if elem == "mxfp4" else ModelBase._nvfp4_pack(w, s)[0]
        return torch.from_numpy(blocks)
    return repack


def dequantize_blocks(elem: str, blocks: torch.Tensor, scale2: torch.Tensor | None = None) -> torch.Tensor:
    """gguf-py's f32 dequantization of ggml FP4 blocks [rows, row_bytes], × scale2 (NVFP4's) when given."""
    import gguf

    values = torch.from_numpy(gguf.quants.dequantize(blocks.cpu().numpy(), gguf.GGMLQuantizationType[elem.upper()]))
    return values.float() if scale2 is None else values.float() * scale2.float().cpu().reshape(())


def _llama_cpp(root: str | pathlib.Path) -> Callable:
    """gguf-py's dequantization of llama.cpp's own repack (its converter at `root`), × scale2 for NVFP4."""
    repack = converter(root)

    def decode(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        w, s = raw.get("weight_packed", raw.get("weight")), raw["weight_scale"]
        if s.dtype == torch.uint8:
            return dequantize_blocks("mxfp4", repack("mxfp4", w, s))
        scale2 = raw["weight_scale_2"].float() if "weight_scale_2" in raw else 1.0 / raw["weight_global_scale"].float()
        return dequantize_blocks("nvfp4", repack("nvfp4", w, s), scale2)
    return decode


def repack_check(path: str | pathlib.Path, llama_cpp: str | pathlib.Path | None = None, device: str = "cuda",
                 limit: int | None = None) -> dict:
    """Acceptance 4: every FP4 module the engine's weight source yields, repacked by kernels.ggml_cuda (its `repack`,
    what its `prepare` holds, on `device`): gguf-py's dequantization of those blocks × scale2 against the reference
    (the largest difference in ulps, and the modules off it: none expected), and, given a llama.cpp checkout, the
    blocks against its converter's bytes (the modules that differ: none expected)."""
    from holon.api import QTensor
    from holon_plugins.kernels.ggml_cuda.plugin import FP4, repack
    from holon_std.weights.hf.source import HFSafetensors

    theirs = converter(llama_cpp) if llama_cpp is not None else None
    report = {"modules": 0, "max_ulps": 0, "over": [], "bytes_differ": [] if theirs else None}
    quantized = ((n, t) for n, t in HFSafetensors(path).tensors()
                 if isinstance(t, QTensor) and t.layout.scheme.elem in FP4)
    for name, q in itertools.islice(quantized, limit):
        elem, module = q.layout.scheme.elem, name.removesuffix(".weight")
        blocks = repack(elem, {r: p.to(device) for r, p in q.parts.items()}).cpu()
        d = int(ulps(dequantize_blocks(elem, blocks, q.parts.get("scale2")), reference.dequantize(elem, q.parts)).max())
        report["modules"] += 1
        report["max_ulps"] = max(report["max_ulps"], d)
        if d:
            report["over"].append(module)
        if theirs and not torch.equal(blocks, theirs(elem, q.parts["weight"], q.parts["scale"])):
            report["bytes_differ"].append(module)
    return report
