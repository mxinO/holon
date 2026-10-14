"""GGUF files on the judge's side (M0.1 design §4.6, M0.2 §4.6): the dense HF model a file stands for, built with
gguf-py's own dequantization through the judge's own GGUF → HF name map (never the engine's loader or weight map) —
expert stacks assembled as transformers stores them — and the gate that checks what both maps believe (orientation,
no Q/K permutation, expert order, gate before up) against transformers' own loading of the checkpoint.
"""
from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import re
from collections.abc import Callable, Iterator

import gguf
import numpy as np
import torch

GATE = 0.5          # NMSE(dequantized, HF bf16) per tensor: a correct quant explains at least half the energy
GATES = {"mxfp4": 0.1, "q8_0": 1e-3}     # docs/m06-lowp.md §4.8: FP4 experts; Q8_0 from its prediction (≈ 5e-5)
# llama.cpp's tensor names (src/llama-arch.cpp) for the dense-transformer tensors -> HF Qwen3 names.
_GLOBAL = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}
_LAYER = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
          "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
          "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj",
          "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "ffn_gate_inp": "mlp.gate"}
# Expert stacks [E, out, in] -> transformers' fused parameters: gate_up_proj = per expert gate rows then up rows.
_EXPERTS = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}
# Qwen3.5/3.8 (`qwen35`): its own norm name and the Gated DeltaNet tensors (llama.cpp conversion/qwen.py), by full
# suffix since `ssm_a` and `ssm_dt.bias` are not `.weight`s.
_QWEN35 = {"post_attention_norm.weight": "post_attention_layernorm.weight",
           "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
           "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
           "ssm_out.weight": "linear_attn.out_proj.weight", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
           "ssm_norm.weight": "linear_attn.norm.weight", "ssm_a": "linear_attn.A_log",
           "ssm_dt.bias": "linear_attn.dt_bias"}
NAME_MAP_HASH = hashlib.sha256(json.dumps([_GLOBAL, _LAYER, _EXPERTS, _QWEN35], sort_keys=True).encode()
                               ).hexdigest()[:16]


def hf_name(name: str) -> str:
    if name in _GLOBAL:
        return _GLOBAL[name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m is not None and m[2] in _QWEN35:
        return f"model.layers.{m[1]}.{_QWEN35[m[2]]}"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
    if m is None or m[2] not in _LAYER:
        raise KeyError(f"no HF name for GGUF tensor {name!r}")
    return f"model.layers.{m[1]}.{_LAYER[m[2]]}.weight"


def _qwen35(fields) -> dict | None:
    """The `qwen35` facts the judge needs to undo llama.cpp's conversion (None for other architectures)."""
    if bytes(fields["general.architecture"].parts[-1]).decode() != "qwen35":
        return None
    value = lambda k: int(fields[f"qwen35.{k}"].parts[-1][0])                  # noqa: E731
    blocks = value("block_count") - (value("nextn_predict_layers") if "qwen35.nextn_predict_layers" in fields else 0)
    hk, hv = value("ssm.group_count"), value("ssm.time_step_rank")
    return {"layers": blocks, "hk": hk, "hv": hv, "dv": value("ssm.inner_size") // hv, "dk": value("ssm.state_size")}


def _untile(x: torch.Tensor, f: dict, dim: int, start: int = 0) -> torch.Tensor:
    """Value heads from llama.cpp's tiled order (v-head j reads k-head j mod Hk) back to HF's grouped one (j // r),
    along `dim` from `start` on, each head a block of `x.shape[dim]`'s per-head width."""
    hk, hv = f["hk"], f["hv"]
    r = hv // hk
    tiled = [s * hk + h for h in range(hk) for s in range(r)]                  # grouped position g = h·r + s
    head = (x.shape[dim] - start) // hv
    index = torch.tensor([start + t * head + i for t in tiled for i in range(head)])
    return torch.cat([x.narrow(dim, 0, start), x.index_select(dim, index)], dim=dim)


def _qwen35_order(name: str, x: torch.Tensor, f: dict) -> torch.Tensor:
    """`x` (a tensor of `name`'s rows × columns, or anything laid out like it, e.g. its q8_1 offsets per group) with
    value heads back in grouped order: rows of the per-v-head projections, the input columns of `ssm_out`."""
    part = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if part in ("ssm_a", "ssm_dt.bias", "ssm_beta.weight", "ssm_alpha.weight", "attn_gate.weight"):
        return _untile(x, f, 0)
    if part in ("attn_qkv.weight", "ssm_conv1d.weight"):
        return _untile(x, f, 0, start=2 * f["hk"] * f["dk"])
    if part == "ssm_out.weight":
        return _untile(x, f, 1)
    return x


def _qwen35_value(name: str, x: torch.Tensor, f: dict) -> torch.Tensor:
    """Undo the converter's value changes: zero-centred norms (all but the GDN norm) stored as w + 1, `ssm_a` as
    −exp(A_log), conv1d squeezed, value heads tiled."""
    part = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if part.endswith("norm.weight") and part != "ssm_norm.weight":
        return x - 1
    x = _qwen35_order(name, torch.log(-x) if part == "ssm_a" else x, f)
    return x[:, None, :] if part == "ssm_conv1d.weight" else x


_DENSE = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}


def hf_state(path: str | pathlib.Path, select: Callable[[str], bool] = lambda name: True
             ) -> Iterator[tuple[str, torch.Tensor]]:
    """The HF state-dict entries (fp32) the file's tensors make (those whose GGUF name `select` keeps, e.g. one
    layer's `blk.<i>.`), dequantized by gguf-py one tensor at a time; a layer's gate and up expert stacks are fused as
    transformers stores them (`gate_up_proj` [E, 2I, H])."""
    pending: dict[str, dict[str, torch.Tensor]] = {}
    reader = gguf.GGUFReader(path)
    q35 = _qwen35(reader.fields)
    for t in reader.tensors:
        if q35 is not None and re.match(rf"blk\.{q35['layers']}\.", t.name):
            continue                                   # the MTP block (M0.4)
        if not select(t.name):
            continue
        x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)
        x = torch.from_numpy(x.reshape([int(n) for n in reversed(t.shape)]).copy())
        if q35 is not None:
            x = _qwen35_value(t.name, x, q35)
        m = re.fullmatch(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight", t.name)
        if m is None:
            yield hf_name(t.name), x
        elif _EXPERTS[m[2]] == "down":
            yield f"model.layers.{m[1]}.mlp.experts.down_proj", x
        else:
            pending.setdefault(m[1], {})[_EXPERTS[m[2]]] = x
            if len(pending[m[1]]) == 2:
                p = pending.pop(m[1])
                yield f"model.layers.{m[1]}.mlp.experts.gate_up_proj", torch.cat([p["gate"], p["up"]], dim=1)


def hf_types(path: str | pathlib.Path) -> dict[str, str]:
    """The ggml type (lowercase) of each HF state entry `hf_state` makes (a fused gate/up stack: gate's)."""
    out = {}
    for t in gguf.GGUFReader(path).tensors:
        if m := re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", t.name):
            name = f"model.layers.{m[1]}.mlp.experts.{'down' if m[2] == 'down' else 'gate_up'}_proj"
            if m[2] == "up":
                continue
        else:
            try:
                name = hf_name(t.name)
            except KeyError:
                continue                                       # the MTP block, which hf_state skips
        out[name] = t.tensor_type.name.lower()
    return out


def dequantized(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Every tensor of the file, dequantized by gguf-py (fp32), by HF name."""
    return dict(hf_state(path))


def emulate_q8_1(model: torch.nn.Module, path: str | pathlib.Path, kernel_path: str,
                 prefix: str = "") -> dict[str, str]:
    """Tier B's floor (design §4.6): every quantized linear of `model` (the HF model of `path`, lm_head included, or a
    part of it named `prefix`, e.g. one streamed layer) computes as llama.cpp's int8 dots do on `kernel_path`
    (`q8_1.MMQ` or `q8_1.MMVQ`) — its input rounded to q8_1, the stored-sum offsets of its type — in float32 over
    its weights, writing its own dtype (the head: float32). Returns the ggml type of each linear it changed."""
    from . import q8_1

    linears = {prefix + name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    types = {}
    reader = gguf.GGUFReader(path)
    q35 = _qwen35(reader.fields)
    for t in reader.tensors:
        if q35 is not None and re.match(rf"blk\.{q35['layers']}\.", t.name):
            continue
        if t.name.endswith("_exps.weight"):            # expert stacks: the floors' D3 block (`q8_1_experts`)
            continue
        owner = hf_name(t.name).removesuffix(".weight")
        if t.tensor_type in _DENSE or owner not in linears:
            continue                                   # norms, and the embedding (get_rows reads it exactly)
        kind, blocks = t.tensor_type.name.lower(), np.asarray(t.data)
        g = q8_1.blocking(kind, kernel_path).sum_group
        offs = q8_1.offsets(blocks, kind, g).float() if g else None
        if offs is not None and q35 is not None:          # the weight's value heads were untiled: so are its offsets
            offs = _qwen35_order(t.name, offs, q35)
        offs = None if offs is None else offs.to(linears[owner].weight.device)
        linears[owner].forward = functools.partial(_emulated, linears[owner], kind, kernel_path, offs,
                                                   torch.float32 if owner == "lm_head" else None)
        types[owner] = kind
    return types


def q8_1_experts(path: str | pathlib.Path, kernel_path: str):
    """Tier B's expert stacks for the floors' D3 block (`moe.mirror(experts=…)`): each applies llama.cpp's MUL_MAT_ID
    on `kernel_path` (`q8_1.linear_experts`, f32) with the type and stored-sum offsets of the file's stack; a merged
    gate/up stack takes gate's type (gate and up share one in every file in scope) and both offsets."""
    from . import q8_1

    stacks: dict[str, dict[str, tuple[str, np.ndarray]]] = {}
    for t in gguf.GGUFReader(path).tensors:
        m = re.fullmatch(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight", t.name)
        if m:
            blocks = np.asarray(t.data)                                     # [E, out, row_bytes] -> rows
            stacks.setdefault(f"model.layers.{m[1]}.mlp", {})[_EXPERTS[m[2]]] = (t.tensor_type.name.lower(),
                                                                                blocks.reshape(-1, blocks.shape[-1]))
    cache: dict[str, torch.Tensor | None] = {}

    def offsets(key: str, parts: list[tuple[str, np.ndarray]], e: int, device) -> torch.Tensor | None:
        """Each stack's offsets [E, out, in/group] (merged: gate's then up's rows), once per stack."""
        if key not in cache:
            g = q8_1.blocking(parts[0][0], kernel_path).sum_group
            if not g:
                cache[key] = None
            else:
                per = [q8_1.offsets(b, kind, g).float() for kind, b in parts]
                cache[key] = torch.cat([o.view(e, -1, o.shape[-1]) for o in per], dim=1).to(device)
        return cache[key]

    def experts(name: str, x, weight, ids, glu):
        block, which = name.rsplit(".", 1)
        parts = [stacks[block]["gate"], stacks[block]["up"]] if which == "gate_up" else [stacks[block]["down"]]
        if len({kind for kind, _ in parts}) != 1:
            raise ValueError(f"{block}: gate and up experts of different types")
        offs = offsets(name, parts, weight.shape[0], weight.device)
        return q8_1.linear_experts(x, weight, None, ids, parts[0][0], kernel_path, glu, offs, torch.float32)
    return experts


def mmvq_step(path: str | pathlib.Path) -> int:
    """The largest step at which every quantized matmul of the file takes mmvq on the current GPU — dense tensors by
    llama.cpp's dense limit, expert stacks by its MoE one (the plugin's own predicates): Tier B's decode run."""
    from holon_plugins.kernels.ggml_cuda import extension, plugin

    major, minor = torch.cuda.get_device_capability()
    lib, device = extension.load(10 * major + minor), torch.cuda.current_device()
    limits = []
    for t in gguf.GGUFReader(path).tensors:
        if t.tensor_type in _DENSE or t.name == "token_embd.weight":     # get_rows reads the embedding
            continue
        use = lib.use_mmvq_id if t.name.endswith("_exps.weight") else lib.use_mmvq
        kind = plugin.TYPES[t.tensor_type.name.lower()]
        limits.append(max((n for n in range(1, 9) if use(device, kind, n)), default=0))
    return min(limits)


def _emulated(linear: torch.nn.Linear, kind: str, kernel_path: str, offs: torch.Tensor | None,
              out_dtype: torch.dtype | None, x: torch.Tensor) -> torch.Tensor:
    from . import q8_1

    rows = x.reshape(-1, x.shape[-1])
    y = q8_1.matmul(rows, linear.weight, None, kind, kernel_path, offs, torch.float32)
    return y.to(out_dtype or x.dtype).reshape(*x.shape[:-1], -1)


def gate(path: str | pathlib.Path, hf_dir: str | pathlib.Path) -> dict[str, float]:
    """NMSE of every dequantized tensor against the HF checkpoint, per expert for expert stacks, one layer of the
    checkpoint at a time: its tensors as the judge reads them (`hf_quant.state`, whose fusion of per-expert tensors is
    transformers' own, bit for bit: judge/tests/test_gguf_moe.py::
    test_the_golden_is_hfs_model_of_the_dequantized_weights), so a mistake in this file's assembly (expert order,
    gate before up) fails here even if the engine makes the same one. MoE routers must be exact (F32 in the file, the
    checkpoint's values upcast): a ValueError otherwise."""
    from . import hf_quant

    out, layer, reference = {}, None, {}
    with hf_quant.opened(hf_dir) as (get, names):
        for name, x in hf_state(path):
            if (m := re.match(r"model\.layers\.\d+\.", name)) is None:
                ref = get(name).float()
            else:
                if m[0] != layer:
                    layer, reference = m[0], {}           # the previous layer's tensors freed first
                    reference = hf_quant.state(get, names, lambda n, start=layer: n.startswith(start))[0]
                ref = reference[name]
            if name.endswith(".mlp.gate.weight") and not torch.equal(x.float(), ref):
                raise ValueError(f"{path}: router {name} differs from the checkpoint's")
            if ".mlp.experts." in name:
                for e in range(x.shape[0]):
                    out[f"{name}[{e}]"] = _nmse(x[e], ref[e])
            else:
                out[name] = _nmse(x, ref)
    return out


def _nmse(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x - ref).pow(2).sum() / ref.pow(2).sum()).item()


def file_meta(path: str | pathlib.Path) -> dict:
    """Provenance without local paths: the file (hub repo and stem for a hub-cache file), its sha256 (the blob name
    for a hub-cache symlink), the gguf-py that dequantized it and the name map's hash."""
    from importlib.metadata import version

    p = pathlib.Path(path)
    hub = [part for part in p.parts if part.startswith("models--")]
    model = f"{hub[-1].removeprefix('models--').replace('--', '/')}/{p.stem}" if hub else p.stem
    if p.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", p.resolve().name):
        digest = p.resolve().name
    else:
        with open(p, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"model": model, "gguf_sha256": digest, "gguf_py": version("gguf"), "name_map": NAME_MAP_HASH}
