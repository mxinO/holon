"""Tiny quantized HF checkpoints for CPU tests, one per dialect (docs/dev/m06-lowp.md §5 step A2): a random Qwen3 or
Qwen3-MoE from transformers, its linear weights quantized and saved as each producer stores them — compressed-tensors'
by its own compressors and config, Qwen's FP8, ModelOpt's NVFP4 and Quark's MXFP4 from their specs — in two shards
that split modules' parts between them. `write` also returns every weight as the file means it (fp32, by HF name,
compressed-tensors' decoded by compressed-tensors): the oracle checkpoint transformers runs, which shares nothing with
the engine's dialects or `reference.py`."""
from __future__ import annotations

import json
import pathlib
import re

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

DIALECTS = ("fp8", "ct-fp8", "ct-nvfp4", "ct-mxfp4", "modelopt", "quark")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FUSED = re.compile(r"(.*)\.(q_proj|k_proj|v_proj|gate_proj|up_proj)$")     # modules one scale2 is shared across


def tiny(path: pathlib.Path, moe: bool, seed: int = 0) -> pathlib.Path:
    """A random two-layer Qwen3 (or Qwen3-MoE, 4 experts, top 2) saved in fp32, whose every linear dimension is a
    multiple of 128 and most weights span several FP8 blocks each way (q 2 × 2, gate 3 × 2, down 2 × 3), so a
    transposed block scale is visible."""
    torch.manual_seed(seed)
    common = dict(vocab_size=512, hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=64, tie_word_embeddings=False, eos_token_id=None)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(intermediate_size=384, moe_intermediate_size=256, num_experts=4,
                                               num_experts_per_tok=2, norm_topk_prob=True, **common)) if moe \
        else Qwen3ForCausalLM(Qwen3Config(intermediate_size=384, **common))
    with torch.no_grad():                    # weights large enough that their rounding moves the logits
        for name, p in model.named_parameters():
            if p.dim() >= 2 and "embed" not in name:
                p.mul_(4)
    model.save_pretrained(path)
    return path


def write(checkpoint: pathlib.Path, out: pathlib.Path, dialect: str) -> dict[str, torch.Tensor]:
    """`checkpoint` (a `tiny` one) as `dialect` stores it in `out`; returns every tensor as the file means it."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    dense = load_file(checkpoint / "model.safetensors")
    moe = cfg["model_type"] == "qwen3_moe"
    linear = {n.removesuffix(".weight") for n, t in dense.items()
              if t.dim() == 2 and "embed" not in n and n != "lm_head.weight" and not n.endswith("mlp.gate.weight")}
    if dialect == "quark" and moe:                                   # amd-quark's 30B: experts only
        linear = {m for m in linear if ".experts." in m}
    tensors, meant, qc = WRITERS[dialect](dense, sorted(linear), cfg)
    tensors |= {n: t for n, t in dense.items() if n.removesuffix(".weight") not in linear}
    meant |= {n: t.float() for n, t in dense.items() if n.removesuffix(".weight") not in linear}
    out.mkdir(parents=True, exist_ok=True)
    ignored = sorted({n.removesuffix(".weight") for n, t in dense.items() if t.dim() == 2} - linear)
    _config(out, cfg, dialect, qc, ignored, separate=moe)
    names = sorted(tensors)
    shards = {f"model-0000{i + 1}-of-00002.safetensors": names[i::2] for i in range(2)}     # parts split
    for file, part in shards.items():
        save_file({n: tensors[n].contiguous() for n in part}, out / file)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: f for f, part in shards.items() for n in part}}))
    return meant


def _config(out: pathlib.Path, cfg: dict, dialect: str, qc: dict, ignored: list[str], separate: bool) -> None:
    """config.json with the dialect's `quantization_config`; ModelOpt's MoE writes it in hf_quant_config.json only
    (the other place the source reads it from)."""
    if dialect == "modelopt":
        quant = {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16, "exclude_modules": ignored}
        if separate:
            (out / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"},
                                                                  "quantization": quant}))
            qc = None
        else:
            qc = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ignored}
    elif dialect == "fp8":
        qc = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
              "modules_to_not_convert": ignored}
    elif dialect == "quark":
        spec = {"dtype": "fp4", "group_size": 32, "qscheme": "per_group", "scale_format": "e8m0", "ch_axis": -1}
        qc = {"quant_method": "quark", "exclude": ignored, "export": {"pack_method": "reorder"},
              "global_quant_config": {"weight": spec | {"is_dynamic": False},
                                      "input_tensors": spec | {"is_dynamic": True}},
              "layer_quant_config": {}, "layer_type_quant_config": {}}
    else:
        qc |= {"ignore": [f"re:^{re.escape(m)}$" if ".mlp.gate" in m else m for m in ignored]}
    (out / "config.json").write_text(json.dumps(cfg | ({"quantization_config": qc} if qc else {})))


def _fused(linear: list[str]) -> dict[str, str]:
    """module -> its fused group (q/k/v, a dense gate/up) or itself: producers share one global scale across a group
    so it runs as one GEMM. An expert's gate and up keep their own (shared in the real files read, not guaranteed),
    so a merged stack's [2, E] rows differ."""
    return {m: (f"{g[1]}.{'qkv' if g[2][0] in 'qkv' else 'gate_up'}"
                if (g := FUSED.fullmatch(m)) and ".experts." not in m else m) for m in linear}


def _e2m1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Codes (sign bit 3) and values of x rounded to the nearest e2m1 magnitude (saturating at 6)."""
    idx = (x.abs()[..., None] - E2M1).abs().argmin(-1)
    codes = (idx | (torch.signbit(x).long() << 3)).to(torch.uint8)
    return codes, torch.where(torch.signbit(x), -E2M1[idx], E2M1[idx])


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return codes[:, 0::2] | (codes[:, 1::2] << 4)                # element 2i in the low nibble


def _fp8(dense, linear, cfg):
    """Qwen's FP8 (DeepSeek-V3 recipe): `amax / 448` per 128 × 128 block, BF16 multipliers (F32 for the MoE, as
    Qwen3-30B-A3B-FP8 stores them)."""
    moe = cfg["model_type"] == "qwen3_moe"
    tensors, meant = {}, {}
    for m in linear:
        w = dense[f"{m}.weight"].float()
        blocks = w.unflatten(0, (-1, 128)).unflatten(2, (-1, 128))                    # [R, 128, C, 128]
        s = (blocks.abs().amax((1, 3)) / 448).to(torch.float32 if moe else torch.bfloat16)
        q = (blocks / s.float()[:, None, :, None]).flatten(2).flatten(0, 1).to(torch.float8_e4m3fn)
        tensors |= {f"{m}.weight": q, f"{m}.weight_scale_inv": s}
        meant[f"{m}.weight"] = (q.float().unflatten(0, (-1, 128)).unflatten(2, (-1, 128))
                                * s.float()[:, None, :, None]).flatten(2).flatten(0, 1)
    return tensors, meant, {}


def _modelopt(dense, linear, cfg):
    """ModelOpt NVFP4 (`nvfp4_tensor.py`): `weight_scale_2 = amax / (6 · 448)` shared across q/k/v and gate/up, an
    e4m3 scale per 16 of `amax16 / (6 · weight_scale_2)`, codes of `w / (scale · weight_scale_2)`; F32 `input_scale`
    per module, `k_scale` / `v_scale` of 1.0 on k/v."""
    fused, tensors, meant = _fused(linear), {}, {}
    amax = {}
    for m in linear:
        amax[fused[m]] = max(amax.get(fused[m], 0.0), dense[f"{m}.weight"].abs().max().item())
    for m in linear:
        w = dense[f"{m}.weight"].float()
        scale2 = torch.tensor(amax[fused[m]] / (6 * 448), dtype=torch.float32)
        groups = w.unflatten(1, (-1, 16))
        s = (groups.abs().amax(-1) / (6 * scale2)).clamp(2**-9, 448).to(torch.float8_e4m3fn)
        per_block = s.float() * scale2
        codes, values = _e2m1(groups / per_block[..., None])
        tensors |= {f"{m}.weight": _pack(codes.flatten(1)), f"{m}.weight_scale": s, f"{m}.weight_scale_2": scale2,
                    f"{m}.input_scale": torch.tensor(0.01)}
        tensors |= {f"{m}.{kv}_scale": torch.tensor(1.0) for kv in "kv" if m.endswith(f"{kv}_proj")}
        meant[f"{m}.weight"] = (values * per_block[..., None]).flatten(1)
    return tensors, meant, {}


def _quark(dense, linear, cfg):
    """Quark MXFP4 (OCP MX v1.0): an e8m0 scale per 32 of `floor(log2(amax32)) − 2`, codes of `w / 2^(e − 127)`."""
    tensors, meant = {}, {}
    for m in linear:
        groups = dense[f"{m}.weight"].float().unflatten(1, (-1, 32))
        e = (torch.floor(torch.log2(groups.abs().amax(-1))) - 2 + 127).clamp(0, 254)
        scale = torch.ldexp(torch.tensor(1.0, dtype=torch.float64), e.double() - 127)
        codes, values = _e2m1((groups / scale[..., None]).float())
        tensors |= {f"{m}.weight": _pack(codes.flatten(1)), f"{m}.weight_scale": e.to(torch.uint8)}
        meant[f"{m}.weight"] = (values.double() * scale[..., None]).float().flatten(1)
    return tensors, meant, {}


def _compressed_tensors(preset: str, fmt: str):
    """compressed-tensors' own quantization parameters (`calculate_qparams`, `generate_gparam`, the global shared
    across q/k/v and gate/up as llm-compressor fuses them), compressor and config; the file decoded by its own
    functions in fp32."""
    def write(dense, linear, cfg):
        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.compressors.mx_utils import decompress_mx_scale
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization import QuantizationConfig, preset_name_to_scheme
        from compressed_tensors.quantization.lifecycle.forward import dequantize
        from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

        scheme = preset_name_to_scheme(preset, ["Linear"])
        scheme.format, args, compressor = fmt, scheme.weights, BaseCompressor.get_value_from_registry(fmt)
        fused, tensors, meant, extremes = _fused(linear), {}, {}, {}
        for m in linear:
            w = dense[f"{m}.weight"]
            lo, hi = extremes.get(fused[m], (w.min(), w.max()))
            extremes[fused[m]] = (torch.minimum(lo, w.min()), torch.maximum(hi, w.max()))
        for m in linear:
            w = dense[f"{m}.weight"].float()
            if preset == "FP8_BLOCK":
                blocks = w.unflatten(0, (-1, 128)).unflatten(2, (-1, 128))
                lo, hi, gs = blocks.amin((1, 3)), blocks.amax((1, 3)), None
            else:
                groups = w.unflatten(1, (-1, args.group_size))
                lo, hi = groups.amin(-1), groups.amax(-1)
                gs = generate_gparam(*extremes[fused[m]]) if preset == "NVFP4" else None
            scale, _ = calculate_qparams(lo, hi, args, global_scale=gs)
            state = {"weight": w, "weight_scale": scale} | ({"weight_global_scale": gs} if gs is not None else {})
            packed = compressor.compress(state, scheme)
            tensors |= {f"{m}.{k}": v for k, v in packed.items()}
            if preset == "NVFP4":
                tensors[f"{m}.input_global_scale"] = torch.tensor([100.0])
            if preset == "FP8_BLOCK":
                meant[f"{m}.weight"] = dequantize(packed["weight"].float(), packed["weight_scale"].float(),
                                                  args=args, dtype=torch.float32)
            else:
                codes = unpack_fp4_from_uint8(packed["weight_packed"], *w.shape, dtype=torch.float32)
                s = packed["weight_scale"].float() if preset == "NVFP4" else \
                    decompress_mx_scale(packed["weight_scale"]).float()
                meant[f"{m}.weight"] = dequantize(codes, s, args=args, global_scale=gs, dtype=torch.float32)
        qc = QuantizationConfig(config_groups={"group_0": scheme}, format=fmt, quantization_status="compressed")
        return tensors, meant, qc.model_dump(mode="json")
    return write


WRITERS = {"fp8": _fp8, "modelopt": _modelopt, "quark": _quark,
           "ct-fp8": _compressed_tensors("FP8_BLOCK", "float-quantized"),
           "ct-nvfp4": _compressed_tensors("NVFP4", "nvfp4-pack-quantized"),
           "ct-mxfp4": _compressed_tensors("MXFP4", "mxfp4-pack-quantized")}
