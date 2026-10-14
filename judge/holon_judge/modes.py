"""The activation modes a floor emulates on a quantized HF checkpoint (docs/m06-lowp.md §4.8, D3): every quantized
linear and expert stack takes its input as the kernel path's mode quantizes it, then multiplies in f32 by its
(dequantized) weight. Modes are named as `kernels.ggml_cuda` counts its calls (`<path>/<mode>`):

- `q8_1` (W4A8, either path): `q8_1.py`'s blocking of the path (FP4 types use no stored sums);
- `fp4` (W4A4-dynamic, mmq only): llama.cpp's native FP4 quantizer, emulated by `w4a4.values`;
- `fp4-static` (W4A4-static, mmq only, NVFP4 dense linears: a report): `w4a4.static_nvfp4` over each projection's
  static input scale from the checkpoint.
- `fp8` (W8A8, `kernels.triton_fp8`'s `prefill` path, FP8 weights): `fp8_act`, per row and 128 columns. FP8's decode
  path (W8A16: the activations as given) is scored against the plain bf16 floor (A16), which emulates nothing.

Each input row is quantized once, as the kernels do: a token's input to a merged gate/up stack once for its k slots,
each (token, slot) row of down on its own.
"""
from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from holon_std.moe import reference

from . import q8_1, w4a4

MODES = ("q8_1", "fp4", "fp4-static", "fp8")
PATHS = ("mmq", "mmvq", "prefill")         # kernels.ggml_cuda's two; kernels.triton_fp8's W8A8 one


def fp8_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W8A8's activations (docs/m06-lowp.md D12): per row and 128 columns of f32 `x [R, in]`, a_s = max(amax, 1e-10)
    / 448 and e4m3 of `x / a_s` (IEEE division, clamped to ±448, rounded to nearest even): `(e4m3 [R, in], a_s
    [R, in / 128])`, bitwise `kernels.triton_fp8`'s (its L0)."""
    g = x.float().unflatten(-1, (-1, 128))
    a_s = g.abs().amax(-1).clamp_min(1e-10) / 448
    return (g / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2), a_s


def fp8_values(x: torch.Tensor) -> torch.Tensor:
    """The f32 values W8A8 multiplies: `fp8_act`'s codes times their scales."""
    q, a_s = fp8_act(x)
    return (q.float().unflatten(-1, (-1, 128)) * a_s[..., None]).flatten(-2)


def quantizer(elem: str, path: str, mode: str, scale: torch.Tensor | None = None) -> Callable[[torch.Tensor],
                                                                                              torch.Tensor]:
    """f32 rows [R, in] -> the values the kernel multiplies (f32), for an FP4 weight of `elem` on `path` in `mode`.
    Or, for an FP8 weight, W8A8 on `prefill`; ggml types are the GGUF floors' (`gguf.emulate_q8_1`)."""
    if elem == "fp8_e4m3":
        if (path, mode) != ("prefill", "fp8"):
            raise ValueError(f"{elem}: an FP8 floor emulates W8A8 on prefill (`--q8-1 prefill --act fp8`), not {mode} "
                             f"on {path}; its decode path (W8A16) is scored against the A16 floor")
        return fp8_values
    if elem not in w4a4.GROUP or mode == "fp8":
        raise ValueError(f"{elem}: no {mode} emulation (fp8 is FP8 weights' W8A8)")
    if mode == "q8_1":
        b = q8_1.blocking(elem, path)                    # FP4 types use no stored sums
        return lambda x: q8_1.quantize(x, b).float()
    if path != q8_1.MMQ:
        raise ValueError(f"{mode} is mmq's, not {path}'s")
    if mode == "fp4":
        return functools.partial(w4a4.values, elem)
    if mode == "fp4-static" and elem == "nvfp4" and scale is not None:
        return lambda x: w4a4.static_nvfp4(x, scale.to(x.device))
    raise ValueError(f"no {mode} emulation for {elem}")


def emulate(module: torch.nn.Module, kinds: dict[str, str], path: str, mode: str, prefix: str = "",
            scales: Callable[[str], torch.Tensor] | None = None) -> int:
    """Every quantized linear of `module` (named `prefix` + its name in `kinds`, their elems) computes in `mode` on
    `path`; `scales(name)`: a projection's static input scale (`fp4-static`). Returns the number changed."""
    n = 0
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear) and (elem := kinds.get(prefix + name)) is not None:
            quant = quantizer(elem, path, mode, scales(prefix + name) if mode == "fp4-static" else None)
            m.forward = functools.partial(_linear, m, quant)
            n += 1
    return n


def _linear(linear: torch.nn.Linear, quant: Callable, x: torch.Tensor) -> torch.Tensor:
    y = quant(x.reshape(-1, x.shape[-1]).float()) @ linear.weight.float().T
    return y.to(x.dtype).reshape(*x.shape[:-1], -1)


def experts(kinds: dict[str, str], path: str, mode: str) -> Callable:
    """The floors' MoE block's expert stacks (`moe.mirror(experts=…)`): a quantized stack (`<mlp>.gate_up`, `<mlp>.down`
    in `kinds`) takes its rows quantized in `mode` on `path`, then the f32 product (`reference.linear_experts`); a dense
    one is the reference's."""
    def apply(name: str, x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, glu: str) -> torch.Tensor:
        if (elem := kinds.get(name)) is not None:
            if mode == "fp4-static":
                raise ValueError(f"{name}: the W4A4-static floor is for dense linears (a report on the 8B)")
            x = quantizer(elem, path, mode)(x.reshape(-1, x.shape[-1]).float()).view(x.shape)
        return reference.linear_experts(x, weight, ids, glu)
    return apply
