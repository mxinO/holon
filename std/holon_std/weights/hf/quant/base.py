"""What a dialect declares, and the normalized parts of each scheme (docs/m06-lowp.md D2): every dialect of a scheme
yields the same parts, the primary one (`weight`) first, checked at load, and the logical shape they pack."""
from __future__ import annotations

import dataclasses
from typing import Any

import torch

Normalized = tuple[dict[str, torch.Tensor], tuple[int, int]]          # parts, logical [out, in]


@dataclasses.dataclass(frozen=True)
class Dialect:
    """A producer's naming of one scheme's parts. `parts` maps a checkpoint tensor suffix to its normalized role,
    the primary part's first; `dropped` suffixes are read and not kept, each holding the value given (None: any);
    `inverted` roles are stored as reciprocals; `ignore` names the modules the config leaves dense (module names,
    `*` wildcards, or `re:` regular expressions); `args` are what the scheme's normalization reads from the config
    (FP8: its `block`)."""

    elem: str
    parts: dict[str, str]
    ignore: tuple[str, ...] = ()
    dropped: dict[str, float | None] = dataclasses.field(default_factory=dict)
    inverted: frozenset[str] = frozenset()
    args: dict[str, Any] = dataclasses.field(default_factory=dict)


def fp8_e4m3(weight: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> Normalized:
    """e4m3 `weight` [out, in] (as bits), both dimensions multiples of the config's block, and an f32 multiplier per
    block, [out / bm, in / bn] (BF16 scales upcast exactly)."""
    shape = tuple(weight.shape)
    whole = len(shape) == 2 and not shape[0] % block[0] and not shape[1] % block[1]
    if weight.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.bfloat16, torch.float32) or not whole \
            or scale.shape != (shape[0] // block[0], shape[1] // block[1]):
        raise ValueError(f"FP8 weight {_desc(weight)} and scale {_desc(scale)}: expected an F8_E4M3 [out, in] weight "
                         f"of whole {block[0]} × {block[1]} blocks and a BF16 or F32 scale per block")
    bits = weight.view(torch.uint8)
    _refuse((bits & 0x7F) == 0x7F, "an FP8 weight is NaN (0x7F / 0xFF)")
    return {"weight": bits, "scale": scale.float()}, shape


def nvfp4(weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> Normalized:
    """Packed e2m1 `weight` [out, in/2], an e4m3 `scale` per 16 (as bits) and an f32 `scale2` per tensor, []."""
    if weight.dtype != torch.uint8 or weight.dim() != 2 or scale.dtype != torch.float8_e4m3fn \
            or scale.shape != (weight.shape[0], weight.shape[1] // 8) or weight.shape[1] % 8 or scale2.numel() != 1:
        raise ValueError(f"NVFP4 weight {_desc(weight)}, scale {_desc(scale)}, scale2 {_desc(scale2)}: expected U8 "
                         "[out, in/2], F8_E4M3 [out, in/16] and one value")
    bits = scale.view(torch.uint8)
    _refuse((bits >= 0x7F), "an NVFP4 block scale is negative or NaN (sign bit, 0x7F)")
    shape = (weight.shape[0], 2 * weight.shape[1])
    return {"weight": weight, "scale": bits, "scale2": scale2.float().reshape(())}, shape


def mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> Normalized:
    """Packed e2m1 `weight` [out, in/2] and an e8m0 `scale` per 32."""
    if weight.dtype != torch.uint8 or weight.dim() != 2 or scale.dtype != torch.uint8 \
            or scale.shape != (weight.shape[0], weight.shape[1] // 16) or weight.shape[1] % 16:
        raise ValueError(f"MXFP4 weight {_desc(weight)} and scale {_desc(scale)}: expected U8 [out, in/2] and U8 "
                         "[out, in/32]")
    _refuse(scale == 0xFF, "an MXFP4 scale is NaN (e8m0 0xFF)")
    return {"weight": weight, "scale": scale}, (weight.shape[0], 2 * weight.shape[1])


NORMALIZE = {"fp8_e4m3": fp8_e4m3, "nvfp4": nvfp4, "mxfp4": mxfp4}


def _refuse(bad: torch.Tensor, what: str) -> None:
    if bool(bad.any()):
        raise ValueError(f"{what} at {[int(i) for i in bad.nonzero()[0]]}")


def _desc(t: torch.Tensor) -> str:
    return f"{str(t.dtype).removeprefix('torch.')} {list(t.shape)}"
