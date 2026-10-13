"""What a dialect declares, and the normalized parts of each scheme (docs/m06-lowp.md D2): every dialect of a scheme
yields the same parts, the primary one (`weight`) first, checked at load."""
from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class Dialect:
    """A producer's naming of one scheme's parts. `parts` maps a checkpoint tensor suffix to its normalized role,
    the primary part's first; `dropped` suffixes are read and not kept, each holding the value given (None: any);
    `inverted` roles are stored as reciprocals; `ignore` names the modules the config leaves dense (module names,
    `*` wildcards, or `re:` regular expressions)."""

    elem: str
    parts: dict[str, str]
    ignore: tuple[str, ...] = ()
    dropped: dict[str, float | None] = dataclasses.field(default_factory=dict)
    inverted: frozenset[str] = frozenset()


def fp8_e4m3(weight: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """e4m3 `weight` [out, in] (as bits) and an f32 multiplier per block, the blocks tiling the weight (BF16 scales
    upcast exactly)."""
    tiled = weight.dim() == scale.dim() == 2 and 0 not in scale.shape \
        and not any(n % b for n, b in zip(weight.shape, scale.shape, strict=True))
    if weight.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.bfloat16, torch.float32) or not tiled:
        raise ValueError(f"FP8 weight {_desc(weight)} and scale {_desc(scale)}: expected an F8_E4M3 [out, in] weight "
                         "and a BF16 or F32 scale per block, the blocks tiling it")
    bits = weight.view(torch.uint8)
    _refuse((bits & 0x7F) == 0x7F, "an FP8 weight is NaN (0x7F / 0xFF)")
    return {"weight": bits, "scale": scale.float()}


def nvfp4(weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> dict[str, torch.Tensor]:
    """Packed e2m1 `weight` [out, in/2], an e4m3 `scale` per 16 (as bits) and an f32 `scale2` per tensor, []."""
    if weight.dtype != torch.uint8 or weight.dim() != 2 or scale.dtype != torch.float8_e4m3fn \
            or scale.shape != (weight.shape[0], weight.shape[1] // 8) or weight.shape[1] % 8 or scale2.numel() != 1:
        raise ValueError(f"NVFP4 weight {_desc(weight)}, scale {_desc(scale)}, scale2 {_desc(scale2)}: expected U8 "
                         "[out, in/2], F8_E4M3 [out, in/16] and one value")
    bits = scale.view(torch.uint8)
    _refuse((bits >= 0x7F), "an NVFP4 block scale is negative or NaN (sign bit, 0x7F)")
    return {"weight": weight, "scale": bits, "scale2": scale2.float().reshape(())}


def mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """Packed e2m1 `weight` [out, in/2] and an e8m0 `scale` per 32."""
    if weight.dtype != torch.uint8 or weight.dim() != 2 or scale.dtype != torch.uint8 \
            or scale.shape != (weight.shape[0], weight.shape[1] // 16) or weight.shape[1] % 16:
        raise ValueError(f"MXFP4 weight {_desc(weight)} and scale {_desc(scale)}: expected U8 [out, in/2] and U8 "
                         "[out, in/32]")
    _refuse(scale == 0xFF, "an MXFP4 scale is NaN (e8m0 0xFF)")
    return {"weight": weight, "scale": scale}


NORMALIZE = {"fp8_e4m3": fp8_e4m3, "nvfp4": nvfp4, "mxfp4": mxfp4}


def _refuse(bad: torch.Tensor, what: str) -> None:
    if bool(bad.any()):
        raise ValueError(f"{what} at {[int(i) for i in bad.nonzero()[0]]}")


def _desc(t: torch.Tensor) -> str:
    return f"{str(t.dtype).removeprefix('torch.')} {list(t.shape)}"
