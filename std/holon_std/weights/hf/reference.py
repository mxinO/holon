"""Reference dequantization of the `hf` layouts (docs/m06-lowp.md D10): the oracle every provider of these layouts is
scored against, itself cross-checked against each format owner's decoder (transformers, compressed-tensors, ModelOpt,
amd-quark, gguf-py on llama.cpp's repacks). Group and block sizes come from the parts' shapes, so a weight merged by
rows (D5) dequantizes as its units would; fp32 out, every product rounded once.
"""
from __future__ import annotations

import torch

# e2m1 code -> value: bit 3 the sign, bits 2..0 index the magnitudes (OCP MX v1.0; ModelOpt and compressed-tensors
# share the table).
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequantize(elem: str, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The fp32 [out, in] weight of one `<elem>/hf` tensor's parts (the dialects' normalized parts, D2):

    - `fp8_e4m3`: `weight` (e4m3 bits) · `scale` [out / bm, in / bn], one f32 multiplier per block.
    - `nvfp4`: `(e2m1(weight) · e4m3(scale)) · scale2`, a scale per 16 (`in / scale columns`) and one per tensor.
    - `mxfp4`: `e2m1(weight) · 2^(scale − 127)`, an e8m0 scale per 32.
    """
    w, s = parts["weight"], parts["scale"]
    if elem == "fp8_e4m3":
        rows, cols = w.shape[0] // s.shape[0], w.shape[1] // s.shape[1]
        return w.view(torch.float8_e4m3fn).float() * s.float().repeat_interleave(rows, 0).repeat_interleave(cols, 1)
    codes = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(w.shape[0], -1)       # element 2i in the low nibble
    values = E2M1.to(w.device)[codes.long()]
    group = values.shape[1] // s.shape[1]
    if elem == "nvfp4":
        return values * s.view(torch.float8_e4m3fn).float().repeat_interleave(group, 1) * parts["scale2"].float()
    if elem == "mxfp4":
        return values * e8m0(s).repeat_interleave(group, 1)
    raise ValueError(f"no reference dequantization for {elem!r}/hf")


def dequantize_experts(elem: str, parts: dict[str, torch.Tensor], experts: int) -> torch.Tensor:
    """The fp32 [E, out, in] weights of a `linear_experts@2` stack of `<elem>/hf` experts: each part per expert, [E,
    …]; a per-tensor part is [E] (or [E, 1]) or, in a merged gate/up stack, [2, E], gate's half of every expert's
    rows taking row 0 and up's row 1 (D5)."""
    merged = (2, experts)
    halves = 2 if any(p.shape == merged for p in parts.values()) else 1

    def part(p: torch.Tensor, e: int, h: int) -> torch.Tensor:
        if p.shape == merged:
            return p[h, e]
        return p.reshape(-1)[e] if p.dim() <= 2 and p.numel() == experts else p[e].chunk(halves)[h]
    return torch.stack([torch.cat([dequantize(elem, {r: part(p, e, h) for r, p in parts.items()})
                                   for h in range(halves)]) for e in range(experts)])


def e8m0(s: torch.Tensor) -> torch.Tensor:
    """2^(s − 127) in f32, exactly: the biased exponent placed in the f32 exponent field (s = 0 is the subnormal
    2^-127; 255, NaN, is refused at load)."""
    bits = torch.where(s > 0, s.int() << 23, torch.tensor(1 << 22, dtype=torch.int32, device=s.device))
    return bits.view(torch.float32)
