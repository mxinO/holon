"""llama.cpp's native W4A4 activations (docs/m06-lowp.md §4.8): the FP4 codes and scales its quantizer
(cbe28bb3 quantize.cu, CUDA ≥ 12.8) gives each f32 activation row, emulated, their values, and the tie classes
`-use_fast_math` may round the other way.

- MXFP4 (`quantize_mmq_mxfp4`): per 32 values the e8m0 scale `e = rint(log2(amax)) − 2 + 127` (ties to even;
  clamped to [0, 254]; amax 0 gives e 0 and zero codes), codes the e2m1 rounding of `x · 2^−(e − 127)` (to nearest
  even, saturating at ±6). Fast-math's `log2f` may move `rint` where log2(amax) is near a half-integer.
- NVFP4 (`quantize_mmq_nvfp4`): per row the f32 scale `r = amax / (6 · 448)` (read from the kernel, as its rounding
  is fast-math's); per 16 values `v = x · (1/r)`, and a UE4M3 scale among the codes c0 + {0, −1, 1, −2, 2} (c0 the
  e4m3 rounding of max|v| / 6; out of [0, 0x7E] skipped) with the least squared error Σ (|v| − |e2m1(v · inv)| ·
  2s)² accumulated by FMA in order (s = e4m3(c) / 2, inv = 0.5 / s; the first of equals wins), codes the e2m1
  rounding of `x · (1/r · inv)`. Fast-math's reciprocal and divisions (1/r, max|v| / 6, 0.5 / s) are within a few
  ulp of correctly rounded ones.

A block's values are `e2m1(code) · 2^(e − 127)` (MXFP4) or `e2m1(code) · e4m3(c) · r` (NVFP4). `blocks` are the
kernel's `block_fp4_mmq` [in/256, rows, 144] (`kernels.ggml_cuda`'s test entry): per 256 values 16 scale bytes (MXFP4
two e8m0 per 64-value quad in the low half of each uint32; NVFP4 one UE4M3 per 16), then per scale group its codes,
byte j holding element j low and j + half the group high.
"""
from __future__ import annotations

import itertools

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])            # between e2m1 magnitudes
GROUP = {"mxfp4": 32, "nvfp4": 16}
NUDGE = 4                                                                  # ulps a fast-math result may be off


def e2m1(v: torch.Tensor) -> torch.Tensor:
    """The e2m1 codes of f32 `v` (bit 3 the sign): magnitudes rounded to nearest, ties to even, saturating at 6."""
    a = v.abs()
    lower = (a[..., None] > _BOUNDS.to(v.device)).sum(-1)
    tie = (lower < 7) & (a == _BOUNDS.to(v.device)[lower.clamp(max=6)])
    code = torch.where(tie & (lower % 2 == 1), lower + 1, lower)
    return (code | torch.where(v < 0, 8, 0)).to(torch.uint8)


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """The UE4M3 codes of f32 `x`: to nearest even, saturating at 448 (0x7E); not positive → 0."""
    code = x.clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.where(x > 0, code, 0)


def e4m3_value(code: torch.Tensor) -> torch.Tensor:
    return code.view(torch.float8_e4m3fn).float()


def _nudged(x: torch.Tensor, ulps: int) -> torch.Tensor:
    toward = torch.full_like(x, float("inf") if ulps > 0 else -float("inf"))
    for _ in range(abs(ulps)):
        x = torch.nextafter(x, toward)
    return x


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.double().float()


def quantize_mxfp4(x: torch.Tensor, log2_nudge: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes [rows, in], e8m0 [rows, in/32]) of f32 `x` [rows, in]; `log2_nudge` added to log2(amax) (a tie class)."""
    g = x.float().reshape(len(x), -1, 32)
    amax = g.abs().amax(-1)
    log2 = torch.log2(amax.double()) + log2_nudge
    biased = (torch.round(log2) - 2 + 127).clamp(0, 254)                  # torch.round: half to even
    e = torch.where(amax > 0, biased, 0).to(torch.uint8)
    inv = torch.where(amax > 0, torch.exp2(127 - e.double()), 0.0).float()
    return e2m1(g * inv[..., None]).reshape(x.shape), e


def mxfp4_ties(x: torch.Tensor) -> list[float]:
    """The log2 nudges standing for fast-math's `log2f` (each within NUDGE ulp, at least 2^-20 absolute)."""
    amax = x.float().reshape(len(x), -1, 32).abs().amax(-1)
    log2 = torch.log2(amax[amax > 0].double()).float().abs()          # zero blocks: e 0 whatever log2f gives
    if not len(log2):
        return []
    d = float(torch.maximum(NUDGE * (_nudged(log2, 1) - log2), torch.tensor(2.0**-20)).max())
    return [-d, d]


def quantize_nvfp4(x: torch.Tensor, r: torch.Tensor, nudge: tuple[int, int, int] = (0, 0, 0)
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes [rows, in], UE4M3 scales [rows, in/16]) of f32 `x` [rows, in] with the kernel's row scales `r` [rows];
    `nudge` moves 1/r, max|v| / 6 and 0.5 / s by that many ulps (the tie classes)."""
    rows = len(x)
    xs = x.float().reshape(rows, -1, 16)
    inv_col = torch.where(r > 0, _nudged(_f32(1.0 / r.double()), nudge[0]), 0.0)[:, None, None]
    v = xs * inv_col
    first = e4m3(_nudged(_f32(v.abs().amax(-1).double() / 6), nudge[1])).int()
    cands = first[..., None] + torch.tensor([0, -1, 1, -2, 2], device=x.device)
    valid = (cands >= 0) & (cands <= 0x7E)
    valid[..., 0] = True
    scale = torch.where(valid, e4m3_value(cands.clamp(0, 0x7E).to(torch.uint8)) / 2, 0.0)
    inv = torch.where(scale > 0, _nudged(_f32(0.5 / scale.double()), nudge[2]), 0.0)
    q = E2M1.to(x.device)[e2m1(v[..., None, :] * inv[..., None]).long()].abs()          # [rows, n, 5, 16]
    diff = _f32(v.abs()[..., None, :].double() - q.double() * (2 * scale[..., None]).double())   # one FMA each
    err = torch.zeros(diff.shape[:-1], dtype=torch.float32, device=x.device)
    for k in range(16):
        err = _f32(diff[..., k].double() ** 2 + err.double())
    err = torch.where(valid, err, float("inf"))
    best = err.argmin(-1, keepdim=True)                                     # the first of equals: the kernel's order
    code = cands.gather(-1, best)[..., 0].to(torch.uint8)
    s = inv_col[..., 0] * inv.gather(-1, best)[..., 0]
    return e2m1(xs * s[..., None]).reshape(x.shape), code


def nvfp4_ties() -> list[tuple[int, int, int]]:
    """The nudges standing for fast-math's results: 1/r and 0.5 / s together, max|v| / 6 alone."""
    span = range(-NUDGE, NUDGE + 1)
    return [(a, 0, c) for a, c in itertools.product(span, span)] + [(0, b, 0) for b in span if b]


def unpack(elem: str, blocks: torch.Tensor, row_scale: torch.Tensor | None, k: int
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel's activation blocks [in/256 (padded), rows, 144] as (codes [rows, k], scales [rows, k/group],
    values [rows, k] f64)."""
    b = blocks.permute(1, 0, 2).cpu()                                     # [rows, in/256, 144]
    rows, group = len(b), GROUP[elem]
    d, qs = b[..., :16], b[..., 16:].reshape(rows, -1, 256 // group, group // 2)
    codes = torch.cat((qs & 0x0F, qs >> 4), dim=-1).reshape(rows, -1)[:, :k]
    if elem == "mxfp4":
        scales = d.reshape(rows, -1, 4, 4)[..., :2].reshape(rows, -1)[:, :k // group]
        unit = torch.exp2(scales.double() - 127)
    else:
        scales = d.reshape(rows, -1)[:, :k // group]
        unit = e4m3_value(scales).double() * row_scale.cpu().double()[:, None]
    values = E2M1[codes.long()].double() * unit.repeat_interleave(group, 1)
    return codes, scales, values


def check_codes(elem: str, x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                row_scale: torch.Tensor | None = None) -> tuple[float, int]:
    """The kernel's activation codes and scales against the emulation of f32 `x` [rows, in]: (the fraction of codes
    that differ from it, the scale groups that differ beyond every tie-class nudge). Codes compare by value (−0 is
    0)."""
    x, codes, scales = x.float().cpu(), codes.cpu(), scales.cpu()
    group = GROUP[elem]

    def same(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:          # per scale group
        by_value = torch.where(c == 8, 0, c) == torch.where(codes == 8, 0, codes)
        return by_value.reshape(len(x), -1, group).all(-1) & (s == scales)

    if elem == "mxfp4":
        exact, nudged = quantize_mxfp4(x), [quantize_mxfp4(x, d) for d in mxfp4_ties(x)]
    else:
        r = row_scale.float().cpu()
        exact, nudged = quantize_nvfp4(x, r), [quantize_nvfp4(x, r, n) for n in nvfp4_ties()]
    ok = same(*exact)
    differ = (torch.where(exact[0] == 8, 0, exact[0]) != torch.where(codes == 8, 0, codes)).float().mean().item()
    explained = ok.clone()
    for c, s in nudged:
        explained |= same(c, s)
    return differ, int((~explained).sum())
