"""The W4A4 activation emulator (holon_judge.w4a4) on the CPU: against a scalar transliteration of llama.cpp's FP4
quantizers (cbe28bb3 quantize.cu, CUDA ≥ 12.8: `quantize_mmq_nvfp4`, `quantize_mmq_mxfp4`, element by element in
f32), so the NVFP4 emulation is checked off Blackwell too, with MXFP4's tie class beside a zero block; `unpack`
against blocks packed as those kernels store them."""
import math

import numpy as np
import torch
from holon_judge import w4a4

F32 = np.float32
LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1(v: float) -> int:
    """cvt.rn.satfinite.e2m1: nearest, ties to the even code, saturating; the sign in bit 3."""
    a, best = abs(float(v)), 0
    for i in range(1, 8):
        if abs(a - LUT[i]) < abs(a - LUT[best]) or (abs(a - LUT[i]) == abs(a - LUT[best]) and i % 2 == 0):
            best = i
    return best | (8 if v < 0 else 0)


def e4m3(x: float) -> int:
    return 0 if not x > 0 else int(torch.tensor(min(x, 448.0)).to(torch.float8_e4m3fn).view(torch.uint8))


def half_scale(code: int) -> F32:                         # ggml_cuda_ue4m3_to_fp32: the e4m3 value / 2
    return F32(float(torch.tensor(code, dtype=torch.uint8).view(torch.float8_e4m3fn).float()) / 2)


def nvfp4_row(x: np.ndarray, r: F32) -> tuple[list[int], list[int]]:
    inv_col = F32(1) / r if r > 0 else F32(0)
    codes, scales = [], []
    for b in range(len(x) // 16):
        vals = x[16 * b:16 * b + 16].astype(F32)
        v = (vals * inv_col).astype(F32)
        first = e4m3(F32(np.max(np.abs(v)) / F32(6)))
        best, best_err = first, None
        for off in (0, -1, 1, -2, 2):
            c = first + off
            if off and not 0 <= c <= 0x7E:
                continue
            s = half_scale(c)
            inv = F32(F32(0.5) / s) if s > 0 else F32(0)
            err = F32(0)
            for k in range(16):
                diff = F32(abs(float(v[k])) - LUT[e2m1(F32(v[k] * inv)) & 7] * float(F32(2) * s))   # one FMA
                err = F32(float(diff) * float(diff) + float(err))                                    # fmaf
            if best_err is None or err < best_err:
                best, best_err = c, err
        s = half_scale(best)
        scale = F32(inv_col * (F32(F32(0.5) / s) if s > 0 else F32(0)))
        codes += [e2m1(F32(vals[k] * scale)) for k in range(16)]
        scales.append(best)
    return codes, scales


def mxfp4_row(x: np.ndarray) -> tuple[list[int], list[int]]:
    codes, scales = [], []
    for b in range(len(x) // 32):
        vals = x[32 * b:32 * b + 32].astype(F32)
        amax = float(np.max(np.abs(vals)))
        e = 0 if not amax > 0 else min(max(round(math.log2(amax)) - 2 + 127, 0), 254)   # round: half to even
        inv = F32(0) if amax == 0 else F32(2.0 ** (127 - e))
        codes += [e2m1(F32(val * inv)) for val in vals]
        scales.append(e)
    return codes, scales


def by_value(codes) -> list[int]:
    return [0 if c == 8 else int(c) for c in codes]


def test_the_emulator_is_llama_cpp_s_quantizer():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(6, 512, generator=gen) * 10 ** torch.empty(6, 1).uniform_(-3, 3, generator=gen)
    x[1] = 0
    x[2, 7] = 1e5
    r = (x.abs().amax(1) / (6 * 448)).float()
    nv_codes, nv_scales = w4a4.quantize_nvfp4(x, r)
    mx_codes, mx_scales = w4a4.quantize_mxfp4(x)
    for i in range(len(x)):
        codes, scales = nvfp4_row(x[i].numpy(), F32(r[i]))
        assert by_value(codes) == by_value(nv_codes[i].tolist()) and scales == nv_scales[i].tolist(), i
        codes, scales = mxfp4_row(x[i].numpy())
        assert by_value(codes) == by_value(mx_codes[i].tolist()) and scales == mx_scales[i].tolist(), i
    # a code no nudge explains (the largest of a sub-block flipped in sign) is counted, its scale group unexplained
    flipped = nv_codes.clone()
    flipped[0, 0] ^= 8 if flipped[0, 0] & 7 else 1
    assert w4a4.check_codes("nvfp4", x, nv_codes, nv_scales, r) == (0.0, 0)
    differ, unexplained = w4a4.check_codes("nvfp4", x, flipped, nv_scales, r)
    assert math.isclose(differ, 1 / x.numel(), rel_tol=1e-6) and unexplained == 1
    # MXFP4's tie class: a block with log2(amax) at a half-integer may take either exponent, and an all-zero block
    # beside it must not turn the nudges into NaN (which would leave the class empty)
    x = torch.zeros(1, 64)
    x[0, 0] = 2 ** 2.5
    nudges = w4a4.mxfp4_ties(x)
    assert len(nudges) == 2 and all(math.isfinite(d) for d in nudges)
    assert {int(w4a4.quantize_mxfp4(x, d)[1][0, 0]) for d in nudges} == {127, 128}


def test_unpack_reads_the_blocks_as_the_kernels_store_them():
    """block_fp4_mmq per 256 values: NVFP4 16 scale bytes then per 16 values byte j = code j | code j + 8 << 4;
    MXFP4 per 64 values a uint32 with its two e8m0 scales in the low half, then per 32 byte j = code j | code j + 16."""
    gen = torch.Generator().manual_seed(1)
    rows, k = 3, 512
    codes = torch.randint(0, 16, (rows, k), generator=gen, dtype=torch.uint8)
    for elem, group in (("nvfp4", 16), ("mxfp4", 32)):
        top = 0x7F if elem == "nvfp4" else 255
        scales = torch.randint(1, top, (rows, k // group), generator=gen, dtype=torch.uint8)
        c = codes.view(rows, k // group, group)
        qs = (c[..., :group // 2] | (c[..., group // 2:] << 4)).reshape(rows, k // 256, 128)
        d = scales.view(rows, k // 256, -1)
        if elem == "mxfp4":                                   # [s0, s1, 0, 0] per 64-value quad
            d = torch.cat((d.view(rows, k // 256, 4, 2), torch.zeros(rows, k // 256, 4, 2, dtype=torch.uint8)), -1)
        blocks = torch.cat((d.reshape(rows, k // 256, 16), qs), -1).permute(1, 0, 2).contiguous()
        row_scale = torch.rand(rows, generator=gen)
        got_codes, got_scales, values = w4a4.unpack(elem, blocks, row_scale, k)
        assert torch.equal(got_codes, codes) and torch.equal(got_scales, scales), elem
        unit = (w4a4.e4m3_value(scales).double() * row_scale.double()[:, None] if elem == "nvfp4"
                else torch.exp2(scales.double() - 127))
        assert torch.equal(values, w4a4.E2M1[codes.long()].double() * unit.repeat_interleave(group, 1)), elem
