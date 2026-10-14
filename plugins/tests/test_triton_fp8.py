"""L0 for kernels.triton_fp8 (docs/m06-lowp.md D12, §4.6, §4.8): FP8 block-scaled `qlinear` against its modes'
emulations, on random e4m3 bytes (every finite code, subnormals and ±448 included) with distinct random block scales.

- The bit decode: all 254 finite e4m3 codes to bf16 equal torch's `float8_e4m3fn` values (any class).
- `act_quant` (sm ≥ 89): bitwise torch's e4m3 of `x / a_s` (IEEE division, ±448) and its scales, rows of zeros,
  outliers and values that round to e4m3 subnormals included.
- `qlinear` through the plugin's instance, rows T ∈ {0, 1, 7, 8, 9, 16, 17, 64, 512, 513, 2049} (the 16 / 17 mode
  switch) on a small shape and the 8B's projections (their pinned configurations): `l0.check` against W8A16 (the
  bf16 activations, T ≤ 16) or W8A8 (activations quantized as `act_quant`, T > 16) in f64, and the call counted on
  that path and mode. Below sm_89 only the decode path runs (FP8 there is v1).
- CUDA graphs replay bitwise like eager calls on both paths, and calls are run-to-run identical.
"""
import pytest
import torch
from holon.api import Layout, QTensor, Scheme
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0
from holon_plugins.kernels.triton_fp8 import kernel
from holon_plugins.kernels.triton_fp8.plugin import QLinear
from holon_std.weights.hf import reference as hf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ROWS = (0, 1, 7, 8, 9, 16, 17, 64, 512, 513, 2049)
SHAPES = ((384, 256),) + ((6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288))   # Qwen3-8B's, merged
FP8_DOT = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
FINITE = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)


@pytest.fixture(scope="module")
def ctx() -> BuildContext:
    return BuildContext(PluginSet(discover()), torch.device("cuda"))


def weight(out: int, k: int, seed: int) -> dict[str, torch.Tensor]:
    """Random finite e4m3 bytes and distinct block scales spread over 2^-12 … 2^-4."""
    g = torch.Generator().manual_seed(seed)
    w = FINITE[torch.randint(0, len(FINITE), (out, k), generator=g)]
    s = torch.rand(out // 128, k // 128, generator=g) * torch.exp2(torch.randint(-12, -4, (out // 128, k // 128),
                                                                                    generator=g).float())
    return {"weight": w.cuda(), "scale": s.cuda()}


def activations(t: int, k: int, seed: int) -> torch.Tensor:
    """bf16 rows: normal ones and, from 8 rows, an all-zero row, a row of tiny values, one with a large outlier."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(t, k, generator=g)
    if t >= 8:
        x[-1] = 0.0
        x[-2] *= 1e-6
        x[-3, k // 3] = 3000.0
    return x.bfloat16().cuda()


def act_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W8A8's activations in torch: per row and 128 columns, a_s = max(amax, 1e-10) / 448, e4m3 of x / a_s."""
    g = x.float().unflatten(-1, (-1, kernel.BLOCK))
    a_s = g.abs().amax(-1).clamp_min(1e-10) / 448
    return (g / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2), a_s


def emulated(x: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """f64 `x @ W.T` in the mode of the path T rows take: W8A16 up to 16 rows, else W8A8."""
    w = hf.dequantize("fp8_e4m3", parts).double()
    if len(x) <= kernel.DECODE_ROWS:
        return x.double() @ w.T
    q, a_s = act_quant_reference(x)
    return (q.double().unflatten(-1, (-1, kernel.BLOCK)) * a_s.double()[..., None]).flatten(-2) @ w.T


def test_e4m3_decode_is_exact():
    import triton
    import triton.language as tl

    @triton.jit
    def decode(w_ptr, y_ptr, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(y_ptr + i, kernel.e4m3_to_bf16(tl.load(w_ptr + i)))

    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    y = torch.empty(256, dtype=torch.bfloat16, device="cuda")
    decode[(1,)](codes, y, 256)
    finite = FINITE.cuda().long()
    want = codes.view(torch.float8_e4m3fn).bfloat16()
    assert torch.equal(y[finite].view(torch.int16), want[finite].view(torch.int16))


@pytest.mark.skipif(not FP8_DOT, reason="e4m3 conversion needs sm >= 89")
def test_act_quant_is_torchs():
    x = activations(67, 1024, 0)
    x[5] = torch.linspace(-1e-3, 1e-3, 1024).bfloat16()          # 448 · tiny / amax: e4m3 subnormals
    q, a_s = kernel.act_quant(x)
    q_ref, a_s_ref = act_quant_reference(x)
    assert torch.equal(a_s, a_s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


@pytest.mark.parametrize("out,k", SHAPES)
def test_qlinear(ctx, out, k):
    parts = weight(out, k, out + k)
    lin = QLinear(ctx, layout="fp8_e4m3/hf", **{"in": k, "out": out}, act_dtype="bfloat16", out_dtype="bfloat16",
                  act_quant="a8")
    assert lin.prepare(q := QTensor(Layout(Scheme("fp8_e4m3"), "hf"), (out, k), parts)) is q
    with pytest.raises(ValueError, match="128 × 128"):
        lin.prepare(QTensor(q.layout, (out, k), parts | {"scale": parts["scale"].repeat_interleave(2, 1)}))
    for t in ROWS if (out, k) == SHAPES[0] else (1, 16, 17, 513):
        if t > kernel.DECODE_ROWS and not FP8_DOT:
            continue
        x = activations(t, k, t)
        before = lin.calls.copy()
        y = lin(x, parts)
        assert y.shape == (t, out) and y.dtype == torch.bfloat16
        if not t:
            continue
        path = "decode/bf16" if t <= kernel.DECODE_ROWS else "prefill/fp8"
        assert lin.calls - before == {path: 1}
        report = l0.check(y, emulated(x, parts), torch.bfloat16, k)
        assert report.passed, (t, report)


def test_graphs_and_run_to_run(ctx):
    out, k = 384, 1024
    parts = weight(out, k, 1)
    lin = QLinear(ctx, layout="fp8_e4m3/hf", **{"in": k, "out": out}, act_dtype="bfloat16", out_dtype="float32",
                  act_quant="a8")
    for t in (1, 9, 64) if FP8_DOT else (1, 9):
        x = activations(t, k, 7)
        eager = lin(x, parts)
        assert torch.equal(lin(x, parts), eager)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = lin(x, parts)
        x.copy_(activations(t, k, 8))
        g.replay()
        assert torch.equal(captured, lin(x, parts))
