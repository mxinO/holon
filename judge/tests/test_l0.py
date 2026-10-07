"""L0 budgets: scale with dtype and reduction length; legitimate low-precision results pass, small bugs fail."""
import pytest
import torch
from holon_judge.l0 import budget, check


def test_budget_scales_with_dtype_and_reduction_length():
    assert budget(torch.float32, 4096).atol > budget(torch.float32, 16).atol
    assert budget(torch.float16, 1).rtol > budget(torch.float32, 1).rtol
    assert budget(torch.bfloat16, 1).rtol > budget(torch.float16, 1).rtol
    # bf16 storage error dwarfs fp32 accumulation error, so the reduction length barely matters there
    assert budget(torch.bfloat16, 4096).atol < 1.1 * budget(torch.bfloat16, 1).atol
    with pytest.raises(ValueError):
        budget(torch.int8, 1)
    with pytest.raises(ValueError):
        budget(torch.float32, 0)


@pytest.fixture(scope="module")
def matmul():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(256, 1024, generator=gen)
    b = torch.randn(1024, 256, generator=gen) / 32
    return a, b


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_legitimate_matmul_passes_and_scale_bug_fails(matmul, dtype):
    a, b = (t.to(dtype) for t in matmul)
    ref = (a.double() @ b.double()).float()          # fp32-rounded reference of the same low-precision inputs
    good = check(a @ b, ref, dtype, 1024)
    assert good.passed and good.usage < 0.7
    bug = 0.02 if dtype is torch.bfloat16 else 0.002   # below one bf16 ulp per element, caught by the mean
    assert not check((a @ b).float() * (1 + bug), ref, dtype, 1024).passed


def test_local_error_and_non_finite_values_fail(matmul):
    ref = matmul[0] @ matmul[1]
    out = ref.clone()
    out[3, 7] += ref.abs().mean()                      # one wrong element
    r = check(out, ref, torch.bfloat16, 1024)
    assert not r.passed and r.max_abs == pytest.approx(ref.abs().mean().item(), rel=1e-6)
    out = ref.clone()
    out[0, 0] = float("nan")
    assert not check(out, ref, torch.float32, 1024).passed
    ref_inf = ref.clone()
    ref_inf[0, 0] = float("inf")
    assert check(ref_inf.clone(), ref_inf, torch.float32, 1024).passed     # matching non-finite values are fine
    with pytest.raises(ValueError, match="shape"):
        check(ref[:1], ref, torch.float32, 1)
