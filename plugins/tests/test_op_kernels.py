"""L0: FlashInfer kernels for the per-token ops against the std oracles (bf16/fp16)."""
import pytest
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_testing.ref_kernels.kernels import BY_OP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DTYPES = [torch.bfloat16, torch.float16]


def _pair(op: str, **facts):
    """The selected fast kernel and the test-only reference kernel (the oracle) for `op`."""
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    fast = ctx.select_kernel(op, site="t", **facts)
    assert fast.provider == "kernels.flashinfer"
    return fast.impl(ctx, **facts), BY_OP[op.split("@")[0]](None, **facts)


def _close(a, b):
    torch.testing.assert_close(a.float(), b.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("offset", [0.0, 1.0])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_and_fused_add(dtype, offset):
    facts = {"dim": 4096, "eps": 1e-6, "offset": offset, "dtype": str(dtype).removeprefix("torch.")}
    x, r = (torch.randn(37, 4096, device="cuda", dtype=dtype) for _ in range(2))
    w = torch.randn(4096, device="cuda", dtype=dtype)
    fast, ref = _pair("rmsnorm@1", **facts)
    _close(fast(x, w), ref(x, w))
    q = torch.randn(37, 8, 128, device="cuda", dtype=dtype)                     # per-head (q/k norm) shape
    facts_h = {**facts, "dim": 128}
    fast, ref = _pair("rmsnorm@1", **facts_h)
    _close(fast(q, w[:128]), ref(q, w[:128]))
    fast, ref = _pair("add_rmsnorm@1", **facts)
    want_y, want_r = ref(x.clone(), r.clone(), w)
    got_y, got_r = fast(x.clone(), r.clone(), w)
    _close(got_r, want_r)
    _close(got_y, want_y)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rope(dtype):
    facts = {"head_dim": 128, "rotary_dim": 128, "theta": 1e6, "max_positions": 8192,
             "dtype": str(dtype).removeprefix("torch.")}
    fast, ref = _pair("rope@1", **facts)
    q, k = torch.randn(29, 32, 128, device="cuda", dtype=dtype), torch.randn(29, 8, 128, device="cuda", dtype=dtype)
    pos = torch.tensor([0, 1, 2, 15, 16, 1000, 4095, 8191] + list(range(100, 121)), device="cuda")
    want = ref(q.clone(), k.clone(), pos)
    got = fast(q.clone(), k.clone(), pos)
    _close(got[0], want[0])
    _close(got[1], want[1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_mul(dtype):
    facts = {"inter": 12288, "dtype": str(dtype).removeprefix("torch.")}
    fast, ref = _pair("silu_mul@1", **facts)
    gate_up = torch.randn(17, 2 * 12288, device="cuda", dtype=dtype)
    _close(fast(gate_up), ref(gate_up))
