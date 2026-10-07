"""L0: every attention kernel against the std.mixers.gqa oracle on paged KV (ragged lengths, page boundaries,
prefill chunks that continue earlier context, padding rows), plus run-to-run identity."""
import numpy as np
import pytest
import torch
from holon.api import Batch, DeviceBatch, Pages
from holon.core.batching import host_plan
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE, MAX_SEQS = 16, 8
GEOMETRIES = [(32, 8, 128), (16, 4, 256), (24, 4, 256),    # Qwen3-8B, Qwen3.5 / Qwen3.8 full-attention layers
              (16, 8, 64)]                                  # every head_dim the fast kernels accept


def _case(rows: list[tuple[int, int]], heads: int, kv_heads: int, head_dim: int, dtype=torch.bfloat16, seed=0):
    """rows: (tokens before this step, tokens this step) per batch row; returns q, kv, batch, pages."""
    g = torch.Generator().manual_seed(seed)
    slots = np.random.default_rng(seed).permutation(MAX_SEQS)[: len(rows)].astype(np.int32)
    max_pages = -(-max(s + n for s, n in rows) // PAGE)
    units = MAX_SEQS * max_pages + 7                      # a few units no row uses
    table = torch.from_numpy(np.random.default_rng(seed + 1).permutation(units)[: MAX_SEQS * max_pages]
                             .reshape(MAX_SEQS, max_pages).astype(np.int32))
    kv = torch.randn(units, 2, kv_heads, PAGE, head_dim, generator=g).to(dtype).cuda()
    starts = np.array([s for s, _ in rows], dtype=np.int32)
    n_query = np.array([n for _, n in rows], dtype=np.int32)
    plan = host_plan(0, slots, starts, n_query, np.ones_like(n_query), np.zeros(int(n_query.sum()), np.int64))
    qstart = np.concatenate([[0], np.cumsum(n_query)])
    dev = DeviceBatch(input_ids=None, positions=None, query_start=torch.as_tensor(qstart).cuda(),
                      slots=torch.as_tensor(slots).cuda(), logits_index=None, logits_rows=None)
    lens = torch.as_tensor(starts + n_query).cuda()
    pages = Pages(PAGE, table.cuda(), table.numpy(), lens, write_index=None)
    q = torch.randn(int(n_query.sum()), heads, head_dim, generator=g).to(dtype).cuda()
    return q, kv, Batch(plan, dev), pages


def _kernel(op: str, provider: str, heads: int, kv_heads: int, head_dim: int, dtype: str = "bfloat16"):
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"), options={"kernels": {op: provider}})
    facts = {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "page": PAGE, "q_dtype": dtype,
             "kv_dtype": dtype, "layout": "kv.hnd", "max_seqs": MAX_SEQS}
    return ctx.select_kernel(f"{op}@1", site="test", **facts).impl(ctx, **facts)


def _run(kernel, q, kv, batch, pages):
    return kernel.run(q, kv, kernel.plan(batch, pages))


PREFILL_ROWS = [(0, 1), (0, 17), (5, 16), (31, 33), (0, 64), (100, 7), (16, 16)]
DECODE_ROWS = [(0, 1), (15, 1), (16, 1), (99, 1), (999, 1), (4096, 1)]
CASES = [("attention.paged_prefill", "kernels.flashinfer", PREFILL_ROWS),
         ("attention.paged_decode", "kernels.flashinfer", DECODE_ROWS),
         ("attention.paged_decode", "kernels.triton_attn", DECODE_ROWS)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize(("op", "provider", "rows"), CASES)
def test_kernel_matches_oracle(op, provider, rows, geometry, dtype):
    q, kv, batch, pages = _case(rows, *geometry, dtype=dtype)
    name = str(dtype).removeprefix("torch.")
    ours = _run(_kernel(op, provider, *geometry, dtype=name), q, kv, batch, pages)
    ref = _run(_kernel(op, "std.mixers.gqa", *geometry, dtype=name), q, kv, batch, pages)
    # bf16 output: a rounding step is ~0.4% of the value, and short rows output near-raw V values (|v| ~ 3).
    torch.testing.assert_close(ours.float(), ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(("op", "provider", "rows"), CASES)
def test_kernel_is_run_to_run_identical(op, provider, rows):
    q, kv, batch, pages = _case(rows, *GEOMETRIES[0])
    kernel = _kernel(op, provider, *GEOMETRIES[0])
    assert torch.equal(_run(kernel, q, kv, batch, pages), _run(kernel, q, kv, batch, pages))


@pytest.mark.parametrize("provider", ["kernels.triton_attn", "kernels.flashinfer"])
def test_decode_padding_rows_are_zero(provider):
    q, kv, batch, pages = _case([(40, 1), (7, 1), (0, 1)], *GEOMETRIES[0])
    batch.plan.start_upper[1:], batch.plan.n_query[1:] = 0, 0      # graph padding: length 0, any valid slot
    pages.lens[1:] = 0
    out = _run(_kernel("attention.paged_decode", provider, *GEOMETRIES[0]), q, kv, batch, pages)
    assert torch.isfinite(out).all() and not out[1:].any()


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_every_selected_kernel_constructs(head_dim, dtype):
    # Whatever supports() accepts on this device must also build (a selected kernel may not fail later).
    for op in ("attention.paged_prefill", "attention.paged_decode"):
        ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
        facts = {"heads": 16, "kv_heads": 8, "head_dim": head_dim, "page": PAGE, "q_dtype": dtype,
                 "kv_dtype": dtype, "layout": "kv.hnd", "max_seqs": MAX_SEQS}
        assert ctx.select_kernel(f"{op}@1", site="t", **facts).impl(ctx, **facts) is not None


def test_selection_prefers_fast_kernels_and_falls_back_on_cpu():
    facts = {"heads": 32, "kv_heads": 8, "head_dim": 128, "page": PAGE, "q_dtype": "bfloat16",
             "kv_dtype": "bfloat16", "layout": "kv.hnd", "max_seqs": MAX_SEQS}
    gpu = BuildContext(PluginSet(discover()), torch.device("cuda"))
    assert gpu.select_kernel("attention.paged_prefill@1", site="t", **facts).provider == "kernels.flashinfer"
    assert gpu.select_kernel("attention.paged_decode@1", site="t", **facts).provider == "kernels.triton_attn"
    fp32 = {**facts, "q_dtype": "float32", "kv_dtype": "float32"}
    assert gpu.select_kernel("attention.paged_decode@1", site="t", **fp32).provider == "std.mixers.gqa"
    cpu = BuildContext(PluginSet(discover()), torch.device("cpu"))
    assert cpu.select_kernel("attention.paged_prefill@1", site="t", **facts).provider == "std.mixers.gqa"
    assert "kernels.flashinfer" not in cpu.plugins._loaded      # unavailable plugins are never imported
