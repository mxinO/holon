"""L0: every attention kernel against the std.mixers.gqa oracle on paged KV (ragged lengths, page boundaries,
prefill chunks that continue earlier context, padding rows), plus run-to-run identity; `kernels.triton_attn`'s launch
configs within every class's shared memory, its row tiles and pipeline stages bitwise."""
import dataclasses

import numpy as np
import pytest
import torch
from holon.api import Batch, DeviceBatch, Pages
from holon.core.batching import host_plan
from holon.core.build import BuildContext
from holon.core.plugins import PluginError, PluginSet, discover
from holon_plugins.kernels.triton_attn import kernel as triton_attn
from holon_plugins.kernels.triton_attn.plugin import FORMAT, Verify
from holon_std.mixers.gqa.kernels import kv_dtype
from holon_testing.ref_kernels.kernels import PagedAttention

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

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


def _kernel(op: str, provider: str, heads: int, kv_heads: int, head_dim: int, dtype: str = "bfloat16",
            layout: str = "kv.hnd", **masks):
    """The kernel `provider` offers for `op`; `masks`: verify's `causal` / `window` facts, given when not default."""
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"), options={"kernels": {op: provider}})
    facts = {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "page": PAGE, "q_dtype": dtype,
             "kv_dtype": kv_dtype(layout, dtype), "layout": layout, "max_seqs": MAX_SEQS} | masks
    return ctx.select_kernel(f"{op}@1", site="test", **facts).impl(ctx, **facts)


def _run(kernel, q, kv, batch, pages):
    handle = kernel.plan(batch, pages)
    if hasattr(kernel, "prepare"):          # the device half of the plan (the engine runs it after its device prep)
        kernel.prepare(handle)
    return kernel.run(q, kv, handle)


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
    ref = _run(PagedAttention(None), q, kv, batch, pages)
    # bf16 output: a rounding step is ~0.4% of the value, and short rows output near-raw V values (|v| ~ 3).
    torch.testing.assert_close(ours.float(), ref.float(), atol=1e-2, rtol=1e-2)


# Blocks of 8 rows (a drafter's block, M0.4 D4) from an empty context to past the widest window, across page bounds.
BLOCK_ROWS = [(0, 8), (5, 8), (13, 8), (200, 8), (2040, 8), (2041, 8), (3000, 8)]
MASKS = [{}, {"causal": False}, {"window": 16}, {"causal": False, "window": 16}, {"causal": False, "window": 100},
         {"causal": False, "window": 2048}]


@pytest.mark.parametrize("masks", MASKS, ids=lambda m: "-".join(f"{k}={v}" for k, v in m.items()) or "default")
@pytest.mark.parametrize(("op", "rows", "geometry"),     # decode runs the verify kernel: one geometry is enough
                         [("attention.paged_verify", BLOCK_ROWS, g) for g in GEOMETRIES]
                         + [("attention.paged_decode", DECODE_ROWS, GEOMETRIES[2])])
def test_masks_match_oracle(op, rows, geometry, masks):
    q, kv, batch, pages = _case(rows, *geometry)
    kernel = _kernel(op, "kernels.triton_attn", *geometry, **masks)
    ours, ref = _run(kernel, q, kv, batch, pages), _run(PagedAttention(None, **masks), q, kv, batch, pages)
    torch.testing.assert_close(ours.float(), ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(ours, _run(kernel, q, kv, batch, pages))


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


def test_selection_prefers_fast_kernels_and_never_falls_back_to_a_reference():
    facts = {"heads": 32, "kv_heads": 8, "head_dim": 128, "page": PAGE, "q_dtype": "bfloat16",
             "kv_dtype": "bfloat16", "layout": "kv.hnd", "max_seqs": MAX_SEQS}
    gpu = BuildContext(PluginSet(discover()), torch.device("cuda"))
    assert gpu.select_kernel("attention.paged_prefill@1", site="t", **facts).provider == "kernels.flashinfer"
    assert gpu.select_kernel("attention.paged_decode@1", site="t", **facts).provider == "kernels.triton_attn"
    fp32 = {**facts, "q_dtype": "float32", "kv_dtype": "float32"}
    with pytest.raises(PluginError, match="no kernel for attention.paged_decode@1 on cuda"):
        gpu.select_kernel("attention.paged_decode@1", site="t", **fp32)       # no production fp32 kernel: an error
    cpu = BuildContext(PluginSet(discover()), torch.device("cpu"))         # the test-only references serve the CPU
    assert cpu.select_kernel("attention.paged_prefill@1", site="t", **facts).provider == "testing.ref_kernels"
    assert "kernels.flashinfer" not in cpu.plugins._loaded      # unavailable plugins are never imported


@pytest.mark.parametrize("layout", ["kv.hnd", "kv.hnd.q8_0"])
def test_masks_select_only_kernels_that_implement_them(layout):
    # A sliding-window or non-causal site never runs on a kernel that would ignore its facts: no prefill implements
    # them on the GPU yet (the build fails), Triton's decode and verify do.
    facts = {"heads": 32, "kv_heads": 8, "head_dim": 128, "page": PAGE, "q_dtype": "bfloat16",
             "kv_dtype": kv_dtype(layout, "bfloat16"), "layout": layout, "max_seqs": MAX_SEQS}
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    for masks in ({"window": 16}, {"causal": False}):
        with pytest.raises(PluginError, match="no kernel for attention.paged_prefill@1"):
            ctx.select_kernel("attention.paged_prefill@1", site="t", **facts, **masks)
        for op in ("attention.paged_decode@1", "attention.paged_verify@1"):
            assert ctx.select_kernel(op, site="t", **facts, **masks).provider == "kernels.triton_attn"


@pytest.mark.parametrize("geometry", [(32, 1, 128), (24, 1, 256)])
@pytest.mark.parametrize("op", ["attention.paged_prefill", "attention.paged_decode"])
def test_flashinfer_workspace_holds_the_widest_split(op, geometry):
    # One KV head and long contexts: FlashInfer splits the KV into up to 2 CTAs per SM, whose f32 partials exceed the
    # 128 MiB floor on a 64-SM card (Qwen3-30B-A3B's 4 KV heads did on a 128-SM RTX 4090). At head_dim 256 its query
    # tile is 64 rows, not 128 (FA2DetermineCtaTileQ): the workspace sized so (Flash-Next's 24 / 2 / 256: half of the
    # 5090's 512 MiB) still holds a split prefill of rows past 64 packed queries.
    q, kv, batch, pages = _case([(2000, 3 if op.endswith("prefill") else 1)] * MAX_SEQS, *geometry)
    ours = _run(_kernel(op, "kernels.flashinfer", *geometry), q, kv, batch, pages)
    torch.testing.assert_close(ours.float(), _run(PagedAttention(None), q, kv, batch, pages).float(),
                               atol=1e-2, rtol=1e-2)


# The lock classes' SMs (each SM's shared memory per block: `triton_attn.SHARED`): compiled for each offline.
CLASSES = {"rtx-3090": 86, "rtx-4090": 89, "rtx-5080": 120, "rtx-5090": 120}
WIDTHS = (1, 4, 8, 16)             # decode, MTP's verify (k 3), DFlash's verify and block, a 16-row block
GROUP = {128: 8, 256: 6}           # query heads per KV head: Qwen3-30B-A3B; Qwen3.5 / 3.8 (the widest of each)


@pytest.mark.parametrize("layout", ["kv.hnd", "kv.hnd.fp8", "kv.hnd.q8_0"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("cls", sorted(CLASSES))
def test_triton_attn_configs_fit_every_class(cls, head_dim, layout):
    """docs/dev/m07-perf.md §4: the launch config a class's verify kernel runs at each width a deployment declares,
    compiled
    for the class's SM, takes at most its shared memory — the config `fitted` at build (on the class's own device;
    here with its fingerprint), checked at one split and several for the widest width that runs each config (shared
    memory grows with BLOCK_M, and so with the width). Before the fix the RTX 5090's pinned (64, 4, 3) took 106,752 B
    at head_dim 256, 6 heads per KV head, 8 rows, bf16 KV (DFlash2 failed graph capture)."""
    sm, g = CLASSES[cls], GROUP[head_dim]
    if layout == "kv.hnd.fp8" and sm < 89:
        pytest.skip("fp8 KV from sm_89")
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"), options={"max_draft_tokens": max(WIDTHS) - 1})
    ctx.fingerprint = dataclasses.replace(ctx.fingerprint, gpu=cls, sm=sm)
    kernel = Verify(ctx, heads=4 * g, kv_heads=4, head_dim=head_dim, page=PAGE, q_dtype="bfloat16",
                    kv_dtype=kv_dtype(layout, "bfloat16"), layout=layout, max_seqs=MAX_SEQS)
    widest = {kernel.cfg(width): width for width in sorted(WIDTHS)}           # each config at its widest width
    for cfg, width in widest.items():
        for splits in (1, 2):
            used = triton_attn.shared_bytes(cfg, sm, nq=width, g=g, d=head_dim, page=PAGE,
                                            fmt=FORMAT[layout], causal=True, window=0, dtype="bfloat16", splits=splits)
            assert used <= triton_attn.SHARED[sm], (width, splits, cfg, used)


@pytest.mark.parametrize("masks", [{}, {"causal": False, "window": 16}], ids=["causal", "block-window"])
@pytest.mark.parametrize("geometry", [(32, 4, 128), (24, 4, 256)])
def test_triton_attn_tiles_and_stages_are_bitwise(geometry, masks):
    """One pipeline stage gives three stages' bits (`fitted` steps stages down first), one split or several. Rows tiled
    over the grid (16 rows x 8 or 6 heads: two tiles of 64 pairs) give today's single tile's bits where Triton keeps
    its warp layout — at head_dim 256 (the 27B's 16-row blocks). At head_dim 128 a 128-pair tile puts its warps along
    the rows and a 64-pair one along the columns (Triton's chained-dot rule: rows >= head_dim), so the row sums run in
    another order: one bf16 ulp in a few values (docs/dev/m07-perf.md §12.4); no lock runs over 64 pairs."""
    q, kv, batch, pages = _case([(s, 16) for s in (0, 5, 200, 2041)], *geometry)
    for splits in (1, 3):
        def run(cfg, block=None, splits=splits):
            return triton_attn.paged_attention(q, kv, pages.table, batch.dev.slots, pages.lens, nq=16, splits=splits,
                                               cfg=cfg, block=block, **masks)
        tiled = run(triton_attn.Config(32, 4, 1))
        assert torch.equal(tiled, run(triton_attn.Config(32, 4, 3)))
        whole = run(triton_attn.Config(32, 4, 1), block=128)
        if geometry[2] == 256:
            assert torch.equal(tiled, whole)
        else:
            torch.testing.assert_close(tiled, whole, rtol=2**-7, atol=0)
