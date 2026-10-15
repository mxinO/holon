"""L0 for quantized KV (docs/m03-qwen38.md D6): `kv_write@1` bitwise against the reference write (the model-dtype
layout's Triton copy too, docs/m07-perf.md D3), `attention.qk_rope_kv@1` against the judge's composition of the units'
references (D3: partial rotary, the gated q layout, 1-token rows; its q8_0 write bitwise the reference's write of
its own model-dtype k), the GQA unit's decode row against its last prefill row (bitwise KV), and every kernel
that reads q8_0 or fp8 against `reference.paged_attention` reading the same stored cache — the reference dequantizes
it, so the comparison isolates the kernel's error from the format's — within the attention budget of
test_attention_kernels. Plants that must fail: q8_0 scales read from another head, and a step attending to its own
tokens unquantized instead of through the cache.

Caches are laid out as std.mixers.gqa lays out one layer (q8_0: each unit's values, then its scales, in one byte
segment). K and V have per-head magnitudes (×½, ×1, ×2), so a scale from another head is far outside the budget, and
K an outlier channel per 32 values, as real keys do.
"""
import numpy as np
import pytest
import torch
from holon.api import SamplingParams, Seq
from holon.core.batching import host_plan
from holon.core.build import BuildContext
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import PluginError, PluginSet, discover
from holon_judge import fused, l0
from holon_std.mixers.gqa import kernels as gqa_kernels
from holon_std.mixers.gqa import reference as gqa
from holon_std.mixers.gqa.units import GQA
from holon_testing.qwen3_5_files import save, tiny_hf
from holon_testing.ref_kernels.kernels import PagedAttention

from .test_attention_kernels import (
    BLOCK_ROWS,
    DECODE_ROWS,
    GEOMETRIES,
    MAX_SEQS,
    PAGE,
    PREFILL_ROWS,
    _case,
    _kernel,
    _run,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PREFILL, DECODE, VERIFY = "attention.paged_prefill", "attention.paged_decode", "attention.paged_verify"
TRITON, FLASHINFER = "kernels.triton_attn", "kernels.flashinfer"
MODEL, Q8, FP8 = "kv.hnd", "kv.hnd.q8_0", "kv.hnd.fp8"
VERIFY_ROWS = [(0, 3), (13, 3), (14, 3), (99, 3), (998, 3), (4093, 3)]       # 1 + 2 drafts, across page boundaries
CASES = {Q8: [(PREFILL, TRITON, PREFILL_ROWS + [(2000, 129)]), (DECODE, TRITON, DECODE_ROWS),
              (VERIFY, TRITON, VERIFY_ROWS)],
         FP8: [(PREFILL, FLASHINFER, PREFILL_ROWS + [(2000, 129)]), (DECODE, TRITON, DECODE_ROWS),
               (DECODE, FLASHINFER, DECODE_ROWS), (VERIFY, TRITON, VERIFY_ROWS)]}
ROWS = {op: rows for op, _, rows in CASES[Q8]}


def fp8_device() -> bool:
    return torch.cuda.get_device_capability() >= (8, 9)


needs_fp8 = pytest.mark.skipif(not torch.cuda.is_available() or not fp8_device(),
                               reason="fp8 KV kernels are selected from sm_89 on")


def usage(ours: torch.Tensor, ref: torch.Tensor) -> float:
    """The attention budget (test_attention_kernels): |ours - ref| <= 1e-2 + 1e-2 |ref|; its largest use."""
    return ((ours.float() - ref.float()).abs() / (1e-2 + 1e-2 * ref.float().abs())).max().item()


def empty_cache(units: int, kv_heads: int, head_dim: int, layout: str, dtype: torch.dtype = torch.bfloat16):
    """One layer's pages in `layout` (`dtype`: the model dtype's), as std.mixers.gqa's `_kv_layer` views them."""
    shape = (units, 2, kv_heads, PAGE, head_dim)
    if layout in (MODEL, FP8):
        return torch.zeros(shape, dtype=gqa.FP8 if layout == FP8 else dtype, device="cuda")
    n = 2 * kv_heads * PAGE * head_dim
    seg = torch.zeros(units, gqa_kernels.page_bytes(layout, PAGE, kv_heads, head_dim, 1), dtype=torch.uint8,
                      device="cuda")
    return seg[:, :n].view(torch.int8).view(shape), seg[:, n:].view(torch.float16).view(*shape[:-1], head_dim // 32)


def store(raw: torch.Tensor, layout: str):
    """`raw` [units, 2, Hkv, page, D] as the reference stores it."""
    cache = empty_cache(raw.shape[0], raw.shape[2], raw.shape[4], layout)
    if layout == FP8:
        return cache.copy_(gqa.to_fp8(raw))
    values, scales = gqa.quantize_q8_0(raw)
    cache[0].copy_(values), cache[1].copy_(scales)
    return cache


def shaped(x: torch.Tensor) -> torch.Tensor:
    """Per-head magnitudes ×½, ×1, ×2 and, in K, an outlier channel per 32 values (×4). `x`: a cache [units, 2, Hkv,
    page, D] or k [T, Hkv, D]."""
    hdim = 2 if x.dim() == 5 else 1
    heads = 2.0 ** (torch.arange(x.shape[hdim]) % 3 - 1)
    outlier = torch.where(torch.arange(x.shape[-1]) % 32 == 7, 4.0, 1.0)
    if x.dim() == 5:
        scale = heads.view(-1, 1, 1) * torch.stack([outlier, torch.ones_like(outlier)]).view(2, 1, 1, -1)
    else:
        scale = heads.view(-1, 1) * outlier
    return (x.float() * scale.to(x.device)).to(x.dtype)


def quant_case(rows, heads: int, kv_heads: int, head_dim: int, layout: str, dtype=torch.bfloat16, seed=0):
    """test_attention_kernels' case with its KV shaped and stored in `layout`: q, raw KV, cache, batch, pages."""
    q, raw, batch, pages = _case(rows, heads, kv_heads, head_dim, dtype=dtype, seed=seed)
    raw = shaped(raw)
    return q, raw, store(raw, layout), batch, pages


def step_slots(rows, pages, batch):
    """(unit, off) of each of this step's tokens, as the core derives them (std.mixers.gqa.units.Planner)."""
    slots = batch.plan.slots
    pos = [(int(slots[b]), p) for b, (start, n) in enumerate(rows) for p in range(start, start + n)]
    unit = torch.tensor([int(pages.host_table[s, p // PAGE]) for s, p in pos], device="cuda")
    return unit, torch.tensor([p % PAGE for _, p in pos], device="cuda")


def dequantized(cache) -> torch.Tensor:
    return gqa.dequantize_q8_0(*cache) if isinstance(cache, tuple) else cache.float()


def writer(layout: str, kv_heads: int, head_dim: int, dtype: str = "bfloat16"):
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    facts = {"layout": layout, "kv_heads": kv_heads, "head_dim": head_dim, "page": PAGE, "dtype": dtype}
    spec = ctx.select_kernel(gqa_kernels.KV_WRITE, site="test", **facts)
    assert spec.provider == TRITON
    return spec.impl(ctx, **facts)


def raw_bytes(cache) -> torch.Tensor:
    """Every byte of the cache (q8_0: the whole segment both views live in)."""
    if isinstance(cache, tuple):
        values = cache[0]
        return torch.as_strided(values.view(torch.uint8), (values.shape[0], values.stride(0)),
                                (values.stride(0), 1)).cpu()
    return cache.view(torch.uint8).cpu()


def on_cpu(cache):
    if isinstance(cache, tuple):                  # keep the views on one segment
        seg = raw_bytes(cache).clone()
        n = cache[0][0].numel()
        return (seg[:, :n].view(torch.int8).view(cache[0].shape),
                seg[:, n: n + cache[1][0].numel() * 2].view(torch.float16).view(cache[1].shape))
    return cache.cpu()


# kv_write ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("layout", [MODEL, Q8, FP8])
def test_kv_write_is_the_reference_bitwise(layout, head_dim, dtype):
    # The reference runs on the CPU: there it is ggml's arithmetic (test_kv_formats: equal to gguf-py). On CUDA torch
    # divides by the scalar 127 as a multiplication by its reciprocal, which moves d by an ulp now and then.
    t, kv_heads, units = 300, 4, 40
    g = torch.Generator().manual_seed(head_dim)
    qkv = torch.randn(t, 3 * kv_heads * head_dim, generator=g).to(dtype).cuda()
    k, v = (qkv[:, i * kv_heads * head_dim:(i + 1) * kv_heads * head_dim].view(t, kv_heads, head_dim)
            for i in (1, 2))                     # strided rows, as the fused projection's split gives them
    k.copy_(shaped(k)), v.copy_(shaped(v))
    assert not k.is_contiguous() and not v.is_contiguous()
    k[0, 0], v[1, 1] = 0, 0                                                       # all-zero blocks
    k[2, 0, :32] = (torch.arange(32) - 15.5) / 8                                  # ties: with amax 127/8 the scale
    k[2, 0, 31] = 127 / 8                                                         # is 1/8, so x / d = ±n.5
    if layout == FP8:
        k[3, 0, :4] = torch.tensor([1000.0, -1000.0, 448.0, 464.0])               # saturate (464 rounds past 448)
    unit = torch.randperm(units, generator=g)[:t // PAGE + 1].repeat_interleave(PAGE)[:t]
    off = torch.arange(t) % PAGE
    perm = torch.randperm(t, generator=g)                                         # tokens in any order
    unit, off = unit[perm].cuda(), off[perm].cuda()
    ours = empty_cache(units, kv_heads, head_dim, layout, dtype)
    for x in ours if isinstance(ours, tuple) else (ours,):
        x.view(torch.uint8).fill_(0x5A)                                           # garbage: unwritten bytes stay
    want = on_cpu(ours)
    writer(layout, kv_heads, head_dim, str(dtype).removeprefix("torch."))(k, v, ours, unit, off)
    gqa.kv_write(k.cpu(), v.cpu(), want, unit.cpu(), off.cpu())
    assert torch.equal(raw_bytes(ours), raw_bytes(want))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("heads", "kv_heads", "head_dim", "rotary_dim", "gated"),
                         [(32, 8, 128, 128, False), (24, 4, 256, 64, True), (16, 8, 64, 64, False)])
def test_qk_rope_kv_is_the_composition(heads, kv_heads, head_dim, rotary_dim, gated, dtype):
    """D3: the fused kernel, selected for the facts the units answer (the table: FlashInfer's rope instance's),
    against the judge's composition on the same inputs — q' and the model-dtype cache within the L0 budget of a
    norm over head_dim, v copied exactly, no other byte touched; its q8_0 write is the reference write of its own
    model-dtype k, bitwise. T = 1 (a decode row), 300 rows across pages in any order, and 0."""
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"), dtype, options={"max_model_len": 4096})
    name = str(dtype).removeprefix("torch.")
    rope = ctx.select_kernel("rope@1", site="t", head_dim=head_dim, rotary_dim=rotary_dim, theta=1e6,
                             max_positions=4096, dtype=name).impl(ctx, head_dim=head_dim, rotary_dim=rotary_dim,
                                                                  theta=1e6, max_positions=4096, dtype=name)
    norm = {"norm": "rmsnorm@1", "eps": 1e-6, "offset": 1.0 if gated else 0.0}
    units = 40
    for t in (1, 300, 0):
        g = torch.Generator().manual_seed(t + head_dim)
        qw = 2 * (heads * head_dim * (2 if gated else 1))
        x = torch.randn(t, qw + 2 * kv_heads * head_dim, generator=g).to(dtype).cuda()
        q = x[:, :qw // 2].view(t, heads, qw // 2 // heads)          # strided rows, as the fused projection's split
        k, v = (x[:, qw // 2 + i * kv_heads * head_dim:][:, :kv_heads * head_dim].view(t, kv_heads, head_dim)
                for i in (0, 1))
        weights = [(torch.randn(head_dim, generator=g) * 0.1 + (0 if gated else 1)).to(dtype).cuda() for _ in "qk"]
        positions = torch.randint(0, 4096, (t,), generator=g).cuda()
        unit = torch.randperm(units, generator=g)[:t // PAGE + 1].repeat_interleave(PAGE)[:t]
        perm = torch.randperm(t, generator=g)
        unit, off = unit[perm].cuda(), (torch.arange(t) % PAGE)[perm].cuda()
        caches = {}
        for layout in (MODEL, Q8):
            facts = {"layout": layout, "kv_heads": kv_heads, "head_dim": head_dim, "page": PAGE, "dtype": name,
                     "heads": heads, "gated": gated, **norm, "rope": "rope@1", "rotary_dim": rotary_dim, "theta": 1e6,
                     "max_positions": 4096, "cos_sin": rope.cos_sin}
            spec = ctx.select_kernel(gqa_kernels.QK_ROPE_KV, site="t", **facts)
            assert spec.provider == TRITON
            caches[layout] = ours = empty_cache(units, kv_heads, head_dim, layout, dtype)
            for c in ours if isinstance(ours, tuple) else (ours,):
                c.view(torch.uint8).fill_(0x5A)                           # garbage: unwritten bytes stay
            q_out = spec.impl(ctx, **facts)(q, k, v, positions, *weights, ours, unit, off)
            if layout == MODEL:
                want = empty_cache(units, kv_heads, head_dim, layout, dtype).view(torch.uint8).fill_(0x5A).view(dtype)
                ref = fused.qk_rope_kv(q, k, v, positions, *weights, want, unit, off, eps=1e-6,
                                       offset=norm["offset"], theta=1e6, rotary_dim=rotary_dim)
                assert q_out.shape == (t, heads, head_dim) and q_out.is_contiguous()
                if t:
                    for got, exp in ((q_out, ref), (ours[unit, 0, :, off], want[unit, 0, :, off])):
                        r = l0.check(got, exp, dtype, head_dim)
                        assert r.passed, (t, r)
                assert torch.equal(ours[unit, 1, :, off], v)                                       # v: a copy
                ours[unit, 0, :, off] = want[unit, 0, :, off]
                assert torch.equal(raw_bytes(ours), raw_bytes(want))                               # nothing else
            else:
                want = on_cpu(empty_cache(units, kv_heads, head_dim, Q8, dtype))
                for c in want:
                    c.view(torch.uint8).fill_(0x5A)
                if t:
                    gqa.kv_write(caches[MODEL][unit, 0, :, off].cpu(), v.cpu(), want, unit.cpu(), off.cpu())
                assert torch.equal(raw_bytes(ours), raw_bytes(want)), t


@pytest.mark.parametrize("layout", [MODEL, Q8])
def test_a_decode_row_is_the_last_prefill_row(tmp_path, layout):
    """The GQA unit with its selected kernels (D3's fused op included) on a tiny Qwen3.5 — partial rotary, the gated q
    layout, a norm offset: prefilling T tokens against prefilling T − 1 then decoding the last as a one-token row.
    The last token's output agrees within the attention budget (another kernel computes it), and the KV it wrote is
    bitwise: inputs in {−1, 0, 1} and projection weights in {−1, 0, 1} / 8 make every projection exact whatever the
    matmul's order, so the same write lands at the same position; T − 1 = 16 opens a page. A decode-only position
    error, which every relative judge misses (docs/m07-perf.md §13.7), fails the KV check.

    Captured too (docs/m07-perf.md §17): one request's decode row captured into a CUDA graph at position P, then
    replayed at P, P + 1 (a page's first token) and P + 2, each step staged on the host as the engine stages a graphed
    step; output and KV bitwise another request's eager decode rows of the same tokens. A host-side decision captured
    into the graph (stale on replay) fails it."""
    path = save(tiny_hf(4, seed=1, dim=64), tmp_path)
    engine = Engine(EngineConfig(str(path), dtype=torch.bfloat16, cuda_graphs=False, overlap=False, page_size=PAGE,
                                 max_seqs=2, max_step_tokens=64, max_model_len=128, state_memory=8 << 20,
                                 plugin_options={"kv.format": "q8_0"} if layout == Q8 else {}))
    gqa_unit = next(m for m in engine.executor.model.modules() if isinstance(m, GQA))
    assert gqa_unit._fused is not None and gqa_unit.gated
    table = next(iter(engine.state.tables.values())).host
    torch.manual_seed(0)
    for linear in (m for m in gqa_unit.modules() if hasattr(m, "in_features") and m.weight is not None):
        linear.weight.copy_(torch.randint(-1, 2, linear.weight.shape) / 8)

    def stage(seq, t):
        """The host half of a step of `t` tokens of `seq` (static device buffers, as a graphed step's)."""
        assert engine.state.reserve(seq, t)
        n = np.array([t], np.int32)
        return engine.executor._context(host_plan(0, np.array([seq.slot], np.int32),
                                                  np.array([seq.num_computed], np.int32), n, n * 0 + 1,
                                                  np.zeros(t, np.int64)))

    def run(seq, x):
        ctx = stage(seq, len(x))
        engine.executor._prepare(ctx)
        seq.num_computed += len(x)
        return gqa_unit(x, ctx)[-1]

    def written(seq, pos):
        cache = gqa_unit._kv_layer()
        return [c[table[seq.slot, pos // PAGE], :, :, pos % PAGE] for c in (cache if layout == Q8 else (cache,))]

    with torch.no_grad():
        for t in (17, 40):
            x = torch.randint(-1, 2, (t, 64), device="cuda").to(torch.bfloat16)
            prefill, decode = seqs = [Seq(str(i), [0] * t, SamplingParams(seed=0)) for i in range(2)]
            assert all(engine.state.admit(s) for s in seqs)
            last = run(prefill, x)
            run(decode, x[:-1])
            assert usage(run(decode, x[-1:]), last) <= 1, t
            got, want = written(decode, t - 1), written(prefill, t - 1)
            assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True)), t
            for s in seqs:
                engine.state.release(s)
        start = 2 * PAGE - 1                                    # replays at 31, 32 (opens a page) and 33
        x = torch.randint(-1, 2, (start + 3, 64), device="cuda").to(torch.bfloat16)
        graphed, eager = seqs = [Seq(str(i), [0] * (start + 3), SamplingParams(seed=0)) for i in (2, 3)]
        assert all(engine.state.admit(s) for s in seqs)
        run(graphed, x[:start]), run(eager, x[:start])
        static, graph = x[start:start + 1].clone(), torch.cuda.CUDAGraph()
        for pos in range(start, start + 3):
            want = run(eager, x[pos:pos + 1])                   # eager first: every kernel warm before the capture
            ctx = stage(graphed, 1)
            static.copy_(x[pos:pos + 1])
            if pos == start:
                with torch.cuda.graph(graph):
                    engine.executor._prepare(ctx)
                    out = gqa_unit(static, ctx)
            graph.replay()
            graphed.num_computed += 1
            assert torch.equal(out[-1], want), pos
            assert all(torch.equal(a, b) for a, b in zip(written(graphed, pos), written(eager, pos), strict=True)), pos
    engine.close()


def test_kv_write_q8_0_is_ggufs_own():
    gguf = pytest.importorskip("gguf")
    k = shaped(torch.randn(64, 2, 256, generator=torch.Generator().manual_seed(1))).cuda()
    cache = empty_cache(4, 2, 256, Q8)
    writer(Q8, 2, 256, "float32")(k, k, cache, torch.arange(64, device="cuda") // PAGE,
                                   torch.arange(64, device="cuda") % PAGE)
    blocks = gguf.quants.quantize(k.cpu().numpy(), gguf.GGMLQuantizationType.Q8_0).reshape(4, PAGE, 2, 8, 34)
    values, scales = (x[:, 0].transpose(1, 2).cpu() for x in cache)               # [unit, page, Hkv, ...]
    assert torch.equal(values.reshape(4, PAGE, 2, 8, 32), torch.from_numpy(blocks[..., 2:].copy()).view(torch.int8))
    assert torch.equal(scales, torch.from_numpy(blocks[..., :2].copy()).view(torch.float16)[..., 0])


# attention -----------------------------------------------------------------------------------------------------------

def _cases(layout):
    return [pytest.param(layout, *case, marks=[needs_fp8] if layout == FP8 else [],
                         id=f"{layout}-{case[0]}-{case[1]}")
            for case in CASES[layout]]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize(("layout", "op", "provider", "rows"), _cases(Q8) + _cases(FP8))
def test_kernel_reads_the_stored_format(layout, op, provider, rows, geometry, dtype):
    q, _, cache, batch, pages = quant_case(rows, *geometry, layout, dtype=dtype)
    kernel = _kernel(op, provider, *geometry, dtype=str(dtype).removeprefix("torch."), layout=layout)
    ours, want = _run(kernel, q, cache, batch, pages), _run(PagedAttention(None), q, cache, batch, pages)
    assert usage(ours, want) <= 1


@pytest.mark.parametrize("op", [DECODE, VERIFY])
def test_triton_reads_fp8_on_every_sm(op):
    # fp8 is selected from sm_89 on, but the Triton kernel decodes e4m3 from its bits on any SM: checked here too.
    from holon_plugins.kernels.triton_attn import plugin
    geometry = (24, 4, 256)
    q, _, cache, batch, pages = quant_case(ROWS[op], *geometry, FP8)
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    facts = {"heads": 24, "kv_heads": 4, "head_dim": 256, "page": PAGE, "q_dtype": "bfloat16",
             "kv_dtype": "float8_e4m3fn", "layout": FP8, "max_seqs": MAX_SEQS}
    kernel = (plugin.Decode if op == DECODE else plugin.Verify)(ctx, **facts)
    assert usage(_run(kernel, q, cache, batch, pages), _run(PagedAttention(None), q, cache, batch, pages)) <= 1


@pytest.mark.parametrize("masks", [{"causal": False, "window": 16}, {"causal": False, "window": 2048}],
                         ids=["window16", "window2048"])
@pytest.mark.parametrize("layout", [Q8, FP8])
def test_verify_masks_read_the_stored_format(layout, masks):
    # A drafter's block (8 rows, bidirectional, sliding window) on the KV formats a 24 GB card serves it with.
    if layout == FP8 and not fp8_device():
        pytest.skip("fp8 KV kernels are selected from sm_89 on")
    geometry = GEOMETRIES[0]
    q, _, cache, batch, pages = quant_case(BLOCK_ROWS, *geometry, layout)
    kernel = _kernel(VERIFY, TRITON, *geometry, layout=layout, **masks)
    assert usage(_run(kernel, q, cache, batch, pages), _run(PagedAttention(None, **masks), q, cache, batch, pages)) <= 1


@pytest.mark.parametrize(("op", "rows"), [(op, rows) for op, _, rows in CASES[Q8]])
def test_q8_0_padding_rows_are_zero(op, rows):
    q, _, cache, batch, pages = quant_case(rows, *GEOMETRIES[2], Q8)
    pages.lens[1::2] = 0                                   # graph padding: length 0 on a valid slot
    ours = _run(_kernel(op, TRITON, *GEOMETRIES[2], layout=Q8), q, cache, batch, pages)
    want = _run(PagedAttention(None), q, cache, batch, pages)
    assert torch.isfinite(ours).all() and usage(ours, want) <= 1
    starts = batch.dev.query_start.tolist()
    assert not any(ours[starts[b]:starts[b + 1]].any() for b in range(1, len(rows), 2))


@pytest.mark.parametrize(("layout", "op", "provider", "rows"), _cases(Q8) + _cases(FP8))
def test_kernel_is_run_to_run_identical(layout, op, provider, rows):
    q, _, cache, batch, pages = quant_case(rows, *GEOMETRIES[2], layout)
    kernel = _kernel(op, provider, *GEOMETRIES[2], layout=layout)
    assert torch.equal(_run(kernel, q, cache, batch, pages), _run(kernel, q, cache, batch, pages))


def test_q8_0_decode_step_replays_from_a_graph():
    """kv_write, the device plan and decode captured once, replayed for the next step (new q, K/V and lengths)."""
    geometry, rows = GEOMETRIES[2], [(40, 1), (7, 1), (300, 1), (0, 1)]
    q, raw, cache, batch, pages = quant_case(rows, *geometry, Q8)
    kernel, write = _kernel(DECODE, TRITON, *geometry, layout=Q8), writer(Q8, *geometry[1:])
    unit, off = step_slots(rows, pages, batch)
    k, v = (raw[unit, i, :, off].contiguous() for i in (0, 1))
    handle = kernel.plan(batch, pages)
    kernel.prepare(handle)                                  # compiled and warm before the capture
    write(k, v, cache, unit, off)
    kernel.run(q, cache, handle)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):                           # torch's default error mode: a host sync would fail
        kernel.prepare(handle)
        write(k, v, cache, unit, off)
        out = kernel.run(q, cache, handle)
    nxt = [(s + 1, 1) for s, _ in rows]                     # the next step: one more token per row
    pages.lens += 1
    new_unit, new_off = step_slots(nxt, pages, batch)
    unit.copy_(new_unit), off.copy_(new_off)
    q.copy_(torch.randn_like(q)), k.copy_(shaped(torch.randn_like(k))), v.copy_(torch.randn_like(v))
    graph.replay()
    want_cache = on_cpu(cache)
    gqa.kv_write(k.cpu(), v.cpu(), want_cache, unit.cpu(), off.cpu())
    assert torch.equal(raw_bytes(cache), raw_bytes(want_cache))
    assert torch.equal(out, kernel.run(q, cache, handle))                      # replay = eager
    assert usage(out, _run(PagedAttention(None), q, cache, batch, pages)) <= 1


# plants --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("op", [PREFILL, DECODE, VERIFY])
def test_plant_q8_0_scales_from_the_wrong_head_fail(op):
    q, _, cache, batch, pages = quant_case(ROWS[op], *GEOMETRIES[2], Q8)
    kernel = _kernel(op, TRITON, *GEOMETRIES[2], layout=Q8)
    want = _run(PagedAttention(None), q, cache, batch, pages)
    assert usage(_run(kernel, q, cache, batch, pages), want) <= 1
    planted = (cache[0], cache[1].roll(1, dims=2))          # each kv head reads its neighbour's scales
    assert usage(_run(kernel, q, planted, batch, pages), want) > 1


@pytest.mark.parametrize("layout", [Q8, FP8])
@pytest.mark.parametrize("op", [PREFILL, DECODE, VERIFY])
def test_a_step_reads_its_own_tokens_through_the_cache(op, layout):
    """The step's K/V go in through kv_write over garbage, then the kernel reads them back stored (the reference
    reads the same cache); the plant — attending to the step's own K/V unquantized, as a prefill fed this chunk's
    bf16 K/V would — must fail."""
    if layout == FP8 and not fp8_device():
        pytest.skip("fp8 KV kernels are selected from sm_89 on")
    geometry, rows = (24, 4, 256), ROWS[op]
    q, raw, cache, batch, pages = quant_case(rows, *geometry, layout)
    unit, off = step_slots(rows, pages, batch)
    k, v = raw[unit, 0, :, off], raw[unit, 1, :, off]
    for x in cache if isinstance(cache, tuple) else (cache,):
        x[unit, :, :, off] = x[(unit + 1) % len(x), :, :, off]                      # stale contents
    writer(layout, *geometry[1:])(k, v, cache, unit, off)
    want = _run(PagedAttention(None), q, store(raw, layout), batch, pages)          # everything stored, then read
    provider = next(p for o, p, _ in CASES[layout] if o == op)
    assert usage(_run(_kernel(op, provider, *geometry, layout=layout), q, cache, batch, pages), want) <= 1
    mixed = dequantized(cache)
    mixed[unit, 0, :, off], mixed[unit, 1, :, off] = k.float(), v.float()            # the plant: this step raw
    assert usage(_run(PagedAttention(None), q, mixed, batch, pages), want) > 1


# selection -----------------------------------------------------------------------------------------------------------

def _facts(layout, op_dtype="bfloat16"):
    return {"heads": 24, "kv_heads": 4, "head_dim": 256, "page": PAGE, "q_dtype": op_dtype,
            "kv_dtype": gqa_kernels.kv_dtype(layout, op_dtype), "layout": layout, "max_seqs": MAX_SEQS}


def test_selection_per_layout():
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    pick = {op: ctx.select_kernel(f"{op}@1", site="t", **_facts(Q8)).provider for op in (PREFILL, DECODE, VERIFY)}
    assert pick == {PREFILL: TRITON, DECODE: TRITON, VERIFY: TRITON}
    bf16 = {op: ctx.select_kernel(f"{op}@1", site="t", **_facts("kv.hnd")).provider for op in (PREFILL, DECODE)}
    assert bf16 == {PREFILL: FLASHINFER, DECODE: TRITON}                      # the model-dtype choices are unchanged
    if fp8_device():
        pick = {op: ctx.select_kernel(f"{op}@1", site="t", **_facts(FP8)).provider for op in (PREFILL, DECODE, VERIFY)}
        assert pick == {PREFILL: FLASHINFER, DECODE: TRITON, VERIFY: TRITON}       # decode: graph-safe
    else:
        for op in (PREFILL, DECODE, VERIFY):
            with pytest.raises(PluginError, match="no kernel"):
                ctx.select_kernel(f"{op}@1", site="t", **_facts(FP8))
