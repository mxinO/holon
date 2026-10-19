"""std.hyper's units on the CPU (docs/v1-flash-next.md D3, §5): `gr_block@1` returns R complete — the attention
half's combine applied by the FFN half's mix, its own combine before it returns (the trunk contract: what taps, the
drafter and the next block see) — equal to the ops' oracles composed by hand around the same mixer and FFN; `expand@1`
repeats the inner embedding into n float32 streams; `gr_mix@1` (the model's norm) is the output mixer."""
import torch
from holon.api import Ref
from holon.core.build import BuildContext, StepContext
from holon.core.plugins import PluginSet, discover
from holon_std.hyper import reference as hc

N, H, RANK, T, EPS = 4, 64, 16, 11, 1e-6


def _ctx() -> BuildContext:
    return BuildContext(PluginSet(discover()), torch.device("cpu"), torch.float32)


def _fill(module: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * (0.1 if p.dim() == 1 else p.shape[-1] ** -0.5)


@torch.no_grad()
def test_gr_block_returns_r_complete():
    ctx = _ctx()
    mixer, ffn = Ref("toy.running_sum@1"), Ref("std.ffn.swiglu@1", inter=32)
    block = ctx.unit(Ref("std.hyper.gr_block@1", n=N, rank=RANK), layer=0, hidden=H, mixer=mixer, ffn=ffn,
                     norm=Ref("std.ops.rmsnorm@1", eps=EPS))
    _fill(block, 0)
    ids = torch.randint(0, 128, (T,), generator=torch.Generator().manual_seed(1))
    step = StepContext.whole_sequences([ids.tolist()], "cpu")
    r = torch.randn(T, N * H, generator=torch.Generator().manual_seed(2))
    want = r.clone()
    out = block(r.clone(), step)
    a, f = block.attn_hc, block.ffn_hc
    x, w = hc.hc_gr_mix(want, a.norm, a.down, a.up, a.inject, None, None, N, EPS, torch.float32)
    y = block.mixer(x, step)
    x, w_ffn = hc.hc_gr_mix(want, f.norm, f.down, f.up, f.inject, y, w, N, EPS, torch.float32)
    hc.hc_gr_combine(want, block.ffn(x), w_ffn)
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_expand_and_the_output_mixer():
    ctx = _ctx()
    expand = ctx.unit(Ref("std.hyper.expand@1", inner=Ref("std.ops.embedding@1"), n=N), vocab=128, hidden=H)
    mix = ctx.unit(Ref("std.hyper.gr_mix@1", n=N, rank=RANK), dim=H)
    _fill(expand, 3)
    _fill(mix, 4)
    ids = torch.arange(T)
    r = expand(ids)
    assert r.dtype == torch.float32 and torch.equal(r, expand.inner.weight[ids].repeat(1, N))
    x, w = hc.hc_gr_mix(r, mix.hc.norm, mix.hc.down, mix.hc.up, None, None, None, N, EPS, torch.float32)
    assert w is None and torch.equal(mix(r), x)


@torch.no_grad()
def test_qsa_sparse_path_selecting_every_block_is_the_dense_attention():
    """`std.mixers.qsa@1` in a model-only build: with a budget above every sequence, the sparse path (pages made for
    the call, index, select, attend with the output gate) selects every block, so it equals gqa's dense attention."""
    ctx = _ctx()
    norm = Ref("std.ops.rmsnorm@1", eps=EPS, offset=1.0)
    qsa = ctx.unit(Ref("std.mixers.qsa@1", heads=4, kv_heads=2, head_dim=64, qk_norm=norm, output_gate="sigmoid",
                       rope=Ref("std.ops.rope@1", theta=1e7, rotary_dim=16), index_heads=2, index_head_dim=32,
                       budget=2048), layer=0, hidden=H)
    _fill(qsa, 5)
    ids = [list(range(1, 41)), list(range(50, 73))]
    step = StepContext.whole_sequences(ids, "cpu")
    x = torch.randn(sum(map(len, ids)), H, generator=torch.Generator().manual_seed(6))
    dense = qsa(x, step)
    qsa.limit = 0                                             # every step sparse
    torch.testing.assert_close(qsa(x, step), dense, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_ffn_sum_of_routed_parts_and_a_gated_shared_expert():
    ctx = _ctx()
    parts = (Ref("std.ffn.swiglu@1", inter=32), Ref("std.ffn.gated_swiglu@1", inter=16))
    ffn = ctx.unit(Ref("std.ffn.sum@1", parts=parts), hidden=H, layer=0)
    _fill(ffn, 7)
    x = torch.randn(T, H, generator=torch.Generator().manual_seed(8))
    shared = ffn.parts[1]
    want = ffn.parts[0](x) + shared.down_proj(torch.nn.functional.silu(shared.gate_proj(x)) * shared.up_proj(x)) \
        * torch.sigmoid(shared.gate(x))
    torch.testing.assert_close(ffn(x), want, rtol=1e-5, atol=1e-5)
