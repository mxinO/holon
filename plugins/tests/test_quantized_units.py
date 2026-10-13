"""Quantized weights in std units (docs/m06-lowp.md D3–D6), on the CPU with a toy layout (fixtures/toy_quant): a
unit hands its kernel every part by role (`qlinear@2`, `linear_experts@1`), binds after merges with the `act_quant`
fact, keeps only what the kernel's `prepare` returns, merges parts by shape, and stacks experts slice by slice."""
import pathlib

import pytest
import torch
from holon.api import Layout, Manifest, QTensor, Ref, Scheme
from holon.core.build import BuildContext
from holon.core.plugins import Installed, PluginSet, discover
from holon_std.moe import reference as moe

from .fixtures.toy_quant.plugin import FOLDED, PREPARED, ROWS, G

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "toy_quant"
E, H, INTER = 4, 8, 4


def _ctx(**plugin_options) -> BuildContext:
    found = discover()
    manifest = Manifest.load(FIXTURE / "plugin.toml")
    found[manifest.name] = Installed(manifest, "tests.fixtures.toy_quant", FIXTURE)
    return BuildContext(PluginSet(found), torch.device("cpu"), torch.float32, options={"plugins": plugin_options})


def _toy(out: int, inp: int, seed: int, scale2: float = 0.5, rows_per_scale: int = G) -> QTensor:
    g = torch.Generator().manual_seed(seed)
    return QTensor(ROWS, (out, inp), {"q": torch.randint(-128, 128, (out, inp), dtype=torch.int8, generator=g),
                                      "scale": torch.rand(out // rows_per_scale, generator=g),
                                      "scale2": torch.tensor(scale2)})


def _x(rows: int, seed: int) -> torch.Tensor:
    return torch.randn(rows, H, generator=torch.Generator().manual_seed(seed))


def _weight(q: QTensor) -> torch.Tensor:
    p = q.parts
    return p["q"].float() * p["scale"].repeat_interleave(len(p["q"]) // len(p["scale"]))[:, None] * p["scale2"]


def _unit(ctx, ref: str, q: QTensor, **site):
    with torch.device("meta"):
        unit = ctx.unit(Ref(ref, **site))
    assert unit.adopt(ctx, "weight", q)
    return unit


def _linear(ctx, q: QTensor):
    return _unit(ctx, "std.ops.linear@1", q, out_features=q.logical_shape[0], in_features=q.logical_shape[1])


def test_a_linear_reads_the_parts_its_kernel_prepared():
    ctx, q = _ctx(), _toy(4, H, seed=0)
    unit = _linear(ctx, q)
    assert unit.parts.keys() == {"q", "scale", "scale2"} and unit.parts["q"] is unit.weight
    PREPARED.clear()
    unit.bind_kernels(ctx)
    assert len(PREPARED) == 1 and PREPARED[0][0].layout == ROWS
    assert unit.parts.keys() == {"q", "scale"} and unit.parts["q"] is unit.weight      # the prepared primary part
    assert unit.weight.data_ptr() != q.parts["q"].data_ptr()                             # the source is let go
    x = _x(3, seed=0)
    torch.testing.assert_close(unit(x), x @ _weight(q).T)
    (key, site), = ((k, s) for k, s in ctx.kernels.items() if k.endswith("/qlinear@2"))
    assert site["provider"] == "tests.toy_quant" and site["facts"]["act_quant"] == "a8", key


def test_quant_act_sets_the_fact_and_a16_is_refused():
    ctx = _ctx(**{"quant.act": "a4"})
    _linear(ctx, _toy(4, H, seed=0)).bind_kernels(ctx)
    assert [s["facts"]["act_quant"] for s in ctx.kernels.values()] == ["a4"]
    with pytest.raises(ValueError, match=r"quant.act 'a16': one of \['a4', 'a8'\]"):
        _linear(_ctx(**{"quant.act": "a16"}), _toy(4, H, seed=0))


def test_a_tied_head_prepares_its_own_copy_of_the_embedding_parts():
    ctx, q = _ctx(), _toy(16, H, seed=1)
    embed = _unit(ctx, "std.ops.embedding@1", q, vocab=16, hidden=H)
    with torch.device("meta"):
        head = ctx.unit(Ref("std.ops.lm_head@1", tied=True), vocab=16, hidden=H, embed=embed)
    source = {r: t.clone() for r, t in embed.parts.items()}
    head.finalize_weights(ctx)
    embed.bind_kernels(ctx)
    head.bind_kernels(ctx)
    assert embed.parts.keys() == source.keys() and all(torch.equal(embed.parts[r], t) for r, t in source.items())
    assert head.parts.keys() == {"q", "scale"} and head.weight is None
    h = _x(2, seed=1)
    torch.testing.assert_close(head(h), h @ _weight(q).T)


@pytest.mark.parametrize("second, merges", [
    (dict(scale2=0.5), True),                       # rows concatenate (scale: one per G rows), scale2 equal
    (dict(scale2=0.25), False),                     # scale2 differs: the units stay apart
    (dict(rows_per_scale=1), False),                # its scale rows are not out / G: neither rows nor equal
])
def test_linears_merge_parts_by_shape(second, merges):
    ctx = _ctx()
    qs = [_toy(4, H, seed=2), _toy(2, H, seed=3, **second), _toy(2, H, seed=4)]
    units = [_linear(ctx, q) for q in qs]
    merged = type(units[0]).merge(*units)
    if not merges:
        assert merged is None and all(u.weight is not None for u in units)
        return
    assert {r: tuple(t.shape) for r, t in merged.parts.items()} == {"q": (8, H), "scale": (4,), "scale2": ()}
    assert all(u.weight is None and u.parts is None for u in units)
    merged.bind_kernels(ctx)
    x = _x(3, seed=2)
    torch.testing.assert_close(merged(x), torch.cat([x @ _weight(q).T for q in qs], dim=-1))


def test_a_quantized_moe_stacks_its_experts_slice_by_slice():
    ctx = _ctx()
    with torch.device("meta"):
        unit = ctx.unit(Ref("std.moe@1", experts=E, top_k=2, inter=INTER, renormalize=True), hidden=H, layer=0)
    router = torch.nn.Parameter(torch.randn(E, H, generator=torch.Generator().manual_seed(5)), requires_grad=False)
    torch.utils.swap_tensors(unit.router, router)
    shapes = {"gate": (INTER, H), "up": (INTER, H), "down": (H, INTER)}
    slices = {(p, e): _toy(*shapes[p], seed=10 * e + i, scale2=0.25 * (1 + e + i)) for e in range(E)
              for i, p in enumerate(shapes)}
    for e in (3, 1, 0, 2):
        for p in shapes:
            assert getattr(unit, p).adopt(ctx, "weight", slices[p, e], index=e)
    other = QTensor(Layout(Scheme("toy.int8"), "cols"), shapes["gate"], slices["gate", 0].parts)
    assert not unit.gate.adopt(ctx, "weight", other, index=0)                          # another layout
    assert not unit.gate.adopt(ctx, "weight", _toy(*shapes["gate"], seed=0, rows_per_scale=1), index=0)  # shapes
    assert {r: tuple(t.shape) for r, t in unit.gate.parts.items()} == {"q": (E, INTER, H), "scale": (E, 2),
                                                                       "scale2": (E,)}
    unit.finalize_weights(ctx)
    scale2 = [torch.stack([slices[p, e].parts["scale2"] for e in range(E)]) for p in ("gate", "up")]
    assert torch.equal(unit._gate_up.parts["scale2"], torch.stack(scale2))           # [2, E]: gate's row, up's
    PREPARED.clear()
    for m in unit.modules():
        if hasattr(m, "bind_kernels"):
            m.bind_kernels(ctx)
    assert len(PREPARED) == 2 and unit._gate_up.parts.keys() == {"q", "scale"} and PREPARED[0][0].layout == ROWS
    assert unit._gate_up._kernel.prepare(QTensor(ROWS, (E, 2 * INTER, H), PREPARED[0][1])).layout == FOLDED
    facts = [s["facts"] for k, s in ctx.kernels.items() if k.endswith("/linear_experts@1")]
    assert [f["act_quant"] for f in facts] == ["a8", "a8"] and {f["glu"] for f in facts} == {"swiglu", "none"}
    stack = {p: torch.stack([_weight(slices[p, e]) for e in range(E)]) for p in shapes}
    x = _x(5, seed=3)
    weights, ids = moe.moe_topk(x @ unit.router.T, 2, True)
    h = moe.linear_experts(x[:, None], torch.cat([stack["gate"], stack["up"]], dim=1), ids, "swiglu")
    want = moe.moe_combine(moe.linear_experts(h, stack["down"], ids), weights, x.dtype)
    torch.testing.assert_close(unit(x), want, rtol=1e-5, atol=1e-2)       # |y| up to ~1e5 (SwiGLU of int8 rows)
