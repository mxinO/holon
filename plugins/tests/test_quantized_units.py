"""Quantized weights in std units (docs/m06-lowp.md D3–D6), on the CPU with a toy layout (fixtures/toy_quant): a
unit hands its kernel every part by role (`qlinear@2`, `linear_experts@2`), binds after merges with the `act_quant`
fact, keeps only what the kernel's `prepare` returns, merges parts by shape, and stacks experts slice by slice."""
import pathlib
import types

import pytest
import torch
from holon.api import Layout, Manifest, QTensor, Ref, Scheme, WeightMap, WeightRule
from holon.core.build import BuildContext, load_weights
from holon.core.plugins import Installed, PluginError, PluginSet, discover
from holon_std.moe import reference as moe

from .fixtures.toy_quant.plugin import PREPARED, ROWS, G

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "toy_quant"
E, H, INTER = 4, 8, 4


def _ctx(**plugin_options) -> BuildContext:
    found = discover()
    manifest = Manifest.load(FIXTURE / "plugin.toml")
    found[manifest.name] = Installed(manifest, "tests.fixtures.toy_quant", FIXTURE)
    return BuildContext(PluginSet(found), torch.device("cpu"), torch.float32, options={"plugins": plugin_options})


def _toy(out: int, inp: int, seed: int, scale2: float = 0.5, rows_per_scale: int = G, shape2=()) -> QTensor:
    g = torch.Generator().manual_seed(seed)
    return QTensor(ROWS, (out, inp), {"q": torch.randint(-128, 128, (out, inp), dtype=torch.int8, generator=g),
                                      "scale": torch.rand(out // rows_per_scale, generator=g),
                                      "scale2": torch.full(shape2, scale2)})


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


@pytest.mark.parametrize("scale2", [0.25, 0.5])
def test_a_one_element_part_is_per_tensor_in_a_merge(scale2):
    """Two units of equal `out` with a [1] per-tensor scale: equal, kept once; unequal, the units stay apart (never
    concatenated as if it were one row per unit)."""
    ctx = _ctx()
    qs = [_toy(4, H, seed=6, shape2=(1,)), _toy(4, H, seed=7, scale2=scale2, shape2=(1,))]
    units = [_linear(ctx, q) for q in qs]
    merged = type(units[0]).merge(*units)
    if scale2 != 0.5:
        assert merged is None
        return
    assert tuple(merged.parts["scale2"].shape) == (1,)
    merged.bind_kernels(ctx)
    x = _x(3, seed=4)
    torch.testing.assert_close(merged(x), torch.cat([x @ _weight(q).T for q in qs], dim=-1))


@pytest.mark.parametrize("ref, site", [("std.ops.linear@1", dict(out_features=4, in_features=H)),
                                       ("std.ops.embedding@1", dict(vocab=4, hidden=H)),
                                       ("std.ops.lm_head@1", dict(vocab=4, hidden=H))])
def test_a_unit_without_stacks_refuses_a_slice(ref, site):
    ctx = _ctx()
    model = torch.nn.Module()
    with torch.device("meta"):
        model.unit = ctx.unit(Ref(ref, **({"tied": False} if "lm_head" in ref else {})), **site)
    model.spec = types.SimpleNamespace(weights=WeightMap("hf_safetensors", rules=(
        WeightRule("u.{e}", "unit.weight", stack="e"),)))
    with pytest.raises(PluginError, match="cannot read layout"):
        load_weights(model, ctx, [("u.0", _toy(4, H, seed=0))])


@pytest.mark.parametrize("shape2", [(), (1,)])
def test_a_quantized_moe_stacks_its_experts_slice_by_slice(shape2):
    ctx = _ctx()
    with torch.device("meta"):
        unit = ctx.unit(Ref("std.moe@1", experts=E, top_k=2, inter=INTER, renormalize=True), hidden=H, layer=0)
    router = torch.nn.Parameter(torch.randn(E, H, generator=torch.Generator().manual_seed(5)), requires_grad=False)
    torch.utils.swap_tensors(unit.router, router)
    shapes = {"gate": (INTER, H), "up": (INTER, H), "down": (H, INTER)}
    slices = {(p, e): _toy(*shapes[p], seed=10 * e + i, scale2=0.25 * (1 + e + i), shape2=shape2) for e in range(E)
              for i, p in enumerate(shapes)}
    for e in (3, 1, 0, 2):
        for p in shapes:
            assert getattr(unit, p).adopt(ctx, "weight", slices[p, e], index=e)
    other = QTensor(Layout(Scheme("toy.int8"), "cols"), shapes["gate"], slices["gate", 0].parts)
    assert not unit.gate.adopt(ctx, "weight", other, index=0)                          # another layout
    assert not unit.gate.adopt(ctx, "weight", _toy(*shapes["gate"], seed=0, rows_per_scale=1), index=0)  # shapes
    assert {r: tuple(t.shape) for r, t in unit.gate.parts.items()} == {"q": (E, INTER, H), "scale": (E, 2),
                                                                       "scale2": (E, *shape2)}
    unit.finalize_weights(ctx)
    scale2 = [torch.stack([slices[p, e].parts["scale2"].reshape(()) for e in range(E)]) for p in ("gate", "up")]
    assert torch.equal(unit._gate_up.parts["scale2"], torch.stack(scale2))           # [2, E]: gate's row, up's
    PREPARED.clear()
    for m in unit.modules():
        if hasattr(m, "bind_kernels"):
            m.bind_kernels(ctx)
    assert [tuple(source["scale2"].shape) for _, source in PREPARED] == [(E, *shape2), (2, E)]     # down, gate/up
    assert unit._gate_up.parts.keys() == {"q", "scale"} and unit.down.parts.keys() == {"q", "scale"}
    facts = [s["facts"] for k, s in ctx.kernels.items() if k.endswith("/linear_experts@2")]
    assert [f["act_quant"] for f in facts] == ["a8", "a8"] and {f["glu"] for f in facts} == {"swiglu", "none"}
    stack = {p: torch.stack([_weight(slices[p, e]) for e in range(E)]) for p in shapes}
    x = _x(5, seed=3)
    weights, ids = moe.moe_topk(x @ unit.router.T, 2, True)
    h = moe.linear_experts(x[:, None], torch.cat([stack["gate"], stack["up"]], dim=1), ids, "swiglu")
    want = moe.moe_combine(moe.linear_experts(h, stack["down"], ids), weights, x.dtype)
    torch.testing.assert_close(unit(x), want, rtol=1e-5, atol=1e-2)       # |y| up to ~1e5 (SwiGLU of int8 rows)


def _block_scaled(out: int, inp: int, seed: int, **extra: torch.Tensor) -> QTensor:
    """FP8-shaped: int8 rows and a scale per 128 × 128 block, [out/128, in/128] (a 128 × 128 unit's is [1, 1])."""
    g = torch.Generator().manual_seed(seed)
    return QTensor(ROWS, (out, inp), {"q": torch.randint(-128, 128, (out, inp), dtype=torch.int8, generator=g),
                                      "scale": torch.rand(out // 128, inp // 128, generator=g), **extra})


def test_a_block_scale_of_one_block_merges_as_rows():
    """[1, 1] is one row of blocks, not a per-tensor value: two 128-row units merge to a [2, 1] scale (a kept [1, 1]
    would leave the second block row's scale out of range)."""
    ctx = _ctx()
    q = _block_scaled(128, 128, seed=1)
    units = [_linear(ctx, QTensor(ROWS, (128, 128), dict(q.parts))) for _ in range(2)]        # equal scales too
    merged = type(units[0]).merge(*units)
    assert tuple(merged.parts["scale"].shape) == (2, 1) and tuple(merged.parts["q"].shape) == (256, 128)


def _stacks(ctx, slice_of):
    """A std.moe unit (E experts, inter = hidden = 128) whose gate and up stacks adopt `slice_of(seed)` per expert."""
    with torch.device("meta"):
        unit = ctx.unit(Ref("std.moe@1", experts=E, top_k=2, inter=128, renormalize=True), hidden=128, layer=0)
    for e in range(E):
        for i, p in enumerate(("gate", "up")):
            assert getattr(unit, p).adopt(ctx, "weight", slice_of(10 * e + i), index=e)
    return unit


def test_expert_block_scales_of_one_block_merge_per_expert():
    """[E, 1, 1] is one block row per expert: gate/up concatenate it per expert, [E, 2, 1], not [2, E]."""
    unit = _stacks(_ctx(), lambda seed: _block_scaled(128, 128, seed))
    merged = type(unit.gate).merge(unit.gate, unit.up)
    assert tuple(merged.parts["scale"].shape) == (E, 2, 1) and tuple(merged.parts["q"].shape) == (E, 256, 128)


def test_an_expert_part_with_an_empty_dimension_does_not_merge():
    unit = _stacks(_ctx(), lambda seed: _block_scaled(128, 128, seed, empty=torch.zeros(0)))
    assert type(unit.gate).merge(unit.gate, unit.up) is None
