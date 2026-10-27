"""Quantized weights in std units (docs/dev/m06-lowp.md D3–D6), on the CPU with a toy layout (fixtures/toy_quant): a
unit hands its kernel every part by role (`qlinear@2`, `linear_experts@2`), binds after merges with the `act_quant`
fact, keeps only what the kernel's `prepare` returns, merges parts by shape, and stacks experts slice by slice; a
merged gate/up binds the fused activation, and one input prepares once for several linears (docs/dev/m07-perf.md D6)."""
import dataclasses
import pathlib
import types

import pytest
import torch
from holon.api import Layout, Manifest, QTensor, Ref, Scheme, WeightMap, WeightRule
from holon.core.build import BuildContext, load_weights
from holon.core.plugins import Installed, PluginError, PluginSet, discover
from holon_std.moe import reference as moe

from .fixtures.toy_quant.plugin import CALLS, PREPARED, ROWS, G, Handle

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
    assert unit.adopt_weights(ctx, "weight", q)
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


@pytest.mark.parametrize("act", ["a8", "a4"])
def test_mixed_steps_refuse_kernels_with_two_activation_modes(act):
    """docs/dev/m07-perf.md §3.1: with `chunk_with_decode` > 0 a quantized linear and an expert stack refuse, at bind, a
    kernel whose paths compute in different activation modes (the toy's a4 spec: chunks at 4 bits) — generically,
    from its spec's `config["act"]`; one mode on both paths (a8) binds, and every kernel binds without mixed steps
    (the option at 0: the engine passes 0 with a drafter, whose steps alternate). The units read that option alone,
    not `max_draft_tokens` (§17)."""
    for chunk, drafts in ((0, 3), (256, 3), (256, 0)):
        ctx = _ctx(**{"quant.act": act})
        ctx.options |= {"chunk_with_decode": chunk, "max_draft_tokens": drafts}
        stacks = _stacks(ctx, lambda seed: _toy(128, 128, seed))
        for unit in (_linear(ctx, _toy(4, H, seed=0)), type(stacks.gate).merge(stacks.gate, stacks.up)):
            if chunk and act == "a4":
                with pytest.raises(ValueError, match=r"needs one activation mode on every path; tests.toy_quant"):
                    unit.bind_kernels(ctx)
            else:
                unit.bind_kernels(ctx)


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


@pytest.mark.parametrize("second, shape2, merges", [
    (dict(scale2=0.5), (), True),                   # rows concatenate (scale: one per G rows), scale2 equal
    (dict(scale2=0.25), (), False),                 # scale2 differs: the units stay apart
    (dict(rows_per_scale=1), (), False),            # its scale rows are not out / G: neither rows nor equal
    # equal outs, where a [1] scale2 would also read as one row per unit: it is per tensor, kept once
    (dict(scale2=0.5, out=4), (1,), True),
    (dict(scale2=0.25, out=4), (1,), False)])
def test_linears_merge_parts_by_shape(second, shape2, merges):
    ctx = _ctx()
    second = dict(second)
    out = second.pop("out", 2)
    qs = [_toy(4, H, seed=2, shape2=shape2), _toy(out, H, seed=3, shape2=shape2, **second),
          _toy(out, H, seed=4, shape2=shape2)]
    units = [_linear(ctx, q) for q in qs]
    merged = type(units[0]).merge(*units)
    if not merges:
        assert merged is None and all(u.weight is not None for u in units)
        return
    rows = 4 + 2 * out
    assert {r: tuple(t.shape) for r, t in merged.parts.items()} == {"q": (rows, H), "scale": (rows // G,),
                                                                    "scale2": shape2}
    assert all(u.weight is None and u.parts is None for u in units)
    merged.bind_kernels(ctx)
    x = _x(3, seed=2)
    torch.testing.assert_close(merged(x), torch.cat([x @ _weight(q).T for q in qs], dim=-1))


def _swiglu(ctx, up_scale2: float):
    """A toy SwiGLU, its weights adopted and bound (gate and up merge when their scale2 agree), and its dense
    weights."""
    with torch.device("meta"):
        ffn = ctx.unit(Ref("std.ffn.swiglu@1", inter=INTER), hidden=H)
    qs = {"gate": _toy(INTER, H, seed=6), "up": _toy(INTER, H, seed=7, scale2=up_scale2),
          "down": _toy(H, INTER, seed=8)}
    for n, q in qs.items():
        assert getattr(ffn, f"{n}_proj").adopt_weights(ctx, "weight", q)
    ffn.finalize_weights(ctx)
    for m in ffn.modules():
        if hasattr(m, "bind_kernels"):
            m.bind_kernels(ctx)
    return ffn, {n: _weight(q) for n, q in qs.items()}


def test_a_merged_gate_up_fuses_its_activation_and_linears_share_a_prepared_input():
    """D6: SwiGLU's merged gate/up binds `qlinear_glu@1` (a lock site) because its `qlinear@2` spec lists it in
    `config["fused"]`; at one row the output is the fused kernel's, at two the composition (the toy fuses one row, as
    ggml).
    Unmerged (another scale2 apiece), gate and up share one `prepare_input`, which a dense linear takes too."""
    for scale2, merged in ((0.5, True), (0.25, False)):
        ctx = _ctx()
        ffn, weights = _swiglu(ctx, scale2)
        assert (ffn._gate_up is not None) == merged
        assert any(k.endswith("/qlinear_glu@1") for k in ctx.kernels) == merged
        for rows in (1, 2):
            CALLS.clear()
            x = _x(rows, seed=rows)
            g, u = x @ weights["gate"].T, x @ weights["up"].T
            want = (torch.nn.functional.silu(g) * u) @ weights["down"].T
            torch.testing.assert_close(ffn(x), want, rtol=1e-5, atol=1e-3)
            fused = ["qlinear_glu", "qlinear"] if merged and rows == 1 else []
            shared = [] if merged else ["prepare_input", "qlinear(handle)", "qlinear(handle)", "qlinear"]
            assert CALLS == (fused or (["qlinear", "qlinear"] if merged else shared)), (merged, rows, CALLS)
    dense = torch.nn.Linear(H, 2, bias=False)
    with torch.device("meta"):
        linear = ctx.unit(Ref("std.ops.linear@1", in_features=H, out_features=2))
    torch.utils.swap_tensors(linear.weight, torch.nn.Parameter(dense.weight.detach().clone(), requires_grad=False))
    x = _x(3, seed=9)
    assert linear.prepare_input(x) is x and torch.equal(linear(Handle(x)), dense(x))   # a dense linear: x itself


@pytest.mark.parametrize("ours", [False, True], ids=["another-only", "another-chosen"])
def test_a_glu_kernel_of_another_provider_binds_no_site(monkeypatch, ours):
    """A merged gate/up's `qlinear_glu@1` must be its `qlinear@2` kernel's provider's, which reads the parts that
    provider prepared (`BuildCtx.candidates`, docs/dev/m07-perf.md §17): when only another provider serves it, no site
    is
    recorded and SwiGLU composes; when the deployment would choose another provider's over it, the build refuses."""
    ctx = _ctx()
    kernels, get = ctx.plugins.regs.kernels, ctx.plugins.regs.kernels.get

    def candidates(op: str) -> list:
        if op != "qlinear_glu":
            return get(op)
        other = [dataclasses.replace(s, provider="tests.other", priority=20) for s in get(op)]
        return other + (get(op) if ours else [])
    monkeypatch.setattr(kernels, "get", candidates)
    if ours:
        with pytest.raises(ValueError, match="would run tests.other's kernel on tests.toy_quant's prepared parts"):
            _swiglu(ctx, 0.5)
        return
    ffn, weights = _swiglu(ctx, 0.5)
    assert ffn._gate_up is not None and not any(k.endswith("/qlinear_glu@1") for k in ctx.kernels)
    CALLS.clear()
    x = _x(1, seed=1)
    want = (torch.nn.functional.silu(x @ weights["gate"].T) * (x @ weights["up"].T)) @ weights["down"].T
    torch.testing.assert_close(ffn(x), want, rtol=1e-5, atol=1e-3)
    assert CALLS == ["qlinear", "qlinear"]


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
            assert getattr(unit, p).adopt_weights(ctx, "weight", slices[p, e], index=e)
    other = QTensor(Layout(Scheme("toy.int8"), "cols"), shapes["gate"], slices["gate", 0].parts)
    assert not unit.gate.adopt_weights(ctx, "weight", other, index=0)                  # another layout
    assert not unit.gate.adopt_weights(ctx, "weight", _toy(*shapes["gate"], seed=0, rows_per_scale=1), index=0)  # shape
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
            assert getattr(unit, p).adopt_weights(ctx, "weight", slice_of(10 * e + i), index=e)
    return unit


def test_expert_block_scales_of_one_block_merge_per_expert():
    """[E, 1, 1] is one block row per expert: gate/up concatenate it per expert, [E, 2, 1], not [2, E]; a part with
    an empty dimension does not merge."""
    unit = _stacks(_ctx(), lambda seed: _block_scaled(128, 128, seed))
    merged = type(unit.gate).merge(unit.gate, unit.up)
    assert tuple(merged.parts["scale"].shape) == (E, 2, 1) and tuple(merged.parts["q"].shape) == (E, 256, 128)
    unit = _stacks(_ctx(), lambda seed: _block_scaled(128, 128, seed, empty=torch.zeros(0)))
    assert type(unit.gate).merge(unit.gate, unit.up) is None
