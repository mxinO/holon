"""The stacking weight rule (docs/m02-moe.md D7): per-expert checkpoint tensors fill one [E, …] parameter; quantized
ones are adopted slice by slice by the unit owning it (docs/m06-lowp.md D6)."""
import types

import pytest
import torch
from holon.api import Layout, QTensor, Scheme, WeightMap, WeightRule
from holon.core.build import BuildContext, _fill_slice, load_weights, map_name
from holon.core.plugins import PluginError, PluginSet, discover
from torch import nn

RULES = WeightMap("hf_safetensors", rules=(
    WeightRule("m.layers.{i}.experts.{e}.w", "layers.{i}.ffn.w", stack="e"),
    WeightRule("m.layers.{i}.{rest}", "layers.{i}.{rest}"),
))


def test_the_stack_capture_is_an_integer_index():
    assert map_name("m.layers.3.experts.12.w", RULES) == ("layers.3.ffn.w", None, 12)
    assert map_name("m.layers.3.experts.1.w", RULES)[2] == 1               # "1" and "12" are distinct slices
    assert map_name("m.layers.3.experts.x.w", RULES) == ("layers.3.experts.x.w", None, None)   # digits only
    assert map_name("m.layers.3.norm", RULES) == ("layers.3.norm", None, None)


def test_a_stack_must_name_a_capture():
    bad = WeightMap("hf_safetensors", rules=(WeightRule("m.experts.{e}.w", "ffn.w", stack="x"),))
    with pytest.raises(PluginError, match="stack 'x' is not one of its captures"):
        map_name("m.experts.0.w", bad)


@pytest.fixture
def ctx():
    return BuildContext(PluginSet(discover()), torch.device("cpu"))


def test_slices_fill_a_meta_stack_in_place(ctx):
    with torch.device("meta"):
        param = nn.Parameter(torch.empty(3, 2, 4, dtype=torch.bfloat16), requires_grad=False)
    got: set[int] = set()
    for e in (2, 0, 1):
        _fill_slice(None, param, "w", e, f"x.{e}", torch.full((2, 4), float(e)), None, ctx, got)
    assert not param.is_meta and param.dtype == torch.bfloat16 and got == {0, 1, 2}
    assert [param[e].float().unique().item() for e in range(3)] == [0.0, 1.0, 2.0]


def test_bad_slices_are_load_errors(ctx):
    with torch.device("meta"):
        param = nn.Parameter(torch.empty(3, 2, 4), requires_grad=False)
    got: set[int] = set()
    _fill_slice(None, param, "w", 0, "x.0", torch.zeros(2, 4), None, ctx, got)
    with pytest.raises(PluginError, match="filled twice"):
        _fill_slice(None, param, "w", 0, "x.0", torch.zeros(2, 4), None, ctx, got)
    with pytest.raises(PluginError, match="outside the 3 slices"):
        _fill_slice(None, param, "w", 3, "x.3", torch.zeros(2, 4), None, ctx, got)
    with pytest.raises(PluginError, match="a slice of"):
        _fill_slice(None, param, "w", 1, "x.1", torch.zeros(4, 2), None, ctx, got)


class _Model(nn.Module):
    def __init__(self, weights: WeightMap) -> None:
        super().__init__()
        self.spec = types.SimpleNamespace(weights=weights)
        with torch.device("meta"):
            self.w = nn.Parameter(torch.empty(2, 3), requires_grad=False)


@pytest.mark.parametrize("whole_first", [True, False])
def test_a_parameter_is_filled_whole_or_by_slices_not_both(ctx, whole_first):
    rules = WeightMap("hf_safetensors", rules=(WeightRule("e.{e}.w", "w", stack="e"), WeightRule("w", "w")))
    slices = [("e.0.w", torch.zeros(3)), ("e.1.w", torch.zeros(3))]
    whole = [("w", torch.ones(2, 3))]
    with pytest.raises(PluginError, match="filled both whole and by stack slices"):
        load_weights(_Model(rules), ctx, whole + slices if whole_first else slices + whole)



ROWS = Layout(Scheme("toy.int8"), "rows")     # int8 rows, an f32 scale per row


class _Rows(nn.Module):
    """A toy stacked unit reading one two-part layout into [S, …] stacks of each part, whole or slice by slice (the
    first slice materializes them), its parameter holding the first part."""

    def __init__(self, slices: int, rows: int, cols: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(slices, rows, cols), requires_grad=False)
        self.parts: dict[str, torch.Tensor] = {}

    def adopt_weights(self, ctx, name: str, q: QTensor, index: int | None = None) -> bool:
        if name != "w" or q.layout != ROWS:
            return False
        if index is None or self.w.is_meta:
            self.parts = dict(q.parts) if index is None else \
                {r: torch.empty(len(self.w), *t.shape, dtype=t.dtype) for r, t in q.parts.items()}
            torch.utils.swap_tensors(self.w, nn.Parameter(self.parts["q"], requires_grad=False))
        if index is not None:
            for role, t in q.parts.items():
                self.parts[role][index].copy_(t)
        return True

    def read(self) -> torch.Tensor:
        return self.parts["q"].float() * self.parts["scale"][..., None]


class _Stacked(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = types.SimpleNamespace(weights=WeightMap("hf_safetensors", rules=(
            WeightRule("e.{e}.w", "ffn.w", stack="e"), WeightRule("w", "ffn.w"),
            WeightRule("t.{e}.w", "ffn.w", stack="e", transform="neg_exp"))))
        with torch.device("meta"):
            self.ffn = _Rows(3, 2, 4)


def _slice(e: int, layout: Layout = ROWS) -> QTensor:
    g = torch.Generator().manual_seed(e)
    return QTensor(layout, (2, 4), {"q": torch.randint(-128, 128, (2, 4), dtype=torch.int8, generator=g),
                                    "scale": torch.rand(2, generator=g)})


def test_a_quantized_stack_is_adopted_slice_by_slice(ctx):
    model = _Stacked()
    load_weights(model, ctx, [(f"e.{e}.w", _slice(e)) for e in (2, 0, 1)])
    assert model.ffn.w.dtype == torch.int8 and model.ffn.w.shape == (3, 2, 4)
    want = torch.stack([_slice(e).parts["q"].float() * _slice(e).parts["scale"][:, None] for e in range(3)])
    assert torch.equal(model.ffn.read(), want)


WHOLE = QTensor(ROWS, (3, 2, 4), {"q": torch.zeros(3, 2, 4, dtype=torch.int8), "scale": torch.ones(3, 2)})


@pytest.mark.parametrize("tensors, match", [
    ([("e.0.w", _slice(0)), ("e.3.w", _slice(3))], "outside the 3 slices"),
    ([("e.0.w", _slice(0)), ("e.1.w", _slice(1)), ("e.0.w", _slice(0))], "filled twice"),
    ([("e.0.w", _slice(0)), ("e.2.w", _slice(2))], r"stack slices not loaded: \[1\]"),
    ([("w", WHOLE), ("e.0.w", _slice(0))], "filled both whole and by stack slices"),
    ([("t.0.w", _slice(0))], "transform 'neg_exp' cannot apply to a quantized tensor"),
    ([("e.0.w", _slice(0)), ("e.1.w", _slice(1, Layout(Scheme("toy.int8"), "cols")))], "_Rows cannot read layout"),
], ids=["out-of-range", "twice", "missing", "whole-then-sliced", "transform", "refused"])
def test_bad_quantized_slices_are_load_errors(ctx, tensors, match):
    with pytest.raises(PluginError, match=match):
        load_weights(_Stacked(), ctx, tensors)
