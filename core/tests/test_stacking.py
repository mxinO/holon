"""The stacking weight rule (docs/m02-moe.md D7): per-expert checkpoint tensors fill one [E, …] parameter."""
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
        _fill_slice(param, "w", e, f"x.{e}", torch.full((2, 4), float(e)), None, ctx, got)
    assert not param.is_meta and param.dtype == torch.bfloat16 and got == {0, 1, 2}
    assert [param[e].float().unique().item() for e in range(3)] == [0.0, 1.0, 2.0]


def test_bad_slices_are_load_errors(ctx):
    with torch.device("meta"):
        param = nn.Parameter(torch.empty(3, 2, 4), requires_grad=False)
    got: set[int] = set()
    _fill_slice(param, "w", 0, "x.0", torch.zeros(2, 4), None, ctx, got)
    with pytest.raises(PluginError, match="filled twice"):
        _fill_slice(param, "w", 0, "x.0", torch.zeros(2, 4), None, ctx, got)
    with pytest.raises(PluginError, match="outside the 3 slices"):
        _fill_slice(param, "w", 3, "x.3", torch.zeros(2, 4), None, ctx, got)
    with pytest.raises(PluginError, match="a slice of"):
        _fill_slice(param, "w", 1, "x.1", torch.zeros(4, 2), None, ctx, got)
    q = QTensor(Layout(Scheme("ggml.q8_0"), "gguf.native"), (2, 4), {"blocks": torch.zeros(2, 34, dtype=torch.uint8)})
    with pytest.raises(PluginError, match="quantized tensor cannot fill a slice"):
        _fill_slice(param, "w", 2, "x.2", q, None, ctx, got)


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
