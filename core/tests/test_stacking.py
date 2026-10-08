"""The stacking weight rule (docs/m02-moe.md D7): per-expert checkpoint tensors fill one [E, …] parameter."""
import pytest
import torch
from holon.api import Layout, QTensor, Scheme, WeightMap, WeightRule
from holon.core.build import BuildContext, _fill_slice, map_name
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
