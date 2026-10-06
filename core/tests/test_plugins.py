"""Loader and builder error paths, using the installed std plugins."""
import pytest
import torch
from holon.api import Ref, WeightMap, WeightRule
from holon.core.build import BuildContext, map_name
from holon.core.plugins import PluginError, PluginSet, Registry, discover


@pytest.fixture(scope="module")
def plugins():
    return PluginSet(discover())


def test_registry_refuses_duplicates():
    r = Registry("unit")
    r.add("a", 1)
    with pytest.raises(PluginError, match="duplicate"):
        r.add("a", 2)
    with pytest.raises(PluginError, match="unknown"):
        r.get("b")


def test_units_resolve_through_declared_provides(plugins):
    assert plugins.provider("unit", Ref("std.ops.rmsnorm@1")) == "std.ops"
    with pytest.raises(PluginError, match="provides unit.std.ops.rmsnorm@1"):
        plugins.provider("unit", Ref("std.ops.rmsnorm@2"))
    with pytest.raises(PluginError, match="exactly one provider"):
        plugins.provider("unit", Ref("std.ops.nonexistent@1"))


def test_plugins_cannot_register_undeclared_names(plugins):
    from holon.core.plugins import _scoped
    scoped = _scoped(plugins.regs, plugins.installed["std.ffn"].manifest)
    with pytest.raises(PluginError, match="without declaring"):
        scoped.units.add("std.ffn.not_declared", object)


def test_static_buffers_are_stable_and_checked(plugins):
    ctx = BuildContext(plugins, torch.device("cpu"))
    a = ctx.static("x", (4,), torch.int32)
    assert ctx.static("x", (4,), torch.int32).data_ptr() == a.data_ptr()
    with pytest.raises(PluginError, match="two different"):
        ctx.static("x", (8,), torch.int32)


def test_map_name_rules_root_and_skip():
    wm = WeightMap("hf_safetensors", root="model.", skip=("visual.",), rules=(
        WeightRule("embed_tokens.weight", "embed.weight"),
        WeightRule("layers.{i}.mlp.{rest}", "layers.{i}.ffn.{rest}", transform="t"),
    ))
    assert map_name("model.embed_tokens.weight", wm) == ("embed.weight", None)
    assert map_name("model.layers.12.mlp.up_proj.weight", wm) == ("layers.12.ffn.up_proj.weight", "t")
    assert map_name("model.visual.patch.weight", wm) is None
    with pytest.raises(PluginError, match="no weight rule"):
        map_name("model.unknown.weight", wm)
