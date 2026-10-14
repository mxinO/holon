"""Loader and builder error paths, using the installed std plugins."""
import importlib.metadata

import pytest
import torch
from holon.api import Manifest, Ref, WeightMap, WeightRule
from holon.core.build import BuildContext, map_name
from holon.core.plugins import Installed, PluginError, PluginSet, Registry, discover


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
    assert map_name("model.embed_tokens.weight", wm) == ("embed.weight", None, None)
    assert map_name("model.layers.12.mlp.up_proj.weight", wm) == ("layers.12.ffn.up_proj.weight", "t", None)
    assert map_name("model.visual.patch.weight", wm) is None
    with pytest.raises(PluginError, match="no weight rule"):
        map_name("model.unknown.weight", wm)


def test_a_plugin_of_another_abi_cannot_load(tmp_path, monkeypatch):
    """Its manifest fails at discovery as a PluginError (a deployment that cannot start: no retry helps)."""
    (tmp_path / "other_abi").mkdir()
    (tmp_path / "other_abi" / "__init__.py").write_text("")
    (tmp_path / "other_abi" / "plugin.toml").write_text(
        '[plugin]\nname="x"\nversion="1"\nentry="p:P"\nslot="unit"\nlicense="MIT"\ncore_abi=">=1.0,<2"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = importlib.metadata.EntryPoint("x", "other_abi", "holon.plugins")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [entry])
    with pytest.raises(PluginError, match="does not admit this core's plugin ABI"):
        discover()


def _package(root, name, manifest="", **files):
    """A plugin package `name` under `root` (on the path): its plugin.toml and modules."""
    (root / name).mkdir()
    (root / name / "__init__.py").write_text("")
    (root / name / "plugin.toml").write_text(f'[plugin]\nname="{name}"\nversion="1"\nentry="plugin:Plugin"\n'
                                             f'slot="kernel"\nlicense="MIT"\ncore_abi=">=0.1,<0.2"\n{manifest}')
    for module, text in files.items():
        (root / name / f"{module}.py").write_text(text)
    return Installed(Manifest.load(root / name / "plugin.toml"), name, root / name)


BUILD = """import pathlib, sys
sm = sys.argv[sys.argv.index("--sm") + 1]
with open(pathlib.Path(__file__).parent / "built", "a") as f:
    f.write(sm + "\\n")
sys.exit(1 if sm == "{fail}" else 0)
"""


def test_prebuild_runs_each_build_module_once_per_sm(tmp_path, monkeypatch):
    """`holon prebuild`: `python -m <package>.<[build] module> --sm N` per (plugin, SM); a failure is reported and
    the command exits non-zero after the rest have run."""
    from holon.core import cli

    installed = {"toy_ok": _package(tmp_path, "toy_ok", '[build]\nmodule="build"\n', build=BUILD.format(fail="")),
                 "toy_bad": _package(tmp_path, "toy_bad", '[build]\nmodule="build"\n', build=BUILD.format(fail="89")),
                 "toy_none": _package(tmp_path, "toy_none")}
    monkeypatch.setattr(cli, "discover", lambda: installed)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(SystemExit, match=r"failed: toy_bad \(sm 89\)$"):
        cli.prebuild([86, 89])
    assert [(tmp_path / p / "built").read_text().split() for p in ("toy_ok", "toy_bad")] == [["86", "89"]] * 2
    assert not (tmp_path / "toy_none" / "built").exists()


def test_failed_kinds_tell_resolver_errors_from_a_plugin_failing_to_load(tmp_path, monkeypatch):
    """A start failing in the resolver fails every retry ("plugin": roll back at once); a plugin's own import error
    may be transient ("error": retried)."""
    from holon.core.ipc import _DECODE, _failed

    monkeypatch.syspath_prepend(str(tmp_path))
    plugins = PluginSet({"toy_broken": _package(tmp_path, "toy_broken", plugin="import not_a_module_anywhere\n")})
    kinds = []
    for fails in (lambda: plugins.load("toy_broken"), lambda: plugins.load("not.installed")):
        with pytest.raises(PluginError) as e:
            fails()
        kinds.append(_DECODE[b"failed"].decode(_failed(e.value)[1])[0])
    assert kinds == ["error", "plugin"]
