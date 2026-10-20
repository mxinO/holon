"""Loader and builder error paths, using the installed std plugins."""
import dataclasses
import importlib.metadata

import pytest
import torch
from holon.api import Fingerprint, Manifest, Ref, WeightMap, WeightRule
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
        '[plugin]\nname="x"\nversion="1"\nentry="p:P"\nslot="unit"\nlicense="MIT"\ncore_abi=">=0.2,<0.3"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = importlib.metadata.EntryPoint("x", "other_abi", "holon.plugins")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [entry])
    with pytest.raises(PluginError, match="does not admit this core's plugin ABI"):
        discover()


def _package(root, name, manifest="", slot="kernel", **files):
    """A plugin package `name` under `root` (on the path): its plugin.toml and modules."""
    (root / name).mkdir()
    (root / name / "__init__.py").write_text("")
    (root / name / "plugin.toml").write_text(f'[plugin]\nname="{name}"\nversion="1"\nentry="plugin:Plugin"\n'
                                             f'slot="{slot}"\nlicense="MIT"\ncore_abi=">=1.0,<2.0"\n{manifest}')
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


RESOLVES = """from holon.api import Resolved

class Plugin:
    def __init__(self):
        self.asked, self.fetched = [], 0

    def register(self, regs):
        pass

    def resolve(self, ref):
        self.asked.append(ref)
        if ref.startswith("x/"):
            raise ValueError("{name} cannot serve " + ref)
        if ref.split("/")[0] in ("{name}", "both"):
            return Resolved("{name}:" + ref, 2**40 if ref.endswith("big") else 10, self.fetch)
        return None

    def fetch(self):
        self.fetched += 1
        return "/fetched/{name}"
"""
GPU = Fingerprint("rtx-4090", 89, 24, "cu130-torch2.11")


@pytest.mark.parametrize("ref, kind, fp, want", [
    ("<path>", None, GPU, "<path>"),                       # an existing path: no plugin is asked
    ("a/m", None, GPU, "/fetched/a"),
    ("a/big", None, GPU, "--no-size-check"),               # 1 TiB on a 24 GiB card: refused before the fetch
    ("a/big", None, None, "/fetched/a"),                   # --no-size-check
    ("a/big", None, Fingerprint("cpu", 0, 0, "cu130-torch2.11"), "/fetched/a"),    # a CPU device: no check
    ("a/big+host", None, GPU, "/fetched/a"),               # --host-memory > 0: the build's per-pool report judges
    ("x/m", None, GPU, "no weight source resolves it\n  a: a cannot serve x/m\n  b: b cannot serve x/m"),
    ("both/m", None, GPU, "several weight sources resolve it"),
    ("both/m", "b", GPU, "/fetched/b"),                    # --source b asks only b
])
def test_model_references_resolve_through_one_weight_source(tmp_path, monkeypatch, ref, kind, fp, want):
    """`holon serve <ref>` (release design §14.3): a path is itself; else the weight sources' `resolve` hooks claim it
    (a ValueError is a hint shown when none does), exactly one must, and its fetch runs once, unless the claim is larger
    than the GPU and the deployment gives plugins no host memory (docs/v1-offload.md, the CLI row)."""
    from holon.core import cli

    monkeypatch.syspath_prepend(str(tmp_path))
    plugins = PluginSet({name: _package(tmp_path, name, f'[resources]\nprovides=["weight_source.{name}@1"]\n',
                                        "weight_source", plugin=RESOLVES.replace("{name}", name)) for name in "ab"})
    ref, host = ref.replace("<path>", str(tmp_path)).removesuffix("+host"), int(ref.endswith("+host")) << 30
    if want.startswith("/") or want == "<path>":
        assert cli.local_model(ref, plugins, kind, [], fp, host) == want.replace("<path>", str(tmp_path))
    else:
        with pytest.raises((SystemExit, PluginError), match=want):
            cli.local_model(ref, plugins, kind, [], fp, host)
    asked = {name: plugins.load(name).asked for name in plugins.loaded}
    assert asked == ({} if want == "<path>" else {"b": [ref]} if kind else {"a": [ref], "b": [ref]})
    assert sum(plugins.load(name).fetched for name in asked) == want.startswith("/")


def test_serve_help_and_option_checks_follow_the_declarations(tmp_path, monkeypatch, capsys):
    """Every engine flag has its field's help line; the options installed plugins declare, and the plugins `--slots`
    selects among, are listed; a flag outside a declaring plugin's `choices` (any of them) fails before the model is
    fetched; an undeclared option is warned about and kept, a lock's value outside the choices drops that lock from
    the history; options only frontends declare are not planned."""
    from holon.core import cli, plans
    from holon.core.engine import EngineConfig

    options = {"toy_eng": ('"k.fmt" = { help = "a format", choices = ["x", "y"] }\n'
                           '"k.both" = { help = "both read it", choices = ["x", "y"] }', "unit"),
               "toy_eng2": ('"k.both" = { help = "both", choices = ["x"] }', "unit"),
               "toy_front": ('"f.name" = { help = "the served name" }', "frontend"),
               "toy_sampler": ('', "sampler")}
    installed = {name: _package(tmp_path, name, f"[options]\n{body}\n", slot) for name, (body, slot) in options.items()}
    monkeypatch.setattr(cli, "discover", lambda: installed)
    with pytest.raises(SystemExit):
        cli.main(["serve", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    for f in dataclasses.fields(EngineConfig):
        if f.name not in ("model", "clamp_context"):
            assert f"--{f.name.replace('_', '-')}" in text and " ".join(f.metadata["help"].split()) in text
    assert "k.fmt a format (one of: x, y) [toy_eng]" in text and "f.name the served name [toy_front]" in text
    assert text.endswith("selectable plugins (--slots SLOT=PLUGIN): sampler toy_sampler")

    monkeypatch.setattr(cli, "local_model", lambda *a: pytest.fail("fetched before the flags were checked"))
    with pytest.raises(SystemExit, match=r"k.both=y: not one of x \(toy_eng2\)"):
        cli.main(["serve", "org/repo:TAG", "--device", "cpu", "--plugin-options", "k.both=y"])

    declared = cli.declared_options(installed)
    history = [(EngineConfig("m", plugin_options={"zz": "1", "k.fmt": "x"}), None),
               (EngineConfig("m", plugin_options={"zz": "1", "k.fmt": "z"}), "1.lock")]
    cli.check_flags({"plugin_options": {"zz": "1", "k.fmt": ""}}, declared)      # undeclared, or deleting: no error
    assert cli.checked_options(history, declared) == history[:1]
    err = capsys.readouterr().err
    assert "WARNING: no installed plugin declares the option zz" in err
    assert "1.lock: k.fmt=z: not one of x, y (toy_eng); serving defaults" in err
    assert plans.unplanned_options(installed) == {"f.name"}
