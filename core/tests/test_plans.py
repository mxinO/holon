"""Plans and the lock history on the core's side: unit source hashes, the lock `holon serve` starts from and how flags
layer over it, the engine's context clamp and soft kernel pins (release design D2). Toy locks are written with the
tools' writer."""
import dataclasses
import shutil
import types

import pytest
import torch
from holon.api import Binding, Fingerprint, Lock, Plan, SamplingParams, UnitPin
from holon.core import cli, plans
from holon.core.engine import Engine, EngineConfig
from holon.core.memory import OutOfUnits
from holon.core.plugins import PluginError
from holon_tools import locks


def test_src_hash_is_content_only(tmp_path):
    unit = tmp_path / "unit"
    (unit / "sub").mkdir(parents=True)
    (unit / "a.py").write_text("x = 1\n")
    (unit / "sub" / "b.py").write_text("y = 2\n")
    first = plans.src_hash(unit)
    assert first.startswith("sha256:")
    # caches, build products and hidden files are not source
    for junk in ("__pycache__/a.cpython-312.pyc", "build/lib/a.py", "x.egg-info/PKG-INFO", ".hidden/f", "k.so",
                 "sub/c.pyc"):
        (unit / junk).parent.mkdir(parents=True, exist_ok=True)
        (unit / junk).write_text("junk")
    assert plans.src_hash(unit) == first
    copy = tmp_path / "installed" / "unit"                 # an installed copy elsewhere hashes the same
    shutil.copytree(unit, copy)
    assert plans.src_hash(copy) == first
    (unit / "sub" / "b.py").write_text("y = 3\n")
    assert plans.src_hash(unit) != first
    (unit / "sub" / "b.py").write_text("y = 2\n")
    assert plans.src_hash(unit) == first
    (unit / "sub" / "b.py").rename(unit / "b.py")           # paths count, not only contents
    assert plans.src_hash(unit) != first


# -- `holon serve` from the lock (release design D2) -----------------------------------------------------------------
FP = Fingerprint(gpu="rtx-4090", sm=89, vram_gib=24, toolchain="cu130-torch2.11")
INSTALLED = {"holon": "core", "std.ops": "s1"}
BUILD = "120307ba529eb243"


def _lock(root, model_id, cls="rtx-4090/cu130-torch2.11", src="s1", sm=89):
    units = {"holon": UnitPin("0.0.1", "core"), "std.ops": UnitPin("1", src)}
    engine = {"dtype": "bfloat16", "max_model_len": 4096, "max_seqs": 8, "plugin_options": {"a": "1", "b": "2"}}
    plan = Plan(model_id, cls, "sha256:r", "c1", "none", units, {"s/rmsnorm@1": Binding("kernels.fast")}, engine)
    return locks.append(root, Lock(plan, sm=sm))


@pytest.mark.parametrize("case", ["lock", "flags", "borrowed", "outdated", "none", "two ids", "corrupt"])
def test_serve_starts_from_the_lock_of_its_build(tmp_path, capsys, monkeypatch, case):
    """Found by the build alone (the file downloaded under another name); flags over the lock, dict flags per key and
    `key=` deleting one; the lock's pins as preferences; defaults first in the history, at SERVE_MAX_SEQS and the
    model's `auto` length; a history that cannot be read is warned about and defaults serve. `holon models` shows the
    same verdict per model, and a repository reference names the one model under it locked for this card (an existing
    local path stays a path)."""
    written = None
    if case != "none":
        written = _lock(tmp_path, f"org/repo/Qwen3-8B-Q4_K_M@{BUILD}", **{
            "borrowed": {"cls": "rtx-5080/cu130-torch2.11", "sm": 120}, "outdated": {"src": "s0"}}.get(case, {}))
    if case == "two ids":
        _lock(tmp_path, f"Qwen3-8B-Q4_K_M@{BUILD}")
    if case == "flags":                                    # a second file of the repository, also locked here
        _lock(tmp_path, "org/repo/Qwen3-8B-UD-IQ3_XXS@0123456789abcdef")
    if case == "corrupt":
        written.write_text(written.read_text().replace("4096", "4097"))
    listed = cli.locked_models(tmp_path, FP, INSTALLED)
    rows = dict(row.split(maxsplit=1) for row in cli.models_table(listed))
    want = {"lock": "this card's lock 1; context 4096, 8 sequences", "flags": "this card's lock 1",
            "borrowed": "rtx-5080's lock 1, unverified", "outdated": "outdated", "corrupt": "unreadable",
            "two ids": "unreadable: "}.get(case)                  # as `holon serve` reads it: two ids, one build
    assert rows["org/repo/Qwen3-8B-Q4_K_M"].startswith(want) if want else rows == {}
    assert rows.get("Qwen3-8B-Q4_K_M", "not servable by name").endswith("not servable by name")
    if case == "flags":
        with pytest.raises(SystemExit, match="several models have a lock for this GPU, name one: org/repo/Qwen3-8B-Q"):
            cli.repo_ref("org/repo", listed)
    else:
        assert cli.repo_ref("org/repo", listed) == ("org/repo/Qwen3-8B-Q4_K_M" if case == "lock" else "org/repo")
    assert cli.repo_ref("org/rep", listed) == "org/rep"                         # whole path segments only
    assert cli.repo_ref("org", listed) == "org"                                 # a repository, not an organisation
    monkeypatch.chdir(tmp_path)
    if case == "lock":                                     # `org/repo` is a directory here: a path, never expanded
        assert cli.local_model("org/repo", None, None, listed, FP) == "org/repo"
    source = types.SimpleNamespace(config=lambda: {"n": 1000}, kind="toy", identity=lambda: f"local@{BUILD}")
    found = cli.find_lock(tmp_path, source, FP, INSTALLED)
    if case in ("two ids", "corrupt"):
        err = capsys.readouterr().err
        assert found is None and "WARNING: unreadable lock history" in err
        assert (str(written) if case == "corrupt" else "several model ids") in err
        return
    flags = {"plugin_options": {"b": "", "c": "3"}, "kernels": {"rope": "x"}, "max_seqs": 4} if case == "flags" else {}
    model = types.SimpleNamespace(trained_context=lambda cfg, kind: cfg["n"]) if case == "none" else object()
    history = cli.serve_history("m", flags, found, cli.auto_context(model, source))
    defaults = history[0][0]
    assert history[0][1] is None and defaults.clamp_context and defaults.kernel_preferences == {}
    assert defaults.max_model_len == (1000 if case == "none" else cli.AUTO_CONTEXT)
    assert defaults.max_seqs == (4 if case == "flags" else cli.SERVE_MAX_SEQS)
    if case in ("outdated", "none"):                            # an outdated lock is a missing one (design §4.4)
        assert found is None and len(history) == 1
        return
    config, path = history[1]
    assert path == str(written) and found[1] == (case == "borrowed")    # borrowed: unverified, the same otherwise
    assert (config.max_model_len, config.dtype, config.kernel_preferences) == (4096, torch.bfloat16,
                                                                            {"rmsnorm": "kernels.fast"})
    assert (config.max_seqs, config.kernels, config.plugin_options) == (
        (4, {"rope": "x"}, {"a": "1", "c": "3"}) if case == "flags" else (8, {}, {"a": "1", "b": "2"}))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                 eos_token_id=None)).save_pretrained(path)
    return path


def test_the_clamp_serves_what_the_pools_hold_and_lock_pins_are_soft(tiny):
    """KV for 32 tokens (2 layers x K, V x 2 heads x 16 dims x 4 B per token: 512 B) under a 64-token request: a pool
    sized for one full request does not fit; `clamp_context` serves 32 and plans it. A lock's pin that cannot run here
    is a preference (default priority); the same pin given as `kernels` fails."""
    cfg = EngineConfig(str(tiny), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2, max_step_tokens=8,
                       max_model_len=64, state_memory=(32 + 4) * 512, kernel_preferences={"rmsnorm": "kernels.fast"})
    with pytest.raises(OutOfUnits):
        Engine(cfg)
    engine = Engine(dataclasses.replace(cfg, clamp_context=True))
    assert (engine.cfg.max_model_len, engine.sched.max_model_len, cfg.max_model_len) == (32, 32, 64)
    plan = plans.plan_of(engine)          # a lock of the requested length: the clamp is the served plan's only diff
    lock = locks.append(tiny / "plans", Lock(dataclasses.replace(plan, engine=plan.engine | {"max_model_len": 64})))
    assert plans.served(engine, str(lock))["diff"] == ["engine:max_model_len: 64 -> 32"]
    kv = next(b for b in engine.state.bound if b.spec.name == "kv")       # past the clamp: only the null unit, where
    assert (engine.state.tables[kv.spec.group].host[:, 32 // 4:] == kv.pool.null).all()   # a verify row's overflow goes
    with pytest.raises(ValueError, match="no room under max_model_len 32"):
        engine.add(list(range(32)))
    out = engine.generate([list(range(30))], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    assert len(out[0]) == 2                                        # stopped at the clamp ("length")
    with pytest.raises(PluginError, match="pinned provider 'kernels.fast'"):
        Engine(dataclasses.replace(cfg, clamp_context=True, kernels={"rmsnorm": "kernels.fast"}))


UV_LOCK = """
[[package]]
name = "holonlm"
version = "0.0.1"
source = { editable = "core" }
dependencies = [{ name = "torch", extra = ["cuda"] }]

[[package]]
name = "torch"
version = "2.11.0"
source = { registry = "https://example/cu130" }

[package.optional-dependencies]
cuda = [{ name = "nvrtc" }]

[[package]]
name = "nvrtc"
version = "13.0.1"
source = { registry = "https://example/simple" }

[[package]]
name = "pytest"
version = "9.0"
source = { registry = "https://example/simple" }
"""


def test_uv_lock_hashes_the_runtime_closure_extras_included(tmp_path):
    def hashed(text):
        (tmp_path / "uv.lock").write_text(text)
        return plans.uv_lock(tmp_path / "uv.lock")

    base = hashed(UV_LOCK)
    assert hashed(UV_LOCK.replace('"9.0"', '"9.1"')) == base             # a dev-only pin moves no plan
    assert hashed(UV_LOCK.replace('"13.0.1"', '"13.0.2"')) != base       # reached through torch's extra
    assert plans.uv_lock(tmp_path / "absent.lock") == "none"
