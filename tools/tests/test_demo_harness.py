"""The agent demo's harness (`bench/demo/run.py`, release design D7) and its proxy (`bench/capture_proxy.py`): what a
run is scored and recorded by, on the CPU, without a server or Claude Code."""
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = _module("demo_run", "bench/demo/run.py")
proxy = _module("capture_proxy", "bench/capture_proxy.py")
FIX = ("{cents % 100}\"", "{cents % 100:02d}\"")


@pytest.fixture(scope="module")
def project():
    return demo.load_project()


def _work(tmp_path, project, fixed_money=False):
    work = tmp_path / "work"
    demo.write_tree(project, work)
    if fixed_money:
        money = work / "shop" / "money.py"
        money.write_text(money.read_text().replace(*FIX))
    return work


def test_scoring_reads_only_the_agents_regular_files_and_never_raises(tmp_path, project, monkeypatch):
    tests = {k: v for k, v in project.items() if k.startswith("tests/")}
    assert {k: demo.score(_work(tmp_path / "a", project), tests)[k] for k in demo.SEEDED} == demo.SEEDED
    # a module linked to a fixed copy outside the package is not followed: scored as missing
    outside = tmp_path / "outside.py"
    outside.write_text(project["shop/money.py"].decode().replace(*FIX))
    work = _work(tmp_path / "b", project)
    (work / "shop" / "money.py").unlink()
    (work / "shop" / "money.py").symlink_to(outside)
    (work / "shop" / "gone.py").symlink_to(tmp_path / "missing.py")                    # dangling: no crash
    linked = demo.score(work, tests)
    assert not linked["success"] and linked["symlinks_skipped"] == 2 and linked["passed"] == 0
    # whatever fails while scoring is a recorded miss
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    failed = demo.score(_work(tmp_path / "c", project, fixed_money=True), tests)
    assert not failed["success"] and failed["error"] == "OSError"


def test_the_project_and_the_scoring_environment_are_checked(tmp_path, project, monkeypatch):
    assert demo.digest(project) == demo.PROJECT_DIGEST
    changed = tmp_path / "project"
    demo.write_tree(project, changed)
    (changed / ".pytest_cache").mkdir()                     # a local pytest run's cache is not the project
    (changed / ".pytest_cache" / "README.md").write_text("x")
    assert demo.digest(demo.load_project(changed)) == demo.PROJECT_DIGEST
    (changed / "tests" / "test_money.py").write_text("")
    with pytest.raises(ValueError, match="pre-registered"):
        demo.load_project(changed)
    site = tmp_path / "site"
    (site / "_pytest").mkdir(parents=True)
    (site / "_pytest" / "main.py").write_text("x = 1\n")
    (site / "other").mkdir()
    before = demo.site_digest([site])
    (site / "other" / "unrelated.py").write_text("")
    assert demo.site_digest([site]) == before
    for tamper in ("hook.pth", "sitecustomize.py", "_pytest/main.py"):
        (site / tamper).write_text("import os\n")
        assert demo.site_digest([site]) != before
        before = demo.site_digest([site])
    guarded = demo.site_digest()                            # a run's validity: the same digest, and it still runs
    assert demo.invalid_after(guarded) is None
    assert "changed" in demo.invalid_after("sha256:other")
    monkeypatch.setattr(demo, "scoring_modules", lambda: (_ for _ in ()).throw(subprocess.TimeoutExpired("p", 1)))
    assert "TimeoutExpired" in demo.invalid_after(guarded)
    monkeypatch.undo()
    loaded = demo.scoring_modules()                         # everything a scoring run imports, the stdlib included
    for module in ("iniconfig/__init__.py", "pluggy/__init__.py", "_pytest/junitxml.py", "xml/etree/ElementTree.py"):
        assert any(f.endswith(module) for f in loaded), module


def test_claude_code_is_pinned_by_path_version_and_hash(tmp_path, monkeypatch):
    binary = tmp_path / "versions" / demo.CLAUDE_CODE
    binary.parent.mkdir()
    binary.write_text(f"#!/bin/sh\necho '{demo.CLAUDE_CODE} (Claude Code)'\n")
    binary.chmod(0o755)
    (tmp_path / "claude").symlink_to(binary)
    monkeypatch.setenv("PATH", str(tmp_path))
    path, sha = demo.resolve_claude("claude")
    assert path == binary and sha == "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    demo.check_claude(path, sha)
    binary.write_text("#!/bin/sh\necho '9.9.9 (Claude Code)'\n")
    with pytest.raises(SystemExit, match="changed"):
        demo.check_claude(path, sha)


def test_the_record_keeps_counts_and_shapes_never_text_or_host_paths(tmp_path, project):
    assert demo.scrub("plan not computed: OSError: [Errno 2] /data/u/x/plans/a/b/1.lock and /opt/y/z.py") == \
        "plan not computed: OSError: [Errno 2] plans/a/b/1.lock and <path>"
    assert demo.lock_class("plans/unsloth/R/F@b/rtx-5080/cu130-torch2.11/5.lock") == "rtx-5080"
    assert demo.lock_class(None) is None and demo.lock_class("<path>") is None
    events = [{"type": "assistant", "message": {"id": i, "content": []}} for i in ("m1", "m1", "m2")]
    assert demo.turns(events) == 2 and demo.turns([*events, {"type": "result", "num_turns": 7}]) == 7
    rows = [{"method": "POST", "path": "/v1/messages?beta=true", "status": 200, "ttft_ms": 10.0, "total_ms": 110.0,
             "usage": {"input_tokens": 900, "output_tokens": 101}, "stop_reason": "end_turn"},
            {"method": "POST", "path": "/v1/messages", "status": 400, "usage": {}, "length_refused": True},
            {"method": "POST", "path": "/v1/messages/count_tokens", "status": 200, "usage": {}}]
    served = demo.requests_summary(rows)
    assert served["statuses"] == {"200": 2, "400": 1} and served["length_refusals"] == 1
    assert served["peak_context"] == 1001 and served["requests"][0]["tpot_ms"] == 1.0
    assert [r["input_tokens"] for r in served["requests"]] == [900, None]
    work = _work(tmp_path, project, fixed_money=True)
    diff = demo.shop_diff(project, work)
    assert "+    return f\"{sign}{cents // 100}.{cents % 100:02d}\"" in diff and demo.private_ok(diff)
    assert not demo.private_ok(f"see {pathlib.Path.home()}/notes")
    outside = tmp_path / "elsewhere"                         # a linked shop/ is not read
    outside.mkdir()
    (outside / "secret.txt").write_text("not the agent's")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "shop").symlink_to(outside)
    assert demo.shop_diff(project, linked) == "<no shop/ package>"


def test_the_proxy_records_stop_reasons_and_length_refusals_without_text():
    sse = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12}}}\n\n'
           b'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":7}}\n\n')
    assert proxy.facts(sse) == {"usage": {"input_tokens": 12, "output_tokens": 7}, "stop_reason": "max_tokens"}
    refused = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                     "message": "prompt of 40000 tokens leaves no room under "
                                                                "max_model_len 32768"}}).encode()
    assert proxy.facts(refused) == {"usage": {}, "error_type": "invalid_request_error", "length_refused": True}
    assert proxy.facts(b'{"message": "oops", "usage": "x"}') == {"usage": {}}


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_each_run_gets_its_own_venv_with_pytest(tmp_path):
    bin_dir = demo.make_venv(tmp_path / "venv")
    out = subprocess.run([str(bin_dir / "python"), "-m", "pytest", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "pytest" in out.stdout + out.stderr
    assert (bin_dir / "pytest").exists()


def test_a_series_runs_only_on_a_clean_harness_and_its_reference_environment(tmp_path, monkeypatch, project):
    repo = tmp_path / "repo"
    for path in ("bench/demo/run.py", "scripts/ci/check_private.py", "docs/x.md"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "-C", str(repo), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(repo), "commit", "-qm", "x"], check=True)
    assert demo.harness_dirty(repo) == []
    (repo / "docs" / "x.md").write_text("y\n")                       # outside the harness: fine
    (repo / "bench" / "new.py").write_text("")
    (repo / "scripts" / "ci" / "check_private.py").write_text("y\n")
    assert demo.harness_dirty(repo) == ["bench/new.py", "scripts/ci/check_private.py"]
    # one reference for the whole series: a run starts only in it
    monkeypatch.setattr(demo, "site_digest", lambda dirs=None: "sha256:now")
    demo.check_environment("sha256:now")
    with pytest.raises(SystemExit, match="restore"):
        demo.check_environment("sha256:reference")
    # an invalid run is a miss, published as such; a scored run's success is its scoring's
    assert demo.outcome("changed", None) == {"success": False, "invalid": "changed", "scored": None}
    assert demo.outcome(None, {"success": True})["success"]
    # the shop/ diff never loses the run
    monkeypatch.setattr(demo, "tree", lambda root: (_ for _ in ()).throw(OSError("boom")))
    work = tmp_path / "work"
    (work / "shop").mkdir(parents=True)
    assert demo.shop_diff(project, work) == "<diff failed: OSError>"


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_every_runs_venv_has_the_first_runs_pytest(tmp_path):
    first = demo.venv_digest(demo.make_venv(tmp_path / "a"))
    second = demo.make_venv(tmp_path / "b")
    assert demo.venv_digest(second) == first
    (next(second.parent.glob("lib/python*/site-packages")) / "_pytest" / "main.py").write_text("")
    assert demo.venv_digest(second) != first
