"""`holon serve`'s values fitted to the machine (docs/dev/v1-auto-config.md): the host RAM it reads, the pin lock, the
shared-GPU test, the GPU slack, the history with a spill and the lock's files, restart admission, the refusal's
hints."""
import asyncio
import os
import sys
import types

import pytest
import torch
from holon.core import cli, platform
from holon.core.executor import Executor
from holon.core.memory import Ledger
from holon.core.supervisor import Supervisor

GiB = 1 << 30


def _tree(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


@pytest.mark.parametrize("case", ["none", "limited", "container root", "v1"])
def test_host_ram_is_meminfo_and_the_cgroup_limits_less_what_they_hold(tmp_path, case):
    """Clean file pages count as free (dirty and writeback pages do not); the lowest limit, an ancestor's included,
    and `memory.high`, bound both figures; a cgroup path missing here (cgroupns=host) reads the mount's root; v1 is
    not read."""
    proc, cg = tmp_path / "proc", tmp_path / "cg"
    group = {"v1": "4:memory:/user\n0::/user.slice/app\n"}.get(case, "0::/user.slice/app\n")
    _tree(proc, {"meminfo": f"MemTotal: {64 << 20} kB\nMemAvailable: {40 << 20} kB\n", "self/cgroup": group})
    stat = f"active_file {2 * GiB}\ninactive_file {GiB}\nfile_dirty {GiB // 2}\nfile_writeback 0\n"
    level = {"memory.max": "max", "memory.high": str(24 * GiB), "memory.current": str(20 * GiB), "memory.stat": stat}
    if case == "limited":                  # the app's own cgroup has no limit; its parent's binds
        _tree(cg, {f"user.slice/{k}": v for k, v in level.items()} | {"user.slice/app/memory.max": "max"})
    if case == "container root":           # /proc/self/cgroup names a host path; the mount is the container's
        _tree(cg, level)
    available, total, why = platform.host_ram(str(proc), str(cg))
    if case in ("none", "v1"):
        assert (available, total) == (40 * GiB, 64 * GiB) and ("cgroup v1" in why) == (case == "v1")
    else:                                  # 24 − 20 + 2 + 1 − 0.5
        assert (available, total) == (int(6.5 * GiB), 24 * GiB) and why.startswith("cgroup /")
    assert platform.host_headroom(total) == 6 * GiB and platform.host_headroom(256 * GiB) == int(12.8 * GiB)


def test_the_pin_lock_never_fails_a_start(tmp_path):
    """Created readable and writable by everyone past the umask; an existing file opened without O_CREAT (another
    user's in sticky /tmp); a lock that cannot be taken is a warning."""
    said, path = [], tmp_path / "pin.lock"
    old = os.umask(0o077)
    try:
        with platform.pin_lock(said.append, str(path)):
            pass
    finally:
        os.umask(old)
    assert path.stat().st_mode & 0o777 == 0o666
    path.chmod(0o444)                                        # read-only: opening it with O_CREAT|O_RDWR would fail
    with platform.pin_lock(said.append, str(path)):
        pass
    with platform.pin_lock(said.append, str(tmp_path / "missing" / "pin.lock")):
        pass
    assert len(said) == 1 and said[0].startswith("WARNING: no host-memory lock")


@pytest.mark.parametrize("case", ["display", "other", "own", "invisible", "no nvml"])
def test_a_gpu_is_shared_by_a_display_or_a_process_outside_holon(monkeypatch, case):
    proc = types.SimpleNamespace
    pids = {"other": [1], "own": [os.getpid(), os.getppid()], "invisible": [2**22 + 7]}.get(case, [])
    nvml = types.SimpleNamespace(
        nvmlInit=lambda: None if case != "no nvml" else 1 / 0, nvmlDeviceGetHandleByUUID=lambda uuid: "gpu",
        nvmlDeviceGetDisplayActive=lambda gpu: case == "display",
        nvmlDeviceGetComputeRunningProcesses=lambda gpu: [proc(pid=p) for p in pids],
        nvmlDeviceGetGraphicsRunningProcesses=lambda gpu: [])
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: proc(uuid="u"))
    want = {"display": "drives a display", "other": "processes [1] use it"}.get(case, "")
    assert platform.gpu_shared(torch.device("cuda")) == want


@pytest.mark.parametrize("case", ["no fill pool", "fill at its preference", "fill short, headless",
                                  "fill short, shared"])
def test_the_slack_is_measured_only_where_a_cache_is_short(monkeypatch, case):
    """D2: 10% of the free memory (today's 0.9) unless a fill pool would stay below its preference there; then the
    allocator's overshoot and a reserve (1 GiB on a shared card, 256 MiB headless), at most those 10%."""
    ledger = Ledger(torch.device("cpu"))
    ledger.request("state.kv", unit_bytes=GiB // 8, min_units=1, tokens_per_unit=16)
    if case != "no fill pool":
        ledger.plugin_pool("cache", unit_bytes=GiB // 4, pref_units=64 if "short" in case else 4, fill=True)
    monkeypatch.setattr(platform, "gpu_shared", lambda device: "drives a display" if "shared" in case else "")
    executor = types.SimpleNamespace(device=torch.device("cpu"))
    slack = Executor.slack(executor, ledger, free=10 * GiB, peak=GiB, out={"overshoot": 64 << 20})
    want = {"fill short, headless": (64 << 20) + (256 << 20)}      # shared: 1 GiB + 64 MiB, capped at the 10%
    assert slack == want.get(case, GiB) and executor.memory_note


def _found(sources=None, options=None, drafts=3):
    plan = types.SimpleNamespace(sources=sources or {}, engine={"max_draft_tokens": drafts,
                                                                "plugin_options": options or {}}, sites={})
    return types.SimpleNamespace(plan=plan), False, "plans/m/1.lock"


class _Owner:
    def __init__(self, spill=None, source=None):
        self._spill, self._source = spill, source

    def spill(self, options, checkpoint):
        return self._spill

    def make_source(self, identity, directory):
        if isinstance(self._source, Exception):
            raise self._source
        return self._source


def test_the_lock_is_fitted_with_its_files_a_spill_and_manual_notes(tmp_path, monkeypatch, capsys):
    """D1, D5: a lock's file found or made by identity; one that cannot be had serves without speculation (both lock
    entries), never a refusal; a spill becomes a memory-only entry before the lock, dropped when a flag sets one of
    its keys; an empty spill is a note for the refusal only; no defaults entry when asked."""
    monkeypatch.setattr(cli, "CACHE", tmp_path)
    found = _found({"mtp.path": "head@0123456789abcdef"}, {"rows.place": "t:host"})
    owners = [_Owner(({"rows.place": "t:ssd"}, "slower"), "/x/head.gguf"), _Owner(({}, "manual"))]
    extra, spill, notes, manual = cli.fit_lock(found, {}, owners, "m")
    assert extra == {"plugin_options": {"mtp.path": "/x/head.gguf"}} and spill == {"rows.place": "t:ssd"}
    assert notes == ["rows.place=t:ssd slower"] and manual == ["manual"]
    assert cli.fit_lock(found, {"plugin_options": {"rows.place": "t:host"}}, owners, "m")[1] == {}      # flags win
    offline = [_Owner(source=ValueError("offline"))]
    assert cli.fit_lock(found, {}, offline, "m")[0] == {"max_draft_tokens": 0}
    assert "head@0123456789abcdef unavailable (offline): speculation off" in capsys.readouterr().err
    assert cli.fit_lock(found, {"max_draft_tokens": 0}, offline, "m")[0] == {"plugin_options": {}}  # never asked
    with pytest.raises(ValueError, match="not a file identity"):
        cli.source_file("../x@0123456789abcdef", owners)
    monkeypatch.setattr(cli.plans, "lock_overrides", lambda lock: dict(lock.plan.engine))
    history = cli.serve_history("m", {"host_memory": 7}, found, 64, lock_extra={"max_draft_tokens": 0},
                                spill=spill, keep_defaults=False)
    assert [(c.plugin_options, c.max_draft_tokens, c.host_memory, c.memory_fraction) for c, _ in history] == [
        ({"rows.place": "t:ssd"}, 0, 7, 0.0), ({"rows.place": "t:host"}, 0, 7, 0.0)]


def test_a_refused_start_says_what_to_do_and_a_restart_is_admitted_against_live_ram(tmp_path, monkeypatch):
    rows = [("org/repo/big@1", (None, False, "1.lock")), ("org/repo/small@2", (None, False, "2.lock")),
            ("org/repo/lent@3", (None, True, "3.lock"))]
    monkeypatch.setattr(cli, "_fits", lambda name, plugins, fp: name == "org/repo/small")
    reason = "OutOfUnits: host memory 10.000 GiB too small: plugin.x needs 9 units (12.500 GiB), 10.000 GiB left"
    hints = cli.refusal_hints(reason, 10 * GiB, "org/repo/big@1", rows, None, None, ["manual mode"])
    assert hints.splitlines()[1:] == [
        "  free at least 2.5 GiB of RAM and start again",
        "  or serve a smaller model this card's lock holds without pinned RAM: org/repo/small",
        f"  or start anyway at the risk of swapping: --host-memory {13 * GiB}", "  manual mode"]
    assert cli.refusal_hints("CUDA out of memory", 0, "m@1", rows, None, None, []) == ""
    for i in (1, 2):
        (tmp_path / f"head-0000{i}-of-00002.gguf").write_bytes(b"x" * i)
    assert cli.checkpoint_bytes(str(tmp_path / "head-00001-of-00002.gguf")) == 3
    monkeypatch.setattr(platform, "host_ram", lambda: (20 * GiB, 32 * GiB, "meminfo"))
    assert cli.admit_restart({"host_bytes": 17 * GiB}) is None              # 20 − 6 / 2 = 17 GiB room
    assert "a restart needs 18.0 GiB" in cli.admit_restart({"host_bytes": 18 * GiB})


def test_a_spill_entry_is_rolled_back_to_only_for_memory_and_a_refused_restart_gives_up(capsys):
    sup = Supervisor([("defaults", None), ("spill", "l"), ("lock", "l")], memory_only={1: "(spilled: x)"})
    assert sup._roll_back() and sup.version == 0                           # crashes skip the spill
    sup.version = 2
    assert sup._roll_back("memory") and sup.version == 1
    assert "after a memory failure (spilled: x)" in capsys.readouterr().err
    sup = Supervisor([("lock", "l")], admit=lambda ready: "host memory: a restart needs 9 GiB, 4 free")
    sup._first_ready = {"host_bytes": 9 * GiB}
    asyncio.run(sup._launch())
    assert "a restart needs 9 GiB" in sup._failure
