"""`holon serve`'s values fitted to the machine (docs/dev/v1-auto-config.md): the host RAM it reads, the pin lock, the
shared-GPU test, the GPU slack, the history with a spill and the lock's files, restart admission, the refusal's
hints."""
import asyncio
import contextlib
import os
import sys
import types

import pytest
import torch
from holon.api import OutOfUnits
from holon.core import cli, platform
from holon.core.executor import Executor
from holon.core.ipc import StartFailed
from holon.core.memory import Ledger
from holon.core.supervisor import Supervisor

GiB = 1 << 30


def _tree(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


@pytest.mark.parametrize("case", ["none", "limited", "container root", "v1", "unreadable"])
def test_host_ram_is_meminfo_and_the_cgroup_limits_less_what_they_hold(tmp_path, case):
    """Clean file pages count as free (dirty and writeback pages do not); the lowest limit, an ancestor's included,
    and `memory.high`, bound both figures; a cgroup path missing here (cgroupns=host) reads the mount's root; v1 is
    not read; one that cannot be read falls back to meminfo, saying so."""
    proc, cg = tmp_path / "proc", tmp_path / "cg"
    group = {"v1": "4:memory:/user\n0::/user.slice/app\n"}.get(case, "0::/user.slice/app\n")
    _tree(proc, {"meminfo": f"MemTotal: {64 << 20} kB\nMemAvailable: {40 << 20} kB\n", "self/cgroup": group})
    stat = f"active_file {2 * GiB}\ninactive_file {GiB}\nfile_dirty {GiB // 2}\nfile_writeback 0\n"
    level = {"memory.max": "max", "memory.high": str(24 * GiB), "memory.current": str(20 * GiB), "memory.stat": stat}
    if case == "limited":                  # the app's own cgroup has no limit; its parent's binds
        _tree(cg, {f"user.slice/{k}": v for k, v in level.items()} | {"user.slice/app/memory.max": "max"})
    if case in ("container root", "unreadable"):   # /proc/self/cgroup names a host path; the mount is the container's
        _tree(cg, level | ({"memory.stat": "not a stat\n"} if case == "unreadable" else {}))
    available, total, why = platform.host_ram(str(proc), str(cg))
    if case in ("none", "v1", "unreadable"):
        assert (available, total) == (40 * GiB, 64 * GiB) and ("cgroup v1" in why) == (case == "v1")
        assert ("cgroup not read" in why) == (case == "unreadable")
    else:                                  # 24 − 20 + 2 + 1 − 0.5
        assert (available, total) == (int(6.5 * GiB), 24 * GiB) and why.startswith("cgroup /")
    assert platform.host_headroom(total) == 6 * GiB and platform.host_headroom(256 * GiB) == int(12.8 * GiB)


def test_the_pin_lock_never_fails_a_start(tmp_path, monkeypatch, capsys):
    """Per user, in the cache: a second holder waits, then goes on with a warning; no cache to write is a warning."""
    monkeypatch.setattr(cli, "CACHE", tmp_path)
    with cli.pin_lock(), cli.pin_lock(wait=1):          # another file description: the second waits its 1 s
        pass
    monkeypatch.setattr(cli, "CACHE", tmp_path / "pin.lock" / "x")    # under a file: cannot be made
    with cli.pin_lock():
        pass
    err = capsys.readouterr().err
    assert "waiting for another `holon serve`" in err and err.count("WARNING: no host-memory lock") == 2


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
    want = {"display": "drives a display", "other": "processes [1] use it", "no nvml": None}.get(case, "")
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

    def source_hint(self, identity):
        return self._source


def test_the_lock_is_fitted_with_its_files_a_spill_and_manual_notes(tmp_path, monkeypatch, capsys):
    """D1, D5: a lock's file found by identity in the cache; one missing serves without speculation (both lock
    entries) with the plugin's hint, never a refusal, and an unreadable cache too; a spill becomes a memory-only entry
    before the lock, dropped when a flag sets one of its keys; an empty spill is a note for the refusal only; no
    defaults entry when asked."""
    monkeypatch.setattr(cli, "CACHE", tmp_path)
    found = _found({"mtp.path": "head@0123456789abcdef"}, {"rows.place": "t:host"})
    owners = [_Owner(({"rows.place": "t:ssd"}, "slower"), "make it so"), _Owner(({}, "manual"))]
    (pack := tmp_path / "sources" / "head@0123456789abcdef" / "head.gguf").parent.mkdir(parents=True)
    pack.write_bytes(b"head")
    extra, spill, notes, manual = cli.fit_lock(found, {}, owners, "m")
    assert extra == {"plugin_options": {"mtp.path": str(pack)}} and spill == {"rows.place": "t:ssd"}
    assert notes == ["rows.place=t:ssd slower"] and manual == ["manual"]
    assert cli.fit_lock(found, {"plugin_options": {"rows.place": "t:host"}}, owners, "m")[1] == {}      # flags win
    pack.unlink()
    assert cli.fit_lock(found, {}, owners, "m")[0] == {"max_draft_tokens": 0}
    assert "head@0123456789abcdef unavailable (not in" in (err := capsys.readouterr().err) and "; make it so" in err
    assert cli.fit_lock(found, {"max_draft_tokens": 0}, owners, "m")[0] == {"plugin_options": {}}  # never looked for
    monkeypatch.setattr(cli, "CACHE", tmp_path / "unreadable")
    (tmp_path / "unreadable").write_text("")                     # a cache that is not a directory: never a crash
    assert cli.fit_lock(found, {}, owners, "m")[0] == {"max_draft_tokens": 0}
    with pytest.raises(ValueError, match="not a file identity"):
        cli.source_file("../x@0123456789abcdef", owners)
    monkeypatch.setattr(cli.plans, "lock_overrides", lambda lock: dict(lock.plan.engine))
    history = cli.serve_history("m", {"host_memory": 7}, found, 64, lock_extra={"max_draft_tokens": 0},
                                spill=spill, keep_defaults=False)
    assert [(c.plugin_options, c.max_draft_tokens, c.host_memory, c.memory_fraction) for c, _ in history] == [
        ({"rows.place": "t:ssd"}, 0, 7, 0.0), ({"rows.place": "t:host"}, 0, 7, 0.0)]


def test_a_refusal_says_what_to_do_and_a_restart_is_admitted_against_its_own_entry(tmp_path, monkeypatch):
    """The ledger's refusal names the shortfall and the override; a restart, under the pin lock, is refused (memory)
    when the RAM no longer holds what its entry pinned, and an entry never started gets its derived cap lowered to what
    is free above half the headroom, not checked against another entry's bytes (A C-M3)."""
    ledger = Ledger(torch.device("cpu"), host_memory=GiB)
    with pytest.raises(OutOfUnits, match=rf"Short by 1.0 GiB at least: .* --host-memory {2 * GiB} or more"):
        ledger.plugin_pool("big", unit_bytes=GiB, min_units=2, pref_units=2, device="host")
    for i in (1, 2):
        (tmp_path / f"head-0000{i}-of-00002.gguf").write_bytes(b"x" * i)
    assert cli.checkpoint_bytes(str(tmp_path / "head-00001-of-00002.gguf")) == 3
    monkeypatch.setattr(cli, "CACHE", tmp_path)
    monkeypatch.setattr(platform, "host_ram", lambda: (20 * GiB, 32 * GiB, "meminfo"))     # 17 GiB above 6 / 2
    config = cli.EngineConfig(model="m", host_memory=30 * GiB)
    with cli.admit_restart(config, {"host_bytes": 17 * GiB}, derived=True) as got:
        assert got is config
    with pytest.raises(StartFailed, match="a restart needs 18.0 GiB"), cli.admit_restart(
            config, {"host_bytes": 18 * GiB}, derived=True):
        pass
    with cli.admit_restart(config, None, derived=True) as got:         # never started: its cap meets the RAM now
        assert got.host_memory == 17 * GiB
    with cli.admit_restart(config, {"host_bytes": 18 * GiB}, derived=False) as got:    # --host-memory: the user's
        assert got is config


def test_a_pool_viewed_whole_keeps_no_unit_bookkeeping():
    """A host table of many rows (Qwen3.8-Flash-Next's PLE: 320M of 90 B) costs its bytes only: the free list and the
    refcounts, ≈ 40 B a unit outside the budget, are built on first use, and again after a resize."""
    pool = Ledger(torch.device("cpu"), host_memory=GiB).plugin_pool("rows", unit_bytes=1, min_units=1 << 20,
                                                                    pref_units=1 << 20, device="host")
    assert "_free" not in vars(pool) and "_refs" not in vars(pool)
    assert list(pool.alloc(2)) == [0, 1] and pool.num_free == (1 << 20) - 2 and pool.refcount([1]) == [1]
    pool._set(4)
    assert pool.num_free == 4 and "_refs" not in vars(pool)


def test_a_spill_entry_is_rolled_back_to_only_for_memory_and_restarts_are_admitted_per_entry(capsys):
    sup = Supervisor([("defaults", None), ("spill", "l"), ("lock", "l")], memory_only={1: "(spilled: x)"})
    assert sup._roll_back() and sup.version == 0                           # crashes skip the spill
    sup.version = 2
    assert sup._roll_back("memory") and sup.version == 1
    assert "after a memory failure (spilled: x)" in capsys.readouterr().err
    asked = []

    @contextlib.contextmanager
    def admit(config, ready):
        asked.append((config, ready))
        raise StartFailed("memory", "host memory: a restart needs 9 GiB")
        yield config
    sup = Supervisor([("spill", "l"), ("lock", "l")], memory_only={0: ""}, admit=admit)
    sup._start_s, sup._reports = [1.0], {1: {"host_bytes": 9 * GiB}}
    asyncio.run(sup._launch())
    assert asked == [("lock", {"host_bytes": 9 * GiB}), ("spill", None)] and "a restart needs 9 GiB" in sup._failure
