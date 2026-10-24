"""kernels.cpu_task's `cpu_task@1` (docs/v1-offload.md D6): a CUDA graph hands tasks to the native worker and waits for
them — each signal runs its slot's task once, on every thread of the pool, before the wait returns —, and the
protocol fails open: a failing task, a hung task (the spin wait's time bound) and a dead worker (the memop flavour's
watchdog) each raise the error word, and once the flag is set every later wait returns at once."""
import time

import numpy as np
import pytest
import torch
from holon.core import platform

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
PREBUILT = ("cpu_task",)  # CUDA extensions built if missing (conftest)

LINE = 64


@pytest.fixture(scope="module")
def lib():
    from holon_plugins.kernels.cpu_task import extension

    return extension.load(platform.fingerprint(torch.device("cuda")).sm)


def _worker(lib, probes, memop=False, timeout=5.0, threads=3):
    """A worker over `probes` (a slot each) and its mapped words."""
    device = torch.device("cuda")
    words = platform.host_memory((len(probes) + 1) * LINE, device, mapped=True)
    alias = platform.device_alias(words, device)
    cores = sorted(__import__("os").sched_getaffinity(0))[-threads:]
    worker = lib.Worker(words.data_ptr(), alias.data_ptr(), [p.handle() for p in probes], cores, int(timeout * 1e9),
                        memop)
    return worker, (words, alias)


@pytest.mark.parametrize("memop", [False, True])
def test_a_graph_runs_each_slots_task_once_per_signal_on_every_thread(lib, memop):
    outs = [torch.zeros(2, dtype=torch.int64) for _ in range(3)]
    probes = [lib.Probe(20_000 * s, 0, out) for s, out in enumerate(outs)]
    worker, keep = _worker(lib, probes, memop)
    errors = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        worker.signal(0, errors)                     # eager first (the graph's capture runs nothing)
        worker.wait(0, errors)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for s in range(3):
                worker.signal(s, errors)
                worker.wait(s, errors)
    torch.cuda.synchronize()
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    assert [int(o[0]) for o in outs] == [51, 50, 50] and all(int(o[1]) == 3 for o in outs)   # every thread ran it
    assert worker.served() == 151 and worker.flag() == 0 and int(errors[0]) == 0
    worker.stop()


@pytest.mark.parametrize("memop", [False, True])
def test_the_protocol_fails_open(lib, memop):
    """A failing task raises bit 1 and the flag; then a slow task's wait returns at once and the worker answers it
    without running it. A task outliving the time bound raises bit 2 (spin: the wait gives up; memop: the watchdog
    answers), and a stop does not wait for it past the bound; a dead worker raises bit 2; every wait after returns at
    once."""
    out = [torch.zeros(2, dtype=torch.int64) for _ in range(3)]
    probes = [lib.Probe(0, 0, out[0]), lib.Probe(0, 1, out[1]), lib.Probe(int(3e9), 0, out[2])]
    worker, keep = _worker(lib, probes, memop, timeout=0.3)
    errors = torch.zeros(1, dtype=torch.int32, device="cuda")
    for s in range(3):
        worker.signal(s, errors)
        worker.wait(s, errors)
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    assert int(errors[0]) == 1 and worker.flag() == 1 and time.perf_counter() - t0 < 1.0
    assert int(out[2][0]) == 0                                        # answered failed, never run
    worker.stop()

    probes = [lib.Probe(0, 0, out[0]), lib.Probe(int(2e9), 0, out[2])]
    worker, keep = _worker(lib, probes, memop, timeout=0.3)
    errors.zero_()
    t0 = time.perf_counter()
    worker.signal(1, errors)
    worker.wait(1, errors)
    worker.signal(0, errors)
    worker.wait(0, errors)
    torch.cuda.synchronize()
    assert int(errors[0]) & 2 and worker.flag() & 2 and 0.25 < time.perf_counter() - t0 < 1.5   # (memop: abandoned)
    t0 = time.perf_counter()
    assert worker.stop()                   # the coordinator, still in the slow task, is left behind after the bound
    assert time.perf_counter() - t0 < 1.0
    while not worker.served():             # its words and task outlive it (the plugin leaks them on purpose; here:
        time.sleep(0.05)                   # until it has answered, its last touch of them)

    worker, keep = _worker(lib, [lib.Probe(0, 0, out[0])], memop, timeout=0.3)
    worker.kill()                                                     # the coordinator stops answering
    errors.zero_()
    t0 = time.perf_counter()
    for _ in range(5):
        worker.signal(0, errors)
        worker.wait(0, errors)
    torch.cuda.synchronize()
    assert int(errors[0]) & 2 and time.perf_counter() - t0 < 1.5      # one time bound, not five
    worker.stop()


def test_the_copy_task_reads_each_requested_piece_into_its_unit(lib, tmp_path):
    """F9's copy task: per role, the requested pieces' extents (unaligned, two per piece; an O_DIRECT extent read
    through its aligned superset, a page-cache one — O_DIRECT off, a page-cache tier's — directly) read in order into
    the requested units, on two threads; nothing else written; the
    oracle (std.moe.reference `copy_task`) agrees; a request past the pieces fails the task."""
    import os

    from holon_std.moe import reference

    data = np.random.default_rng(0).integers(0, 256, 3 * 4096 * 10 + 123, dtype=np.uint8)
    path = tmp_path / "experts.bin"
    path.write_bytes(data.tobytes())
    plain = os.open(path, os.O_RDONLY)
    try:
        direct, flag = os.open(path, os.O_RDONLY | os.O_DIRECT), 1
    except OSError:                                         # a filesystem without O_DIRECT
        direct, flag = plain, 0
    pieces = [[(7, 1000), (5000, 2000)], [(9001, 3000), (40000, 0)], [(100, 50), (120000, 2950)]]
    desc = torch.tensor([[(fd, f, off, n) for off, n in piece] for piece in pieces for fd, f in [(direct, flag)]],
                        dtype=torch.int64)
    desc[1, :, 0:2] = torch.tensor([plain, 0])              # one piece through the page cache (O_DIRECT off)
    want = torch.zeros(5, 3000, dtype=torch.uint8)
    req = torch.tensor([[2, 0], [2, 4], [0, 1], [0, 0]], dtype=torch.int32)
    units = want.clone()
    reference.copy_task([(desc, want, req)])
    for unit, piece in ((4, 2), (1, 0)):
        expect = np.concatenate([data[off:off + n] for off, n in pieces[piece]])
        assert np.array_equal(want[unit].numpy(), expect)
    task = lib.CopyTask([(desc, units, req)])
    assert task.run(2) == 0 and torch.equal(units, want)
    req[1] = torch.tensor([3, 0])                           # a piece past the descriptors
    assert task.run(2) != 0
    os.close(plain)
    if flag:
        os.close(direct)
