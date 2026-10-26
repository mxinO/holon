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


def _landing(tmp_path, size=3 * 4096 * 10 + 123):
    """A file and its pieces (two extents each, unaligned), as O_DIRECT descriptors where the filesystem has it."""
    import os

    data = np.random.default_rng(1).integers(0, 256, size, dtype=np.uint8)
    path = tmp_path / "experts.bin"
    path.write_bytes(data.tobytes())
    try:
        fd, flag = os.open(path, os.O_RDONLY | os.O_DIRECT), 1
    except OSError:
        fd, flag = os.open(path, os.O_RDONLY), 0
    pieces = [[(7, 1000), (5000, 2000)], [(9001, 3000), (40000, 0)], [(100, 50), (120000, 2950)]]
    desc = torch.tensor([[(fd, flag, off, n) for off, n in piece] for piece in pieces], dtype=torch.int64)
    return data, pieces, desc, fd


def _check_landing(data, pieces, rows, runs, keep, req):
    """Each requested piece's extents where `runs` says, in its landing row; kept pieces canonical in their unit."""
    for piece, row, kept in req[1:1 + int(req[0, 0])].tolist():
        canon = np.concatenate([data[off:off + n] for off, n in pieces[piece]])
        got = np.concatenate([rows[row, a:a + n].numpy() for a, n in runs[row].tolist()])
        assert np.array_equal(got, canon) and all(a % 4096 == off % 4096 for (a, _), (off, n) in
                                                  zip(runs[row].tolist(), pieces[piece], strict=True) if n)
        if kept >= 0:
            assert np.array_equal(keep[kept, :len(canon)].numpy(), canon)


def test_the_copy_task_reads_pieces_into_landing_rows_and_keeps(lib, tmp_path):
    """docs/v1-glm.md D8f's landing rows: each O_DIRECT extent's aligned superset read in place at the row's next
    4 KiB-aligned offset (no bounce), (offset, nbytes) of its bytes in `runs`, a kept piece copied canonical into its
    keep unit; the oracle agrees on every byte it names; a row too small for a piece's supersets fails the task."""
    import os

    from holon_std.moe import reference

    data, pieces, desc, fd = _landing(tmp_path)
    req = torch.tensor([[2, 0, 0], [2, 1, 3], [0, 3, -1]], dtype=torch.int32)
    rows = platform.host_memory(4 * 4 * 4096, torch.device("cuda"), mapped=False).view(torch.uint8).view(4, 4 * 4096)
    runs, keep = torch.zeros(4, 2, 2, dtype=torch.int64), torch.zeros(4, 3000, dtype=torch.uint8)
    task = lib.CopyTask([(desc, rows, req, runs, keep)])
    assert task.run(2) == 0
    _check_landing(data, pieces, rows, runs, keep, req)
    want_rows, want_runs, want_keep = rows.clone().zero_(), torch.zeros_like(runs), torch.zeros_like(keep)
    reference.copy_task([(desc, want_rows, req, want_runs, want_keep)])
    assert torch.equal(want_runs, runs) and torch.equal(want_keep, keep)
    _check_landing(data, pieces, want_rows, want_runs, want_keep, req)
    small = rows[:, :4096]                                  # (one page a row: two supersets cannot fit)
    assert lib.CopyTask([(desc, small.contiguous(), req, runs[:, :, :], keep)]).run(1) != 0
    os.close(fd)


def _io_worker(lib, tasks, backend, urgent=(), depth=4, chunk=8192, memop=False, timeout=5.0):
    device = torch.device("cuda")
    words = platform.host_memory((len(tasks) + 1) * LINE, device, mapped=True)
    alias = platform.device_alias(words, device)
    core = sorted(__import__("os").sched_getaffinity(0))[-1:]
    worker = lib.Worker(words.data_ptr(), alias.data_ptr(), [t.native() for t in tasks], core, int(timeout * 1e9),
                        memop, depth, backend, list(urgent) or [False] * len(tasks), chunk)
    return worker, (words, alias)


@pytest.mark.parametrize("backend", ["uring", "aio"])
def test_the_io_worker_reads_asynchronously_and_serves_urgent_requests_first(lib, tmp_path, backend):
    """The `io` pool (docs/v1-glm.md D8f): one completion-driven thread over copy tasks, io_uring or native AIO (forced
    here), small chunks in flight: canonical units (bounced) and landing rows (in place, kept pieces copied) as the
    oracle reads them, through graph-free signals and waits; a failing request fails open. An urgent slot's request,
    signalled while a large request's chunks are in flight, is answered before it (the large one's unsubmitted chunks
    held meanwhile)."""
    import os

    from holon_std.moe import reference

    words = torch.zeros(2 * LINE, dtype=torch.uint8)       # (a worker of no tasks: is the interface there?)
    try:
        lib.Worker(words.data_ptr(), words.data_ptr(), [], [0], 1, False, 1, backend).stop()
    except RuntimeError as e:
        pytest.skip(f"{backend}: {e}")
    data, pieces, desc, fd = _landing(tmp_path)
    req = torch.tensor([[2, 0], [2, 4], [0, 1]], dtype=torch.int32)
    units, want = torch.zeros(5, 3000, dtype=torch.uint8), torch.zeros(5, 3000, dtype=torch.uint8)
    reference.copy_task([(desc, want, req)])
    lreq = torch.tensor([[2, 0, 0], [2, 1, 1], [0, 0, -1]], dtype=torch.int32)
    rows = platform.host_memory(2 * 4 * 4096, torch.device("cuda"), mapped=False).view(torch.uint8).view(2, 4 * 4096)
    runs, keep = torch.zeros(2, 2, 2, dtype=torch.int64), torch.zeros(2, 3000, dtype=torch.uint8)
    tasks = [lib.CopyTask([(desc, units, req)]), lib.CopyTask([(desc, rows, lreq, runs, keep)])]
    worker, held = _io_worker(lib, tasks, backend)
    assert worker.backend() == {"uring": "io_uring", "aio": "aio"}[backend]
    errors = torch.zeros(1, dtype=torch.int32, device="cuda")
    for s in range(2):
        worker.signal(s, errors)
        worker.wait(s, errors)
    torch.cuda.synchronize()
    assert int(errors[0]) == 0 and torch.equal(units, want)
    _check_landing(data, pieces, rows, runs, keep, lreq)
    stats = worker.io_stats()                               # its counters (step 1's review C S1-M3)
    spans = sum(-(-(off + n) // 4096) * 4096 - off // 4096 * 4096 for p in (2, 0) for off, n in pieces[p] if n)
    assert stats["backend"] == worker.backend() and stats["bytes"] >= spans and stats["busy_ns"] > 0, stats
    assert stats["requests_2-3"] == 2 and stats["latency_ns_2-3"] > 0, stats
    req[1] = torch.tensor([3, 0])                           # a piece past the descriptors: fails open
    worker.signal(0, errors)
    worker.wait(0, errors)
    torch.cuda.synchronize()
    assert int(errors[0]) == 1 and worker.flag() == 1
    worker.stop()
    os.close(fd)

    big = tmp_path / "big.bin"                              # a large read in flight, then an urgent one
    big.write_bytes(np.random.default_rng(2).integers(0, 256, 64 << 20, dtype=np.uint8).tobytes())
    try:
        bfd, bflag = os.open(big, os.O_RDONLY | os.O_DIRECT), 1
    except OSError:
        bfd, bflag = os.open(big, os.O_RDONLY), 0
    os.posix_fadvise(bfd, 0, 0, os.POSIX_FADV_DONTNEED)
    n = 16
    bdesc = torch.tensor([[(bfd, bflag, i * (4 << 20), 4 << 20)] for i in range(n)], dtype=torch.int64)
    breq = torch.tensor([[n, 0]] + [[i, i] for i in range(n)], dtype=torch.int32)
    bunits = torch.zeros(n, 4 << 20, dtype=torch.uint8)
    ureq = torch.tensor([[1, 0], [0, 0]], dtype=torch.int32)
    uunits = torch.zeros(1, 4 << 20, dtype=torch.uint8)
    tasks = [lib.CopyTask([(bdesc, bunits, breq)]), lib.CopyTask([(bdesc, uunits, ureq)])]
    worker, (words, _) = _io_worker(lib, tasks, backend, urgent=(False, True), depth=2, chunk=256 << 10)
    word = words.view(torch.int32)
    word[0] = 1                                             # REQ, from the host: the large request
    time.sleep(0.01)
    word[16] = 1                                            # the urgent one, while the large one's reads go on
    order = []
    while len(order) < 2:
        order += [s for s in (0, 1) if s not in order and int(word[16 * s]) >= 2]
    assert order == [1, 0] and int(word[0]) == 2 and int(word[16]) == 2, order
    assert np.array_equal(uunits[0].numpy(), bunits[0].numpy())
    stats = worker.io_stats()
    assert stats["requests_1"] == 1 and stats["requests_4+"] == 1 and stats["bytes"] >= (n + 1) * (4 << 20), stats
    worker.stop()
    os.close(bfd)


def test_an_abandoned_read_never_answers_the_slots_next_request(lib, tmp_path):
    """Step 1's review A S1-m1: the memop watchdog abandons a request whose read is still in flight (a FIFO with no
    data yet); the slot is signalled again; when the old read lands, its answer is not the new request's — the new one
    is started after the old one drains and fails at once under the timeout flag (or the flagged watchdog abandons it
    first)."""
    import os

    words = torch.zeros(2 * LINE, dtype=torch.uint8)
    try:
        lib.Worker(words.data_ptr(), words.data_ptr(), [], [0], 1, False, 1, "uring").stop()
    except RuntimeError as e:
        pytest.skip(f"uring: {e}")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    fd = os.open(fifo, os.O_RDWR)                           # (read and write ends: neither open blocks)
    desc = torch.tensor([[(fd, 0, 0, 4096)]], dtype=torch.int64)
    units = torch.zeros(1, 4096, dtype=torch.uint8)
    task = lib.CopyTask([(desc, units, torch.tensor([[1, 0], [0, 0]], dtype=torch.int32))])
    worker, (words, _) = _io_worker(lib, [task], "uring", depth=1, chunk=4096, memop=True, timeout=0.1)
    word = words.view(torch.int32)
    word[0] = 1                                             # REQ: its read waits on the FIFO
    t0 = time.perf_counter()
    while int(word[0]) != 4 and time.perf_counter() - t0 < 5:              # ABANDONED by the watchdog
        time.sleep(0.01)
    assert int(word[0]) == 4 and worker.flag() & 2
    word[0] = 1                                             # the slot's next request, while the old read is in flight
    os.write(fd, b"x" * 4096)                               # the old read lands
    t0 = time.perf_counter()
    while int(word[0]) == 1 and time.perf_counter() - t0 < 5:
        time.sleep(0.01)
    assert int(word[0]) in (3, 4)                           # FAILED (the flag) or abandoned (the watchdog, flagged):
    #                                                         never DONE by the old read's answer
    worker.stop()
    os.close(fd)
