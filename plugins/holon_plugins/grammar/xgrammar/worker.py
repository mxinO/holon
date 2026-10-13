"""A vetting worker (docs/m05-grammar.md D5), run by `vetting.Vetting` as `python -m <this module> <parent pid>
<tokenizer file> <vocab size> <stop ids>`: it compiles each constraint it is sent and reports only the outcome — the
compile time, the compiled size and how far the compile grew its resident set at its peak — never the grammar
(shipping and loading one costs more than compiling it). One JSON line per job on stdin, one per outcome on stdout.

It dies with the thread that started it (PR_SET_PDEATHSIG) and offers itself first to the kernel's OOM killer.
"""
from __future__ import annotations

import ctypes
import json
import os
import resource
import signal
import sys
import time

PR_SET_PDEATHSIG = 1


def _peak_reset() -> None:
    """Restart the kernel's peak-RSS count (VmHWM) for the next compile (Linux 4.0+; else the lifetime peak)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def _status(field: str) -> int:
    """A size from /proc/self/status (VmRSS, VmHWM) in bytes; else the lifetime peak."""
    try:
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) << 10 for line in f if line.startswith(field + ":"))
    except (OSError, StopIteration):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss << 10


def main(parent: int, tokenizer_file: str, vocab_size: int, stops: list[int]) -> None:
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != parent:          # the parent died before prctl: nobody will signal us
        sys.exit(1)
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    reply = os.fdopen(os.dup(1), "w")          # the outcomes' channel; anything else printed goes to stderr
    os.dup2(2, 1)
    import xgrammar as xgr

    from .constraint import grammar

    with open(tokenizer_file) as f:
        info = xgr.TokenizerInfo.deserialize_json(f.read())
    if info.vocab_size != vocab_size or sorted(info.stop_token_ids) != stops:
        raise SystemExit(f"tokenizer mismatch: {info.vocab_size} / {info.stop_token_ids} vs {vocab_size} / {stops}")
    compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
    print(json.dumps({"ready": True}), file=reply, flush=True)
    for line in sys.stdin:
        _peak_reset()
        base, start = _status("VmRSS"), time.perf_counter()
        try:
            size = compiler.compile_grammar(grammar(json.loads(line))).memory_size_bytes
            out = {"ok": True, "seconds": time.perf_counter() - start, "size": size,
                   "growth": _status("VmHWM") - base}
        except Exception as e:          # a compile error: the constraint's own fault
            out = {"ok": False, "error": str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__}
        print(json.dumps(out), file=reply, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]))
