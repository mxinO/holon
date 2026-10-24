"""FN §23.8's launch-failure check: one deployment serving one long prompt and then MTP verify passes, for
`compute-sanitizer` (memcheck: an access outside any allocation or an illegal address faults the run). Graphs on,
as served. Usage:

    compute-sanitizer --tool memcheck --error-exitcode 9 python spikes/fn7/sanitize.py <model> '<flags>' \\
        --ctx 8192 --tokens 24"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=24)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    sys.path.insert(0, str(ROOT / "spikes/fn7"))
    from _engine import engine
    from breakdown import prompt
    from holon.api import SamplingParams

    t0 = time.perf_counter()
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=a.ctx + a.tokens + 64)
    print(f"built in {time.perf_counter() - t0:.0f} s", flush=True)
    t0 = time.perf_counter()
    out = e.generate([prompt(a.ctx)], SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True))
    stats = {k: v for k, v in e.stats().items() if k.startswith("spec_verify")}
    print(f"served {a.ctx} + {a.tokens} tokens in {time.perf_counter() - t0:.0f} s: {stats}", flush=True)
    print(out[0] if out else None, flush=True)
    e.close()


if __name__ == "__main__":
    main()
