"""What selecting a logits processor costs a deployment before any request is constrained (docs/m05-grammar.md §6):
graphs captured, capture time, the graph pool, and the KV tokens the larger pool and the masks' memory profile take
from the state pools (evidence record).

Each configuration starts an engine (default settings: CUDA graphs, overlap, `memory_fraction` 0.9) without a
processor, with the test-only toy processor (`testing.toy_mask`, whose `apply` is a dense masked fill materializing
[rows, vocab] intermediates), and with the toy's `apply` replaced by a no-op — the core's own cost (twice the graphs,
the staged indices) apart from any kernel's. One engine at a time in this process, each closed before the next;
`--rounds` interleaved rounds (the pools are sized from a measured free memory, which varies by a few MiB).

Usage (from the repository root):
  python bench/processor_overhead.py --gpu 2 --model <checkpoint dir>
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import time
from unittest import mock

from run import ROOT, git, gpu_class, model_id

SPEC = {"max_draft_tokens": 3, "plugin_options": {"spec.k": "3", "spec.fallback": 2},
        "slots": {"drafter": "drafters.ngram", "draft_policy": "std.spec.fixed_k", "verifier": "std.spec.greedy"}}


def measure(model: str, spec: bool, processor: str | None) -> dict:
    import torch
    from holon.core.engine import Engine, EngineConfig
    from holon.core.executor import Executor
    from holon_testing.toy_mask.plugin import ToyMask

    capture, timed = {}, Executor.capture_graphs

    def capture_graphs(self):
        torch.cuda.synchronize()
        reserved, start = torch.cuda.memory_reserved(), time.perf_counter()
        timed(self)
        torch.cuda.synchronize()
        capture.update(s=time.perf_counter() - start, pool=torch.cuda.memory_reserved() - reserved)

    kw = dict(SPEC) if spec else {}
    if processor:
        kw["slots"] = {**kw.get("slots", {}), "logits_processor": "testing.toy_mask"}
    null = mock.patch.object(ToyMask, "apply", lambda self, logits, mask, rows: None)
    with mock.patch.object(Executor, "capture_graphs", capture_graphs), \
            null if processor == "null" else contextlib.nullcontext():
        start = time.perf_counter()
        engine = Engine(EngineConfig(model=model, **kw))
        init_s = time.perf_counter() - start
    kv = engine.ledger.pool("state.kv")
    out = {"graphs": len(engine.executor.graphs), "capture_s": round(capture["s"], 2),
           "graph_pool_mib": round(capture["pool"] / 2**20, 1), "init_s": round(init_s, 2),
           "kv_tokens": kv.capacity * engine.cfg.page_size}
    engine.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results: dict[str, list[dict]] = {}
    for _ in range(args.rounds):
        for spec in (False, True):
            for processor in (None, "toy", "null"):
                name = f"{'ngram-k3' if spec else 'plain'}/{processor or 'none'}"
                results.setdefault(name, []).append(measure(args.model, spec, processor))
                print(name, results[name][-1], flush=True)
    date, gpu = datetime.date.today().isoformat(), gpu_class(args.gpu)
    record = {"workload": "processor_overhead", "date": date, "gpu_class": gpu, "model": model_id(args.model),
              "command": f"python bench/processor_overhead.py --gpu {args.gpu} --model {model_id(args.model)} "
                         f"--rounds {args.rounds}",
              "commit": git("rev-parse", "HEAD"), "results": results}
    name = model_id(args.model).split("@")[0].split("/")[-1].lower()
    out = ROOT / "evidence" / f"processor_overhead_{name}_{gpu}_{date}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
