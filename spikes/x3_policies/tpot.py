"""Decode latency under borrowing (panel C, step 3b): the agent trace (bench/traces/agent.jsonl, natural-text
prompts cut from judge/prompts/v1.jsonl) replayed through the offload store with mixed steps (`chunk_with_decode`
512, `max_seqs` 8), with borrowing on (the store as built: prompt steps lend cache slots, D4 / D8) and off (the
layout step 3a had: staging slots for a whole prefill pass, nothing lent, 512-token passes — set here by patching the
store, not an engine option). Per arm and round: TPOT p50 / p99 over requests, the per-request maximum
inter-token gap p50 / p99, the steps that borrowed; arms interleaved by round, each a fresh engine:

    python spikes/x3_policies/tpot.py <model.gguf> <out.json> <rounds> <cache> <arm>...    # arm: on | off
"""
import contextlib
import gc
import json
import pathlib
import statistics
import sys
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import workloads  # noqa: E402
from backends.holon import Backend  # noqa: E402
from driver import replay, warmup_sessions  # noqa: E402
from holon.core.engine import Engine, EngineConfig  # noqa: E402
from holon_std.tiered.store import TieredStore  # noqa: E402

TRACE = ROOT / "bench/traces/agent.jsonl"
CORPUS = ROOT / "judge/prompts/v1.jsonl"


@contextlib.contextmanager
def no_borrowing():
    """The store without lending: a prefill pass's staging slots, no ring, 512-token passes."""
    resolver, materialized = TieredStore.resolver, TieredStore.after_materialize

    def no_lending_resolver(self, group):
        if self.residency is None:
            self.ring_tokens, self.staging = 0, self.prefill
            for c in self.classes.values():
                c.staging = self.prefill
        return resolver(self, group)

    def after(self, ctx):
        materialized(self, ctx)
        self.lending = False
    with mock.patch.object(TieredStore, "resolver", no_lending_resolver), \
            mock.patch.object(TieredStore, "after_materialize", after):
        yield


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> None:
    model, out, rounds, cache, arms = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:]
    sessions, corpus = workloads.load(TRACE), workloads.corpus(CORPUS)
    longest = max(workloads.context_len(s) for s in sessions)
    results = []
    for r in range(rounds):
        for arm in arms if r % 2 == 0 else arms[::-1]:
            with no_borrowing() if arm == "off" else contextlib.nullcontext():
                backend = Backend.__new__(Backend)
                backend.engine = Engine(EngineConfig(
                    model=model, dtype=torch.bfloat16, max_seqs=8, max_step_tokens=4096,
                    max_model_len=-(-(longest + 1) // 256) * 256, chunk_with_decode=512, host_memory=20 << 30,
                    plugin_options={"moe.store": "std.moe.offload@1", "moe.cache": cache}))
                backend._rids, backend.temperature = {}, 0.0
                store = backend.engine.ctx.shared_objects["std.moe.offload"]
                replay(backend, warmup_sessions(sessions), salt="agent/warmup", corpus=corpus)
                before = dict(store.borrowed)
                done, steps = replay(backend, sessions, salt="agent/run", corpus=corpus)
            tpot = [(q.last - q.first) / (len(q.generated) - 1) * 1e3 for q in done if len(q.generated) > 1]
            gaps = [q.max_gap * 1e3 for q in done]
            row = {"arm": arm, "round": r, "requests": len(done), "steps": len(steps),
                   "borrowed": {k: v - before[k] for k, v in store.borrowed.items()},
                   "tpot_ms": {"p50": round(statistics.median(tpot), 2), "p99": round(pct(tpot, 0.99), 2)},
                   "max_gap_ms": {"p50": round(statistics.median(gaps), 1), "p99": round(pct(gaps, 0.99), 1)},
                   "mixed_steps": backend.engine.stats().get("mixed_steps")}
            print(json.dumps(row), flush=True)
            results.append(row)
            backend.engine.close()
            del backend, store
            gc.collect()
            torch.cuda.empty_cache()
    pathlib.Path(out).write_text(json.dumps({"cache": cache, "trace": "agent", "max_seqs": 8,
                                             "chunk_with_decode": 512, "runs": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
