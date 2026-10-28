"""M6 step 1: routing traces of a MoE model served by Holon's engine (spike; docs/dev/v1-offload.md §8 M6).

Runs the engine eagerly (no CUDA graphs, so Python wrappers see every MoE call) and records, per launched step and
per token row: the row's sequence, whether it is a decode row, and for every MoE layer its top-k expert ids (routing
order) and weights. On every `--recall-every`-th step with decode rows it also applies layer l+1's router to layer
l's MoE input for those rows (D11's next-layer predictor) and keeps its top k+4 ids. Wrappers live in this process
only: `MoE._topk` and `MoE.forward` per instance, `Engine._launch` for the step's rows. Nothing in core/std changes.

Workloads:
  agent   bench/traces/claude_code.jsonl on corpus text at temperature 1.0 (the Claude-Code-shaped agent trace)
  mixed   bench/traces/agent.jsonl + long_prompt.jsonl, greedy, chunk_with_decode 512 (mixed steps)
  chat    judge/prompts/v1.jsonl chat prompts, batch 1, 2 seeds, up to 1024 tokens (Qwen3 thinking sampling)
  code    the same for the code prompts
  ptime   prefill timing: one MoE layer's forward and the whole step vs chunk size (64 … 4096 tokens)

    CUDA_VISIBLE_DEVICES=<gpu> python spikes/x0_offload/m6/capture.py --model <file.gguf> --workload agent \
        --out local/tmp/x0-m6/agent.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch
from holon.api import SamplingParams, Status
from holon.core.engine import Engine, EngineConfig
from holon_std.moe.units import MoE

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
import driver  # noqa: E402
import workloads  # noqa: E402

CORPUS = ROOT / "judge" / "prompts" / "v1.jsonl"
EXTRA = 4                                      # predicted ids beyond top-k (D11's top-k+4)


class Recorder:
    """Per-step routing of every MoE layer, gathered by wrappers on each layer's top-k and forward."""

    def __init__(self, moes: list[MoE], recall_every: int) -> None:
        self.moes = sorted(moes, key=lambda m: m.layer)
        self.L, self.recall_every = len(self.moes), recall_every
        self.seq_ids: dict[str, int] = {}
        self.rows: list[np.ndarray] = []          # per step: [R, 3] (step, seq, decode)
        self.ids: list[np.ndarray] = []           # per step: [R, L, k] uint8
        self.w: list[np.ndarray] = []             # per step: [R, L, k] float16
        self.rec_rows: list[np.ndarray] = []      # global row indices of the recall sample
        self.rec_pred: list[np.ndarray] = []      # [r, L-1, k+EXTRA] uint8: layer l+1's ids predicted at layer l
        self.n_rows = self.step = self.decode_steps = 0
        self.meta: list[tuple[int, int, bool]] | None = None
        self.n_dec = 0
        self.sample = False
        self.timing: list | None = None            # ptime: (layer, rows, start event, end event)
        for m in self.moes:
            self._wrap(m)

    def _wrap(self, m: MoE) -> None:
        topk, forward, rec = m._topk, m.forward, self
        nxt = self.moes[m.layer + 1].router if m.layer + 1 < self.L else None

        def wrapped_topk(logits: torch.Tensor):
            weights, ids = topk(logits)
            if rec.meta is not None:
                rec.cur_ids[m.layer].append(ids)
                rec.cur_w[m.layer].append(weights)
            return weights, ids

        def wrapped_forward(x: torch.Tensor) -> torch.Tensor:
            if rec.meta is not None and rec.sample and nxt is not None:
                logits = torch.nn.functional.linear(x[:rec.n_dec].float(), nxt)
                rec.cur_pred[m.layer].append(logits.topk(m._gate_up.top_k + EXTRA, dim=-1).indices)
            if rec.timing is not None:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                y = forward(x)
                b.record()
                rec.timing.append((m.layer, x.shape[0], a, b))
                return y
            return forward(x)

        m._topk, m.forward = wrapped_topk, wrapped_forward

    def begin(self, seqs, n_query) -> None:
        self.meta = [(self.seq_ids.setdefault(f"{s.rid}/{s.index}", len(self.seq_ids)), int(n),
                      s.status is Status.RUNNING) for s, n in zip(seqs, n_query, strict=True)]
        dec = [d for _, _, d in self.meta]
        # decode rows lead a mixed step (BatchPlan.n_decode); a step is decode-only or prefill-only otherwise
        self.n_dec = sum(n for _, n, d in self.meta if d)
        assert all(dec[:sum(dec)]) and not any(dec[sum(dec):]), "decode rows must lead the step"
        self.sample = self.n_dec > 0 and self.decode_steps % self.recall_every == 0
        self.decode_steps += self.n_dec > 0
        self.cur_ids = [[] for _ in range(self.L)]
        self.cur_w = [[] for _ in range(self.L)]
        self.cur_pred = [[] for _ in range(self.L - 1)]

    def end(self) -> None:
        R = sum(n for _, n, _ in self.meta)
        ids = torch.stack([torch.cat(c) for c in self.cur_ids], 1)          # [R, L, k]
        w = torch.stack([torch.cat(c) for c in self.cur_w], 1)
        assert ids.shape[0] == R, (ids.shape, R)
        self.ids.append(ids.to(torch.uint8).cpu().numpy())
        self.w.append(w.to(torch.float16).cpu().numpy())
        rows = np.array([(self.step, s, d) for s, n, d in self.meta for _ in range(n)], dtype=np.int32)
        self.rows.append(rows)
        if self.sample:
            pred = torch.stack([torch.cat(c) for c in self.cur_pred], 1)     # [n_dec, L-1, k+EXTRA]
            self.rec_pred.append(pred.to(torch.uint8).cpu().numpy())
            self.rec_rows.append(np.arange(self.n_rows, self.n_rows + self.n_dec, dtype=np.int64))
        self.n_rows += R
        self.step += 1
        self.meta = None

    def save(self, path: pathlib.Path, info: dict) -> None:
        rows = np.concatenate(self.rows)
        k = self.ids[0].shape[-1]
        np.savez_compressed(
            path, ids=np.concatenate(self.ids), w=np.concatenate(self.w), step=rows[:, 0], seq=rows[:, 1],
            decode=rows[:, 2].astype(bool),
            rec_rows=np.concatenate(self.rec_rows) if self.rec_rows else np.zeros(0, np.int64),
            rec_pred=np.concatenate(self.rec_pred) if self.rec_pred else np.zeros((0, self.L - 1, k + EXTRA), np.uint8),
            info=json.dumps(info))


def install(engine: Engine, recall_every: int) -> Recorder:
    moes = [m for m in engine.executor.model.modules() if isinstance(m, MoE)]
    assert moes and [m.layer for m in sorted(moes, key=lambda m: m.layer)] == list(range(len(moes)))
    rec = Recorder(moes, recall_every)
    launch = Engine._launch

    def recorded_launch(self, plan, seqs):
        rec.begin(seqs, plan.n_query)
        try:
            return launch(self, plan, seqs)
        finally:
            rec.end()

    Engine._launch = recorded_launch
    return rec


class Adapter:
    """bench/driver.py's Backend over an engine built here."""

    def __init__(self, engine: Engine, temperature: float) -> None:
        self.engine, self.temperature, self._rids = engine, temperature, {}

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        params = SamplingParams(temperature=self.temperature, max_tokens=max_tokens, ignore_eos=True, seed=rid)
        self._rids[self.engine.add(prompt, params)] = rid

    def step(self):
        out = []
        for d in self.engine.step():
            done = d.finish_reason is not None
            out.append((self._rids.pop(d.rid) if done else self._rids[d.rid], d.token_ids, done))
        return out

    def has_work(self) -> bool:
        return self.engine.has_work()

    def info(self) -> dict:
        return {}


def run_traces(engine: Engine, names: list[str], temperature: float) -> dict:
    corpus = workloads.corpus(CORPUS)
    out = {}
    for name in names:
        sessions = workloads.load(ROOT / "bench" / "traces" / f"{name}.jsonl")
        done, steps = driver.replay(Adapter(engine, temperature), sessions, salt=f"m6/{name}", corpus=corpus)
        out[name] = {"requests": len(done), "generated": sum(len(r.generated) for r in done),
                     "repeated": sum(r.row()[-1] for r in done), "steps": len(steps)}
    return out


def run_prompts(engine: Engine, category: str, seeds: int, max_tokens: int) -> dict:
    prompts = [json.loads(line) for line in CORPUS.read_text().splitlines() if line]
    prompts = [p for p in prompts if p["category"] == category]
    generated = repeated = 0
    for seed in range(seeds):
        for p in prompts:
            ids = p["ids"][:p["gen_start"]]
            params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=seed)
            engine.add(ids, params)
            got = []
            while engine.has_work():
                for d in engine.step():
                    got += d.token_ids
            generated += len(got)
            repeated += driver.repeated(ids, got)
    return {"prompts": [p["id"] for p in prompts], "seeds": seeds, "generated": generated, "repeated": repeated}


def run_ptime(engine: Engine, rec: Recorder, sizes: list[int], reps: int) -> dict:
    """Prefill of fresh corpus spans at each size, `reps` times: the median MoE forward per layer and the step."""
    corpus = workloads.corpus(CORPUS)
    rec.timing = []
    out = {}
    for n in sizes:
        moe, step = [], []
        for r in range(reps + 1):                    # the first rep warms up (JIT, autotune)
            ids = workloads.tokens(f"m6/ptime/{n}/{r}", "p", n, corpus)
            rec.timing.clear()
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.add(ids, SamplingParams(temperature=0.0, max_tokens=1))
            while engine.has_work():
                engine.step()
            torch.cuda.synchronize()
            if r:
                step.append((time.perf_counter() - t) * 1e3)
                per = [a.elapsed_time(b) for layer, rows, a, b in rec.timing if rows == n]
                moe.append(statistics.median(per))
        out[n] = {"step_ms": statistics.median(step), "moe_layer_ms": statistics.median(moe)}
    rec.timing = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", required=True, choices=["agent", "mixed", "chat", "code", "ptime"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--recall-every", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    batch1 = args.workload in ("chat", "code", "ptime")
    cfg = EngineConfig(model=args.model, cuda_graphs=False, max_seqs=1 if batch1 else 4,
                       max_model_len=8192 if args.workload in ("chat", "code") else 32768, max_step_tokens=4096,
                       chunk_with_decode=512 if args.workload == "mixed" else 0)
    t = time.perf_counter()
    engine = Engine(cfg)
    load_s = time.perf_counter() - t
    rec = install(engine, args.recall_every)
    t = time.perf_counter()
    if args.workload == "agent":
        result = run_traces(engine, ["claude_code"], temperature=1.0)
    elif args.workload == "mixed":
        result = run_traces(engine, ["agent", "long_prompt"], temperature=0.0)
    elif args.workload == "ptime":
        result = run_ptime(engine, rec, [64, 256, 1024, 4096], reps=3)
    else:
        result = run_prompts(engine, args.workload, args.seeds, args.max_tokens)
    info = {"workload": args.workload, "config": {k: v for k, v in vars(cfg).items() if k in (
        "max_seqs", "max_model_len", "max_step_tokens", "chunk_with_decode", "cuda_graphs", "prefix_cache",
        "overlap")}, "result": result, "load_s": round(load_s, 1), "run_s": round(time.perf_counter() - t, 1),
        "gpu": torch.cuda.get_device_name(), "layers": rec.L, "recall_every": args.recall_every,
        "stats": {k: v for k, v in engine.stats().items() if k in ("prefix_tokens_saved", "mixed_steps")}}
    out = pathlib.Path(args.out)
    if args.workload == "ptime":
        out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    else:
        rec.save(out, info)
    print(json.dumps(info))
    engine.close()


if __name__ == "__main__":
    main()
