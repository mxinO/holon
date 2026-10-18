"""FN step 0's oracle feasibility check (docs/v1-flash-next.md N4, D2): one `v1-long` prompt through the judge's
streamed fp32 HF driver (`golden._streamed`, transformers 5.19.0, one decoder layer resident, PLE's rows gathered
lazily, the indexer's tie rule) on the full GGUF, on the CPU under a memory cap: wall time per layer, peak RSS, and
the indexer's boundary-tie rate (a query whose k-th and (k+1)-th block scores are equal) on each QSA layer. If HF's
attention runs out of memory at this length, the record says so: D2's query-block mode is then needed.

    python spikes/fn0/n4_oracle.py --gguf <...-00001-of-00002.gguf> --config <Qwen3.8-Flash-Next snapshot> \\
        --prompt recall-32k --threads 32 --out <record.json>       # in the golden venv, under systemd-run MemoryMax
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import resource
import time

import torch
import transformers
from holon_judge import gguf, golden, qwen4exp
from holon_judge.recording import Prompt, PromptSet

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", default="recall-32k")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = [json.loads(x) for x in (ROOT / "judge/prompts/v1-long.jsonl").read_text().splitlines() if x]
    row = next(r for r in rows if r["id"] == a.prompt)
    prompts = PromptSet("v1-long.jsonl", "", [Prompt(row["id"], row["category"], row["ids"])])
    path = pathlib.Path(a.gguf)
    model = golden._skeleton(pathlib.Path(a.config))
    ties = {"sparse": 0, "boundary_ties": 0}             # per sparse selection (the record of 2026-10-18 also counted
                                                         # every top-k call: "queries")

    def count(tied: bool) -> None:
        ties["sparse"] += 1
        ties["boundary_ties"] += tied
    qwen4exp.prepare(model, path, "cpu", torch.float32, on_tie=count)
    times, t0 = [], [time.perf_counter()]
    record = {"name": "fn0_n4_oracle", "date": datetime.date.today().isoformat(), "device": "cpu",
              "threads": a.threads, "torch": torch.__version__, "transformers": transformers.__version__,
              "model": gguf.file_meta(path)["model"], "prompt": a.prompt, "tokens": len(row["ids"]),
              "command": f"python spikes/fn0/n4_oracle.py --prompt {a.prompt} --threads {a.threads}",
              "status": "running", "layers": times, "indexer_ties": ties}

    def save() -> None:            # after every layer: a run the memory cap kills keeps how far it got
        record["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        pathlib.Path(a.out).write_text(json.dumps(record, indent=1) + "\n")

    def stamp(module, i: int) -> None:
        times.append({"layer": i, "type": module.layer_type, "s": round(time.perf_counter() - t0[0], 1),
                      "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)})
        print(times[-1], flush=True)
        save()
        t0[0] = time.perf_counter()
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda m, a_, o, i=i: stamp(m, i))
    save()
    start = time.perf_counter()
    try:
        out, wall = golden._streamed(model, lambda select: dict(gguf.hf_state(path, select)), "blk.", prompts, 20,
                                     torch.float32, "cpu", False, lambda m, p: None)
        record |= {"status": "done", "wall_s": round(wall, 1), "target_lp_mean": round(out[0][2].mean().item(), 4)}
    except (RuntimeError, MemoryError) as e:
        record |= {"status": f"failed: {type(e).__name__}: {str(e)[:200]}"}
    record["total_s"] = round(time.perf_counter() - start, 1)
    save()


if __name__ == "__main__":
    main()
