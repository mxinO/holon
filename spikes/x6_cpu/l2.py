"""F5b's bounded L2 (docs/dev/v1-offload.md §6): the vehicle with the CPU's share at a frozen placement
(`std.tiered.static@1`) scored through the judge at one token per step (Tier B's decode path: every expert matmul
mmvq, and F5b's rule per token), its split (the plan and the static placement) written for the path-aware floor:

    python spikes/x6_cpu/l2.py <model.gguf> <candidate.safetensors> <split.json> "<plugin options k=v;k=v>" [plant]

then `holon_judge golden … --q8-1 mmvq --cpu-split <split.json>` records the floor and `holon_judge compare` judges."""
import json
import pathlib
import sys

import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import gguf as judge_gguf
from holon_judge import golden
from holon_judge.engine import score
from holon_judge.placement import cpu_split
from holon_judge.planted import PLANTS
from holon_judge.recording import PromptSet


def main() -> None:
    model, out, split_out, opts = sys.argv[1:5]
    plant = sys.argv[5] if len(sys.argv) > 5 else ""
    prompts = PromptSet.load("judge/prompts/v1.jsonl")
    options = dict(kv.split("=", 1) for kv in opts.split(";") if kv)
    max_len = max(len(p.ids) for p in prompts.prompts) + 1
    with PLANTS[plant]() if plant else __import__("contextlib").nullcontext():
        engine = Engine(EngineConfig(model=model, device="cuda", dtype=torch.bfloat16, max_model_len=max_len,
                                     max_step_tokens=1, max_seqs=1, host_memory=21 << 30, plugin_options=options))
        split = cpu_split(engine)
        pathlib.Path(split_out).write_text(json.dumps(split) + "\n")
        meta = judge_gguf.file_meta(pathlib.Path(model)) | {
            "weight_source": engine.source.kind, "max_step_tokens": 1, "gpu_class": engine.ctx.fingerprint.gpu,
            "dtype": "bfloat16", "device": "cuda", "hardware": golden.hardware(torch.device("cuda")),
            "torch": torch.__version__, "plant": plant}
        rec = score(engine, prompts, meta=meta)
    rec.save(out)
    worker = engine.ctx.shared_objects["cpu_task.compute"].native
    print(json.dumps({"positions": int(rec.offsets[-1]), "wall_s": rec.meta["wall_s"], "served": worker.served(),
                      "flag": worker.flag(), "calls": rec.meta.get("calls")}))
    engine.close()


if __name__ == "__main__":
    main()
