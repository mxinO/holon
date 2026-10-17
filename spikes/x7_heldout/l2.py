"""A1's L2 (docs/v1-offload.md §6: `approx`, human-gated): the vehicle with `moe.route_transform = tail_skip` at a frozen
placement (`std.tiered.static@1`, 25% cached) scored through the judge at one token per step (Tier B's decode path,
every expert matmul mmvq), so the plain mmvq floor is its floor:

    python spikes/x7_heldout/l2.py <model.gguf> <candidate.safetensors> "<plugin options k=v;k=v>"

then `holon_judge compare --golden … --candidate <it> --floor judge/goldens/qwen3-30b-a3b-q4_k_m/v1-bfloat16-mmvq-floor
.safetensors` judges. Teacher-forced tokens are prompt rows: a device-owned policy would never admit on them, so the
placement is frozen either way."""
import json
import pathlib
import sys

import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import gguf as judge_gguf
from holon_judge import golden
from holon_judge.engine import score
from holon_judge.recording import PromptSet


def main() -> None:
    model, out, opts = sys.argv[1:4]
    prompts = PromptSet.load("judge/prompts/v1.jsonl")
    options = dict(kv.split("=", 1) for kv in opts.split(";") if kv)
    max_len = max(len(p.ids) for p in prompts.prompts) + 1
    engine = Engine(EngineConfig(model=model, device="cuda", dtype=torch.bfloat16, max_model_len=max_len,
                                 max_step_tokens=1, max_seqs=1, host_memory=21 << 30, plugin_options=options))
    meta = judge_gguf.file_meta(pathlib.Path(model)) | {
        "weight_source": engine.source.kind, "max_step_tokens": 1, "gpu_class": engine.ctx.fingerprint.gpu,
        "dtype": "bfloat16", "device": "cuda", "hardware": golden.hardware(torch.device("cuda")),
        "torch": torch.__version__, "options": options}
    rec = score(engine, prompts, meta=meta)
    rec.save(out)
    print(json.dumps({"positions": int(rec.offsets[-1]), "wall_s": rec.meta["wall_s"]}))
    engine.close()


if __name__ == "__main__":
    main()
