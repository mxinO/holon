"""GLM step 5's review (C, the clamp): does the SwiGLU clamp (±`swiglu_clamp_exp`, 10) ever bind on the real model?
Gate (c)'s four prompts scored teacher-forced in 1-token steps (every routed expert call on the decode path, position
0 included), twice: clean, recording each routed layer's largest gate (the clamp binds where gate > L) and largest
|up| (where |up| > L) from an unclamped copy of each gate/up call; and under `glu_clamp_off` (the decode path's clamp
dropped). The two runs' top-20 recordings compared bitwise: identical means the clamp never bound.

    python spikes/glm5/clamp.py <maya-s -00001 shard> '<engine flags>' --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPTS = ("summary-6k", "recall-8k", "recall-16k", "chat-multiturn-20k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import engine as build
    from holon_judge.engine import score
    from holon_judge.planted import PLANTS
    from holon_judge.recording import PromptSet
    from holon_plugins.kernels.ggml_cuda import plugin

    full = PromptSet.load(ROOT / "judge/prompts/v1-long-glm5_next.jsonl")
    keep = {p.id: p for p in full.prompts}
    prompts = PromptSet(full.name, full.sha256, [keep[p] for p in PROMPTS])
    longest = max(len(p.ids) for p in prompts.prompts) + 8

    gate_max: dict[int, float] = collections.defaultdict(lambda: float("-inf"))
    up_max: dict[int, float] = collections.defaultdict(float)
    call = plugin.LinearExperts.__call__

    def recorded(self, x, parts, ids, **kw):
        if self.swiglu and self.limit and 0 < len(ids) <= self.mmvq_tokens:
            xs = x.x if not isinstance(x, torch.Tensor) else x
            y = torch.empty(len(ids), self.top_k, self.out, device=xs.device)
            self.lib.mmvq_id(self.type, parts["blocks"].view(-1, parts["blocks"].shape[-1]), xs.contiguous(),
                             ids.contiguous(), y, False, None, None, self.workspace, null_slots=self.null_slots)
            valid = (ids >= 0)[..., None]
            g, u = y[..., :self.out_y], y[..., self.out_y:]
            gate_max[id(self)] = max(gate_max[id(self)], float(torch.where(valid, g, -torch.inf).max()))
            up_max[id(self)] = max(up_max[id(self)], float(torch.where(valid, u.abs(), 0).max()))
        return call(self, x, parts, ids, **kw)

    def run(plant: str | None):
        e = build(a.model, a.flags, max_model_len=longest, max_seqs=1, prefix_cache=False, max_step_tokens=1)
        try:
            if plant:
                with PLANTS[plant]():
                    return score(e, prompts, 20), {}
            layers = {}
            for name, m in e.executor.model.named_modules():
                k = getattr(m, "_kernel", None)
                if isinstance(k, plugin.LinearExperts) and k.swiglu:
                    layers[id(k)] = int(name.split(".")[1])
            plugin.LinearExperts.__call__ = recorded
            try:
                return score(e, prompts, 20), layers
            finally:
                plugin.LinearExperts.__call__ = call
        finally:
            e.close()
    clean, layers = run(None)
    planted, _ = run("glu_clamp_off")
    limit = 10.0
    per = {layers[k]: {"gate_max": round(gate_max[k], 3), "up_abs_max": round(up_max[k], 3)} for k in layers}
    gates, ups = [v["gate_max"] for v in per.values()], [v["up_abs_max"] for v in per.values()]
    out = {"limit": limit, "positions": int(clean.offsets[-1]),
           "bitwise_without_the_clamp": clean.same_tensors(planted),
           "binds_anywhere": max(gates) > limit or max(ups) > limit, "worst_gate": max(gates), "worst_up": max(ups),
           "per_layer": dict(sorted(per.items()))}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_layer"}, indent=1))


if __name__ == "__main__":
    main()
