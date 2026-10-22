"""Two deployments' scores, bitwise (step 7's levers that move no bits): a prompt's target and top-k logprobs in one
prefill step and in 1-token decode steps (graphed), under the base flags plus each side's plugin options. Usage:
bitwise.py <model> '<engine flags>' '<k=v ...>' '<k=v ...>' [prompt tokens]"""
from __future__ import annotations

import json
import sys

import torch
from holon_judge.engine import score
from holon_judge.recording import Prompt, PromptSet

sys.path.insert(0, "spikes/fn6")
from _engine import engine  # noqa: E402


def main() -> None:
    model, flags, a, b = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    ids = json.loads(open("judge/prompts/v1-long.jsonl").readlines()[2])["ids"][:n]
    prompts = PromptSet("p", "", [Prompt("p", "c", ids)])
    out = {}
    for step in (n, 1):
        recs = []
        for side in (a, b):
            opts = dict(x.split("=", 1) for x in side.split())
            e = engine(model, flags, max_seqs=1, max_model_len=n + 8, max_step_tokens=step, plugin_options=opts)
            recs.append(score(e, prompts, 20))
            e.close()
            del e
            torch.cuda.empty_cache()
        out[f"step_{step}"] = [torch.equal(recs[0].target_lp, recs[1].target_lp),
                               torch.equal(recs[0].topk_lp, recs[1].topk_lp)]
        print(step, out[f"step_{step}"], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
