"""A prefill streamed through the per-class ring scores bitwise as one staged throughout (`moe.ring_tokens=0`): an 8K
prompt's target and top-k logprobs. Usage: ring_exact.py <model> '<engine flags>'"""
from __future__ import annotations

import json
import sys

import torch
from _engine import STORE, engine
from holon_judge.engine import score
from holon_judge.recording import Prompt, PromptSet


def main() -> None:
    model, flags = sys.argv[1:3]
    ids = json.loads(open("judge/prompts/v1-long.jsonl").readlines()[2])["ids"][:8192]
    recs = []
    for extra in ({}, {"moe.ring_tokens": "0"}):
        e = engine(model, flags, max_seqs=1, max_model_len=8200, plugin_options=extra)
        recs.append(score(e, PromptSet("p", "", [Prompt("p", "c", ids)]), 20))
        print(extra, e.ctx.shared_objects[STORE].borrowed, flush=True)
        e.close()
        del e
        torch.cuda.empty_cache()
    print("bitwise", torch.equal(recs[0].target_lp, recs[1].target_lp), torch.equal(recs[0].topk_lp, recs[1].topk_lp))


if __name__ == "__main__":
    main()
