"""FN §22.9: gate (c)'s first pass alone (`consistency.py`'s `scored` at a reference step) — its four `v1-long` prompts
scored with prompt logprobs on a deployment's engine; how the RTX 5090's illegal address was bisected. Usage:

    python spikes/fn7/prefill_pass.py <model> '<engine flags>' <step> [--prompts N]"""
import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spikes/fn7"))
import consistency as c  # noqa: E402
from holon_judge.recording import PromptSet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("flags")
ap.add_argument("step", type=int)
ap.add_argument("--prompts", type=int, default=4)
x = ap.parse_args()
full = PromptSet.load(ROOT / "judge/prompts/v1-long.jsonl")
keep = {p.id: p for p in full.prompts}
prompts = PromptSet(full.name, full.sha256, [keep[p] for p in c.PROMPTS[:x.prompts]])
a = argparse.Namespace(model=x.model, pack="", holon_flags=x.flags,
                       max_len=max(20480, max(len(p.ids) for p in prompts.prompts) + 264))
print("prompt lengths", [len(p.ids) for p in prompts.prompts], flush=True)
t = time.time()
rec = c.scored(a, prompts, x.step)
print(f"scored {len(rec.target_lp)} positions in {time.time() - t:.0f} s", flush=True)
