"""FN step 7: per verify step, the missed experts fetched and the CPU's (F5b), over 120 tokens of a 1K v1-long prompt.
Usage: cpu_count.py <model> '<engine flags>'"""
import json
import sys

sys.path.insert(0, "spikes/fn6")
from _engine import STORE, engine
from holon.api import SamplingParams

model, flags = sys.argv[1], sys.argv[2]
e = engine(model, flags, max_seqs=1, max_model_len=2048)
s = e.ctx.shared_objects[STORE]
prev, tot = {"map": s.residency.host.copy()}, {"steps": 0, "missed": 0, "cpu": 0, "admitted": 0}
def tap(step, record):
    c = record["counts"]
    if c.sum() > 8 and c.max() <= 4:
        res = (prev["map"] >= 0).all(-1)
        cpu = record["cpu"]
        tot["steps"] += 1
        tot["missed"] += int(((c > 0) & ~res & ~cpu).sum())
        tot["cpu"] += int(cpu.sum())
        recs = record["records"]                        # (group, a, b): a admitted (b its victim, or an empty slot)
        tot["admitted"] += int(((recs[:, 1] >= 0) & (recs[:, 2] != -1)).sum()) if len(recs) else 0
    prev["map"] = s.residency.host.copy()
s.taps.append(tap)
rows = [json.loads(x) for x in open("judge/prompts/v1-long.jsonl") if x.strip()]
ids = max((r["ids"][:r["gen_start"]] for r in rows), key=len)[:1024]
e.generate([ids], SamplingParams(temperature=0.0, max_tokens=120, ignore_eos=True))
print(json.dumps({k: (v / tot["steps"] if k != "steps" else v) for k, v in tot.items()}))
