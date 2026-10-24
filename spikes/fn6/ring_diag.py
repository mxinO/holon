"""Does a prefill step lend and stream through the ring (FN step 6, §20.1)? The store's lending settings and slot
classes (cache slots incl. null and staging, pieces, cap), then per ring step of an 8K prompt (twice): `ring_ok`, the
error bits and each `lend` round's per-class needs. Usage: ring_diag.py <model> '<engine flags>'"""
from __future__ import annotations

import json
import sys
import time

import torch
from _engine import STORE, engine
from holon.api import SamplingParams
from holon_std.tiered import ring


def main() -> None:
    model, flags = sys.argv[1:3]
    seen = []
    begin = ring.Lent.begin

    def traced(self) -> None:
        begin(self)
        s = self.store
        seen.append({"mode": self.mode, "tail": self.tail, "ring_ok": int(s.ring_ok.item()),
                     "errors": int(s.errors.item()), "needs": s.lend_needs.tolist(), "n_before": int(s.n_before.sum())})

    ring.Lent.begin = traced
    e = engine(model, flags, max_seqs=1, max_model_len=16384)
    s = e.ctx.shared_objects[STORE]
    print("ring_tokens", s.ring_tokens, "lending", s.lending, "tail", s.tail, "prefill", s.prefill, "staging",
          s.staging, "ring_copy", s.ring_copy)
    print("classes", [(n, c.size, len(c.pieces), c.cap) for n, c in s.classes.items()])
    ids = json.loads(open("judge/prompts/v1-long.jsonl").readlines()[2])["ids"][:8192]
    for i in range(2):
        seen.clear()
        t0 = time.time()
        e.generate([ids[i:]], SamplingParams(temperature=0.0, max_tokens=1))
        torch.cuda.synchronize()
        print("prefill", round(time.time() - t0, 2), "s", "peak misses", s.peak_misses)
        for x in seen:
            print(" ", json.dumps(x))


if __name__ == "__main__":
    main()
