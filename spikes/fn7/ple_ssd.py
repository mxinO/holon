"""FN acceptance 4, the PLE table's RAM and SSD tiers byte-identical (docs/v1-flash-next.md D5): the deployment's
flags with `rows.place` ple:host, then ple:ssd (each its own engine), greedy over the bench plan's warm-up and its
first `--prompts` 1K prompts, `--tokens` decode tokens each (graphed steps; MTP where the flags draft), top-`K`
logprobs at every generated position: the token ids and the logprobs' bits must agree. Also each request's tok/s
(its wall over prefill and decode: the SSD reads' cost), and with `--plant` the `rows_off_by_one` plant on the SSD
run (it must differ).
Usage: ple_ssd.py <model> '<engine flags>' --requests <requests.json> [--prompts 2] [--tokens 128] [--plant]"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
import time
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
K = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--plant", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import engine
    from holon.api import SamplingParams
    from holon_judge.bounded import _generate
    from holon_std.tiered import rows

    plan = json.loads(pathlib.Path(a.requests).read_text())
    prompts = [plan["warmup"]["ids"]] + [r["ids"] for r in plan["requests"] if r["key"].startswith("1024/")][:a.prompts]
    params = SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True, logprobs=K)
    if "rows.place=" not in a.flags:
        raise SystemExit("the flags name rows.place (ple:host is replaced by each run's place)")
    runs, out = {}, {}
    for name, place, plant in (("host", "ple:host", 0), ("ssd", "ple:ssd", 0)) + ((("plant", "ple:ssd", 1),)
                                                                                      if a.plant else ()):
        with mock.patch.object(rows._Reads, "plant", plant) if plant else contextlib.nullcontext():
            e = engine(a.model, a.flags.replace("ple:host", place), max_seqs=1, max_model_len=4096,
                       prefix_cache=False)
            got, tps = [], []
            for ids in prompts:
                t0 = time.perf_counter()
                tokens, lps = _generate(e, [ids], params)[0]
                tps.append(a.tokens / (time.perf_counter() - t0))
                got.append((tokens, [(_list(i), _list(x)) for lp in lps for i, x in zip(lp.top_ids, lp.top,
                                                                                           strict=True)]))
            e.close()
        runs[name] = got
        out[name] = {"tok_s": [round(x, 2) for x in tps]}
    out["identical"] = runs["host"] == runs["ssd"]
    if a.plant:
        out["plant_differs"] = runs["plant"] != runs["host"]
    print(json.dumps(out))


def _list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


if __name__ == "__main__":
    main()
