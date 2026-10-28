"""GLM step 0, N2 (docs/dev/v1-glm.md §8): drive Project Maya's `--serve` engine over a request set so that its decode
route log (`STRATA_GLM_ROUTE_LOG=<prefix>`, in the environment) records a trace for `maya_routes.py`. Greedy, a
fixed number of new tokens per request (`--eos-ids 2147483647`), one conversation; Maya's own flags follow `--`.

    STRATA_GLM_ROUTE_LOG=<prefix> python spikes/glm0/maya_capture.py --maya <engine> --requests <json> \\
        --tokens 384 [--contexts 1024 4096] --out <timings.json> -- --glm-pack <dir> --max-context 33792 ...

`--requests`: bench/offload_ctx.py's requests.json or the same form ({"warmup": {...}, "requests": [{"key", "ids"}]}).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

NO_EOS = 2147483647


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maya", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--contexts", type=int, nargs="*", help="keep only requests of these contexts (offload_ctx plans)")
    ap.add_argument("--out", required=True)
    ap.add_argument("flags", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    plan = json.load(open(a.requests))
    reqs = [r for r in plan["requests"] if not a.contexts or r.get("context") in a.contexts]
    flags = a.flags[1:] if a.flags[:1] == ["--"] else a.flags
    engine = subprocess.Popen([a.maya, "--serve", "--eos-ids", str(NO_EOS), *flags], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in engine.stdout:
        if line.startswith("READY"):
            break
    else:
        sys.exit(f"maya exited with {engine.wait()} before READY")

    def gen(ids: list[int], n: int) -> dict:
        t0 = time.perf_counter()
        engine.stdin.write(f"GEN {n} temperature=0 {','.join(map(str, ids))}\n")
        engine.stdin.flush()
        for line in engine.stdout:
            if line.startswith("ERR"):
                sys.exit(line.strip())
            if line.startswith("DONE "):
                f = line.split()
                return {"produced": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]),
                        "decode_ms": float(f[4]), "reused": int(f[8]), "wall_s": time.perf_counter() - t0}
        sys.exit(f"maya exited with {engine.wait()} during a request")

    out = {"warmup": gen(plan["warmup"]["ids"], plan["warmup"]["output"]), "requests": {}}
    for r in reqs:
        out["requests"][r["key"]] = gen(r["ids"], a.tokens)
        print(r["key"], json.dumps(out["requests"][r["key"]]), flush=True)
    engine.stdin.write("QUIT\n")
    engine.stdin.flush()
    engine.wait(300)
    json.dump(out | {"tokens": a.tokens, "flags": flags}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
