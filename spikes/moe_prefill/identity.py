"""The prefill identity check of docs/v1-moe-prefill.md §5 (D0 / D1 / D2 / D4: bitwise): the prompt logprobs of every
prompt position of one-token requests at 1,027 and 4,096 tokens, hashed, for the engine this Python imports — run it
once with main's tree and once with the branch's (the same script, `PYTHONPATH` naming the tree), on the same GPU and
flags, and compare the hashes (`compare`). Modelled on `holon_judge.kernel_identity`, end to end.

- Configurations (`identity_configs.json`): the 5090 lock's resident flags, the 5080's natural-fit offload lock, and
  the capped fit with F5b's CPU share (x8_close's arm).
- Prompts: natural text (`bench/offload_ctx.py`'s corpus) at the lengths given, distinct starts; prefix cache off,
  one sequence, the context `--ctx`.
- The record names each expert store's pass at the longest length (`pass_tokens`).
- The hash: SHA-256 over each position's chosen-token logprob, top-k ids and top-k logprobs (the f32 values, exactly,
  as float64), in order; `--runs` repeats the requests in one engine (run-identical) and the record keeps every run's.

    python spikes/moe_prefill/identity.py run --model <gguf> [--config resident|natural|capped_f5b |
        --flags '<engine flags>'] [--lengths 1027 4096] [--logprobs 5] [--runs 2] [--ctx 8192] --json <out>
    python spikes/moe_prefill/identity.py compare <main.json> <branch.json>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import shlex
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402  (the corpus)
import workloads  # noqa: E402

STARTS = {1027: 3001, 4096: 13001}                 # corpus offsets per length (others: 23001)


def digest(lp) -> str:
    h = hashlib.sha256()
    for token, ids, top in zip(lp.token, lp.top_ids, lp.top, strict=True):
        h.update(struct.pack(f"<d{len(ids)}q{len(top)}d", token, *ids, *top))
    return h.hexdigest()


def tree(root: pathlib.Path) -> str:
    """The engine tree's commit, `+` when its tracked files differ from it."""
    git = ["git", "-C", str(root)]
    sha = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run([*git, "diff", "--quiet", "HEAD"]).returncode != 0
    return sha + "+" * dirty


def run(a: argparse.Namespace) -> None:
    import holon
    import torch
    from holon.api import SamplingParams
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    if a.config:
        a.flags = json.loads((pathlib.Path(__file__).parent / "identity_configs.json").read_text())[a.config]
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.flags)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=1, max_model_len=a.ctx, prefix_cache=False))
    corpus = workloads.corpus(offload_ctx.CORPUS)
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=a.logprobs)
    runs = []
    for _ in range(a.runs):
        hashes = {}
        for n in a.lengths:
            start = STARTS.get(n, 23001)
            rid = engine.add([corpus[(start + i) % len(corpus)] for i in range(n)], params)
            got = None
            while engine.has_work():
                got = next((d.prompt_logprobs for d in engine.step() if d.rid == rid and d.prompt_logprobs), got)
            assert got is not None and len(got.token) == n - 1, n
            hashes[str(n)] = digest(got)
        runs.append(hashes)
    stores = {type(m._store).__name__: m._store.pass_tokens(max(a.lengths)) for m in engine.executor.model.modules()
              if hasattr(m, "_store") and hasattr(m._store, "pass_tokens")}
    engine.close()
    record = {"expert_pass": stores, "gpu": torch.cuda.get_device_name(),
              "tree": tree(pathlib.Path(holon.__file__).resolve().parents[2]), "model": pathlib.Path(a.model).name,
              "config": a.config, "flags": a.flags, "lengths": a.lengths, "logprobs": a.logprobs, "runs": runs,
              "run_identical": all(r == runs[0] for r in runs)}
    print(json.dumps(record))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(record, indent=1) + "\n")


def compare(a: argparse.Namespace) -> int:
    main, branch = (json.loads(pathlib.Path(p).read_text()) for p in (a.main, a.branch))
    same = main["runs"][0] == branch["runs"][0] and main["run_identical"] and branch["run_identical"]
    print(json.dumps({"identical": same, "main": main["runs"][0], "branch": branch["runs"][0]}))
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--flags", default="")
    r.add_argument("--config", help="a configuration of identity_configs.json (in place of --flags)")
    r.add_argument("--lengths", type=int, nargs="+", default=[1027, 4096])
    r.add_argument("--logprobs", type=int, default=5)
    r.add_argument("--runs", type=int, default=2)
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--json")
    c = sub.add_parser("compare")
    c.add_argument("main")
    c.add_argument("branch")
    a = ap.parse_args()
    if a.cmd == "compare":
        return compare(a)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
