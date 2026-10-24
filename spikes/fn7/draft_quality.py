"""FN's speed round E, the draft-quality check: Holon's MTP head and Strata's on identical contexts. Strata serves the
bench plan's prompts greedily (`--serve`, its MTP arm's flags), each `--tokens` long, reporting its drafts accepted
and tested per depth; then Holon decodes the same prompts forced along Strata's tokens (every token Holon emits is
Strata's: the sampler's draw replaced by it, so every verify pass judges Holon's drafts on Strata's context), its
drafts' acceptance counted per depth. Per prompt and pooled: accepted / tested at depths 1–3, and tokens per pass.
Usage: draft_quality.py <model> '<holon MTP flags>' --strata <engine> --strata-flags '<flags>' [--strata-env K=V ...]
       --requests <requests.json> [--ctx 1024] [--prompts 8] [--tokens 256] [--max-context 33792]"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
NO_EOS = 0x7FFFFFFF


def strata(a, prompts: list[list[int]], log: pathlib.Path) -> list[dict]:
    """Strata's tokens and per-depth acceptance for each prompt (after the plan's warm-up)."""
    ple = a.model.replace("-00001-of-00002.gguf", "-00002-of-00002.gguf")
    cmd = [a.strata, "--serve", "--native", a.model, "--ple-gguf", ple, "--max-context", str(a.max_context),
           "--eos-ids", str(NO_EOS), "--prompt-cache", "0", *shlex.split(a.strata_flags)]
    env = dict(os.environ, STRATA_SPEC_DEPTH="1", **dict(kv.split("=", 1) for kv in a.strata_env))
    with open(log, "w") as err:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                             env=env)
        for line in p.stdout:
            if line.startswith("READY"):
                break

        def gen(ids: list[int], n: int) -> list[int]:
            p.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n")
            p.stdin.flush()
            out = []
            for line in p.stdout:
                if line.startswith("T "):
                    out += [int(x) for x in line.split()[1:]]
                elif line.startswith(("DONE", "ERR")):
                    break
            return out
        gen(json.loads(pathlib.Path(a.requests).read_text())["warmup"]["ids"], 64)
        tokens = [gen(ids, a.tokens) for ids in prompts]
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(120)
    depth = [[tuple(map(int, d.split("/"))) for d in m.group(1).split()]
             for m in re.finditer(r"spec depth: accepted/tested ((?:\d+/\d+ ?)+)", log.read_text())][1:]   # (warm-up)
    return [{"tokens": t, "depth": d[:3]} for t, d in zip(tokens, depth, strict=True)]


def holon(a, prompts: list[list[int]], forced_tokens: list[list[int]]) -> list[dict]:
    """Holon's per-depth acceptance on each prompt, its sequence forced along `forced_tokens`."""
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import engine
    from holon.api import SamplingParams
    from holon_judge.bounded import _generate
    from holon_std.sampling import sampler

    forced = torch.zeros(a.max_context + 8, dtype=torch.int64, device="cuda")
    sample = sampler.Sampler.sample

    def sample_forced(self, logits, ctx):
        out = sample(self, logits, ctx)
        dev = ctx.batch.dev
        at = (dev.positions[dev.logits_index].long() + 1).clamp(max=len(forced) - 1)
        out.tokens.copy_(forced[at])                    # the token Strata emitted there
        return out
    sampler.Sampler.sample = sample_forced
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=a.max_context)
    e.generate([json.loads(pathlib.Path(a.requests).read_text())["warmup"]["ids"]],
               SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))

    def rows() -> dict[int, int]:
        return {int(k.rsplit("_", 1)[1]): n for k, n in e.stats().items() if k.startswith("spec_rows_k3_accepting_")}
    out = []
    for ids, toks in zip(prompts, forced_tokens, strict=True):
        seq = ids + toks
        forced.zero_()
        forced[: len(seq)] = torch.tensor(seq)
        before = rows()
        tokens = _generate(e, [ids], SamplingParams(temperature=0.0, max_tokens=len(toks), ignore_eos=True))[0][0]
        after = rows()
        by = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        depth = [(sum(n for k, n in by.items() if k >= d), sum(n for k, n in by.items() if k >= d - 1))
                 for d in (1, 2, 3)]
        out.append({"followed": tokens == toks, "depth": depth})
    e.close()
    sampler.Sampler.sample = sample
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--strata", required=True)
    ap.add_argument("--strata-flags", required=True)
    ap.add_argument("--strata-env", nargs="*", default=[])
    ap.add_argument("--requests", required=True)
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=33792)
    ap.add_argument("--log", default="draft_quality_strata.log")
    a = ap.parse_args()
    plan = json.loads(pathlib.Path(a.requests).read_text())
    prompts = [r["ids"] for r in plan["requests"] if r["key"].split("/")[0] == a.ctx][: a.prompts]
    s = strata(a, prompts, pathlib.Path(a.log))
    h = holon(a, prompts, [r["tokens"] for r in s])
    rate = lambda ds, d: round(sum(x[d][0] for x in ds) / max(sum(x[d][1] for x in ds), 1), 3)   # noqa: E731
    per = [{"strata": [rate([x["depth"]], d) for d in range(3)], "holon": [rate([y["depth"]], d) for d in range(3)],
            "followed": y["followed"]} for x, y in zip(s, h, strict=True)]
    pooled = {"strata": [rate([x["depth"] for x in s], d) for d in range(3)],
              "holon": [rate([y["depth"] for y in h], d) for d in range(3)]}
    print(json.dumps({"ctx": a.ctx, "prompts": len(prompts), "tokens": a.tokens, "per_prompt": per,
                      "pooled": pooled}))


if __name__ == "__main__":
    main()
