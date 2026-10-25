"""GLM's task-accuracy gate (docs/v1-glm.md acceptance 1(d), (e); §6.2): `glm_tasks.jsonl` through Holon and through
Project Maya, greedy, thinking on (the template's default), at most 16,384 new tokens, on the same GGUF, card and
prompt ids (GLM's chat template through `models.glm5_next`'s bundle), stopping at the model's end-of-turn ids; then
scored as FN's (`spikes/fn7/tasks/run.py`: the last `Final answer:` after the reasoning; number, text, letter, code;
degenerate = the limit hit inside the reasoning, no answer, or a 40-character span repeated 6 times).

- `generate --engine holon`: Holon's `Engine` in process, `--holon-flags` the deployment's (mode (i)), several
  sequences at once.
- `generate --engine maya`: Maya's engine in `--serve` mode (as `bench/offload_ctx.py` drives it: `GEN <n>
  temperature=0 <ids>`, `T <id>` per token, `DONE ...`), one request at a time, its end-of-turn ids given
  (`--eos-ids`); `--maya-flags` its step-0 default arm's, `--maya-env` that arm's environment.
- `score`: per engine and category; the bar, stated before the first run (§6.2): Holon within 5 items of Maya's
  score, and no Holon answer degenerate where Maya's is not, judged on pooled rates (the owner's ruling of
  2026-10-24: Holon-only degenerate items are variance; the clause is met when Holon's rate does not exceed Maya's);
  `--samples` a short Markdown dump of 8 chosen items' Holon answers (e).

    python spikes/glm5/tasks/run.py generate --engine holon --model <...-00001-of-00003.gguf> --holon-flags '...' \\
        --out <answers.jsonl>
    python spikes/glm5/tasks/run.py generate --engine maya --model <...> --maya <engine> --maya-flags '...' \\
        --maya-env 'K=V ...' --out <answers.jsonl>
    python spikes/glm5/tasks/run.py score --answers holon=<a.jsonl> maya=<b.jsonl> --out <json> --samples <md>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
TASKS = HERE / "glm_tasks.jsonl"
MAX_TOKENS, MAX_LEN, TAIL = 16384, 49152, 800
SAMPLES = ("math-04", "math-13", "code-08", "code-13", "reasoning-02", "reasoning-07", "recall-09", "multiturn-04")


def fn_run():
    """FN's runner (its scoring: `final`, `correct`, `degenerate`, unchanged)."""
    spec = importlib.util.spec_from_file_location("fn_run", HERE.parents[1] / "fn7" / "tasks" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def items() -> list[dict]:
    rows = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    return rows[1:]


def bundle_and_eos(model: str):
    from holon.core.assembly import resolve_model
    from holon.core.engine import EngineConfig
    from holon.core.plugins import PluginSet, discover
    source, plugin = resolve_model(PluginSet(discover()), EngineConfig(model))
    return plugin.bundle(source), tuple(sorted(source.eos_token_ids))


def holon(a, prompts: list[list[int]], eos: tuple[int, ...]) -> list[dict]:
    from holon.api import SamplingParams
    from holon.core.cli import layered
    from holon.core.engine import Engine, EngineConfig
    e = Engine(EngineConfig(a.model, **layered({"max_model_len": MAX_LEN, "max_seqs": 4},
                                               fn_run().flags(a.holon_flags))))
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop_token_ids=eos)
    index = {e.add(p, params): i for i, p in enumerate(prompts)}
    out = [{"tokens": [], "finish": None, "wall_s": None} for _ in prompts]
    t0 = time.perf_counter()
    while e.has_work():
        for d in e.step():
            o = out[index[d.rid]]
            o["tokens"] += d.token_ids
            if d.finish_reason is not None:
                o["finish"], o["wall_s"] = d.finish_reason, round(time.perf_counter() - t0, 1)
                print(sum(x["finish"] is not None for x in out), len(o["tokens"]), d.finish_reason, flush=True)
    e.close()
    return out


def maya(a, prompts: list[list[int]], eos: tuple[int, ...]) -> list[dict]:
    cmd = [a.maya, "--serve", "--max-context", str(MAX_LEN), "--kv-resident", str(MAX_LEN), "--eos-ids",
           ",".join(map(str, eos)), *shlex.split(a.maya_flags)]
    env = os.environ | dict(kv.split("=", 1) for kv in shlex.split(a.maya_env))
    engine = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                              bufsize=1, env=env, cwd=a.maya_cwd or None)
    out = []
    try:
        for line in engine.stdout:
            if line.startswith("READY"):
                break
        else:
            raise RuntimeError(f"maya exited with {engine.wait()} before READY")
        for ids in prompts:
            t0 = time.perf_counter()
            engine.stdin.write(f"GEN {MAX_TOKENS} temperature=0 {','.join(map(str, ids))}\n")
            engine.stdin.flush()
            tokens, finish = [], None
            for line in engine.stdout:
                if line.startswith("T "):
                    tokens.append(int(line.split()[1]))
                elif line.startswith("ERR"):
                    raise RuntimeError(line.strip())
                elif line.startswith("DONE "):
                    finish = line.split()[5]
                    break
            else:
                raise RuntimeError(f"maya exited with {engine.wait()} during a request")
            out.append({"tokens": tokens, "finish": finish, "wall_s": round(time.perf_counter() - t0, 1)})
            print(len(out), len(tokens), finish, flush=True)
    finally:
        try:
            engine.stdin.write("QUIT\n")
            engine.stdin.flush()
            engine.wait(300)
        except (OSError, subprocess.TimeoutExpired):
            engine.kill()
    return out


def generate(a) -> None:
    b, eos = bundle_and_eos(a.model)
    rows = [r for r in items() if not a.only or r["id"] in a.only]
    prompts = [b.render_chat(r["messages"]) for r in rows]
    if a.prompts_out:
        pathlib.Path(a.prompts_out).write_text(json.dumps({r["id"]: p for r, p in zip(rows, prompts, strict=True)}))
    t0 = time.perf_counter()
    runs = (holon if a.engine == "holon" else maya)(a, prompts, eos)
    total = round(time.perf_counter() - t0, 1)
    with open(a.out, "w") as f:
        for r, p, o in zip(rows, prompts, runs, strict=True):
            text = b.tokenizer.decode([t for t in o["tokens"] if t not in eos])
            finish = "length" if o["finish"] == "length" or len(o["tokens"]) >= MAX_TOKENS else o["finish"]
            f.write(json.dumps({"id": r["id"], "engine": a.engine, "prompt_tokens": len(p), "tokens": len(o["tokens"]),
                                "finish": finish, "wall_s": o["wall_s"], "run_wall_s": total, "text": text}) + "\n")


def score(a) -> None:
    fn = fn_run()
    rows = {r["id"]: r for r in items()}
    runs = {}
    for spec in a.answers:
        name, path = spec.split("=", 1)
        runs[name] = {x["id"]: x for x in map(json.loads, pathlib.Path(path).read_text().splitlines())}
    table, totals = {}, {}
    cats = sorted({r["category"] for r in rows.values()})
    for name, answers in runs.items():
        per = {}
        for i, r in rows.items():
            body, value = fn.final(answers[i]["text"])
            per[i] = {"correct": fn.correct(r, body, value), "value": value,
                      "degenerate": fn.degenerate(answers[i], body), "tokens": answers[i]["tokens"],
                      "finish": answers[i]["finish"]}
        table[name] = per
        some = next(iter(answers.values()))
        totals[name] = {"correct": sum(x["correct"] for x in per.values()), "of": len(per),
                        "degenerate": sum(x["degenerate"] for x in per.values()),
                        "tokens": sum(x["tokens"] for x in per.values()), "run_wall_s": some.get("run_wall_s"),
                        **{c: sum(per[i]["correct"] for i, r in rows.items() if r["category"] == c) for c in cats}}
    out = {"totals": totals, "items": table}
    if {"holon", "maya"} <= set(runs):
        h, m = totals["holon"], totals["maya"]
        only = [i for i in rows if table["holon"][i]["degenerate"] and not table["maya"][i]["degenerate"]]
        out["bar"] = {"tolerance_items": 5, "difference": h["correct"] - m["correct"],
                      "degenerate_rates": [h["degenerate"] / h["of"], m["degenerate"] / m["of"]],
                      "holon_only_degenerate": only,           # reported: variance, judged on the pooled rates
                      "pass": (h["correct"] >= m["correct"] - 5
                               and h["degenerate"] / h["of"] <= m["degenerate"] / m["of"])}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"totals": totals, "bar": out.get("bar")}, indent=1))
    if a.samples and "holon" in runs:
        lines = ["# GLM task samples: Holon's answers (greedy, thinking on)", "",
                 "Eight items from `spikes/glm5/tasks/glm_tasks.jsonl` (synthetic), answered by Holon on Maya-S "
                 "(`spikes/glm5/tasks/run.py`). Long prompts are cut in the middle; of the reasoning only its last "
                 f"{TAIL} characters are shown.", ""]
        for i in SAMPLES:
            r, ans = rows[i], runs["holon"][i]
            q = r["messages"][-1]["content"]
            q = q if len(q) < 1500 else q[:600] + f"\n[... {len(q) - 1200} characters of filler ...]\n" + q[-600:]
            think, done, rest = ans["text"].rpartition("</think>")
            think = think if done else ans["text"]
            answer = rest.strip() if done else "(none: the token limit was reached inside the reasoning)"
            lines += [f"## {i} ({r['category']}; expected: {r['answer'] if r['check'] != 'code' else 'tests pass'}; "
                      f"{'correct' if table['holon'][i]['correct'] else 'WRONG'}; {ans['tokens']} tokens)", "",
                      "**Question**", "", "~~~text", q, "~~~", "", "**Reasoning (end)**", "", "~~~text",
                      ("…" if len(think) > TAIL else "") + think[-TAIL:].strip(), "~~~", "", "**Answer**", "", answer,
                      ""]
        pathlib.Path(a.samples).write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--engine", choices=("holon", "maya"), required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--holon-flags", default="", help="engine flags over the defaults (the deployment's)")
    g.add_argument("--maya")
    g.add_argument("--maya-flags", default="")
    g.add_argument("--maya-env", default="", help="K=V ... for Maya's process")
    g.add_argument("--maya-cwd", default="", help="Maya's working directory (its data files)")
    g.add_argument("--prompts-out", help="write the rendered prompt ids, by item")
    g.add_argument("--only", nargs="*", help="these items only (a smoke run)")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--answers", nargs="+", required=True, help="name=answers.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--samples")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else score(a)


if __name__ == "__main__":
    main()
