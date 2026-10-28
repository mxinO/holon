"""The task-accuracy gate (FN: docs/dev/v1-flash-next.md §18 (d), (e); GLM: docs/dev/v1-glm.md acceptance 1(d), (e),
§6.2): a task set (`build.py`'s: `fn_tasks.jsonl`, `glm_tasks.jsonl`) through Holon and through a reference engine,
greedy, on the same GGUF and the same prompt ids (the model's chat template through its bundle, thinking on: the
template's default), stopping at the model's end-of-turn ids, at most `--max-tokens` new tokens; then scored.

- `generate --engine holon`: Holon's `Engine` in process, `--holon-flags` the deployment's, several sequences at once
  (8 unless the flags say otherwise).
- `generate --engine strata|maya`: the reference engine (`--ref`, its binary) in `--serve` mode (as
  `bench/offload_ctx.py` drives it: `GEN <n> [temperature=0] <ids>`, `T <id>` per token, `DONE ...`), one request at
  a time, the end-of-turn ids given (`--eos-ids`); `--ref-flags` the arm's (FN: §17's no-approximation arm; GLM: Maya's
  step-0 default arm), `--ref-env` its environment, `--ref-cwd` its working directory.
- `score`: each answer's text after the reasoning (`</think>`), its last `Final answer:` line checked against the item:
  `number` (exact, commas ignored), `text` (exact, case-insensitive), `letter` (A–E), `code` (the last ```python
  block and the item's asserts run in a fresh interpreter, 20 s). An answer is flagged degenerate if it hits the
  token limit inside its reasoning, has no text after it, or repeats a 40-character span 6 times in a row. Writes the
  per-item table and `--samples`, a short Markdown dump of chosen items' Holon answers (e).

The bar, stated before each gate's first run: Holon scores within 5 items of the reference on the 72, and no Holon
answer is degenerate where the reference's is not (`--degenerate items`, FN's §18) — or, GLM's (the owner's ruling of
2026-10-24: Holon-only degenerate items are variance), Holon's degenerate rate does not exceed the reference's
(`--degenerate rates`).

    python judge/tasks/run.py generate --engine holon --model <gguf> [--tasks <set.jsonl>] --holon-flags '...' \\
        --out <answers.jsonl>
    python judge/tasks/run.py generate --engine strata --model <gguf> --ref <engine> --ref-flags '...' \\
        --out <answers.jsonl>
    python judge/tasks/run.py score [--tasks <set.jsonl>] --answers holon=<a.jsonl> strata=<b.jsonl> --out <json> \\
        --samples <md> [--degenerate items|rates]

GLM's: `--tasks judge/tasks/glm_tasks.jsonl --max-tokens 16384 --max-len 49152`, Holon with `--max-seqs 4`, the
reference `--engine maya`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
TAIL = 800
SAMPLES = ("math-04", "math-13", "code-08", "code-13", "reasoning-02", "reasoning-07", "recall-09", "multiturn-04")


def items(tasks: str) -> list[dict]:
    rows = [json.loads(line) for line in pathlib.Path(tasks).read_text().splitlines() if line.strip()]
    return rows[1:]


def bundle_and_eos(model: str):
    from holon.core.assembly import resolve_model
    from holon.core.engine import EngineConfig
    from holon.core.plugins import PluginSet, discover
    source, plugin = resolve_model(PluginSet(discover()), EngineConfig(model))
    return plugin.bundle(source), tuple(sorted(source.eos_token_ids))


def holon(a, prompts: list[list[int]], eos: tuple[int, ...]) -> list[dict]:
    from holon.api import SamplingParams
    from holon_judge.engine import from_flags
    e = from_flags(a.model, a.holon_flags, {"max_model_len": a.max_len, "max_step_tokens": 8192, "max_seqs": 8})
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_token_ids=eos)
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


def reference(a, prompts: list[list[int]], eos: tuple[int, ...]) -> list[dict]:
    """The reference engine's answers, one request at a time over its `--serve` protocol."""
    ids = ",".join(map(str, eos))
    if a.engine == "strata":       # the model and its PLE file given here; its prompt cache off
        ple = a.model.replace("-00001-of-00002.gguf", "-00002-of-00002.gguf")
        cmd, gen = [a.ref, "--serve", "--native", a.model, "--ple-gguf", ple, "--max-context", str(a.max_len),
                    "--eos-ids", ids, "--prompt-cache", "0"], ""
    else:                          # maya: its model in its flags; the KV resident for the longest request
        cmd, gen = [a.ref, "--serve", "--max-context", str(a.max_len), "--kv-resident", str(a.max_len), "--eos-ids",
                    ids], "temperature=0 "
    env = os.environ | dict(kv.split("=", 1) for kv in shlex.split(a.ref_env))
    engine = subprocess.Popen([*cmd, *shlex.split(a.ref_flags)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=sys.stderr, text=True, bufsize=1, env=env, cwd=a.ref_cwd or None)
    out = []
    try:
        for line in engine.stdout:
            if line.startswith("READY"):
                break
        else:
            raise RuntimeError(f"{a.engine} exited with {engine.wait()} before READY")
        for p in prompts:
            t0 = time.perf_counter()
            engine.stdin.write(f"GEN {a.max_tokens} {gen}{','.join(map(str, p))}\n")
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
                raise RuntimeError(f"{a.engine} exited with {engine.wait()} during a request")
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
    rows = [r for r in items(a.tasks) if not a.only or r["id"] in a.only]
    prompts = [b.render_chat(r["messages"]) for r in rows]
    if a.prompts_out:
        pathlib.Path(a.prompts_out).write_text(json.dumps({r["id"]: p for r, p in zip(rows, prompts, strict=True)}))
    t0 = time.perf_counter()
    runs = (holon if a.engine == "holon" else reference)(a, prompts, eos)
    total = round(time.perf_counter() - t0, 1)
    with open(a.out, "w") as f:
        for r, p, o in zip(rows, prompts, runs, strict=True):
            text = b.tokenizer.decode([t for t in o["tokens"] if t not in eos])
            finish = "length" if o["finish"] == "length" or len(o["tokens"]) >= a.max_tokens else o["finish"]
            f.write(json.dumps({"id": r["id"], "engine": a.engine, "model": pathlib.Path(a.model).name,
                                "prompt_tokens": len(p), "tokens": len(o["tokens"]), "finish": finish,
                                "wall_s": o["wall_s"], "run_wall_s": total, "text": text}) + "\n")


def final(text: str) -> tuple[str, str | None]:
    """The answer after the reasoning, and its last `Final answer:` value."""
    body = text.split("</think>")[-1].strip()
    found = re.findall(r"Final answer:\s*\**\s*(.+)", body)
    return body, (found[-1].strip().strip("*`$. ").strip() if found else None)


def run_code(body: str, r: dict) -> bool:
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    if not blocks:
        return False
    src = blocks[-1] + "\n" + "\n".join(f"assert {t}" for t in r["tests"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "t.py").write_text(src)
        try:
            done = subprocess.run([sys.executable, "-I", "t.py"], cwd=d, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False
        return done.returncode == 0


def correct(r: dict, body: str, value: str | None) -> bool:
    if r["check"] == "code":
        return run_code(body, r)
    if value is None:
        return False
    if r["check"] == "number":
        m = re.search(r"-?\d[\d,]*", value)
        return m is not None and m.group().replace(",", "") == str(r["answer"])
    if r["check"] == "letter":
        m = re.search(r"\b([A-E])\b", value)
        return m is not None and m.group(1) == r["answer"]
    return value.lower() == str(r["answer"]).lower()


def degenerate(ans: dict, body: str) -> bool:
    text = ans["text"]
    looping = re.search(r"(.{40,}?)\1{5,}", text, re.S) is not None
    return (ans["finish"] == "length" and "</think>" not in text) or not body or looping


def score(a) -> None:
    rows = {r["id"]: r for r in items(a.tasks)}
    runs = {}
    for spec in a.answers:
        name, path = spec.split("=", 1)
        runs[name] = {x["id"]: x for x in map(json.loads, pathlib.Path(path).read_text().splitlines())}
    table, totals = {}, {}
    cats = sorted({r["category"] for r in rows.values()})
    for name, answers in runs.items():
        per = {}
        for i, r in rows.items():
            body, value = final(answers[i]["text"])
            per[i] = {"correct": correct(r, body, value), "value": value, "degenerate": degenerate(answers[i], body),
                      "tokens": answers[i]["tokens"], "finish": answers[i]["finish"]}
        table[name] = per
        some = next(iter(answers.values()))
        totals[name] = {"correct": sum(x["correct"] for x in per.values()), "of": len(per),
                        "degenerate": sum(x["degenerate"] for x in per.values()),
                        "tokens": sum(x["tokens"] for x in per.values()), "run_wall_s": some.get("run_wall_s"),
                        **{c: sum(per[i]["correct"] for i, r in rows.items() if r["category"] == c) for c in cats}}
    out = {"totals": totals, "items": table}
    refs = [n for n in runs if n != "holon"]
    if "holon" in runs and len(refs) == 1:
        h, m = totals["holon"], totals[refs[0]]
        only = [i for i in rows if table["holon"][i]["degenerate"] and not table[refs[0]][i]["degenerate"]]
        rates = [h["degenerate"] / h["of"], m["degenerate"] / m["of"]]
        out["bar"] = {"reference": refs[0], "tolerance_items": 5, "difference": h["correct"] - m["correct"],
                      "degenerate": a.degenerate, "degenerate_rates": rates, "holon_only_degenerate": only,
                      "pass": h["correct"] >= m["correct"] - 5
                      and (not only if a.degenerate == "items" else rates[0] <= rates[1])}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"totals": totals, "bar": out.get("bar")}, indent=1))
    if a.samples and "holon" in runs:
        model = next(iter(runs["holon"].values())).get("model")
        model = f"`{model}`" if model else "the model"
        lines = [f"# Task samples ({pathlib.Path(a.tasks).stem}): Holon's answers (greedy, thinking on)", "",
                 f"Eight items from `judge/tasks/{pathlib.Path(a.tasks).name}` (synthetic), answered by Holon on "
                 f"{model} (`judge/tasks/run.py`). Long prompts are cut in the middle; of the reasoning only its "
                 f"last {TAIL} characters are shown.", ""]
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
    g.add_argument("--engine", choices=("holon", "strata", "maya"), required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--max-tokens", type=int, default=8192, help="new tokens a request")
    g.add_argument("--max-len", type=int, default=40960, help="the context (Holon's and the reference's)")
    g.add_argument("--holon-flags", default="", help="engine flags over the defaults (the deployment's)")
    g.add_argument("--ref", "--strata", "--maya", dest="ref", help="the reference engine's binary")
    g.add_argument("--ref-flags", "--strata-flags", "--maya-flags", dest="ref_flags", default="")
    g.add_argument("--ref-env", "--maya-env", dest="ref_env", default="", help="K=V ... for the reference's process")
    g.add_argument("--ref-cwd", "--maya-cwd", dest="ref_cwd", default="", help="its working directory (data files)")
    g.add_argument("--prompts-out", help="write the rendered prompt ids, by item")
    g.add_argument("--only", nargs="*", help="these items only (a smoke run)")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--answers", nargs="+", required=True, help="name=answers.jsonl")
    s.add_argument("--degenerate", choices=("items", "rates"), default="items",
                   help="the bar's degenerate clause: no Holon-only item, or Holon's rate within the reference's")
    s.add_argument("--out", required=True)
    s.add_argument("--samples")
    for p in (g, s):
        p.add_argument("--tasks", default=str(HERE / "fn_tasks.jsonl"), help="the task set (build.py's)")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else score(a)


if __name__ == "__main__":
    main()
