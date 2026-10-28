"""FN's task-accuracy gate (docs/dev/v1-flash-next.md §18 (d), (e)): `fn_tasks.jsonl` through Holon and through Strata,
greedy, on the same GGUF and the same prompt ids (the chat template's rendering, thinking on), then scored.

- `generate --engine holon`: Holon's `Engine` in process, all-resident, 8 sequences, plain decoding.
- `generate --engine strata`: Strata's engine in `--serve` mode (as `bench/offload_ctx.py` drives it: `GEN <n> <ids>`,
  `T <id>` per token, `DONE ...`), one request at a time, stopping at the model's end-of-turn ids; its flags are the
  no-approximation arm's of §17 (`--strata-flags`).
- `score`: each answer's text after the reasoning (`</think>`), its last `Final answer:` line checked against the item:
  `number` (exact, commas ignored), `text` (exact, case-insensitive), `letter` (A–E), `code` (the last ```python
  block and the item's asserts run in a fresh interpreter, 20 s). An answer is flagged degenerate if it hits the
  token limit inside its reasoning, has no text after it, or repeats a 40-character span 6 times in a row. Writes the
  per-item table and `--samples`, a short Markdown dump of chosen items' answers (§18 (e)).

The bar, stated before the first run (§18): Holon scores within 5 items (6.9 points) of Strata on the 72, and no
Holon answer is degenerate where Strata's is not.

    python judge/tasks/run.py generate --engine holon --model <...-00001-of-00002.gguf> --out <answers.jsonl>
    python judge/tasks/run.py generate --engine strata --model <...> --strata <engine> --strata-flags '...' \\
        --out <answers.jsonl>
    python judge/tasks/run.py score --answers holon=<a.jsonl> strata=<b.jsonl> --out <json> --samples <md>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
TASKS = HERE / "fn_tasks.jsonl"
MAX_TOKENS, MAX_LEN, EOS, TAIL = 8192, 40960, (248044, 248046), 800
SAMPLES = ("math-04", "math-13", "code-08", "code-13", "reasoning-02", "reasoning-07", "recall-09", "multiturn-04")


def items() -> list[dict]:
    rows = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    return rows[1:]


def bundle(model: str):
    from holon.core.assembly import resolve_model
    from holon.core.engine import EngineConfig
    from holon.core.plugins import PluginSet, discover
    source, plugin = resolve_model(PluginSet(discover()), EngineConfig(model))
    return plugin.bundle(source)


def holon(a, prompts: list[list[int]]) -> list[dict]:
    from holon.api import SamplingParams
    from holon.core.cli import layered
    from holon.core.engine import Engine, EngineConfig
    e = Engine(EngineConfig(a.model, **layered({"max_model_len": MAX_LEN, "max_step_tokens": 8192, "max_seqs": 8},
                                               flags(a.holon_flags))))
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop_token_ids=EOS)
    index = {e.add(p, params): i for i, p in enumerate(prompts)}
    out = [{"tokens": [], "finish": None} for _ in prompts]
    t0 = time.perf_counter()
    while e.has_work():
        for d in e.step():
            o = out[index[d.rid]]
            o["tokens"] += d.token_ids
            o["finish"] = d.finish_reason or o["finish"]
    for o in out:
        o["wall_s"] = round(time.perf_counter() - t0, 1)
    e.close()
    return out


def flags(text: str) -> dict:
    """`holon serve`-style engine flags (e.g. an offload deployment's) as EngineConfig settings: only those given."""
    import argparse
    import shlex

    from holon.core.cli import engine_flags, engine_settings
    parser = argparse.ArgumentParser()
    engine_flags(parser, given_only=True)
    return engine_settings(parser.parse_args(shlex.split(text)))


def strata(a, prompts: list[list[int]]) -> list[dict]:
    ple = a.model.replace("-00001-of-00002.gguf", "-00002-of-00002.gguf")
    cmd = [a.strata, "--serve", "--native", a.model, "--ple-gguf", ple, "--max-context", str(MAX_LEN), "--eos-ids",
           ",".join(map(str, EOS)), "--prompt-cache", "0", *shlex.split(a.strata_flags)]
    engine = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                              bufsize=1)
    out = []
    try:
        for line in engine.stdout:
            if line.startswith("READY"):
                break
        else:
            raise RuntimeError(f"strata exited with {engine.wait()} before READY")
        for ids in prompts:
            t0 = time.perf_counter()
            engine.stdin.write(f"GEN {MAX_TOKENS} {','.join(map(str, ids))}\n")
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
            out.append({"tokens": tokens, "finish": finish, "wall_s": round(time.perf_counter() - t0, 1)})
            print(len(out), len(tokens), finish, flush=True)
    finally:
        engine.stdin.write("QUIT\n")
        engine.stdin.flush()
        engine.wait(120)
    return out


def generate(a) -> None:
    b = bundle(a.model)
    rows = items()
    prompts = [b.render_chat(r["messages"]) for r in rows]
    runs = (holon if a.engine == "holon" else strata)(a, prompts)
    with open(a.out, "w") as f:
        for r, p, o in zip(rows, prompts, runs, strict=True):
            text = b.tokenizer.decode([t for t in o["tokens"] if t not in EOS])
            f.write(json.dumps({"id": r["id"], "engine": a.engine, "prompt_tokens": len(p), "tokens": len(o["tokens"]),
                                "finish": o["finish"], "wall_s": o["wall_s"], "text": text}) + "\n")


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
    rows = {r["id"]: r for r in items()}
    runs = {}
    for spec in a.answers:
        name, path = spec.split("=", 1)
        runs[name] = {x["id"]: x for x in map(json.loads, pathlib.Path(path).read_text().splitlines())}
    table, totals = {}, {}
    for name, answers in runs.items():
        per = {}
        for i, r in rows.items():
            body, value = final(answers[i]["text"])
            per[i] = {"correct": correct(r, body, value), "value": value, "degenerate": degenerate(answers[i], body),
                      "tokens": answers[i]["tokens"], "finish": answers[i]["finish"]}
        table[name] = per
        cats = sorted({r["category"] for r in rows.values()})
        totals[name] = {"correct": sum(x["correct"] for x in per.values()), "of": len(per),
                        "degenerate": sum(x["degenerate"] for x in per.values()),
                        **{c: sum(per[i]["correct"] for i, r in rows.items() if r["category"] == c) for c in cats}}
    out = {"totals": totals, "items": table}
    if {"holon", "strata"} <= set(runs):
        h, s = totals["holon"], totals["strata"]
        only = [i for i in rows if table["holon"][i]["degenerate"] and not table["strata"][i]["degenerate"]]
        out["bar"] = {"tolerance_items": 5, "difference": h["correct"] - s["correct"], "holon_only_degenerate": only,
                      "pass": h["correct"] >= s["correct"] - 5 and not only}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"totals": totals, "bar": out.get("bar")}, indent=1))
    if a.samples and "holon" in runs:
        lines = ["# FN task samples: Holon's answers (greedy, thinking on)", "",
                 "Eight items from `judge/tasks/fn_tasks.jsonl` (synthetic), answered by Holon on the IQ2_XS file "
                 "(`judge/tasks/run.py`). Long prompts are cut in the middle; of the reasoning only its last "
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
    g.add_argument("--engine", choices=("holon", "strata"), required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--strata")
    g.add_argument("--strata-flags", default="")
    g.add_argument("--holon-flags", default="", help="engine flags over the defaults (an offload deployment's)")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--answers", nargs="+", required=True, help="name=answers.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--samples")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else score(a)


if __name__ == "__main__":
    main()
