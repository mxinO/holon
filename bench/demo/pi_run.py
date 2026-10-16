"""A real-agent test of `holon serve` with the pi coding agent (report-only, not a release gate): pi at a pinned
version, headless (`pi --mode json`), through a custom OpenAI-compatible provider on the server's `/v1`, with
`bench/capture_proxy.py --metrics` between them. Sessions run one at a time, so each request's prefix tokens saved
are the server's counter across it.

Tasks:
- `demo`: D7's task and prompt (`run.py`), scored as D7: the harness's own tests against the agent's `shop/`.
- `workloads`: a copy of Holon's own stdlib-only workload tools (`bench/workloads.py`, `bench/driver.py`, three
  traces) with a seeded suite (`workloads_task/tests`, 9 tests, passing). The agent adds `trace_stats` and a
  `--stats` option, with tests. Success, in fresh directories under the harness's interpreter as D7 scores: the
  agent's modules pass the seeded tests (restored) and the harness's own (`workloads_task/hidden_test_trace_stats.py`,
  never shown to the agent); and the agent's own suite passes, with more tests than the seeded ones and one naming
  `trace_stats`.

pi runs with only the environment it needs (no API keys or proxies; a fresh HOME and config directory: no settings,
credentials, skills or context files), offline (`PI_OFFLINE`: no update check, no telemetry), its default tools
(read, bash, edit, write) and a dummy key. Its provider (`models.json`, recorded) is the served model with
`reasoning: true` and the served context; `--compat documented` adds what pi's docs give for local OpenAI-compatible
Qwen servers (`supportsDeveloperRole: false`, `thinkingFormat: qwen-chat-template`), `--compat none` nothing.

Recorded per run (no message text): success and the scoring's counts, wall time, exit, turns, pi's stop reasons and
error messages (host paths scrubbed), retries and compactions, the tool calls (tool, a file's path inside the project
or the program a bash command runs, whether it failed), the diff of the agent's change (public code only; withheld if
`check_private` matches it), whether the final reply was "done", and per request (from the proxy): status, input,
cached (prefix tokens saved) and output tokens, TTFT, TPOT, total time, finish reason, tool calls and whether their
arguments parsed, markup left in the text, stream integrity; the request shapes pi sent (keys, roles, message
fields, sampling fields). Each session's events, pi's session file and the proxy's lines are kept under the
gitignored `bench/.out/demo/pi/`. `--report` adds the speed and endpoint summary over every series of a record.

Usage:
  python bench/demo/pi_run.py --server http://127.0.0.1:8000 --gpu-class rtx-5090 --pi <prefix>/node_modules/.bin/pi \
      --task demo --runs 3 --out <record.json>             # a second task appends its series to the same record
  python bench/demo/pi_run.py --report <record.json>
"""
from __future__ import annotations

import argparse
import collections
import datetime
import difflib
import importlib.util
import json
import os
import pathlib
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("demo_run", HERE / "run.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)
ROOT, OUT = demo.ROOT, demo.OUT / "pi"
PI = "0.73.1"
TASK = HERE / "workloads_task"
SEEDED_TESTS, HIDDEN_TESTS = 9, 5
TRACES = ("agent", "decode_b8", "long_prompt")
HIDDEN = "tests/test_harness_trace_stats.py"
WORKLOADS_PROMPT = (
    "This directory holds a copy of an LLM inference engine's benchmark workload tools: `workloads.py` (the trace "
    "generator), `driver.py` (the replay), sample traces under traces/ and tests under tests/. Add a function "
    "`trace_stats(sessions)` to workloads.py that summarizes a trace (a list of sessions, as `load` returns) as a dict "
    "with exactly these keys: `sessions` (the number of sessions), `requests` (the number of turns over all "
    "sessions), `prompt_tokens` (the sum over every turn of its prompt length, a turn's prompt being as the module "
    "docstring defines it), `output_tokens` (the sum of every turn's output) and `max_context` (the largest "
    "`context_len` of any session; 0 for an empty trace). Also add a `--stats PATH` option to the command line that "
    "prints the stats of the trace at PATH as JSON and writes no traces. Add tests for both under tests/ and make "
    "`python -m pytest` pass without changing the existing tests. When the tests pass, reply with the single word "
    "done.")
PROMPTS = {"demo": demo.PROMPT, "workloads": WORKLOADS_PROMPT}
COMPAT = {"documented": {"supportsDeveloperRole": False, "thinkingFormat": "qwen-chat-template"}, "none": {}}
TIMEOUT_S = demo.TIMEOUT_S
FILE_TOOLS = {"read", "edit", "write"}
SAMPLING = ("model", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "reasoning_effort",
            "chat_template_kwargs", "stream", "stream_options", "store", "tool_choice", "parallel_tool_calls")


def workloads_project() -> dict[str, bytes]:
    """The `workloads` task's project: Holon's files at this commit and the task's seeded files."""
    files = {name: (ROOT / "bench" / name).read_bytes() for name in ("workloads.py", "driver.py")}
    files |= {f"traces/{n}.jsonl": (ROOT / "bench" / "traces" / f"{n}.jsonl").read_bytes() for n in TRACES}
    return files | {k: v for k, v in demo.tree(TASK).items() if k == "pyproject.toml" or k.startswith("tests/")}


def project(task: str) -> dict[str, bytes]:
    return demo.load_project() if task == "demo" else workloads_project()


def pytest_counts(files: dict[str, bytes]) -> dict:
    """Run the harness's pytest over `files` in a fresh directory, as D7 scores (its interpreter, an empty
    configuration, no conftest or plugin autoload, bytecode only from a fresh directory)."""
    out = {"passed": 0, "failed": 0, "skipped": 0, "pytest_exit": None, "timed_out": False}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        scored, report, ini, pycache = tmp / "scored", tmp / "junit.xml", tmp / "pytest.ini", tmp / "pycache"
        demo.write_tree(files, scored)
        ini.write_text("[pytest]\n")
        pycache.mkdir()
        try:
            run = subprocess.run([sys.executable, "-m", "pytest", *demo._pytest_args(ini, scored, report)], cwd=scored,
                                 env=demo._scoring_env(pycache), capture_output=True, text=True,
                                 timeout=demo.SCORING_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return out | {"timed_out": True}
        cases = ET.parse(report).getroot().iter("testcase") if report.exists() else []
        for case in cases:
            kind = next((c.tag for c in case if c.tag in ("failure", "error", "skipped")), "passed")
            out[{"failure": "failed", "error": "failed"}.get(kind, kind)] += 1
        return out | {"pytest_exit": run.returncode}


def score_workloads(work: pathlib.Path, seeded: dict[str, bytes]) -> dict:
    """The `workloads` task's scoring: the harness's tests (the seeded ones restored, and its own) against the agent's
    modules; then the agent's own suite as it left it, which must pass with more than the seeded tests and one test
    file naming `trace_stats`."""
    try:
        files = demo.tree(work)
        modules = {k: v for k, v in files.items() if "/" not in k and k.endswith(".py")}
        traces = {k: v for k, v in seeded.items() if k.startswith("traces/")}
        harness = pytest_counts(modules | traces | {k: v for k, v in seeded.items() if k.startswith("tests/")}
                                | {HIDDEN: (TASK / "hidden_test_trace_stats.py").read_bytes()})
        agent_tests = {k: v for k, v in files.items() if k.startswith("tests/") and k.endswith(".py")}
        own = pytest_counts(modules | traces | agent_tests)
        names = any(b"trace_stats" in v for v in agent_tests.values())
        success = (harness["pytest_exit"] == 0 and harness["passed"] == SEEDED_TESTS + HIDDEN_TESTS
                   and own["pytest_exit"] == 0 and own["passed"] > SEEDED_TESTS and names)
        return {"success": success, "harness": harness, "agent_suite": own, "agent_tests_name_trace_stats": names}
    except Exception as e:                                 # a run is never lost to its scoring
        return {"success": False, "error": type(e).__name__}


def diff(before: dict[str, bytes], work: pathlib.Path, task: str) -> str:
    """The agent's change to the project's code (the demo: `shop/`; workloads: modules and tests) as a unified diff."""
    keep = (lambda k: k.startswith("shop/")) if task == "demo" else (
        lambda k: k.endswith(".py") and ("/" not in k or k.startswith("tests/")))
    try:
        after = demo.tree(work)
        out = []
        for name in sorted(k for k in before.keys() | after.keys() if keep(k)):
            old, new = (d.get(name, b"").decode(errors="replace").splitlines(True) for d in (before, after))
            out += difflib.unified_diff(old, new, f"a/{name}", f"b/{name}")
        return "".join(out)
    except Exception as e:                                 # a run is never lost to its record
        return f"<diff failed: {type(e).__name__}>"


def pi_identity(pi: pathlib.Path, env: dict[str, str]) -> dict:
    """pi's version (it must be the pinned one), node's, and a digest of the installed packages' code."""
    version = subprocess.run([str(pi), "--version"], capture_output=True, text=True, env=env, check=True).stdout.strip()
    if version != PI:
        raise SystemExit(f"pi {version!r}: this test is pinned at {PI}")
    node = subprocess.run(["node", "--version"], capture_output=True, text=True, env=env, check=True).stdout.strip()
    packages = pathlib.Path(os.path.realpath(pi)).parents[2]          # node_modules/@mariozechner
    return {"pi": version, "node": node, "packages_digest": demo.digest(
        {f"{p.name}/{k}": v for p in sorted(packages.iterdir()) for k, v in demo.tree(p / "dist").items()})}


def base_env(pi: pathlib.Path, home: pathlib.Path, config: pathlib.Path, venv: pathlib.Path | None = None) -> dict:
    """pi's whole environment: nothing inherited but the system's PATH (no API keys, proxies or user config)."""
    node = pathlib.Path(shutil.which("node") or "node").parent
    path = os.pathsep.join([*([str(venv)] if venv else []), str(node), "/usr/local/bin", "/usr/bin", "/bin"])
    return {"PATH": path, "HOME": str(home), "LANG": "C.UTF-8", "TERM": "dumb", "NO_PROXY": "127.0.0.1,localhost",
            "PI_CODING_AGENT_DIR": str(config), "PI_OFFLINE": "1", "PI_TELEMETRY": "0", "PI_SKIP_VERSION_CHECK": "1"
            } | ({"VIRTUAL_ENV": str(venv.parent)} if venv else {})


def models_json(base_url: str, model: str, context: int, compat: dict) -> dict:
    provider = {"baseUrl": base_url, "api": "openai-completions", "apiKey": "holon",
                "models": [{"id": model, "reasoning": True, "contextWindow": context}]}
    return {"providers": {"holon": provider | ({"compat": compat} if compat else {})}}


def session_facts(events: list[dict], work: pathlib.Path) -> dict:
    """What pi's event stream says about the session, without text: turns (assistant messages), stop reasons, error
    messages (scrubbed), retries, compactions, the tool calls, whether the final reply was "done"."""
    work, stops, errors, calls, by_id = work.resolve(), collections.Counter(), [], [], {}
    turns = retries = compactions = 0
    final = ""
    for e in events:
        kind = e.get("type")
        if kind == "message_end" and (m := e.get("message") or {}).get("role") == "assistant":
            turns += 1
            stops[str(m.get("stopReason"))] += 1
            if m.get("errorMessage"):
                errors.append(demo.scrub(str(m["errorMessage"]))[:300])
            final = "".join(b.get("text", "") for b in m.get("content") or [] if b.get("type") == "text")
        elif kind == "tool_execution_start":
            call, args = {"tool": e.get("toolName")}, e.get("args") or {}
            if call["tool"] in FILE_TOOLS and isinstance(args.get("path"), str):
                target = (work / os.path.expanduser(args["path"])).resolve()
                call["path"] = str(target.relative_to(work)) if target.is_relative_to(work) else "<outside>"
            elif call["tool"] == "bash" and isinstance(args.get("command"), str):
                try:
                    words = shlex.split(args["command"])
                except ValueError:
                    words = args["command"].split()
                words = [w for w in words if not demo.ASSIGNMENT.match(w)] or [""]
                call["program"] = pathlib.Path(words[0]).name
            by_id[e.get("toolCallId")] = call
            calls.append(call)
        elif kind == "tool_execution_end" and e.get("toolCallId") in by_id:
            by_id[e["toolCallId"]]["error"] = bool(e.get("isError"))
        retries += kind == "auto_retry_start"
        compactions += kind == "compaction_start"
    return {"turns": turns, "stop_reasons": dict(stops), "errors": errors, "retries": retries,
            "compactions": compactions, "tool_calls": calls, "replied_done": final.strip() == "done"}


def requests_summary(rows: list[dict]) -> dict:
    """The proxy's lines: one entry per chat request, counts per status, the request shapes and sampling fields pi
    sent, totals and the peak context."""
    chats = [r for r in rows if r.get("path", "").split("?")[0] == "/v1/chat/completions" and r.get("method") == "POST"]
    per = []
    for r in chats:
        usage, ttft = r.get("usage") or {}, r.get("ttft_ms")
        n_in, n_out = usage.get("input_tokens"), usage.get("output_tokens")
        per.append({"status": r.get("status"), "input_tokens": n_in, "cached_tokens": r.get("prefix_tokens_saved"),
                    "output_tokens": n_out, "ttft_ms": ttft, "total_ms": r.get("total_ms"),
                    "tpot_ms": round((r["total_ms"] - ttft) / (n_out - 1), 2) if ttft is not None and (n_out or 0) > 1
                    else None, "stop_reason": r.get("stop_reason"), "error_type": r.get("error_type"),
                    "tool_calls": [c["name"] for c in r.get("tool_calls") or []],
                    "tool_args_json": all(c["args_json"] for c in r.get("tool_calls") or []),
                    "markup": r.get("markup") or [], "stream": r.get("stream"),
                    "content_chars": r.get("content_chars"), "reasoning_chars": r.get("reasoning_chars")})
    statuses = collections.Counter(str(r.get("status")) for r in rows)
    shapes = {"keys": sorted({k for r in chats for k in r.get("keys") or []}),
              "roles": sorted({m.get("role") for r in chats for m in r.get("messages") or []}),
              "message_fields": sorted({f"{m.get('role')}.{f}" for r in chats for m in r.get("messages") or []
                                        for f in m.get("fields") or []}),
              "content_kinds": sorted({f"{m.get('role')}:{c if isinstance(c := m.get('content'), str) else 'parts'}"
                                       for r in chats for m in r.get("messages") or []}),
              "tools": sorted({json.dumps(t, sort_keys=True) for r in chats for t in r.get("tools") or []})}
    shapes["tools"] = [json.loads(t) for t in shapes["tools"]]
    sampling = sorted({json.dumps({k: r[k] for k in SAMPLING if k in r}, sort_keys=True) for r in chats})
    contexts = [(p["input_tokens"] or 0) + (p["output_tokens"] or 0) for p in per]
    return {"requests_total": len(rows), "chat_requests": len(chats), "statuses": dict(statuses),
            "peak_context": max(contexts, default=0), "input_tokens": sum(p["input_tokens"] or 0 for p in per),
            "cached_tokens": sum(p["cached_tokens"] or 0 for p in per),
            "output_tokens": sum(p["output_tokens"] or 0 for p in per), "shapes": shapes,
            "sampling": [json.loads(s) for s in sampling], "requests": per}


def run_once(server: str, model: str, pi: pathlib.Path, task: str, files: dict[str, bytes], keep: pathlib.Path,
             reference: str, first_venv: str | None, context: int, compat: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="holon-pi-") as tmp:
        tmp = pathlib.Path(tmp)
        work, home, config, sessions = tmp / "project", tmp / "home", tmp / "pi", tmp / "sessions"
        demo.write_tree(files, work)
        for d in (home, config, sessions):
            d.mkdir()
        venv, made = demo.prepare_venv(tmp / "venv", first_venv)
        port, captured = demo._free_port(), tmp / "requests.jsonl"
        (config / "models.json").write_text(json.dumps(models_json(f"http://127.0.0.1:{port}/v1", model, context,
                                                                   compat), indent=1))
        (config / "settings.json").write_text(json.dumps({"enableInstallTelemetry": False}))
        proxy = subprocess.Popen([sys.executable, str(ROOT / "bench" / "capture_proxy.py"), "--upstream", server,
                                  "--port", str(port), "--out", str(captured), "--metrics"])
        try:
            for _ in range(100):
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"the capture proxy did not start (exit {proxy.poll()})")
            env = base_env(pi, home, config, venv)
            before = demo._get(server, "/metrics")
            start, timed_out = time.perf_counter(), False
            with open(tmp / "events.jsonl", "w") as log:      # a file, not a pipe: what the session leaves running
                session = subprocess.Popen([str(pi), "--mode", "json", "--provider", "holon", "--model", model,
                                            "--session-dir", str(sessions), PROMPTS[task]], cwd=work, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    session.wait(timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:                                      # the session and anything it left running
                    try:
                        os.killpg(session.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    session.wait()
            wall = time.perf_counter() - start
            after = demo._get(server, "/metrics")
        finally:
            proxy.terminate()
            proxy.wait()
        keep.parent.mkdir(parents=True, exist_ok=True)       # transcripts, kept out of the repository
        shutil.copy(tmp / "events.jsonl", keep.with_suffix(".events.jsonl"))
        if captured.exists():
            shutil.copy(captured, keep.with_suffix(".requests.jsonl"))
        for i, f in enumerate(sorted(sessions.rglob("*.jsonl"))):
            shutil.copy(f, keep.with_suffix(f".session{i}.jsonl"))
        events, other = [], 0
        for line in (tmp / "events.jsonl").read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                other += 1                                    # pi's own non-JSON output lines (stderr)
        rows = [json.loads(line) for line in captured.read_text().splitlines()] if captured.exists() else []
        invalid = demo.invalid_after(reference)
        scored = None if invalid else (demo.score(work, {k: v for k, v in files.items() if k.startswith("tests/")})
                                       if task == "demo" else score_workloads(work, files))
        change = diff(files, work, task)
        return {"success": invalid is None and bool(scored and scored["success"]), "invalid": invalid,
                "scored": scored, "venv_pytest": made, "venv_after": demo.venv_change(venv, made),
                "wall_s": round(wall, 1), "timed_out": timed_out,
                "exit_code": None if timed_out else session.returncode,
                "non_json_output_lines": other, **session_facts(events, work),
                "prefix_tokens_saved": after.get("prefix_tokens_saved", 0) - before.get("prefix_tokens_saved", 0),
                "preemptions": after.get("preemptions", 0) - before.get("preemptions", 0),
                "diff": change if demo.private_ok(change) else "<withheld: matches check_private's patterns>",
                "served": requests_summary(rows)}


def pct(values: list[float], q: float) -> float | None:
    """The q-quantile (0..1) of `values`, linearly interpolated; None for none."""
    if not values:
        return None
    s = sorted(values)
    at = (len(s) - 1) * q
    lo = int(at)
    return round(s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (at - lo), 2)


def summarize(record: dict) -> dict:
    """Speed and endpoint facts over every counted run of a record: decode speed (1000 / TPOT) of requests with at
    least 2 output tokens; prefill speed of the uncached input ((input − cached) / TTFT) overall and by context size;
    the prefix-cache hit rate; per-request latency; per-run outcomes; and every request that was not a clean 200
    stream, left markup in its text or made a tool call whose arguments did not parse."""
    rows, runs, issues = [], [], []
    for task, series in record["series"].items():
        for i, run in enumerate(series["runs"]):
            served = run["served"]
            runs.append({"task": task, "run": i + 1, "success": run["success"], "turns": run["turns"],
                         "wall_s": run["wall_s"], "requests": served["chat_requests"],
                         "input_tokens": served["input_tokens"], "cached_tokens": served["cached_tokens"],
                         "output_tokens": served["output_tokens"], "peak_context": served["peak_context"],
                         "tool_calls": len(run["tool_calls"]), "tool_errors": sum(c.get("error", False)
                                                                                 for c in run["tool_calls"]),
                         "retries": run["retries"], "compactions": run["compactions"],
                         "replied_done": run["replied_done"]})
            for j, r in enumerate(served["requests"]):
                rows.append(r)
                clean = r["status"] == 200 and (r["stream"] or {}).get("done") and not (r["stream"] or {}).get(
                    "bad_lines") and r["stop_reason"] in ("stop", "tool_calls", "length")
                if not clean or r["markup"] or not r["tool_args_json"]:
                    issues.append({"task": task, "run": i + 1, "request": j + 1, **{k: r[k] for k in (
                        "status", "stop_reason", "error_type", "stream", "markup", "tool_calls", "tool_args_json",
                        "input_tokens", "output_tokens")}})
    ok = [r for r in rows if r["status"] == 200]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"]]
    timed = [r for r in ok if r["ttft_ms"] and r["input_tokens"] and r["cached_tokens"] is not None]

    def rate(r: dict) -> float:
        return (r["input_tokens"] - r["cached_tokens"]) / r["ttft_ms"] * 1e3
    bins = [(0, 8192), (8192, 16384), (16384, 32768), (32768, 1 << 30)]
    by_context = []
    for lo, hi in bins:
        if sel := [r for r in timed if lo <= r["input_tokens"] < hi]:
            by_context.append({"input_tokens": f"{lo}-{hi if hi < 1 << 30 else ''}", "requests": len(sel),
                               "uncached_p50": pct([r["input_tokens"] - r["cached_tokens"] for r in sel], 0.5),
                               "ttft_ms_p50": pct([r["ttft_ms"] for r in sel], 0.5),
                               "prefill_tok_s_p50": pct([rate(r) for r in sel], 0.5)})
    large = [rate(r) for r in timed if r["input_tokens"] - r["cached_tokens"] >= 512]
    n_in = sum(r["input_tokens"] or 0 for r in ok)
    return {
        "requests": len(rows), "ok": len(ok), "issues": issues, "runs": runs,
        "decode": {"requests": len(tpot), "tpot_ms_p50": pct(tpot, 0.5), "tpot_ms_p90": pct(tpot, 0.9),
                   "tok_s_at_tpot_p50": round(1e3 / pct(tpot, 0.5), 1) if tpot else None,
                   "tok_s_at_tpot_p90": round(1e3 / pct(tpot, 0.9), 1) if tpot else None},
        "prefill": {"requests": len(timed), "uncached_tok_s_p50": pct([rate(r) for r in timed], 0.5),
                    "uncached_at_least_512": {"requests": len(large), "tok_s_p50": pct(large, 0.5),
                                              "tok_s_p10": pct(large, 0.1), "tok_s_p90": pct(large, 0.9)},
                    "by_context": by_context},
        "prefix_cache": {"input_tokens": n_in, "cached_tokens": sum(r["cached_tokens"] or 0 for r in ok),
                         "hit_rate": round(sum(r["cached_tokens"] or 0 for r in ok) / n_in, 4) if n_in else None},
        "latency": {"ttft_ms_p50": pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], 0.5),
                    "ttft_ms_p90": pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], 0.9),
                    "request_ms_p50": pct([r["total_ms"] for r in ok], 0.5),
                    "request_ms_p90": pct([r["total_ms"] for r in ok], 0.9),
                    "output_tokens_p50": pct([r["output_tokens"] for r in ok if r["output_tokens"]], 0.5)},
        "run_wall_s_p50": statistics.median([r["wall_s"] for r in runs]) if runs else None}


def series_header(args: argparse.Namespace, model: str, reference: str, files: dict[str, bytes]) -> dict:
    return {"task": args.task, "prompt": PROMPTS[args.task], "project": demo.digest(files),
            "compat": args.compat, "models_json": models_json("http://127.0.0.1:<proxy>/v1", model, args.context,
                                                              COMPAT[args.compat]),
            "scorer_reference": reference, "runs_planned": args.runs, "runs": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", help="the running `holon serve`, e.g. http://127.0.0.1:8000")
    parser.add_argument("--gpu-class", help="the server's GPU class; must be its lock's (e.g. rtx-5090)")
    parser.add_argument("--pi", type=pathlib.Path, help="pi's executable (node_modules/.bin/pi)")
    parser.add_argument("--task", choices=sorted(PROMPTS), default="demo")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--compat", choices=sorted(COMPAT), default="documented")
    parser.add_argument("--context", type=int, default=49152, help="the served context, given to pi")
    parser.add_argument("--series", help="the series' name in the record (default: the task)")
    parser.add_argument("--out", type=pathlib.Path, help="the record; a new series is added to an existing one")
    parser.add_argument("--report", type=pathlib.Path, help="add the summary to this record and print it")
    args = parser.parse_args()
    if args.report:
        record = json.loads(args.report.read_text())
        record["summary"] = summarize(record)
        args.report.write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps(record["summary"], indent=1))
        return
    if not (args.server and args.gpu_class and args.out and args.pi):
        parser.error("--server, --gpu-class, --pi and --out are needed for runs")
    if dirty := demo.harness_dirty():
        sys.exit(f"the harness differs from its commit: {dirty}")
    reference = demo.site_digest()
    files = project(args.task)
    if args.task == "demo":
        with tempfile.TemporaryDirectory() as tmp:
            demo.write_tree(files, work := pathlib.Path(tmp) / "project")
            seeded = demo.score(work, {k: v for k, v in files.items() if k.startswith("tests/")})
        if {k: seeded[k] for k in demo.SEEDED} != demo.SEEDED:
            sys.exit(f"the project does not fail as pre-registered: {seeded}")
    else:
        seeded = pytest_counts(files)
        if seeded["pytest_exit"] != 0 or seeded["passed"] != SEEDED_TESTS:
            sys.exit(f"the workloads project's seeded suite does not pass as expected: {seeded}")
    (model,) = [m["id"] for m in demo._get(args.server, "/v1/models")["data"]]
    metrics = demo._get(args.server, "/metrics")
    served = {"plan_id": metrics.get("plan_id"), "lock": metrics.get("lock") and demo.scrub(metrics["lock"]),
              "diff": [demo.scrub(str(d)) for d in metrics.get("diff") or []]}
    if demo.lock_class(served["lock"]) != args.gpu_class:
        sys.exit(f"--gpu-class {args.gpu_class} is not the served lock's class ({served['lock']})")
    with tempfile.TemporaryDirectory() as tmp:
        identity = pi_identity(args.pi, base_env(args.pi, pathlib.Path(tmp), pathlib.Path(tmp) / "pi"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = json.loads(args.out.read_text()) if args.out.exists() else {
        "workload": "agent_test_pi", "date": datetime.date.today().isoformat(), "gpu_class": args.gpu_class,
        "commit": commit, "model": model, "served": served,
        "client": identity | {"tools": "read, bash, edit, write (pi's defaults)", "timeout_s": TIMEOUT_S,
                              "command": "pi --mode json --provider holon --model <served id> --session-dir <tmp> "
                                         "<prompt>",
                              "env": sorted(base_env(args.pi, pathlib.Path("<home>"), pathlib.Path("<config>"),
                                                     pathlib.Path("<venv>/bin")))},
        "series": {}}
    for key, want in (("gpu_class", args.gpu_class), ("commit", commit), ("model", model)):
        if record[key] != want:
            sys.exit(f"{args.out}: another record ({key} {record[key]!r} != {want!r})")
    if record["served"]["plan_id"] != served["plan_id"] or record["client"]["packages_digest"] != identity[
            "packages_digest"]:
        sys.exit(f"{args.out}: another record (served plan or pi)")
    name = args.series or args.task
    if name in record["series"]:
        sys.exit(f"{args.out}: series {name!r} exists")
    series = record["series"][name] = series_header(args, model, reference, files)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    first_venv = None
    for i in range(args.runs):
        if dirty := demo.harness_dirty():
            sys.exit(f"the harness changed during the series: {dirty}")
        demo.check_environment(reference)
        keep = OUT / f"{stamp}_{args.gpu_class}_{name}_run{i + 1}"
        run = run_once(args.server, model, args.pi, args.task, files, keep, reference, first_venv, args.context,
                       COMPAT[args.compat])
        first_venv = first_venv or run["venv_pytest"]
        series["runs"].append(run)
        args.out.write_text(json.dumps(record, indent=1) + "\n")      # kept after every run
        print(f"{name} run {i + 1}: success {run['success']}, turns {run['turns']}, wall {run['wall_s']} s, "
              f"statuses {run['served']['statuses']}, stops {run['stop_reasons']}", flush=True)


if __name__ == "__main__":
    main()
