"""The agent demo (release design D7), pre-registered before any run: Claude Code at a pinned version, headless,
against a running `holon serve`, makes a small project's tests pass.

The task: `project/` (the package `shop`, three modules: `money`, `cart`, `discounts`) fails 17 of its 40 tests on a
seeded bug and a missing function. Claude Code gets `PROMPT` in a fresh copy of it, with the tools M0.5's session
allowed, stopped after `TIMEOUT_S`. A run **succeeds** if, after the session (whatever its outcome), pytest passes
all 40 tests in a fresh directory holding only the agent's `shop/` and the harness's own copy of `tests/` (read
before the session), with the harness's configuration: no project configuration, no `conftest.py`, no plugin
autoload, nothing else of the agent's on the import path. Not guarded: the agent runs the harness's own Python, whose
installed packages it could change. Every run is recorded, misses included. Runs are sequential on one server, so a
later run may hit the prefix cache an earlier one filled (the system prompt and tool list are the same), as a user's
next session would. A smaller task, if one is ever needed, is a new pre-registered task reported beside this one.

Recorded per run: success and the test counts, turns, wall time, the tool-call sequence (tool, the file's path
inside the project or the program a Bash command runs, and whether the call failed — never file text or command
arguments), the requests Claude Code sent (their sampling parameters, from `bench/capture_proxy.py`, which sits
between it and the server), the tokens served (input and output, as the server reported them) and the prefix tokens
saved (the engine's counter over the run), TTFT and TPOT per request at the proxy.

Usage (release step 9: `holon serve` the class's 27B lock, then 3 runs):
  python bench/demo/run.py --server http://127.0.0.1:8000 --gpu-class rtx-5080 --runs 3 --out <evidence.json>
  python bench/demo/run.py --score <dir>     # score a project directory as a run is scored
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE / "project"
ROOT = HERE.parents[1]
CLAUDE_CODE = "2.1.300"               # the pinned version (`claude --version`); M0.5's session ran 2.1.296
PROMPT = ("The Python package `shop` in the current directory fails its tests. Fix the package so that "
          "`python -m pytest` passes, without editing any file under tests/. When the tests pass, reply with the "
          "single word done.")
TOOLS = "Read,Write,Edit,Bash,Glob,Grep"
CLIENT_ENV = ["ANTHROPIC_BASE_URL=<the proxy>", "ANTHROPIC_AUTH_TOKEN=holon",
              "ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL, CLAUDE_CODE_SUBAGENT_MODEL=<served id>",
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "DISABLE_AUTOUPDATER=1", "CLAUDE_CONFIG_DIR=<fresh>",
              "NO_PROXY=127.0.0.1,localhost"]
TIMEOUT_S = 1800                      # a session still running after 30 min is stopped and scored as it stands
TESTS = 40
SEEDED = {"passed": 23, "failed": 17}
FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit"}
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
SAMPLING = ("model", "max_tokens", "temperature", "top_p", "top_k", "thinking", "output_config", "stop_sequences")


def tree(root: pathlib.Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name, data in sorted(files.items()):
        h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return "sha256:" + h.hexdigest()


def score(work: pathlib.Path, tests: dict[str, bytes]) -> dict:
    """Run the harness's tests (`tests`, its own copy) against the agent's `shop/` alone, in a fresh directory, with
    the harness's pytest configuration; success is every test passing."""
    out = {"success": False, "passed": 0, "failed": 0, "skipped": 0, "pytest_exit": None, "timed_out": False}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        scored, report, ini = tmp / "scored", tmp / "junit.xml", tmp / "pytest.ini"
        if not (work / "shop").is_dir():
            return out
        shutil.copytree(work / "shop", scored / "shop", ignore=shutil.ignore_patterns("__pycache__"))
        for name, data in tests.items():
            (scored / name).parent.mkdir(parents=True, exist_ok=True)
            (scored / name).write_bytes(data)
        ini.write_text("[pytest]\n")                   # not the project's configuration
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "PYTHON"))}
        env |= {"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        try:
            run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-c", str(ini), "--rootdir", str(scored),
                                  "--noconftest", "-p", "no:cacheprovider", f"--junitxml={report}", "tests"],
                                 cwd=scored, env=env, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return out | {"timed_out": True}
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        if report.exists():
            suite = ET.parse(report).getroot()
            suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
            counts = {k: int(suite.get(k, 0)) for k in counts}
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return out | {"success": run.returncode == 0 and passed == TESTS, "passed": passed, "skipped": counts["skipped"],
                  "failed": counts["failures"] + counts["errors"], "pytest_exit": run.returncode}


def _get(server: str, path: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never through an HTTP proxy
    with opener.open(server.rstrip("/") + path, timeout=60) as r:
        return json.load(r)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _claude_version(claude: str) -> str:
    out = subprocess.run([claude, "--version"], capture_output=True, text=True, check=True).stdout
    return out.split()[0] if out.split() else ""


def tool_calls(events: list[dict], work: pathlib.Path) -> list[dict]:
    """The session's tool calls in order: the tool, a file tool's path inside the project (or "<outside>") or the
    program a Bash command starts, and whether its result was an error. No file text, no arguments."""
    calls, by_id, work = [], {}, work.resolve()
    for event in events:
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                call, args = {"tool": block.get("name")}, block.get("input") or {}
                if call["tool"] in FILE_TOOLS and isinstance(path := args.get("file_path") or args.get(
                        "notebook_path"), str):
                    target = (work / os.path.expanduser(path)).resolve()
                    call["path"] = str(target.relative_to(work)) if target.is_relative_to(work) else "<outside>"
                elif call["tool"] == "Bash" and isinstance(args.get("command"), str):
                    try:
                        words = shlex.split(args["command"])
                    except ValueError:
                        words = args["command"].split()
                    words = [w for w in words if not ASSIGNMENT.match(w)] or [""]      # `VAR=value cmd`
                    call["program"] = pathlib.Path(words[0]).name
                by_id[block.get("id")] = call
                calls.append(call)
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in by_id:
                by_id[block["tool_use_id"]]["error"] = bool(block.get("is_error"))
    return calls


def requests_summary(rows: list[dict]) -> dict:
    """What Claude Code sent and was served, from the proxy's records."""
    messages = [r for r in rows if r.get("path", "").split("?")[0] == "/v1/messages" and r.get("method") == "POST"]
    ttft = [r["ttft_ms"] for r in messages if r.get("ttft_ms") is not None]
    tpot = [round((r["total_ms"] - r["ttft_ms"]) / (r["usage"]["output_tokens"] - 1), 2) for r in messages
            if r.get("ttft_ms") is not None and r.get("usage", {}).get("output_tokens", 0) > 1]
    sampling = sorted({json.dumps({k: r[k] for k in SAMPLING if k in r}, sort_keys=True) for r in messages})
    return {"requests": len(rows), "messages": len(messages),
            "statuses": sorted({r.get("status") for r in rows}),
            "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in messages),
            "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in messages),
            "sampling": [json.loads(s) for s in sampling], "ttft_ms": ttft, "tpot_ms": tpot}


def run_once(server: str, model: str, claude: str, tests: dict[str, bytes]) -> dict:
    with tempfile.TemporaryDirectory(prefix="holon-demo-") as tmp:
        tmp = pathlib.Path(tmp)
        work = tmp / "project"
        shutil.copytree(PROJECT, work, ignore=shutil.ignore_patterns("__pycache__"))
        port, captured = _free_port(), tmp / "requests.jsonl"
        proxy = subprocess.Popen([sys.executable, str(ROOT / "bench" / "capture_proxy.py"), "--upstream", server,
                                  "--port", str(port), "--out", str(captured)])
        try:
            for _ in range(100):
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"the capture proxy did not start (exit {proxy.poll()})")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "PYTEST_"))}
            env |= {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_AUTH_TOKEN": "holon",
                    "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model, "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                    "CLAUDE_CODE_SUBAGENT_MODEL": model, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "DISABLE_AUTOUPDATER": "1", "NO_PROXY": "127.0.0.1,localhost",
                    "CLAUDE_CONFIG_DIR": str(tmp / "claude"),          # no user settings, memory or hooks
                    "PATH": f"{pathlib.Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"}
            saved = _get(server, "/metrics").get("prefix_tokens_saved", 0)
            start, timed_out = time.perf_counter(), False
            with open(tmp / "session.jsonl", "w") as log:   # a file, not a pipe: what the session leaves running
                session = subprocess.Popen([claude, "-p", PROMPT, "--allowedTools", TOOLS, "--output-format",
                                            "stream-json", "--verbose"], cwd=work, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.DEVNULL, start_new_session=True)
                try:
                    session.wait(timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:                             # the session and anything it left running
                    try:
                        os.killpg(session.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    session.wait()
            stdout = (tmp / "session.jsonl").read_text(errors="replace")
            code = None if timed_out else session.returncode
            wall = time.perf_counter() - start
            saved = _get(server, "/metrics").get("prefix_tokens_saved", 0) - saved
        finally:
            proxy.terminate()
            proxy.wait()
        events = []
        for line in stdout.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        result = next((e for e in reversed(events) if e.get("type") == "result"), {})
        rows = [json.loads(line) for line in captured.read_text().splitlines()] if captured.exists() else []
        return {"scored": score(work, tests), "wall_s": round(wall, 1), "timed_out": timed_out, "exit_code": code,
                "turns": result.get("num_turns"), "result_subtype": result.get("subtype"),
                "is_error": result.get("is_error"), "replied_done": (result.get("result") or "").strip() == "done",
                "prefix_tokens_saved": saved, "tool_calls": tool_calls(events, work),
                "served": requests_summary(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", help="the running `holon serve`, e.g. http://127.0.0.1:8000")
    parser.add_argument("--gpu-class", help="the server's GPU class, for the record (e.g. rtx-5080)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=pathlib.Path, help="the evidence record to write")
    parser.add_argument("--claude", default="claude", help="the Claude Code executable")
    parser.add_argument("--score", type=pathlib.Path, help="only score a copy of this project directory")
    args = parser.parse_args()
    tests = {name: data for name, data in tree(PROJECT).items() if name.startswith("tests/")}
    if args.score:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(args.score, work := pathlib.Path(tmp) / "project",
                            ignore=shutil.ignore_patterns("__pycache__"))
            print(json.dumps(score(work, tests)))
        return
    if not (args.server and args.gpu_class and args.out):
        parser.error("--server, --gpu-class and --out are needed for runs")
    if (version := _claude_version(args.claude)) != CLAUDE_CODE:
        sys.exit(f"Claude Code {version!r}: the demo is pre-registered at {CLAUDE_CODE}")
    with tempfile.TemporaryDirectory() as tmp:          # the project as committed fails as pre-registered
        shutil.copytree(PROJECT, work := pathlib.Path(tmp) / "project", ignore=shutil.ignore_patterns("__pycache__"))
        seeded = score(work, tests)
    if {k: seeded[k] for k in SEEDED} != SEEDED:
        sys.exit(f"the project does not fail as pre-registered: {seeded}")
    (model,) = [m["id"] for m in _get(args.server, "/v1/models")["data"]]
    metrics = _get(args.server, "/metrics")
    if lock := metrics.get("lock"):                       # from `plans/` on: no host path in the record
        metrics["lock"] = lock[lock.rfind("plans/"):] if "plans/" in lock else pathlib.Path(lock).name
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {"workload": "agent_demo", "date": datetime.date.today().isoformat(), "gpu_class": args.gpu_class,
              "commit": commit, "model": model, "served": {k: metrics.get(k) for k in ("plan_id", "lock", "diff")},
              "client": {"claude_code": CLAUDE_CODE, "allowed_tools": TOOLS, "timeout_s": TIMEOUT_S, "env": CLIENT_ENV,
                         "command": "claude -p <prompt> --allowedTools <tools> --output-format stream-json --verbose"},
              "task": {"prompt": PROMPT, "project": digest(tree(PROJECT)), "tests": TESTS, "seeded": SEEDED},
              "runs": []}
    for i in range(args.runs):
        record["runs"].append(run_once(args.server, model, args.claude, tests))
        args.out.write_text(json.dumps(record, indent=1) + "\n")      # kept after every run
        print(f"run {i + 1}: {json.dumps(record['runs'][-1]['scored'])}", flush=True)


if __name__ == "__main__":
    main()
