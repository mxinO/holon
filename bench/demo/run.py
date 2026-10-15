"""The agent demo (release design D7), pre-registered before any run: Claude Code at a pinned version, headless,
against a running `holon serve`, makes a small project's tests pass. It is a small three-module fix chosen to fit the
32K context served, not a claim of general agent capability.

The task: `project/` (the package `shop`, three modules: `money`, `cart`, `discounts`) fails 17 of its 40 tests on a
seeded bug and a missing function. Claude Code gets `PROMPT` in a fresh copy of it, with the tools M0.5's session
allowed, stopped after `TIMEOUT_S`.

Isolation and scoring:
- the project is read once into memory and checked against `PROJECT_DIGEST`; each run's copy is written from it;
- each run gets its own disposable venv with pytest (installed offline by uv), first on the agent's `PATH`;
- Claude Code is the binary resolved once (its real path, version `CLAUDE_CODE` and sha256 recorded), re-checked
  before every run;
- a run **succeeds** if, after the session (whatever its outcome), pytest passes all 40 tests in a fresh directory
  holding only the agent's `shop/` (regular files: links are not followed) and the harness's own copy of `tests/`,
  run by the harness's interpreter with its own configuration (no project configuration, `conftest.py` or plugin
  autoload). That interpreter and every module file a scoring run imports, with the `*.pth` files and any
  `sitecustomize` / `usercustomize`, are hashed before and after every run: a change makes the run **invalid**
  (recorded, not scored); not guarded are the shared libraries it links. Anything failing while scoring is a
  recorded miss.
Every run is recorded, misses and invalid runs included. Runs are sequential on one server, so a later run may hit
the prefix cache an earlier one filled (the same system prompt and tools), as a user's next session would. A smaller
task, if one is ever needed, is a new pre-registered task reported beside this one.

Recorded per run: success and the test counts, turns (from the stream when a session times out), wall time, the
tool-call sequence (tool, the file's path inside the project or the program a Bash command runs, and whether the
call failed — never file text or command arguments), the final `shop/` diff (synthetic, public; withheld if
`check_private`'s patterns match), per request (from `bench/capture_proxy.py`, between Claude Code and the server):
status, input and output tokens, TTFT, TPOT, stop reason, a length refusal; counts per status, length refusals, the
peak context; the sampling parameters Claude Code sent; the tokens served and the prefix tokens saved (the engine's
counter over the run). Each session's stream and requests are kept under the gitignored `bench/.out/demo/`.

Usage (release step 9: `holon serve` the class's 27B lock; first one uncounted smoke of M0.5's Claude Code session
task at the pinned version, docs/m05-grammar.md; then 3 counted runs):
  python bench/demo/run.py --server http://127.0.0.1:8000 --gpu-class rtx-5080 --runs 3 --out <evidence.json>
  python bench/demo/run.py --score <dir>     # score a project directory as a run is scored
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import site
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
OUT = ROOT / "bench" / ".out" / "demo"
PROJECT_DIGEST = "sha256:f5c8bb8704fbcc78ba90382828d9d9a4920ff1aa902ef231a98db03165fd2f25"
CLAUDE_CODE = "2.1.300"               # the pinned version (`claude --version`); M0.5's session ran 2.1.296
PROMPT = ("The Python package `shop` in the current directory fails its tests. Fix the package so that "
          "`python -m pytest` passes, without editing any file under tests/. When the tests pass, reply with the "
          "single word done.")
TOOLS = "Read,Write,Edit,Bash,Glob,Grep"
CLIENT_ENV = ["ANTHROPIC_BASE_URL=<the proxy>", "ANTHROPIC_AUTH_TOKEN=holon",
              "ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL, CLAUDE_CODE_SUBAGENT_MODEL=<served id>",
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "DISABLE_AUTOUPDATER=1", "CLAUDE_CONFIG_DIR=<fresh>",
              "NO_PROXY=127.0.0.1,localhost", "PATH=<the run's venv>:…"]
TIMEOUT_S = 1800                      # a session still running after 30 min is stopped and scored as it stands
SCORING_TIMEOUT_S = 600
TESTS = 40
SEEDED = {"passed": 23, "failed": 17}
FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit"}
GUARDED = ("pytest", "_pytest", "pluggy")                 # the scoring's packages, hashed whole around every run
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
ABSOLUTE = re.compile(r"(?<![\w.])/[^\s:'\"(),\]]+")
SAMPLING = ("model", "max_tokens", "temperature", "top_p", "top_k", "thinking", "output_config", "stop_sequences")


def tree(root: pathlib.Path) -> dict[str, bytes]:
    """The regular files under `root` (links not followed; no bytecode caches or hidden files, e.g. `.pytest_cache`),
    by relative path."""
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" not in rel.parts and not any(part.startswith(".") for part in rel.parts) and \
                path.is_file() and not any(
                (root / pathlib.Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            out[str(rel)] = path.read_bytes()
    return out


def write_tree(files: dict[str, bytes], dest: pathlib.Path) -> None:
    for name, data in files.items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes(data)


def digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name, data in sorted(files.items()):
        h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return "sha256:" + h.hexdigest()


def load_project(root: pathlib.Path = PROJECT) -> dict[str, bytes]:
    """The project, read once, as pre-registered."""
    files = tree(root)
    if digest(files) != PROJECT_DIGEST:
        raise ValueError(f"{root}: not the pre-registered project ({digest(files)} != {PROJECT_DIGEST})")
    return files


def _links(root: pathlib.Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_symlink())


def score(work: pathlib.Path, tests: dict[str, bytes]) -> dict:
    """Run the harness's tests (`tests`, its own copy) against the agent's `shop/` alone (regular files, links not
    followed), in a fresh directory, with the harness's interpreter and pytest configuration. Success: every test
    passes. Anything failing here is a miss, recorded."""
    out = {"success": False, "passed": 0, "failed": 0, "skipped": 0, "pytest_exit": None, "timed_out": False,
           "symlinks_skipped": 0}
    try:
        shop = work / "shop"
        if shop.is_symlink() or not shop.is_dir():
            return out | {"error": "no shop/ package"}
        out["symlinks_skipped"] = _links(shop)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            scored, report, ini = tmp / "scored", tmp / "junit.xml", tmp / "pytest.ini"
            write_tree({f"shop/{k}": v for k, v in tree(shop).items()} | tests, scored)
            ini.write_text("[pytest]\n")                   # not the project's configuration
            try:
                run = subprocess.run([sys.executable, "-m", "pytest", *_pytest_args(ini, scored, report)], cwd=scored,
                                     env=_scoring_env(), capture_output=True, text=True, timeout=SCORING_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return out | {"timed_out": True}
            counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
            if report.exists():
                suite = ET.parse(report).getroot()
                suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
                counts = {k: int(suite.get(k, 0)) for k in counts}
        passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        return out | {"success": run.returncode == 0 and passed == TESTS, "passed": passed,
                      "skipped": counts["skipped"], "failed": counts["failures"] + counts["errors"],
                      "pytest_exit": run.returncode}
    except Exception as e:                                 # a run is never lost to its scoring
        return out | {"error": type(e).__name__}


def _scoring_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "PYTHON"))}
    return env | {"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def _pytest_args(ini: pathlib.Path, rootdir: pathlib.Path, report: pathlib.Path) -> list[str]:
    return ["-q", "-c", str(ini), "--rootdir", str(rootdir), "--noconftest", "-p", "no:cacheprovider",
            f"--junitxml={report}", "tests"]


PROBE = """import json, sys, pytest
pytest.main(sys.argv[1:])
print(json.dumps(sorted({f for m in list(sys.modules.values()) if isinstance(f := getattr(m, "__file__", None), str)})))
"""


def scoring_modules() -> list[str]:
    """The files of every module a scoring run imports (pytest, its plugins and dependencies, the stdlib, extension
    modules): a probe scored the same way, on a one-test directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "tests").mkdir()
        (tmp / "tests" / "test_probe.py").write_text("def test_probe():\n    pass\n")
        (ini := tmp / "pytest.ini").write_text("[pytest]\n")
        out = subprocess.run([sys.executable, "-c", PROBE, *_pytest_args(ini, tmp, tmp / "junit.xml")], cwd=tmp,
                             env=_scoring_env(), capture_output=True, text=True, timeout=120, check=True)
        return [f for f in json.loads(out.stdout.strip().splitlines()[-1]) if not f.startswith(str(tmp))]


def site_digest(dirs: list[pathlib.Path] | None = None) -> str:
    """A hash of what the scoring runs on: the interpreter and every module file a scoring run imports
    (`scoring_modules`), plus pytest, `_pytest` and pluggy, the `*.pth` files and any `sitecustomize` /
    `usercustomize` of the site directories (`dirs`: given, only these are hashed). The shared libraries the
    interpreter links (libpython, libc) are not."""
    files: dict[str, bytes] = {}
    if dirs is None:
        dirs = [pathlib.Path(d) for d in (*site.getsitepackages(), site.getusersitepackages(), *sys.path) if d]
        files = {f: pathlib.Path(f).read_bytes() for f in (os.path.realpath(sys.executable), *scoring_modules())
                 if pathlib.Path(f).is_file()}
    for d in dict.fromkeys(dirs):
        if not d.is_dir():
            continue
        for path in sorted(d.iterdir()):
            if path.name in GUARDED and path.is_dir():
                files |= {f"{path}/{k}": v for k, v in tree(path).items()}
            elif path.suffix == ".pth" or path.stem in ("sitecustomize", "usercustomize") and path.suffix == ".py":
                files[str(path)] = path.read_bytes()
    return digest(files)


def make_venv(dest: pathlib.Path) -> pathlib.Path:
    """A disposable venv with this interpreter's pytest version, installed offline (uv's cache); its bin directory."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("the demo needs uv on PATH (each run's venv)")
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(dest)], check=True)
    subprocess.run([uv, "pip", "install", "-q", "--offline", "--python", str(dest / "bin" / "python"),
                    f"pytest=={importlib.metadata.version('pytest')}"], check=True)
    return dest / "bin"


def _claude_version(path: pathlib.Path) -> str:
    out = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=True).stdout
    return out.split()[0] if out.split() else ""


def _sha256(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_claude(name: str) -> tuple[pathlib.Path, str]:
    """The Claude Code binary every run uses: its real path (a versioned binary, not the launcher link) and sha256,
    at the pinned version."""
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"{name}: not found")
    path = pathlib.Path(os.path.realpath(found))
    if (version := _claude_version(path)) != CLAUDE_CODE:
        raise SystemExit(f"Claude Code {version!r}: the demo is pre-registered at {CLAUDE_CODE}")
    return path, _sha256(path)


def check_claude(path: pathlib.Path, sha: str) -> None:
    """Before every run: the same binary, at the pinned version."""
    if _sha256(path) != sha or _claude_version(path) != CLAUDE_CODE:
        raise SystemExit(f"{path}: Claude Code changed since the first run")


def scrub(text: str) -> str:
    """A server message without host paths: a lock's path from `plans/` on, any other absolute path `<path>`."""
    return ABSOLUTE.sub(lambda m: m[0][m[0].rfind("plans/"):] if "plans/" in m[0] else "<path>", text)


def lock_class(lock: str | None) -> str | None:
    """The GPU class of a lock path, `plans/<model id>/<class>/<toolchain>/<n>.lock`; None for anything else."""
    parts = pathlib.PurePosixPath(lock).parts if lock else ()
    return parts[-3] if len(parts) >= 5 and parts[0] == "plans" else None


def turns(events: list[dict]) -> int | None:
    """The session's turns: the result's count, or, when it never came (a timeout), its assistant messages."""
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is not None:
        return result.get("num_turns")
    ids = {(e.get("message") or {}).get("id") for e in events if e.get("type") == "assistant"}
    return len(ids - {None}) or None


def shop_diff(project: dict[str, bytes], work: pathlib.Path) -> str:
    """The agent's change to `shop/` against the project as registered, as a unified diff."""
    shop = work / "shop"
    if shop.is_symlink() or not shop.is_dir():           # as scoring: a linked package is not read
        return "<no shop/ package>"
    before = {k: v for k, v in project.items() if k.startswith("shop/")}
    after = {f"shop/{k}": v for k, v in tree(shop).items()}
    out = []
    for name in sorted(before.keys() | after.keys()):
        old, new = (d.get(name, b"").decode(errors="replace").splitlines(True) for d in (before, after))
        out += difflib.unified_diff(old, new, f"a/{name}", f"b/{name}")
    return "".join(out)


def private_ok(text: str) -> bool:
    """Whether `text` passes `scripts/ci/check_private.py`'s patterns (host paths, machines, private names)."""
    sys.path.insert(0, str(ROOT / "scripts" / "ci"))
    try:
        from check_private import patterns
    finally:
        sys.path.pop(0)
    return patterns().search(text) is None


def _get(server: str, path: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never through an HTTP proxy
    with opener.open(server.rstrip("/") + path, timeout=60) as r:
        return json.load(r)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    """What Claude Code sent and was served, from the proxy's records: one entry per message request, counts per
    status, length refusals, the peak context, the distinct sampling parameters and the token totals."""
    messages = [r for r in rows if r.get("path", "").split("?")[0] == "/v1/messages" and r.get("method") == "POST"]
    per = []
    for r in messages:
        usage, ttft = r.get("usage") or {}, r.get("ttft_ms")
        n_in, n_out = usage.get("input_tokens"), usage.get("output_tokens")
        per.append({"status": r.get("status"), "input_tokens": n_in, "output_tokens": n_out, "ttft_ms": ttft,
                    "tpot_ms": round((r["total_ms"] - ttft) / (n_out - 1), 2) if ttft is not None and (n_out or 0) > 1
                    else None, "stop_reason": r.get("stop_reason"), "length_refused": bool(r.get("length_refused"))})
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
    sampling = sorted({json.dumps({k: r[k] for k in SAMPLING if k in r}, sort_keys=True) for r in messages})
    contexts = [(p["input_tokens"] or 0) + (p["output_tokens"] or 0) for p in per]
    return {"messages": len(messages), "statuses": statuses, "length_refusals": sum(p["length_refused"] for p in per),
            "peak_context": max(contexts, default=0), "input_tokens": sum(p["input_tokens"] or 0 for p in per),
            "output_tokens": sum(p["output_tokens"] or 0 for p in per),
            "sampling": [json.loads(s) for s in sampling], "requests": per}


def run_once(server: str, model: str, claude: pathlib.Path, project: dict[str, bytes], keep: pathlib.Path) -> dict:
    tests = {k: v for k, v in project.items() if k.startswith("tests/")}
    with tempfile.TemporaryDirectory(prefix="holon-demo-") as tmp:
        tmp = pathlib.Path(tmp)
        work = tmp / "project"
        write_tree(project, work)
        venv = make_venv(tmp / "venv")
        guarded = site_digest()
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
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("ANTHROPIC_", "CLAUDE_", "PYTEST_", "VIRTUAL_ENV", "PYTHON"))}
            env |= {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_AUTH_TOKEN": "holon",
                    "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model, "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                    "CLAUDE_CODE_SUBAGENT_MODEL": model, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "DISABLE_AUTOUPDATER": "1", "NO_PROXY": "127.0.0.1,localhost",
                    "CLAUDE_CONFIG_DIR": str(tmp / "claude"),          # no user settings, memory or hooks
                    "VIRTUAL_ENV": str(venv.parent), "PATH": f"{venv}{os.pathsep}{env.get('PATH', '')}"}
            saved = _get(server, "/metrics").get("prefix_tokens_saved", 0)
            start, timed_out = time.perf_counter(), False
            with open(tmp / "session.jsonl", "w") as log:   # a file, not a pipe: what the session leaves running
                session = subprocess.Popen([str(claude), "-p", PROMPT, "--allowedTools", TOOLS, "--output-format",
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
            wall = time.perf_counter() - start
            saved = _get(server, "/metrics").get("prefix_tokens_saved", 0) - saved
        finally:
            proxy.terminate()
            proxy.wait()
        stdout = (tmp / "session.jsonl").read_text(errors="replace")
        keep.parent.mkdir(parents=True, exist_ok=True)              # the transcript, kept out of the repository
        shutil.copy(tmp / "session.jsonl", keep.with_suffix(".session.jsonl"))
        if captured.exists():
            shutil.copy(captured, keep.with_suffix(".requests.jsonl"))
        events = []
        for line in stdout.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        result = next((e for e in reversed(events) if e.get("type") == "result"), {})
        rows = [json.loads(line) for line in captured.read_text().splitlines()] if captured.exists() else []
        invalid = site_digest() != guarded
        diff = shop_diff(project, work)
        return {"invalid": "the scoring interpreter's packages changed during the run" if invalid else None,
                "scored": None if invalid else score(work, tests), "wall_s": round(wall, 1), "timed_out": timed_out,
                "exit_code": None if timed_out else session.returncode, "turns": turns(events),
                "result_subtype": result.get("subtype"), "is_error": result.get("is_error"),
                "replied_done": (result.get("result") or "").strip() == "done", "prefix_tokens_saved": saved,
                "tool_calls": tool_calls(events, work),
                "shop_diff": diff if private_ok(diff) else "<withheld: matches check_private's patterns>",
                "served": requests_summary(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", help="the running `holon serve`, e.g. http://127.0.0.1:8000")
    parser.add_argument("--gpu-class", help="the server's GPU class; must be its lock's (e.g. rtx-5080)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=pathlib.Path, help="the evidence record to write")
    parser.add_argument("--claude", default="claude", help="the Claude Code executable")
    parser.add_argument("--score", type=pathlib.Path, help="only score a copy of this project directory")
    args = parser.parse_args()
    project = load_project()
    tests = {k: v for k, v in project.items() if k.startswith("tests/")}
    if args.score:
        print(json.dumps(score(args.score, tests)))
        return
    if not (args.server and args.gpu_class and args.out):
        parser.error("--server, --gpu-class and --out are needed for runs")
    claude, claude_sha = resolve_claude(args.claude)
    with tempfile.TemporaryDirectory() as tmp:          # the project as registered fails as pre-registered
        write_tree(project, work := pathlib.Path(tmp) / "project")
        seeded = score(work, tests)
    if {k: seeded[k] for k in SEEDED} != SEEDED:
        sys.exit(f"the project does not fail as pre-registered: {seeded}")
    (model,) = [m["id"] for m in _get(args.server, "/v1/models")["data"]]
    metrics = _get(args.server, "/metrics")
    served = {"plan_id": metrics.get("plan_id"), "lock": metrics.get("lock") and scrub(metrics["lock"]),
              "diff": [scrub(str(d)) for d in metrics.get("diff") or []]}
    if lock_class(served["lock"]) != args.gpu_class:
        sys.exit(f"--gpu-class {args.gpu_class} is not the served lock's class ({served['lock']})")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {"workload": "agent_demo", "date": datetime.date.today().isoformat(), "gpu_class": args.gpu_class,
              "commit": commit, "model": model, "served": served,
              "client": {"claude_code": CLAUDE_CODE, "binary_sha256": claude_sha, "allowed_tools": TOOLS,
                         "timeout_s": TIMEOUT_S, "env": CLIENT_ENV,
                         "command": "claude -p <prompt> --allowedTools <tools> --output-format stream-json --verbose"},
              "task": {"prompt": PROMPT, "project": PROJECT_DIGEST, "tests": TESTS, "seeded": SEEDED},
              "runs": []}
    for i in range(args.runs):
        check_claude(claude, claude_sha)
        keep = OUT / f"{stamp}_{args.gpu_class}_run{i + 1}"
        record["runs"].append(run_once(args.server, model, claude, project, keep))
        args.out.write_text(json.dumps(record, indent=1) + "\n")      # kept after every run
        run = record["runs"][-1]
        print(f"run {i + 1}: {run['invalid'] or json.dumps(run['scored'])}", flush=True)


if __name__ == "__main__":
    main()
