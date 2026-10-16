"""The harness's own checks of the task (scoring only; never shown to the agent)."""
import json
import pathlib
import subprocess
import sys

import workloads as w

HERE = pathlib.Path(__file__).resolve().parents[1]
ZERO = {"sessions": 0, "requests": 0, "prompt_tokens": 0, "output_tokens": 0, "max_context": 0}


def test_an_empty_trace_is_all_zeros():
    assert w.trace_stats([]) == ZERO


def test_prompts_grow_by_each_turns_output_and_append():
    multi = {"id": 0, "arrival": 0.0, "system": 50, "turns": [w.turn(10, 5), w.turn(20, 7), w.turn(3, 4)]}
    plain = {"id": 1, "arrival": 0.5, "system": 0, "turns": [w.turn(30, 8)]}
    assert w.trace_stats([multi]) == {"sessions": 1, "requests": 3, "prompt_tokens": 240, "output_tokens": 16,
                                      "max_context": 99}
    assert w.trace_stats([multi, plain]) == {"sessions": 2, "requests": 4, "prompt_tokens": 270,
                                             "output_tokens": 24, "max_context": 99}


def test_the_shipped_traces():
    assert w.trace_stats(w.load(HERE / "traces" / "agent.jsonl")) == {
        "sessions": 24, "requests": 126, "prompt_tokens": 363189, "output_tokens": 13473, "max_context": 6135}
    assert w.trace_stats(w.load(HERE / "traces" / "long_prompt.jsonl")) == {
        "sessions": 4, "requests": 4, "prompt_tokens": 30768, "output_tokens": 3104, "max_context": 30032}


def test_the_input_is_not_changed():
    sessions = w.load(HERE / "traces" / "decode_b8.jsonl")
    before = json.dumps(sessions)
    w.trace_stats(sessions)
    assert json.dumps(sessions) == before


def test_the_command_line_prints_a_traces_stats_and_writes_none():
    before = sorted(p.name for p in (HERE / "traces").iterdir())
    trace = HERE / "traces" / "decode_b8.jsonl"
    out = subprocess.run([sys.executable, str(HERE / "workloads.py"), "--stats", str(trace)], capture_output=True,
                         text=True, timeout=120, cwd=HERE)
    assert out.returncode == 0
    assert json.loads(out.stdout) == {"sessions": 8, "requests": 8, "prompt_tokens": 2048, "output_tokens": 2048,
                                      "max_context": 512}
    assert sorted(p.name for p in (HERE / "traces").iterdir()) == before
