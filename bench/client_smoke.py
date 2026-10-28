"""Real-client smoke at each merge (docs/dev/release.md §22): `holon serve` on a small locked model, driven by the
OpenAI
and Anthropic SDKs in the request shapes real clients send, a few minutes on one GPU. Every request must succeed and its
answer be sane (the right number, a parsable tool call, JSON that meets its schema, a tool result used); the server's
log must hold no traceback.

- OpenAI SDK: models, chat, streamed chat with usage, the legacy completions and the Responses API, tool calls
  (plain and streamed, then the tool's result), a strict `json_schema` response format.
- pi's request shapes (pi itself, `bench/demo/pi_run.py`, needs its pinned npm install; these are the shapes its
  recorded probes sent, `docs/dev/evidence/agent_test_pi_*`): the developer role, `reasoning_effort`,
  `chat_template_kwargs`, `max_completion_tokens`, a replayed assistant turn (null content, `reasoning_content`, tool
  calls) and its tool message, the same request again (a prefix-cache hit in /metrics), `tools: []` with tool history.
- Claude Code's request shapes (as `bench/capture_proxy.py` recorded them, `docs/dev/evidence/claude_code_session_*`;
  Claude Code itself: `bench/demo/run.py`): the beta Messages endpoint with its beta flags, a system of three blocks
  (the per-request billing header, then cached blocks), `$schema` tool schemas without additional properties, adaptive
  thinking with `output_config.effort`, `context_management`, `metadata.user_id`, streamed; a Read tool call, its
  result, then a mid-conversation system message; a side call with `stop_sequences`; `count_tokens`.

    python bench/client_smoke.py [--model unsloth/Qwen3-8B-GGUF:Q4_K_M] [--gpu 0] [--port 0] [--serve-args "..."]
    python bench/client_smoke.py --server http://127.0.0.1:8000        # a server already running

Exit status 1 when a check fails. The server's log goes to bench/.out/client_smoke/serve.log.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

import anthropic
import jsonschema
import openai

OUT = pathlib.Path(__file__).resolve().parent / ".out" / "client_smoke"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
PI_TOOLS = [{"type": "function", "function": {
    "name": "read", "description": "Read a file", "strict": False,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "number"}},
                   "required": ["path"]}}}]
CC_BETAS = ["claude-code-20250219", "interleaved-thinking-2025-05-14", "context-management-2025-06-27",
            "effort-2025-11-24"]
CC_TOOLS = [{"name": name, "description": text, "input_schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
    "properties": {key: {"type": "string", "description": what}}, "required": [key]}}
    for name, text, key, what in (("Read", "Reads a file from the local filesystem.", "file_path", "absolute path"),
                                  ("Bash", "Executes a bash command.", "command", "the command to run"))]
NOTE = "The secret word is PELICAN. The deadline is Friday."


def need(cond: object, what: str) -> None:
    if not cond:
        raise AssertionError(what)


def openai_checks(oa: openai.OpenAI, model: str) -> None:
    r = oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "What is 17 + 25? Reply with "
                                   "the number only."}], max_tokens=32, temperature=0, extra_body=NO_THINKING)
    need("42" in r.choices[0].message.content and r.usage.completion_tokens > 0, f"chat: {r.choices[0].message}")
    text, usage = "", None
    for chunk in oa.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
            max_tokens=32, temperature=0, stream=True, stream_options={"include_usage": True}, extra_body=NO_THINKING):
        text += "".join(c.delta.content or "" for c in chunk.choices)
        usage = chunk.usage or usage
    need("1 2 3 4 5" in text and usage and usage.completion_tokens > 0, f"streamed chat: {text!r}, {usage}")
    r = oa.completions.create(model=model, prompt="The capital of France is", max_tokens=8, temperature=0)
    need("Paris" in r.choices[0].text, f"completions: {r.choices[0].text!r}")
    r = oa.responses.create(model=model, input="What is 6 times 7? Reply with the number only.", temperature=0,
                            max_output_tokens=2048)
    need("42" in r.output_text, f"responses: {r.output_text!r}")

    ask = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    r = oa.chat.completions.create(model=model, messages=ask, tools=[WEATHER], max_tokens=2048, temperature=0)
    msg = r.choices[0].message
    need(r.choices[0].finish_reason == "tool_calls" and msg.tool_calls, f"tool call: {msg}")
    call = msg.tool_calls[0]
    need(call.function.name == "get_weather" and "paris" in json.loads(call.function.arguments)["city"].lower(),
         f"tool call: {call}")
    args = ""
    for chunk in oa.chat.completions.create(model=model, messages=ask, tools=[WEATHER], max_tokens=2048,
                                            temperature=0, stream=True):
        for c in chunk.choices:
            args += "".join(t.function.arguments or "" for t in c.delta.tool_calls or [])
    need("paris" in json.loads(args)["city"].lower(), f"streamed tool call: {args!r}")
    answer = ask + [msg.model_dump(exclude_none=True),
                    {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 31, "sky": "clear"}'}]
    r = oa.chat.completions.create(model=model, messages=answer, tools=[WEATHER], max_tokens=2048, temperature=0)
    reply = r.choices[0].message.content or ""
    need("31" in reply and "<tool_call>" not in reply, f"tool result: {reply!r}")

    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
              "required": ["name", "age"], "additionalProperties": False}
    r = oa.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Invent a person: give a name and an age."}],
        max_tokens=256, temperature=0, extra_body=NO_THINKING,
        response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema, "strict": True}})
    jsonschema.validate(json.loads(r.choices[0].message.content), schema)


def pi_checks(oa: openai.OpenAI, model: str, url: str) -> None:
    def chat(messages: list, **body) -> tuple[str, str, dict[int, str], str | None]:
        """A streamed chat as pi sends it: text, reasoning, tool-call arguments by index, finish reason."""
        text, reasoning, calls, finish = "", "", {}, None
        for chunk in oa.chat.completions.create(model=model, messages=messages, stream=True, store=False,
                                                stream_options={"include_usage": True}, temperature=0,
                                                **{"max_completion_tokens": 2048} | body):
            for c in chunk.choices:
                text += c.delta.content or ""
                reasoning += getattr(c.delta, "reasoning_content", None) or ""
                for t in c.delta.tool_calls or []:
                    calls[t.index] = calls.get(t.index, "") + (t.function.arguments or "")
                finish = c.finish_reason or finish
        return text, reasoning, calls, finish

    system = {"role": "system", "content": "You are a helpful coding assistant."}
    user = {"role": "user", "content": "Name three prime numbers, briefly."}
    for messages, body in (([{"role": "developer", "content": system["content"]}, user], {}),
                           ([system, user], {"reasoning_effort": "medium"}),
                           ([system, user], {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})):
        text, reasoning, _, finish = chat(messages, **body)
        need(finish == "stop" and any(p in text for p in ("2", "3", "5", "7")), f"pi chat {body}: {text!r}")
    _, _, _, finish = chat([system, user], max_completion_tokens=8)
    need(finish == "length", f"pi max_completion_tokens 8: finish {finish}")
    replay = [system, {"role": "user", "content": "Read the file notes.txt and tell me the secret word."},
              {"role": "assistant", "content": None, "reasoning_content": "I should read the file.", "tool_calls": [
                  {"id": "call_1", "type": "function",
                   "function": {"name": "read", "arguments": json.dumps({"path": "notes.txt"})}}]},
              {"role": "tool", "content": NOTE, "tool_call_id": "call_1"}]
    text, *_ = chat(replay, tools=PI_TOOLS)
    need("pelican" in text.lower(), f"pi replay: {text!r}")
    saved = _metrics(url)["prefix_tokens_saved"]
    text, *_ = chat(replay, tools=PI_TOOLS)
    need("pelican" in text.lower() and _metrics(url)["prefix_tokens_saved"] > saved, f"pi replay again: {text!r}")
    text, *_ = chat(replay, tools=[])
    need("pelican" in text.lower(), f"pi tools []: {text!r}")


def claude_code_checks(an: anthropic.Anthropic, model: str) -> None:
    system = [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.300; cch={time.time_ns()};"},
              {"type": "text", "text": "You are Claude Code, a CLI for software engineering.",
               "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "Use the tools to answer. Read files with Read; never guess their contents.",
               "cache_control": {"type": "ephemeral"}}]
    shape = {"model": model, "system": system, "tools": CC_TOOLS, "betas": CC_BETAS, "max_tokens": 32000,
             "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"},
             "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
             "metadata": {"user_id": json.dumps({"device_id": "smoke", "session_id": "smoke"})}}
    messages = [{"role": "user", "content": [{"type": "text", "text": "What is the secret word in /tmp/notes.txt?"}]}]
    with an.beta.messages.stream(messages=messages, **shape) as s:
        first = s.get_final_message()
    uses = [b for b in first.content if b.type == "tool_use"]
    need(first.stop_reason == "tool_use" and uses and uses[0].name == "Read"
         and uses[0].input.get("file_path", "").endswith("notes.txt"), f"Claude Code tool use: {first.content}")
    messages += [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in first.content]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0].id, "content": NOTE}]}]
    with an.beta.messages.stream(messages=messages, **shape) as s:
        second = s.get_final_message()
    text = "".join(b.text for b in second.content if b.type == "text")
    need(second.stop_reason == "end_turn" and "pelican" in text.lower(), f"Claude Code tool result: {text!r}")
    messages += [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in second.content]},
                 {"role": "system", "content": "The user's working directory is /tmp."},
                 {"role": "user", "content": [{"type": "text", "text": "When is the deadline? One word."}]}]
    with an.beta.messages.stream(messages=messages, **shape) as s:
        third = s.get_final_message()
    text = "".join(b.text for b in third.content if b.type == "text")
    need("friday" in text.lower(), f"Claude Code mid-conversation system message: {text!r}")
    side = an.beta.messages.create(model=model, max_tokens=2112, stop_sequences=["</block>"], betas=CC_BETAS,
                                   system=system[:1], messages=[{"role": "user", "content": "Write a five-word title "
                                   "for a chat about pelicans, inside <block></block>."}])
    need(side.content and side.stop_reason in ("stop_sequence", "end_turn"), f"Claude Code side call: {side}")
    count = an.beta.messages.count_tokens(model=model, system=system, tools=CC_TOOLS, messages=messages,
                                          betas=CC_BETAS)
    need(count.input_tokens > 100, f"count_tokens: {count}")


def _metrics(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as r:
        return json.load(r)


def free_port(port: int) -> int:
    """`port` if nothing listens on it (0: one the OS picks), else SystemExit: another server would answer."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"port {port} is busy: another server would answer the smoke") from None
        return s.getsockname()[1]


def serve(a: argparse.Namespace, log: pathlib.Path) -> subprocess.Popen:
    """`holon serve` on the GPU, ready (its /health answers), or SystemExit with the log's tail."""
    env = os.environ | {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(a.gpu)}
    cmd = [sys.executable, "-m", "holon", "serve", a.model, "--port", str(a.port), *shlex.split(a.serve_args)]
    proc = subprocess.Popen(cmd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"holon serve exited ({proc.returncode}):\n{log.read_text()[-3000:]}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=5)
            return proc
        except OSError:
            time.sleep(2)
    proc.kill()
    raise SystemExit(f"holon serve not ready after 900 s:\n{log.read_text()[-3000:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", help="a running server's URL (default: start one)")
    ap.add_argument("--model", default="unsloth/Qwen3-8B-GGUF:Q4_K_M", help="what `holon serve` serves")
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi index of the GPU")
    ap.add_argument("--port", type=int, default=0, help="the server's port (default: a free one)")
    ap.add_argument("--serve-args", default="", help="more `holon serve` flags")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "serve.log"
    started = time.time()
    if not a.server:
        a.port = free_port(a.port)
    proc = None if a.server else serve(a, log)
    url = a.server or f"http://127.0.0.1:{a.port}"
    print(f"server ready in {time.time() - started:.0f} s", flush=True)
    oa = openai.OpenAI(base_url=f"{url}/v1", api_key="holon", max_retries=0, timeout=600)
    an = anthropic.Anthropic(base_url=url, api_key="holon", max_retries=0, timeout=600)
    failed = 0
    try:
        model = oa.models.list().data[0].id
        for name, check in (("openai", lambda: openai_checks(oa, model)), ("pi", lambda: pi_checks(oa, model, url)),
                            ("claude code", lambda: claude_code_checks(an, model))):
            t = time.time()
            try:
                check()
                print(f"ok    {name} ({time.time() - t:.0f} s)", flush=True)
            except Exception as e:  # noqa: BLE001 — every failure is reported, then the next client runs
                failed += 1
                print(f"FAIL  {name} ({time.time() - t:.0f} s): {type(e).__name__}: {e}", flush=True)
    finally:
        if proc is not None and proc.poll() is not None:
            failed += 1
            print(f"FAIL  holon serve exited during the checks ({proc.returncode})", flush=True)
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=60)
    if proc is not None and "Traceback" in (text := log.read_text()):
        failed += 1
        print(f"FAIL  server log holds a traceback ({log}):\n{text[text.index('Traceback'):][:3000]}")
    print(f"{'FAILED' if failed else 'passed'} in {time.time() - started:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
