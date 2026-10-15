"""The HTTP frontend (docs/m05-grammar.md D7): its three dialects over a scripted fake engine and a byte-level toy
tokenizer (CPU only, no model); the format parsers and constraint builders model-free (constraints checked and
compiled by grammar.xgrammar); the Qwen bundles' formats with the real templates; round trips on a tiny Qwen3 (CPU)
and on Qwen3-0.6B (GPU)."""
import asyncio
import dataclasses
import json
import pathlib
import random

import holon_plugins.frontends.http as http_frontend
import pytest
from fastapi.testclient import TestClient
from holon.api import Logprobs, Manifest, Reasoning, SamplingParams, Text, TokenDelta, ToolArgsDelta, ToolCallStart
from holon_plugins.frontends.http.app import build_app
from holon_plugins.frontends.http.detok import Detokenizer
from holon_plugins.frontends.http.formats import TOOLS, Think
from holon_plugins.frontends.http.pipeline import KEY, Chat, Conversation, Parse
from holon_plugins.frontends.http.web import BadRequest

EOS, STOP = 256, 257                       # ids past the 256 byte tokens
SPECIAL = {EOS: "<|eos|>", STOP: "<|stop|>"}
OUTPUTS = ["Hello, wörld! 你好 🙂", "Second answer"]   # sequence i of a request generates OUTPUTS[i], then EOS


class ByteTokenizer:
    """One token per UTF-8 byte, so every multi-byte character is split across tokens."""

    eos_token_ids = frozenset({EOS})

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, ids: list[int]) -> str:
        out, run = [], bytearray()
        for i in ids:
            if i in SPECIAL:
                out += [run.decode(errors="replace"), SPECIAL[i]]
                run.clear()
            else:
                run.append(i)
        return "".join(out) + run.decode(errors="replace")


class Bundle:
    """A template of `<role>content` turns; `tail` ends the prompt (e.g. an opened `<think>`)."""

    model_name = "toy"
    tokenizer = ByteTokenizer()

    def __init__(self, tool_format=None, reasoning_format=None, tail=""):
        self.tool_format, self.reasoning_format, self.tail = tool_format, reasoning_format, tail
        self.rendered = []

    def render_chat(self, messages, tools=None, add_generation_prompt=True, **kw):
        self.rendered.append((messages, tools, kw))
        text = "".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages)
        return self.tokenizer.encode(text + "<assistant>" + self.tail)



class FakeEngine:
    """Streams `outputs` (OUTPUTS) one token per delta, round-robin over the `n` sequences, like the engine:
    max_tokens and stop ids finish a sequence, the EOS/stop token is part of the last delta, prompt logprobs come
    with sequence 0's first delta. Prompts longer than `max_prompt` are rejected; `endless` never finishes; with
    `fail_after`, the request fails (kind "engine") after that many steps; with `pause`, it is quiet that many seconds
    after its first step."""

    def __init__(self, max_prompt=64, endless=False, outputs=OUTPUTS, fail_after=None, pause=0.0):
        self.max_prompt, self.endless, self.outputs, self.fail_after = max_prompt, endless, outputs, fail_after
        self.pause = pause
        self.requests, self.aborted = [], []

    async def generate(self, token_ids, params: SamplingParams, rid=None, resume_output=0):
        self.requests.append((token_ids, params, rid))
        if len(token_ids) > self.max_prompt:
            yield TokenDelta(rid, [], finish_reason="error", error=f"prompt of {len(token_ids)} tokens is too long",
                             kind="client")
            return
        if token_ids in (list(b"busy"), list(b"fail")):        # the server is busy; the engine failed
            kind = "busy" if token_ids == list(b"busy") else None
            yield TokenDelta(rid, [], finish_reason="error", error="no", kind=kind)
            return
        if self.endless:
            while True:
                await asyncio.sleep(0.001)
                yield TokenDelta(rid, [ord("a")])
        seqs = [self._script(self.outputs[i % len(self.outputs)], params) for i in range(params.n)]
        for step in range(max(map(len, seqs))):
            if step == self.fail_after:
                yield TokenDelta(rid, [], finish_reason="error", error="matcher refused", kind="engine")
                return
            if step == 1:
                await asyncio.sleep(self.pause)
            for i, seq in enumerate(seqs):
                if step < len(seq):
                    token, reason = seq[step]
                    yield TokenDelta(rid, [token], index=i, logprobs=self._lp([token], params.logprobs),
                                     prompt_logprobs=self._lp(token_ids[1:], params.prompt_logprobs)
                                     if i == 0 and step == 0 else None, finish_reason=reason)
            await asyncio.sleep(0)

    @staticmethod
    def _script(text, p):
        out = []
        for t in [*text.encode(), EOS]:
            reason = "stop" if t == EOS or t in p.stop_token_ids else None
            if not reason and p.max_tokens is not None and len(out) + 1 >= p.max_tokens:
                reason = "length"
            out.append((t, reason))
            if reason:
                return out

    @staticmethod
    def _lp(ids, k):
        if k is None:
            return None
        return Logprobs([-0.5] * len(ids), [[t, 0][:k] for t in ids], [[-0.5, float("-inf")][:k] for _ in ids])

    async def abort(self, rid):
        await asyncio.sleep(0.01)          # a real client awaits IPC: the abort must survive cancellation
        self.aborted.append(rid)

    async def stats(self):
        return {"waiting": 0, "running": 1}


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def bundle():
    return Bundle()


@pytest.fixture
def http(engine, bundle):
    return TestClient(build_app(engine, bundle))


def _events(resp):
    lines = [line for line in resp.text.split("\n\n") if line]
    assert all(line.startswith("data: ") for line in lines) and lines[-1] == "data: [DONE]"
    return [json.loads(line[6:]) for line in lines[:-1]]


# -- detokenization ---------------------------------------------------------------------------------------------------

def test_detok_releases_only_complete_characters():
    tok = ByteTokenizer()
    text = "naïve 你好 🙂!"
    d = Detokenizer(tok, tok.encode("prompt"))
    pieces = [d.push([t]) for t in tok.encode(text)]
    assert "".join(pieces) == text and not any("�" in p for p in pieces)
    assert pieces[tok.encode(text).index(0xF0)] == ""               # first byte of the 4-byte emoji: held back
    d = Detokenizer(tok)
    assert d.push(tok.encode("ok")) == "ok" and d.push([0xE4]) == ""
    assert d.push([], final=True) == "�"                       # an incomplete tail is released at the end


def test_detok_matches_full_decode_with_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    text = "Ünïcödé — 日本語のテキスト, emoji 👩‍💻🙂, and code: `x = {'a': 1}`\n"
    ids = tok.encode(text)
    d = Detokenizer(tok, tok.encode("Say:"))
    pieces = [d.push([t]) for t in ids]
    assert "".join(pieces) + d.push([], final=True) == tok.decode(ids)
    assert not any("�" in p for p in pieces)


# -- completions ------------------------------------------------------------------------------------------------------

def test_completion(http, engine):
    r = http.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion" and body["model"] == "toy"
    (choice,) = body["choices"]
    assert choice["text"] == OUTPUTS[0] and choice["finish_reason"] == "stop" and choice["logprobs"] is None
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": len(OUTPUTS[0].encode()) + 1,
                             "total_tokens": 2 + len(OUTPUTS[0].encode()) + 1}
    assert engine.requests[0][0] == [ord("H"), ord("i")] and engine.aborted == []


def test_completion_token_prompts_batch_and_length(http, engine):
    r = http.post("/v1/completions", json={"prompt": [[1, 2], [3]], "max_tokens": 5, "n": 2})
    choices = r.json()["choices"]
    assert [c["index"] for c in choices] == [0, 1, 2, 3]
    assert [c["text"] for c in choices] == [OUTPUTS[0][:5], OUTPUTS[1][:5]] * 2
    assert all(c["finish_reason"] == "length" for c in choices)
    assert [req[0] for req in engine.requests] == [[1, 2], [3]]
    assert r.json()["usage"]["prompt_tokens"] == 3 and r.json()["usage"]["completion_tokens"] == 20


@pytest.mark.parametrize("stream", [False, True])
def test_return_token_ids_covers_every_generated_token(http, engine, stream):
    """Clients counting tokens (the bench's HTTP backend) get every token id, in a chunk per engine delta — even
    a delta whose bytes complete no character yet."""
    r = http.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 100, "stream": stream,
                                           "return_token_ids": True})
    choices = [c for e in _events(r) for c in e["choices"]] if stream else r.json()["choices"]
    ids = [t for c in choices for t in c["token_ids"]]
    assert bytes(ids[:-1]).decode() == OUTPUTS[0] and "".join(c["text"] for c in choices) == OUTPUTS[0]
    assert "token_ids" not in http.post("/v1/completions", json={"prompt": "Hi"}).json()["choices"][0]


@pytest.mark.parametrize("stream", [False, True])
def test_stop_string_across_tokens(http, engine, stream):
    # "wö" spans three byte tokens; the stop is found mid-character and the engine request is aborted.
    r = http.post("/v1/completions", json={"prompt": "Hi", "stop": ["xyz", "wö"], "stream": stream, "max_tokens": 100})
    if stream:
        chunks = [e["choices"][0] for e in _events(r)]
        text, reason = "".join(c["text"] for c in chunks), chunks[-1]["finish_reason"]
    else:
        text, reason = r.json()["choices"][0]["text"], r.json()["choices"][0]["finish_reason"]
    assert (text, reason) == ("Hello, ", "stop")
    assert engine.aborted == [engine.requests[0][2]]


def test_stop_string_in_one_choice_keeps_the_other(http, engine):
    r = http.post("/v1/completions", json={"prompt": "Hi", "n": 2, "stop": "ond", "max_tokens": 100})
    assert [c["text"] for c in r.json()["choices"]] == [OUTPUTS[0], "Sec"]
    assert engine.aborted == []                        # the engine had finished both by the time we stopped reading


def test_null_means_default(http, engine):
    r = http.post("/v1/completions", json={"prompt": "Hi", "temperature": None, "n": None, "stream": None,
                                           "stop_token_ids": None, "max_tokens": None, "stop": None})
    assert r.status_code == 200
    p = engine.requests[0][1]
    assert (p.temperature, p.n, p.stop_token_ids, p.max_tokens) == (1.0, 1, (), None)


def test_stop_token_ids_and_params(http, engine):
    r = http.post("/v1/completions", json={
        "prompt": "Hi", "stop_token_ids": [ord(",")], "temperature": 0, "top_p": 0.9, "top_k": -1, "min_p": 0.1,
        "seed": 7, "ignore_eos": True, "max_tokens": 50})
    assert r.json()["choices"][0]["text"] == "Hello" and r.json()["choices"][0]["finish_reason"] == "stop"
    p = engine.requests[0][1]
    assert (p.temperature, p.top_p, p.top_k, p.min_p, p.seed, p.ignore_eos, p.max_tokens, p.stop_token_ids) == (
        0, 0.9, 0, 0.1, 7, True, 50, (ord(","),))
    assert p.logprobs is None and p.prompt_logprobs is None


def test_completion_echo_logprobs(http, engine):
    r = http.post("/v1/completions", json={"prompt": "Hey", "echo": True, "logprobs": 1, "max_tokens": 2})
    (choice,) = r.json()["choices"]
    assert choice["text"] == "Hey" + OUTPUTS[0][:2]
    lp = choice["logprobs"]
    assert lp["tokens"] == list("Hey" + OUTPUTS[0][:2])
    assert lp["token_logprobs"] == [None, -0.5, -0.5, -0.5, -0.5]
    assert lp["top_logprobs"][0] is None and lp["top_logprobs"][1] == {"e": -0.5}
    assert lp["text_offset"] == [0, 1, 2, 3, 4]
    assert engine.requests[0][1].prompt_logprobs == 1 and engine.requests[0][1].logprobs == 1


def test_completion_stream_logprobs_offsets(http):
    r = http.post("/v1/completions", json={"prompt": "Hi", "logprobs": 2, "max_tokens": 3, "stream": True})
    lps = [e["choices"][0]["logprobs"] for e in _events(r)]
    assert [o for lp in lps for o in lp["text_offset"]] == [0, 1, 2]
    assert all(len(top) == 2 for lp in lps for top in lp["top_logprobs"])


# -- chat -------------------------------------------------------------------------------------------------------------

def test_chat(http, bundle):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    r = http.post("/v1/chat/completions", json={
        "model": "toy", "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}], "tools": tools,
        "chat_template_kwargs": {"enable_thinking": False}})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": OUTPUTS[0]}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert bundle.rendered == [([{"role": "user", "content": "Hi"}], tools, {"enable_thinking": False})]
    assert body["usage"]["prompt_tokens"] == len("<user>Hi<assistant>")


def test_chat_stream_n2_usage(http):
    r = http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}], "n": 2, "stream": True,
        "stream_options": {"include_usage": True}})
    events = _events(r)
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    usage = events.pop()
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == sum(len(o.encode()) + 1 for o in OUTPUTS)
    for i, out in enumerate(OUTPUTS):
        chunks = [e["choices"][0] for e in events if e["choices"][0]["index"] == i]
        assert chunks[0]["delta"]["role"] == "assistant"
        assert "".join(c["delta"].get("content", "") for c in chunks) == out
        assert [c["finish_reason"] for c in chunks][-1] == "stop" and chunks[-1]["delta"] == {}
        assert not any("�" in c["delta"].get("content", "") for c in chunks)


def test_chat_logprobs(http, engine):
    r = http.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}], "logprobs": True, "top_logprobs": 2,
        "max_completion_tokens": 4, "max_tokens": 99})
    content = r.json()["choices"][0]["logprobs"]["content"]
    assert [c["token"] for c in content] == list(OUTPUTS[0][:4])
    assert all(c["logprob"] == -0.5 and len(c["top_logprobs"]) == 2 for c in content)
    assert content[0]["bytes"] == [ord("H")] and content[0]["top_logprobs"][0]["token"] == "H"
    assert content[0]["top_logprobs"][1]["logprob"] == -9999           # a masked token: JSON has no -inf
    assert engine.requests[0][1].max_tokens == 4 and engine.requests[0][1].logprobs == 2


# -- errors -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("stream", [False, True])
def test_rejected_request_status_follows_the_error_kind(http, stream):
    r = http.post("/v1/completions", json={"prompt": "x" * 100, "stream": stream})
    assert r.status_code == 400 and "too long" in r.json()["error"]["message"]
    r = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 100}],
                                                "stream": stream})
    assert r.status_code == 400 and "too long" in r.json()["error"]["message"]
    assert http.post("/v1/completions", json={"prompt": "busy", "stream": stream}).status_code == 503
    assert http.post("/v1/completions", json={"prompt": "fail", "stream": stream}).status_code == 500


@pytest.mark.parametrize("body", [
    {"prompt": "Hi", "logit_bias": {"1": 5}},
    {"prompt": "Hi", "n": 0},
    {"prompt": "Hi", "temperature": -1},
    {"prompt": []},
    {"prompt": [[1], [2]], "stream": True},
])
def test_invalid_request_is_400(http, engine, body):
    r = http.post("/v1/completions", json=body)
    assert r.status_code == 400 and r.json()["error"]["message"]
    assert engine.requests == []


# -- other endpoints --------------------------------------------------------------------------------------------------

def test_models_health_metrics(http, engine, bundle, monkeypatch):
    """The served id: the checkpoint name's last segment, or `http.model_name`; the startup hint names it."""
    from holon_plugins.frontends.http.plugin import hint

    assert [m["id"] for m in http.get("/v1/models").json()["data"]] == ["toy"]
    assert http.get("/health").status_code == 200
    assert http.get("/metrics").json() == {"waiting": 0, "running": 1, "billing_headers_dropped": 0}
    monkeypatch.setattr(bundle, "model_name", "org/repo/Qwen3-8B-Q4_K_M")
    for model_id, want in ((None, "Qwen3-8B-Q4_K_M"), ("mine", "mine")):
        client = TestClient(build_app(engine, bundle, model_id=model_id))
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [want]
    assert client.post("/v1/messages", json={"model": "anything", "max_tokens": 2, "messages": [
        {"role": "user", "content": "Hi"}]}).json()["model"] == "mine"
    assert hint("0.0.0.0", 8001, "a b") == (
        "holon: serving a b at http://127.0.0.1:8001 (OpenAI: /v1; Anthropic: /v1/messages)\nholon: Claude Code:\n"
        "  ANTHROPIC_BASE_URL=http://127.0.0.1:8001 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL='a b' "
        "ANTHROPIC_DEFAULT_HAIKU_MODEL='a b' CLAUDE_CODE_SUBAGENT_MODEL='a b' claude")


# -- client disconnect ------------------------------------------------------------------------------------------------

def _asgi_call(app, path, payload, disconnect=None):
    """Drive the app directly (Starlette's TestClient buffers whole responses and never disconnects). The client
    disconnects after the first body chunk ("after_body"), shortly after sending ("soon"), or never (None).
    Returns the response status."""
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
             "query_string": b"", "headers": [(b"content-type", b"application/json")],
             "server": ("test", 80), "client": ("test", 1)}
    messages = [{"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}]
    got_body, status = asyncio.Event(), []

    async def receive():
        if messages:
            return messages.pop()
        if disconnect == "soon":
            await asyncio.sleep(0.05)
        else:
            await got_body.wait()
            if disconnect is None:
                await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])
        if message["type"] == "http.response.body" and message.get("body"):
            got_body.set()

    asyncio.run(asyncio.wait_for(app(scope, receive, send), 5))
    return status[0]


@pytest.mark.parametrize("stream", [False, True])
def test_disconnect_aborts(bundle, stream):
    engine = FakeEngine(endless=True)
    _asgi_call(build_app(engine, bundle), "/v1/completions", {"prompt": "Hi", "stream": stream, "max_tokens": None},
               disconnect="after_body" if stream else "soon")
    assert len(engine.requests) == 1 and engine.aborted == [engine.requests[0][2]]


def test_rejected_prompt_in_batch_aborts_the_others(bundle):
    engine = FakeEngine(endless=True)
    status = _asgi_call(build_app(engine, bundle), "/v1/completions", {"prompt": ["x" * 100, "Hi"], "max_tokens": None})
    assert status == 400 and engine.aborted == [engine.requests[1][2]]


def test_plugin_manifest():
    m = Manifest.load(pathlib.Path(http_frontend.__file__).parent / "plugin.toml")
    assert m.slot == "frontend"
    assert [str(r) for r in m.provides] == ["frontend.openai@1", "frontend.anthropic@1", "frontend.openai_responses@1"]


# -- parsers (formats.py): R §4.2's fixtures; a streamed parse equals the whole text's ---------------------------------

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
TYPED = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
    "s": {"type": "string"}, "n": {"type": "integer"}, "b": {"type": "boolean"}, "t": {"type": "boolean"},
    "u": {}, "z": {"type": ["string", "null"]}, "l": {"type": "array"}, "m": {"type": "string"},
    "o": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}}
HERMES_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
XML_CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n" \
           "</parameter>\n</function>\n</tool_call>"
XML_TYPED = "<tool_call>\n<function=f>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in [
    ("s", "007"), ("n", "3"), ("b", "True"), ("t", "true"), ("u", "True"), ("z", "None"), ("l", "[1, 2]"),
    ("m", "line 1\nline 2"), ("o", "5")]) + "</function>\n</tool_call>"
TYPED_ARGS = {"s": "007", "n": 3, "t": True, "l": [1, 2], "m": "line 1\nline 2", "o": "5"}   # o: Optional[str]
MARKED = {"city": "Paris", "note": "a </tool_call> and a <tool_call> in a string"}
START, OPEN, CLOSED = "<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n", "<think>\n\n</think>\n\n"

PARSES = {   # format, prompt tail, tools, output -> the answer: (kind, text) and ("call", name, arguments)
    "hermes, the model opens its reasoning": (
        "hermes@1", START, [WEATHER], f"<think>\nLet me check.\n</think>\n\n{HERMES_CALL}",
        [("reasoning", "Let me check."), ("call", "get_weather", {"city": "Paris"})]),
    "hermes, text and two calls": (
        "hermes@1", START, [WEATHER], f"Sure.\n{HERMES_CALL}\n{HERMES_CALL.replace('Paris', 'Rome')}",
        [("text", "Sure."), ("call", "get_weather", {"city": "Paris"}), ("call", "get_weather", {"city": "Rome"})]),
    "hermes, markers inside an argument string": (
        "hermes@1", CLOSED, [WEATHER], f'<tool_call>\n{{"name": "get_weather", "arguments": {json.dumps(MARKED)}}}'
                                       f"\n</tool_call>", [("call", "get_weather", MARKED)]),
    "hermes, arguments written as a JSON string": (
        "hermes@1", CLOSED, [WEATHER], '<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}'
                                       "\n</tool_call>", [("call", "get_weather", {"city": "Paris"})]),
    "hermes, a body that is no call stays text": (
        "hermes@1", CLOSED, [WEATHER], "<tool_call>\n{oops}\n</tool_call> done",
        [("text", "<tool_call>\n{oops}\n</tool_call> done")]),
    "hermes, cut inside a call": (
        "hermes@1", CLOSED, [WEATHER], '<tool_call>\n{"name": "get_weather", "arguments": {"ci',
        [("text", '<tool_call>\n{"name": "get_weather", "arguments": {"ci')]),
    "xml, a call ends the reasoning without </think>": (
        "qwen_xml@2", OPEN, [WEATHER], f"I should call it.\n{XML_CALL}",
        [("reasoning", "I should call it."), ("call", "get_weather", {"city": "Paris", "days": 3})]),
    "xml, closed reasoning then text and a call": (
        "qwen_xml@2", OPEN, [WEATHER], f"Hm.\n</think>\n\nCalling.\n\n{XML_CALL}\n",
        [("reasoning", "Hm."), ("text", "Calling."), ("call", "get_weather", {"city": "Paris", "days": 3})]),
    "xml@1, values as Qwen3.5 writes them (| string)": (
        "qwen_xml@1", CLOSED, [TYPED], XML_TYPED, [("call", "f", {**TYPED_ARGS, "b": True, "u": True, "z": None})]),
    "xml@2, values as Qwen3.8 writes them (tojson)": (
        "qwen_xml@2", CLOSED, [TYPED], XML_TYPED,
        [("call", "f", {**TYPED_ARGS, "b": "True", "u": "True", "z": "None"})]),
    "xml, cut inside a parameter": (
        "qwen_xml@2", CLOSED, [WEATHER], "<tool_call>\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n"
                                        "<parameter=city>\nPar", [("call", "get_weather", {"days": 2})]),
    "xml, a marker that starts no call stays text": (
        "qwen_xml@2", CLOSED, [WEATHER], "Use <tool_call> tags, and <think> too.",
        [("text", "Use <tool_call> tags, and <think> too.")]),
}


def _coalesced(events):
    out = []
    for e in events:
        if out and type(e) is type(out[-1]) and isinstance(e, Text | Reasoning | ToolArgsDelta):
            out[-1] = dataclasses.replace(out[-1], text=out[-1].text + e.text)
        else:
            out.append(e)
    return out


def _answer(events):
    """(kind, text) runs and ("call", name, arguments) of coalesced events."""
    out = []
    for e in events:
        if isinstance(e, ToolCallStart):
            out.append(("call", e.name))
        elif isinstance(e, ToolArgsDelta):
            out[-1] += (json.loads(e.text),)
        elif isinstance(e, Text | Reasoning):
            out.append((type(e).__name__.lower(), e.text))
    return out


def _parse(fmt, tail, tools, text, cuts):
    parser = Parse(Think(tail, ("<tool_call>",)), TOOLS[fmt](tools))
    pieces = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]
    return _coalesced([e for i, piece in enumerate(pieces) for e in parser.feed(piece, i == len(pieces) - 1)])


@pytest.mark.parametrize("case", PARSES)
def test_streamed_parse_equals_whole_parse(case):
    fmt, tail, tools, text, expected = PARSES[case]
    whole = _parse(fmt, tail, tools, text, [])
    assert _answer(whole) == expected
    rng = random.Random(0)
    for cuts in [list(range(1, len(text))), *(sorted(rng.sample(range(1, len(text)), k)) for k in (2, 5, 17))]:
        assert _parse(fmt, tail, tools, text, cuts) == whole


# -- constraints (D6, D8): every builder's output passes grammar.xgrammar's check and compiles; what it admits parses --

STRICT = {"type": "function", "function": {"name": "get_weather", "strict": True, "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"], "additionalProperties": False}}}
PARIS = {"type": "object", "properties": {"city": {"enum": ["Paris"]}}, "required": ["city"],
         "additionalProperties": False}
XML_STRICT = XML_CALL                                   # city and days, as STRICT asks
XML_BAD = XML_CALL.replace("\n3\n", "\nthree\n")        # days not an integer
HERMES_BAD = HERMES_CALL.replace('"Paris"', "3")        # city not a string

CONSTRAINED = {  # format, prompt tail, conversation fields -> (admitted outputs and their calls, refused outputs)
    "hermes required, the model may open its reasoning": (
        "hermes@1", START, {"tools": [WEATHER], "tool_choice": "required"},
        [(f"<think>\nhm\n</think>\n\n{HERMES_CALL}", 1), (f"{HERMES_CALL}\n{HERMES_CALL}", 2)],
        ["Paris is sunny.", HERMES_BAD, "<think>\nhm\n</think>\n\nsunny"]),
    "hermes named, thinking off": (
        "hermes@1", CLOSED, {"tools": [WEATHER], "tool_choice": {"name": "get_weather"}},
        [(HERMES_CALL, 1)], [f"{HERMES_CALL}\n{HERMES_CALL}", f"x{HERMES_CALL}", f"<think>\n</think>{HERMES_CALL}"]),
    "xml required, one call, reasoning open": (
        "qwen_xml@2", OPEN, {"tools": [WEATHER], "tool_choice": "required", "parallel_tool_calls": False},
        [(f"I should call it.\n{XML_CALL}", 1), (f"hm\n</think>\n\n{XML_CALL}", 1)],
        [f"hm\n</think>\n\n{XML_CALL}\n{XML_CALL}", "hm\n</think>\n\nsunny", "hm\n" + XML_CALL.replace("get", "set")]),
    "xml@1 auto with a strict tool": (
        "qwen_xml@1", OPEN, {"tools": [STRICT]},
        [("hm\n</think>\n\nIt is sunny.", 0), (f"hm\n</think>\n\nLet me look.\n{XML_STRICT}", 1)],
        [f"hm\n</think>\n\n{XML_BAD}"]),
    "auto, one call at most": (
        "hermes@1", CLOSED, {"tools": [WEATHER], "parallel_tool_calls": False},
        [("It is sunny.", 0), (HERMES_CALL, 1)], [f"{HERMES_CALL}\n{HERMES_CALL}"]),
    "json_schema after open reasoning": (
        "hermes@1", OPEN, {"response_format": {"type": "json_schema", "schema": PARIS, "strict": True}},
        [('hm\n</think>\n\n{"city": "Paris"}', 0)], ['{"city": "Paris"}', 'hm\n</think>\n\n{"city": "Rome"}']),
    "json_object, thinking off": (
        "qwen_xml@2", CLOSED, {"response_format": {"type": "json_object"}}, [('{"a": [1, {"b": null}]}', 0)],
        ["[1]", '"a"', "hm</think>{}"]),
}


@pytest.mark.parametrize("case", CONSTRAINED)
def test_constraints_compile_and_admit_what_parses(case):
    check = pytest.importorskip("holon_plugins.grammar.xgrammar.constraint")
    from xgrammar.testing import _is_grammar_accept_string as admits

    fmt, tail, fields, admitted, refused = CONSTRAINED[case]
    chat = Chat(Bundle(fmt, "think@1", tail), Conversation([{"role": "user", "content": "Weather?"}], **fields))
    check.check(chat.constraint)
    grammar = check.grammar(chat.constraint)
    for text, calls in admitted:
        assert admits(grammar, text), text
        events = chat.parser().feed(text, final=True)
        assert sum(isinstance(e, ToolCallStart) for e in events) == calls, text
    for text in refused:
        assert not admits(grammar, text), text


@pytest.mark.parametrize("fields, error", [
    ({"tools": [WEATHER]}, None),                                                      # auto: free (D8)
    ({"tools": [STRICT], "tool_choice": "none"}, None),                                # none: no tools at all
    ({"tools": [WEATHER], "tool_choice": "required", "response_format": {"type": "json_object"}}, "combined"),
    ({"tool_choice": "required"}, "needs tools"),
    ({"tools": [WEATHER], "tool_choice": {"name": "nope"}}, "not among the tools"),
])
def test_unconstrained_and_refused_choices(fields, error):
    conv = Conversation([{"role": "user", "content": "Weather?"}], **fields)
    if error:
        with pytest.raises(BadRequest, match=error):
            Chat(Bundle("hermes@1", "think@1"), conv)
        return
    chat = Chat(Bundle("hermes@1", "think@1"), conv)
    assert chat.constraint is None and chat.ext == {}
    assert (chat.tools == []) == (fields.get("tool_choice") == "none")


# -- the dialects: tool calls, constraints, failures after the first token ------------------------------------------

REASONED_CALL = f"Checking.\n{XML_CALL}"                 # an answer of a model whose prompt opened its reasoning


def _app(outputs, fail_after=None, tool_format="qwen_xml@2", tail="<think>\n", pause=0.0):
    engine = FakeEngine(max_prompt=4096, outputs=outputs, fail_after=fail_after, pause=pause)
    bundle = Bundle(tool_format, "think@1", tail)
    return TestClient(build_app(engine, bundle)), engine, bundle


def _sse(resp):
    """(event, data) of a stream of named server-sent events."""
    out = []
    for chunk in filter(None, resp.text.split("\n\n")):
        event, data = chunk.split("\n", 1)
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def _without_ids(value):
    if isinstance(value, dict):
        return {k: _without_ids(v) for k, v in value.items() if k not in ("id", "call_id")}
    return [_without_ids(v) for v in value] if isinstance(value, list) else value


def test_chat_tool_calls_streamed_as_whole():
    http, engine, _ = _app([REASONED_CALL])
    body = {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER]}
    choice = http.post("/v1/chat/completions", json=body).json()["choices"][0]
    (call,) = choice["message"]["tool_calls"]
    assert call["id"].startswith("call_") and choice["finish_reason"] == "tool_calls"
    assert choice["message"] == {"role": "assistant", "content": None, "reasoning_content": "Checking.",
                                 "tool_calls": [call]}
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert engine.requests[0][1].ext == {}                                  # auto: unconstrained (D8)

    chunks = [e["choices"][0] for e in _events(http.post("/v1/chat/completions", json=body | {"stream": True}))]
    message = {"role": "assistant", "content": None, "reasoning_content": "", "tool_calls": []}
    for d in (c["delta"] for c in chunks):
        message["reasoning_content"] += d.get("reasoning_content", "")
        for t in d.get("tool_calls", []):
            if "id" in t:
                message["tool_calls"].append({"id": t["id"], "type": t["type"], "function": dict(t["function"])})
            else:
                message["tool_calls"][t["index"]]["function"]["arguments"] += t["function"]["arguments"]
    assert _without_ids(message) == _without_ids(choice["message"]) and chunks[-1]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("fields, constraint", [
    ({"tool_choice": "required"}, {"type": "triggered_tags", "at_least_one": True, "stop_after_first": False}),
    ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, {"type": "tag"}),
    ({"parallel_tool_calls": False}, {"type": "triggered_tags", "at_least_one": False, "stop_after_first": True}),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": PARIS, "strict": True}}},
     {"type": "json_schema", "schema": PARIS, "strict": True}),
    ({"tool_choice": "any"}, 400),
    ({"response_format": {"type": "xml"}}, 400),
])
def test_chat_constraints_reach_the_engine(fields, constraint):
    http, engine, _ = _app(["{}"], tool_format="hermes@1", tail=CLOSED)
    r = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Weather?"}],
                                                "tools": [WEATHER]} | fields)
    if constraint == 400:
        assert r.status_code == 400 and engine.requests == []
        return
    sent = engine.requests[0][1].ext[KEY]
    assert {k: sent[k] for k in constraint} == constraint


DIALECTS = {   # route, body -> the request's
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "tools": [WEATHER]}),
    "messages": ("/v1/messages", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}),
    "responses": ("/v1/responses", {"input": "Hi"}),
}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("dialect", DIALECTS)
def test_failure_after_the_first_token_is_no_client_error(dialect, stream):
    """D7: an engine failure after admission (a refused commit, kind "engine") is a 5xx or a finish reason, never a
    400 — after the first token, OpenAI chat finishes the choice with "error"; the event-stream dialects send their
    error event."""
    http, _, _ = _app([REASONED_CALL], fail_after=3)
    route, body = DIALECTS[dialect]
    r = http.post(route, json=body | {"stream": stream})
    if not stream:
        assert (r.status_code, dialect == "chat" and r.json()["choices"][0]["finish_reason"]) == (
            (200, "error") if dialect == "chat" else (500, False))
    elif dialect == "chat":
        assert r.status_code == 200 and _events(r)[-1]["choices"][0]["finish_reason"] == "error"
    else:
        assert r.status_code == 200 and _sse(r)[-1][0] == ("error" if dialect == "messages" else "response.failed")


@pytest.mark.parametrize("dialect", DIALECTS)
def test_quiet_streams_send_keepalives(monkeypatch, dialect):
    """A stream quiet for KEEPALIVE seconds (a long call is held back until it closes) sends a keepalive — Anthropic's
    `ping` event, an SSE comment otherwise — and goes on to its normal end."""
    from holon_plugins.frontends.http import web

    monkeypatch.setattr(web, "KEEPALIVE", 0.02)
    http, _, _ = _app([REASONED_CALL], pause=0.2)
    route, body = DIALECTS[dialect]
    text = http.post(route, json=body | {"stream": True}).text
    keepalive, end = {"chat": (": keepalive", "data: [DONE]"), "messages": ("event: ping", "event: message_stop"),
                      "responses": (": keepalive", "event: response.completed")}[dialect]
    assert text.index(keepalive) < text.index(end)


ANTHROPIC = {
    "model": "toy", "max_tokens": 50,
    "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=00000"},
               {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}],
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "Use tools."}]},     # Claude Code's inline system
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Look it up.", "signature": "s"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"},
                                     {"type": "text", "text": "<system-reminder>be brief</system-reminder>"},
                                     {"type": "text", "text": "And Rome?"}]},
        {"role": "system", "content": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]}],
    "tools": [{"name": "get_weather", "description": "Weather", "input_schema": WEATHER["function"]["parameters"]}],
    "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    "thinking": {"type": "disabled"},
}


def test_anthropic_request_as_a_conversation():
    http, engine, bundle = _app([XML_CALL], tail=CLOSED)
    assert http.post("/v1/messages", json=ANTHROPIC).status_code == 200
    messages, tools, kwargs = bundle.rendered[0]
    assert messages == [
        {"role": "system", "content": "You are terse.\nUse tools."},      # the billing header dropped
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "Look it up.", "tool_calls": [
            {"id": "toolu_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"},
        {"role": "user", "content": "<system-reminder>be brief</system-reminder>\nAnd Rome?"},
        {"role": "user", "content": "Be brief."}]                          # a later system message: a user turn
    assert tools == [{"type": "function", "function": {"name": "get_weather", "description": "Weather",
                                                       "parameters": WEATHER["function"]["parameters"]}}]
    assert kwargs == {"enable_thinking": False}
    sent = engine.requests[0][1].ext[KEY]                                    # any + no parallel: exactly one call
    assert (sent["type"], sent["at_least_one"], sent["stop_after_first"]) == ("triggered_tags", True, True)
    assert http.get("/metrics").json()["billing_headers_dropped"] == 1
    count = http.post("/v1/messages/count_tokens", json={k: v for k, v in ANTHROPIC.items() if k != "max_tokens"})
    assert count.json() == {"input_tokens": len(engine.requests[0][0])}
    r = http.post("/v1/messages", json={"messages": []})
    assert r.status_code == 400 and r.json()["type"] == "error" and r.json()["error"]["type"] == "invalid_request_error"


def test_anthropic_answer_streamed_as_whole():
    http, _, _ = _app([REASONED_CALL])
    body = {"max_tokens": 200, "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": ANTHROPIC["tools"]}
    message = http.post("/v1/messages", json=body).json()
    assert message["content"][1]["id"].startswith("toolu_") and message["stop_reason"] == "tool_use"
    assert _without_ids(message["content"]) == [
        {"type": "thinking", "thinking": "Checking.", "signature": ""},
        {"type": "tool_use", "name": "get_weather", "input": {"city": "Paris", "days": 3}}]
    events = _sse(http.post("/v1/messages", json=body | {"stream": True}))
    assert [e for e, _ in events[:1] + events[-2:]] == ["message_start", "message_delta", "message_stop"]
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    blocks: dict[int, dict] = {}
    for event, data in events:
        if event == "content_block_start":
            blocks[data["index"]] = dict(data["content_block"]) | {"json": ""}
        elif event == "content_block_delta":
            b, delta = blocks[data["index"]], data["delta"]
            key = {"thinking_delta": "thinking", "text_delta": "text", "input_json_delta": "json",
                   "signature_delta": "signature"}[delta["type"]]
            b[key] += delta["partial_json" if key == "json" else key]
        elif event == "content_block_stop":
            b = blocks[data["index"]]
            b |= {"input": json.loads(b["json"])} if b["type"] == "tool_use" else {}
            del b["json"]
    assert _without_ids([blocks[i] for i in sorted(blocks)]) == _without_ids(message["content"])


# -- OpenAI Responses --------------------------------------------------------------------------------------------------

def test_responses_input_items_and_answer_streamed_as_whole():
    http, engine, bundle = _app([REASONED_CALL])
    body = {"instructions": "Be terse.", "tools": [{"type": "function", **WEATHER["function"]}], "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Look it up."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "Sunny"}]}
    answer = http.post("/v1/responses", json=body).json()
    assert bundle.rendered[0][0] == [
        {"role": "system", "content": "Be terse."}, {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "Checking.", "reasoning_content": "Look it up.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}]
    assert bundle.rendered[0][1] == [WEATHER] and engine.requests[0][1].ext == {}
    assert answer["status"] == "completed" and answer["usage"]["output_tokens"] == len(REASONED_CALL) + 1
    assert _without_ids(answer["output"]) == [
        {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Checking."}]},
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "Paris", "days": 3}',
         "status": "completed"}]
    events = _sse(http.post("/v1/responses", json=body | {"stream": True}))
    assert [d["sequence_number"] for _, d in events] == list(range(len(events)))
    assert [e for e, _ in events[:2]] == ["response.created", "response.in_progress"]
    assert events[-1][0] == "response.completed" and _without_ids(events[-1][1]["response"]["output"]) == \
        _without_ids(answer["output"])
    deltas = "".join(d["delta"] for e, d in events if e == "response.function_call_arguments.delta")
    assert [d["item"] for e, d in events if e == "response.output_item.done"] == events[-1][1]["response"]["output"]
    assert deltas == answer["output"][1]["arguments"]
    assert http.post("/v1/responses", json={"input": "Hi", "store": True}).status_code == 400


# -- the Qwen bundles' formats, read off their templates ------------------------------------------------------------

def _snapshot(name):
    return max(pathlib.Path.home().glob(f".cache/huggingface/hub/models--Qwen--{name}/snapshots/*"), default=None)


@pytest.mark.parametrize("name, module, fmt", [("Qwen3-0.6B", "qwen3", "hermes@1"),
                                               ("Qwen3.5-9B", "qwen3_5", "qwen_xml@1"),
                                               ("Qwen3.8-27B", "qwen3_5", "qwen_xml@2")])
def test_bundle_formats_follow_the_template(name, module, fmt):
    """A bundle names the format its template writes (R §4.1); a call sent with JSON-text arguments (OpenAI's wire
    form, which the XML templates refuse) renders, and the frontend's parser reads the rendered call back."""
    import importlib
    import types

    from transformers import AutoTokenizer

    if (snap := _snapshot(name)) is None:
        pytest.skip(f"needs {name}'s tokenizer in the local Hugging Face cache")
    tok = AutoTokenizer.from_pretrained(snap)
    source = types.SimpleNamespace(name=name, tokenizer=lambda: tok, eos_token_ids=frozenset())
    bundle = importlib.import_module(f"holon_plugins.models.{module}.bundle").Bundle(source)
    assert (bundle.tool_format, bundle.reasoning_format) == (fmt, "think@1")
    args = {"city": "Paris", "days": 3}
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(args)}}
    ids = bundle.render_chat([{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "",
                                                                        "tool_calls": [call]}], tools=[WEATHER],
                             add_generation_prompt=False)
    text = bundle.tokenizer.decode(ids)
    rendered = text[text.rindex("<tool_call>"):].rstrip()               # the template lists the format first
    assert _answer(_coalesced(TOOLS[fmt]([WEATHER]).feed(rendered, final=True))) == [("call", "get_weather", args)]


# -- round trips: a tiny Qwen3 on the CPU, Qwen3-0.6B on the GPU, served through the app ----------------------------

class LocalClient:
    """An EngineClient stepping an in-process engine (one request at a time)."""

    def __init__(self, engine):
        self.engine = engine

    async def generate(self, token_ids, params, rid=None, resume_output=0):
        try:
            self.engine.add(token_ids, params, rid=rid)
        except ValueError as e:
            yield TokenDelta(rid, [], finish_reason="error", error=str(e), kind=getattr(e, "kind", "client"))
            return
        running = params.n
        while running:
            if wait := self.engine.idle_for():
                await asyncio.sleep(wait)
            for d in self.engine.step():
                running -= d.finish_reason is not None
                yield d

    async def abort(self, rid):
        self.engine.abort(rid)

    async def stats(self):
        return self.engine.stats()


def _served(engine):
    from holon_plugins.models.qwen3.bundle import Bundle as Qwen3

    return TestClient(build_app(LocalClient(engine), Qwen3(engine.source)))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny random Qwen3 over Qwen3's vocabulary, tokenizer and template, serving grammar.xgrammar. Whitespace-only
    tokens get zero logits, so its greedy JSON does not run to `max_tokens` in whitespace (D5: no cap)."""
    import torch
    from holon.core.engine import Engine, EngineConfig
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("needs Qwen3-0.6B's tokenizer in the local Hugging Face cache")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, tie_word_embeddings=False, eos_token_id=tok.convert_tokens_to_ids(
            "<|im_end|>")))
    with torch.no_grad():
        model.lm_head.weight[[i for i in range(len(tok)) if not tok.decode([i]).strip()]] = 0
    path = tmp_path_factory.mktemp("tiny_qwen3_chat")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    engine = Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=2,
                                 max_step_tokens=512, max_model_len=1024, state_memory=16 << 20,
                                 slots={"logits_processor": "grammar.xgrammar"}, plugin_options={"grammar.workers": 1}))
    yield _served(engine)
    engine.close()


CITY = {"type": "function", "function": {"name": "get_weather", "strict": True, "parameters": {
    "type": "object", "properties": {"city": {"enum": ["Paris", "Rome"]}}, "required": ["city"],
    "additionalProperties": False}}}


def _round_trip(http, dialect):
    """A forced call, then the conversation with its result: (the call's arguments, the second answer's prompt
    tokens beyond the first's, the second answer's status)."""
    question = "What is the weather in Paris?"
    if dialect == "chat":
        body = {"messages": [{"role": "user", "content": question}], "tools": [CITY], "temperature": 0,
                "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False}}
        first = http.post("/v1/chat/completions", json=body | {"tool_choice": {
            "type": "function", "function": {"name": "get_weather"}}}).json()
        (call,), finish = first["choices"][0]["message"]["tool_calls"], first["choices"][0]["finish_reason"]
        args, n_prompt = json.loads(call["function"]["arguments"]), first["usage"]["prompt_tokens"]
        body["messages"] += [first["choices"][0]["message"], {"role": "tool", "tool_call_id": call["id"],
                                                               "content": "Sunny"}]
        second = http.post("/v1/chat/completions", json=body | {"max_tokens": 4})
        return args, finish == "tool_calls", second.json()["usage"]["prompt_tokens"] - n_prompt, second.status_code
    if dialect == "messages":
        tool = {"name": "get_weather", "input_schema": CITY["function"]["parameters"], "strict": True}
        body = {"messages": [{"role": "user", "content": question}], "tools": [tool], "temperature": 0,
                "max_tokens": 100, "thinking": {"type": "disabled"}}
        events = _sse(http.post("/v1/messages", json=body | {"tool_choice": {"type": "tool", "name": "get_weather"},
                                                             "stream": True}))
        start = next(d["content_block"] for e, d in events if e == "content_block_start")
        args = json.loads("".join(d["delta"]["partial_json"] for e, d in events if e == "content_block_delta"))
        n_prompt = events[0][1]["message"]["usage"]["input_tokens"]
        body["messages"] += [{"role": "assistant", "content": [start | {"input": args}]}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": start["id"], "content": "Sunny"}]}]
        second = http.post("/v1/messages", json=body | {"max_tokens": 4})
        stop = events[-2][1]["delta"]["stop_reason"]
        return args, stop == "tool_use", second.json()["usage"]["input_tokens"] - n_prompt, second.status_code
    body = {"input": question, "tools": [{"type": "function", **CITY["function"]}], "temperature": 0,
            "max_output_tokens": 100}
    first = http.post("/v1/responses", json=body | {"tool_choice": {"type": "function", "name": "get_weather"}}).json()
    (call,) = first["output"]
    body["input"] = [{"role": "user", "content": question}, call,
                     {"type": "function_call_output", "call_id": call["call_id"], "output": "Sunny"}]
    second = http.post("/v1/responses", json=body | {"max_output_tokens": 4})
    n_more = second.json()["usage"]["input_tokens"] - first["usage"]["input_tokens"]
    return json.loads(call["arguments"]), call["type"] == "function_call", n_more, second.status_code


@pytest.mark.parametrize("dialect", DIALECTS)
def test_round_trip_on_a_tiny_model(tiny, dialect):
    args, called, n_more, status = _round_trip(tiny, dialect)
    assert args["city"] in ("Paris", "Rome") and called and n_more > 0 and status == 200


def test_round_trip_on_qwen3_0_6b():
    """A real model through the app (GPU, graphs captured, grammar.xgrammar selected): a required call to a tool,
    the answer from its result (streamed), reasoning the model opens itself (Anthropic, streamed), and a strict
    JSON schema (Responses)."""
    import torch
    from holon.core.engine import Engine, EngineConfig

    if (snap := _snapshot("Qwen3-0.6B")) is None or not torch.cuda.is_available():
        pytest.skip("needs Qwen3-0.6B + GPU")
    engine = Engine(EngineConfig(model=str(snap), dtype=torch.bfloat16, max_seqs=4, max_model_len=2048,
                                 slots={"logits_processor": "grammar.xgrammar"}))
    try:
        http = _served(engine)
        tool = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city",
                                                 "parameters": {"type": "object", "properties": {
                                                     "city": {"type": "string"}}, "required": ["city"]}}}
        body = {"messages": [{"role": "user", "content": "What is the weather like in Paris today?"}],
                "tools": [tool], "temperature": 0, "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False}}
        first = http.post("/v1/chat/completions", json=body | {"tool_choice": "required",
                                                               "parallel_tool_calls": False}).json()["choices"][0]
        (call,) = first["message"]["tool_calls"]
        assert first["finish_reason"] == "tool_calls" and json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        body["messages"] += [first["message"], {"role": "tool", "tool_call_id": call["id"],
                                                "content": "22 degrees Celsius and sunny"}]
        chunks = [e["choices"][0] for e in _events(http.post("/v1/chat/completions", json=body | {"stream": True}))]
        answer = "".join(c["delta"].get("content") or "" for c in chunks)
        assert chunks[-1]["finish_reason"] == "stop" and "22" in answer, answer

        question = {"role": "user", "content": "Is 7 prime? Answer yes or no."}
        events = _sse(http.post("/v1/messages", json={"max_tokens": 1000, "temperature": 0, "stream": True,
                                                      "messages": [question]}))
        kinds = [d["content_block"]["type"] for e, d in events if e == "content_block_start"]
        text = "".join(d["delta"].get("text", "") for e, d in events if e == "content_block_delta")
        assert kinds == ["thinking", "text"] and "yes" in text.lower(), (kinds, text)
        assert events[-2][1]["delta"]["stop_reason"] == "end_turn"

        schema = {"type": "object", "properties": {"capital": {"type": "string"}, "population_millions": {
            "type": "integer"}}, "required": ["capital", "population_millions"], "additionalProperties": False}
        out = http.post("/v1/responses", json={
            "input": "France's capital, as JSON.", "temperature": 0, "max_output_tokens": 300,
            "text": {"format": {"type": "json_schema", "name": "c", "schema": schema, "strict": True}}}).json()
        message = next(item for item in out["output"] if item["type"] == "message")
        assert out["status"] == "completed" and json.loads(message["content"][0]["text"])["capital"] == "Paris"
    finally:
        engine.close()
