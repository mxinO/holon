"""The OpenAI frontend over a scripted fake engine and a byte-level toy tokenizer (CPU only, no model)."""
import asyncio
import json
import pathlib

import holon_plugins.frontends.openai as openai_frontend
import pytest
from fastapi.testclient import TestClient
from holon.api import Logprobs, Manifest, SamplingParams, TokenDelta
from holon_plugins.frontends.openai.app import build_app
from holon_plugins.frontends.openai.detok import Detokenizer

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
    model_name = "toy"
    tokenizer = ByteTokenizer()

    def __init__(self):
        self.rendered = []

    def render_chat(self, messages, tools=None, add_generation_prompt=True, **kw):
        self.rendered.append((messages, tools, kw))
        return self.tokenizer.encode("".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>")

    def reasoning_parser(self):
        return None

    def tool_parser(self):
        return None


class FakeEngine:
    """Streams OUTPUTS one token per delta, round-robin over the `n` sequences, like the engine: max_tokens and
    stop ids finish a sequence, the EOS/stop token is part of the last delta, prompt logprobs come with sequence 0's
    first delta. Prompts longer than `max_prompt` are rejected; `endless` never finishes."""

    def __init__(self, max_prompt=64, endless=False):
        self.max_prompt, self.endless = max_prompt, endless
        self.requests, self.aborted = [], []

    async def generate(self, token_ids, params: SamplingParams, rid=None):
        self.requests.append((token_ids, params, rid))
        if len(token_ids) > self.max_prompt:
            yield TokenDelta(rid, [], finish_reason="error", error=f"prompt of {len(token_ids)} tokens is too long")
            return
        if self.endless:
            while True:
                await asyncio.sleep(0.001)
                yield TokenDelta(rid, [ord("a")])
        seqs = [self._script(OUTPUTS[i % len(OUTPUTS)], params) for i in range(params.n)]
        for step in range(max(map(len, seqs))):
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
        return Logprobs([-0.5] * len(ids), [[t, 0][:k] for t in ids], [[-0.5, -2.0][:k] for _ in ids])

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
    assert engine.requests[0][1].max_tokens == 4 and engine.requests[0][1].logprobs == 2


# -- errors -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("stream", [False, True])
def test_rejected_request_is_400(http, stream):
    r = http.post("/v1/completions", json={"prompt": "x" * 100, "stream": stream})
    assert r.status_code == 400 and "too long" in r.json()["error"]["message"]
    r = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 100}],
                                                "stream": stream})
    assert r.status_code == 400 and "too long" in r.json()["error"]["message"]


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

def test_models_health_metrics(http):
    assert [m["id"] for m in http.get("/v1/models").json()["data"]] == ["toy"]
    assert http.get("/health").status_code == 200
    assert http.get("/metrics").json() == {"waiting": 0, "running": 1}


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
    m = Manifest.load(pathlib.Path(openai_frontend.__file__).parent / "plugin.toml")
    assert m.slot == "frontend" and [str(r) for r in m.provides] == ["frontend.openai@1"]
