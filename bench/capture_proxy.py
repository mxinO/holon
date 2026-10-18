"""A recording proxy between a client (Claude Code, or an OpenAI chat client such as pi) and a server: forwards every
request unchanged, streams the answer back, and writes one JSON line per request describing its shape — never its
text, and no header values except the API version and beta flags (docs/m05-grammar.md R §5.4's unverified items:
headers, `metadata.user_id`'s shape, streaming, the tool list, `cache_control`, side calls). Each line also has the
request's timing at the proxy (the first streamed content, or an OpenAI stream's first chunk, after `ttft_ms`; the
whole answer after `total_ms`) and the token counts the server reported (`usage`), its stop reason, and an error's
type and whether it refused the prompt's length; an OpenAI chat answer also its tool calls and whether their
arguments parse, whether its text carries tool-call or reasoning markup, and whether its stream was well formed
(`openai_facts`). An OpenAI stream's TTFT assumes the server sends its first chunk with the first token, as Holon
does; one that sends a role-only chunk first would read near zero. `--metrics` adds the prefix tokens the server
saved for each request (its `/metrics` counter before and after; `null` when requests overlapped). `--raw` also
keeps the bodies, for local inspection only (never committed). With `--raw`, an OpenAI chat request is forwarded with
`return_token_ids: true` (unless the client set it), and its raw line also keeps the generated token ids (choice 0's,
the terminating token included) and the reported `usage`; the client then sees the stream's added `token_ids` fields
and one chunk per engine delta (some with an empty delta). K-M1 (docs/v1-kv-tiers.md §8) chains it behind
`bench/demo/pi_run.py`'s own proxy: `pi_run.py --server http://127.0.0.1:<this port>`.

Usage:
  python bench/capture_proxy.py --upstream http://<server>:8000 --port 18080 --out <requests.jsonl> [--metrics]
      [--raw <file>]
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHOWN = {"anthropic-version", "anthropic-beta", "content-type", "accept"}
CHAT = "/v1/chat/completions"
TOOL_MARKUP = ("<tool_call>", "<function=")          # a tool call the server left in the text
THINK_MARKUP = ("<think>", "</think>")


def shape(value: str) -> str:
    """A string's form without its content: hex runs, digits and letters replaced."""
    value = re.sub(r"[0-9a-f]{8,}", "<hex>", value)
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "0", value))


def _block(b: dict) -> str:
    kind = b.get("type", "?")
    if kind == "text" and b.get("text", "").lstrip().startswith("<system-reminder>"):
        kind = "text:system-reminder"
    return kind + ("+cache_control" if "cache_control" in b else "")


def summary(method: str, path: str, headers: dict[str, str], body: dict | None, status: int) -> dict:
    out = {"method": method, "path": path, "status": status,
           "headers": {k: (v if k in SHOWN else "<redacted>") for k, v in sorted(headers.items())}}
    if not isinstance(body, dict):
        return out
    out["keys"] = sorted(body)
    for k in ("model", "max_tokens", "stream", "temperature", "top_p", "top_k", "thinking", "output_config",
              "tool_choice", "stop_sequences", "context_management", "max_completion_tokens", "reasoning_effort",
              "chat_template_kwargs", "stream_options", "store", "parallel_tool_calls"):
        if k in body:
            out[k] = body[k]
    if isinstance(body.get("metadata"), dict):
        out["metadata"] = {k: shape(v) if isinstance(v, str) else type(v).__name__ for k, v in body["metadata"].items()}
    system = body.get("system")
    if isinstance(system, list):
        out["system"] = [{"block": _block(b), "chars": len(b.get("text", "")),
                          "billing_header": b.get("text", "").startswith("x-anthropic-billing-header:")}
                         for b in system]
    elif isinstance(system, str):
        out["system"] = [{"block": "string", "chars": len(system)}]
    tools = body.get("tools") or []
    fns = [t.get("function") if isinstance(t.get("function"), dict) else {} for t in tools]     # OpenAI's wrapping
    out["tools"] = [{"name": t.get("name") or fn.get("name"), "type": t.get("type", "custom"),
                     "$schema": "$schema" in (t.get("input_schema") or fn.get("parameters") or {}),
                     "additionalProperties_false": (t.get("input_schema") or fn.get("parameters") or {}).get(
                         "additionalProperties") is False,
                     "cache_control": "cache_control" in t} | ({"strict": fn["strict"]} if "strict" in fn else {})
                    for t, fn in zip(tools, fns, strict=True)]
    out["messages"] = [{"role": m.get("role"), "content": [_block(b) for b in m["content"]]
                        if isinstance(m.get("content"), list) else "string" if isinstance(m.get("content"), str)
                        else "null"}
                       | ({"fields": sorted(set(m) - {"role", "content"})} if set(m) - {"role", "content"} else {})
                       for m in body.get("messages") or []]
    return out


def facts(answer: bytes) -> dict:
    """What an Anthropic answer reports, never its text: the token counts (a whole message's `usage`, or a stream's:
    `message_start` gives the input tokens and the cache reads counted apart, the last `message_delta` the output
    tokens), the stop reason, and for an
    error its type and whether it refused the prompt's length (the engine's `max_model_len` refusal)."""
    out: dict = {"usage": {}}
    for line in answer.decode(errors="replace").splitlines() or [""]:
        text = line[5:].strip() if line.startswith("data:") else line
        try:
            event = json.loads(text)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        body = event["message"] if isinstance(event.get("message"), dict) else event
        usage = body.get("usage")
        out["usage"] |= {k: v for k, v in usage.items() if k in ("input_tokens", "output_tokens",
                                                                 "cache_read_input_tokens")} \
            if isinstance(usage, dict) else {}
        stop = (event.get("delta") or {}).get("stop_reason") if isinstance(event.get("delta"), dict) else None
        if stop := stop or body.get("stop_reason"):
            out["stop_reason"] = stop
        if isinstance(error := event.get("error"), dict):     # an error answer, or a stream's `error` event
            out["error_type"] = error.get("type")
            out["length_refused"] = "max_model_len" in str(error.get("message", ""))
    return out


def openai_facts(answer: bytes) -> dict:
    """What an OpenAI chat answer (a stream of `chat.completion.chunk`s or one `chat.completion`) reports, never its
    text: the token counts as `usage` (`input_tokens`, `output_tokens`, and `cached_tokens` if the server gave
    them), the finish reason, the tool calls made and whether each one's arguments parse as a JSON object, whether
    the content or the reasoning carries tool-call or reasoning markup, their lengths, and for a stream whether it
    ended with `[DONE]` and how many of its data lines were not JSON (`sse`). An error as in `facts`."""
    out: dict = {"usage": {}}
    content, reasoning, calls, stream, bad, done = [], [], {}, False, 0, False
    for line in answer.decode(errors="replace").splitlines() or [""]:
        if line.startswith("data:"):
            stream, text = True, line[5:].strip()
            if text == "[DONE]":
                done = True
                continue
        elif stream or not line.strip():
            continue
        else:
            text = line
        try:
            event = json.loads(text)
        except ValueError:
            bad += stream
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(usage := event.get("usage"), dict):
            out["usage"] = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
            if isinstance(details := usage.get("prompt_tokens_details"), dict) and "cached_tokens" in details:
                out["usage"]["cached_tokens"] = details["cached_tokens"]
        for choice in event.get("choices") or []:
            part = choice.get("delta") or choice.get("message") or {}
            content.append(part.get("content") or "")
            reasoning.append(part.get("reasoning_content") or "")
            for i, call in enumerate(part.get("tool_calls") or []):
                entry = calls.setdefault(call.get("index", i) if stream else i, {"name": None, "args": []})
                fn = call.get("function") or {}
                entry["name"] = entry["name"] or fn.get("name")
                entry["args"].append(fn.get("arguments") or "")
            if choice.get("finish_reason"):
                out["stop_reason"] = choice["finish_reason"]
        if isinstance(error := event.get("error"), dict):
            out["error_type"] = error.get("type")
            out["length_refused"] = "max_model_len" in str(error.get("message", ""))
    content, reasoning = "".join(content), "".join(reasoning)
    out["tool_calls"] = [{"name": c["name"], "args_json": _object("".join(c["args"]))} for c in calls.values()]
    out["content_chars"], out["reasoning_chars"] = len(content), len(reasoning)
    out["markup"] = sorted({f"{where}:{m}" for where, text in (("content", content), ("reasoning", reasoning))
                            for m in (*TOOL_MARKUP, *THINK_MARKUP) if m in text})
    if stream:
        out["sse"] = {"done": done, "bad_lines": bad}
    return out


def openai_token_ids(answer: bytes) -> list[int]:
    """The generated token ids of choice 0 of an OpenAI chat answer asked with `return_token_ids` (stream or not)."""
    ids: list[int] = []
    for line in answer.decode(errors="replace").splitlines():
        try:
            event = json.loads(line[5:].strip() if line.startswith("data:") else line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            if choice.get("index", 0) == 0:
                ids += choice.get("token_ids") or []
    return ids


def _object(text: str) -> bool:
    try:
        return isinstance(json.loads(text or "{}"), dict)
    except ValueError:
        return False


def _saved(upstream: str) -> int | None:
    """The server's prefix-cache counter (`/metrics`), or None if it has none."""
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(upstream + "/metrics", timeout=30) as r:
            return json.load(r).get("prefix_tokens_saved")
    except Exception:  # noqa: BLE001 — a server without the counter
        return None


def serve(upstream: str, port: int, out: str, raw: str | None, metrics: bool = False) -> None:
    target = urllib.parse.urlsplit(upstream)
    lock = threading.Lock()
    flight = {"now": 0, "started": 0}               # requests in flight; requests started so far

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self) -> None:
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            counted = metrics and self.command == "POST"
            if counted:
                with lock:
                    flight["now"] += 1
                    flight["started"] += 1
                    mine, shared = flight["started"], flight["now"] > 1
                saved = _saved(upstream.rstrip("/"))
            after = None
            try:
                body = json.loads(data) if data else None
            except ValueError:
                body = None
            chat = self.path.split("?")[0] == CHAT
            sent = data
            if raw and chat and isinstance(body, dict) and "return_token_ids" not in body:
                sent = json.dumps(body | {"return_token_ids": True}).encode()
            try:
                start, first, answer = time.perf_counter(), None, bytearray()
                conn = http.client.HTTPConnection(target.hostname, target.port, timeout=3600)
                headers = {k: v for k, v in self.headers.items()          # http.client sets the length
                           if k.lower() not in ("host", "connection", "content-length")}
                conn.request(self.command, self.path, sent, headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while chunk := resp.read1(65536):
                    answer += chunk
                    if first is None and (b"content_block_delta" in (window := answer[-len(chunk) - 32:]) or
                                          b"chat.completion.chunk" in window):
                        first = time.perf_counter()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                end = time.perf_counter()
                self.wfile.write(b"0\r\n\r\n")
                after = _saved(upstream.rstrip("/")) if counted else None
            finally:                                   # an aborted request still leaves the server
                if counted:
                    with lock:
                        flight["now"] -= 1
                        alone = not shared and flight["started"] == mine      # none ran beside it
            prefix = {"prefix_tokens_saved": after - saved if alone and None not in (saved, after) else None} \
                if counted else {}
            record = summary(self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body,
                             resp.status) | {"ttft_ms": first and round((first - start) * 1e3, 1),
                                             "total_ms": round((end - start) * 1e3, 1)} | prefix | (
                openai_facts if chat else facts)(answer)
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if raw:
                    extra = {"token_ids": openai_token_ids(answer), "usage": record.get("usage")} if chat else {}
                    with open(raw, "a") as f:
                        f.write(json.dumps({"path": self.path, "body": body} | extra) + "\n")

        do_GET = do_POST = _forward

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw")
    parser.add_argument("--metrics", action="store_true", help="record each request's prefix tokens saved")
    args = parser.parse_args()
    serve(args.upstream, args.port, args.out, args.raw, args.metrics)


if __name__ == "__main__":
    main()
