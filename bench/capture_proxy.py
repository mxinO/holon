"""A recording proxy between a client (Claude Code) and a server: forwards every request unchanged, streams the answer
back, and writes one JSON line per request describing its shape — never its text, and no header values except the
API version and beta flags (docs/m05-grammar.md R §5.4's unverified items: headers, `metadata.user_id`'s shape,
streaming, the tool list, `cache_control`, side calls). Each line also has the request's timing at the proxy (first
streamed content after `ttft_ms`, the whole answer after `total_ms`) and the token counts the server reported
(`usage`). `--raw` also keeps the bodies, for local inspection only.

Usage:
  python bench/capture_proxy.py --upstream http://<server>:8000 --port 18080 --out <requests.jsonl> [--raw <file>]
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHOWN = {"anthropic-version", "anthropic-beta", "content-type", "accept"}


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
              "tool_choice", "stop_sequences", "context_management"):
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
    out["tools"] = [{"name": t.get("name"), "type": t.get("type", "custom"),
                     "$schema": "$schema" in (t.get("input_schema") or {}),
                     "additionalProperties_false": (t.get("input_schema") or {}).get("additionalProperties") is False,
                     "cache_control": "cache_control" in t} for t in tools]
    out["messages"] = [{"role": m.get("role"), "content": [_block(b) for b in m["content"]]
                        if isinstance(m.get("content"), list) else "string"} for m in body.get("messages") or []]
    return out


def usage(answer: bytes) -> dict[str, int]:
    """The token counts an Anthropic answer reports: a whole message's `usage`, or a stream's (`message_start` gives
    the input tokens, the last `message_delta` the output tokens)."""
    out: dict[str, int] = {}
    for line in answer.decode(errors="replace").splitlines() or [""]:
        text = line[5:].strip() if line.startswith("data:") else line
        try:
            event = json.loads(text)
        except ValueError:
            continue
        if isinstance(event, dict):
            body = event["message"] if isinstance(event.get("message"), dict) else event
            out |= body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {k: v for k, v in out.items() if k in ("input_tokens", "output_tokens")}


def serve(upstream: str, port: int, out: str, raw: str | None) -> None:
    target = urllib.parse.urlsplit(upstream)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self) -> None:
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            start, first, answer = time.perf_counter(), None, bytearray()
            conn = http.client.HTTPConnection(target.hostname, target.port, timeout=3600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
            conn.request(self.command, self.path, data, headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while chunk := resp.read1(65536):
                answer += chunk
                if first is None and b"content_block_delta" in answer[-len(chunk) - 32:]:
                    first = time.perf_counter()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            end = time.perf_counter()
            self.wfile.write(b"0\r\n\r\n")
            try:
                body = json.loads(data) if data else None
            except ValueError:
                body = None
            record = summary(self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body,
                             resp.status) | {"ttft_ms": first and round((first - start) * 1e3, 1),
                                             "total_ms": round((end - start) * 1e3, 1), "usage": usage(answer)}
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if raw:
                    with open(raw, "a") as f:
                        f.write(json.dumps({"path": self.path, "body": body}) + "\n")

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
    args = parser.parse_args()
    serve(args.upstream, args.port, args.out, args.raw)


if __name__ == "__main__":
    main()
