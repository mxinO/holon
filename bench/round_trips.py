"""Tool-call round trips through every dialect of `frontends.http` on a served model (docs/dev/m05-grammar.md §1
acceptance 2, step 5; evidence record).

For each dialect (OpenAI chat, Anthropic Messages, OpenAI Responses) and tool choice (auto, required / any, named):
the user asks for the weather, the model calls `get_weather` (strict: its arguments follow the schema), the call's
result goes back, and the model answers. Every request is sent twice, streamed and not, greedy: the streamed events
reassembled must equal the non-streamed answer (ids aside) — the streaming parse equals the whole-text parse. Each
request is sent three times: first streamed (cold: the server's first sight of the conversation and its constraint),
then streamed and whole, both on the prefix the first one cached — the same generation, so that a difference is the
parser's (the cold one may differ: the GGUF kernels are not batch- or chunk-invariant). Per request: what was parsed
(calls, text), the finish / stop reason, TTFT cold and warm. With `--model`, the render and
constraint build of each request (`pipeline.Chat`) is timed in this process with the model's bundle.

Usage (from the repository root, against `holon serve <model> --slots logits_processor=grammar.xgrammar`):
  python bench/round_trips.py --url http://127.0.0.1:8000 --name <name> [--model <the served checkpoint>] [--gpu 0]
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import statistics
import time

import httpx
from run import ROOT, git, gpu_class, model_id

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}},
          "required": ["city", "unit"], "additionalProperties": False}
TOOL = {"name": "get_weather", "description": "The current weather in a city.", "parameters": SCHEMA}
OTHER = {"name": "get_time", "description": "The current local time in a city.",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
ASK = "What is the weather in Paris right now? Use the tool, in celsius."
RESULT = '{"temperature": 18, "condition": "light rain"}'
CHOICES = {"openai": {"auto": "auto", "required": "required", "named": {"type": "function",
                                                                     "function": {"name": "get_weather"}}},
           "anthropic": {"auto": {"type": "auto"}, "required": {"type": "any"},
                         "named": {"type": "tool", "name": "get_weather"}},
           "responses": {"auto": "auto", "required": "required", "named": {"type": "function", "name": "get_weather"}}}


class Dialect:
    """One API's requests, its stream reassembled, and what an answer says (calls, text, finish)."""

    def __init__(self, name: str, max_tokens: int) -> None:
        self.name, self.max_tokens = name, max_tokens

    def first(self, choice: str) -> dict:
        tools = [TOOL | {"strict": True}, OTHER]
        if self.name == "openai":
            return {"messages": [{"role": "user", "content": ASK}], "tools": [{"type": "function", "function": t}
                                                                             for t in tools],
                    "tool_choice": CHOICES["openai"][choice], "max_tokens": self.max_tokens}
        if self.name == "anthropic":
            return {"messages": [{"role": "user", "content": ASK}], "max_tokens": self.max_tokens,
                    "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                              for t in tools], "tool_choice": CHOICES["anthropic"][choice]}
        return {"input": ASK, "tools": [{"type": "function", **t} for t in tools],
                "tool_choice": CHOICES["responses"][choice], "max_output_tokens": self.max_tokens}

    def second(self, body: dict, answer: dict) -> dict:
        """The conversation with the answer's first call and its result appended (tool choice back to auto)."""
        body = copy.deepcopy(body)
        if self.name == "openai":
            message = answer["choices"][0]["message"]
            call = message["tool_calls"][0]
            body["messages"] += [{"role": "assistant", "content": message.get("content") or "", "tool_calls": [call]},
                                 {"role": "tool", "tool_call_id": call["id"], "content": RESULT}]
        elif self.name == "anthropic":
            use = next(b for b in answer["content"] if b["type"] == "tool_use")
            body["messages"] += [{"role": "assistant", "content": answer["content"]},
                                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"],
                                                               "content": RESULT}]}]
        else:
            call = next(i for i in answer["output"] if i["type"] == "function_call")
            body["input"] = [{"role": "user", "content": ASK}, *answer["output"],
                             {"type": "function_call_output", "call_id": call["call_id"], "output": RESULT}]
        body["tool_choice"] = CHOICES[self.name]["auto"]
        return body

    @property
    def path(self) -> str:
        return {"openai": "/v1/chat/completions", "anthropic": "/v1/messages", "responses": "/v1/responses"}[self.name]

    def summary(self, answer: dict) -> dict:
        """What the answer says, ids aside: calls (name, parsed arguments), text, reasoning, finish."""
        if self.name == "openai":
            (choice,) = answer["choices"]
            m = choice["message"]
            return {"calls": [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                              for c in m.get("tool_calls") or []], "text": m.get("content") or "",
                    "reasoning": m.get("reasoning_content") or "", "finish": choice["finish_reason"]}
        if self.name == "anthropic":
            blocks = answer["content"]
            return {"calls": [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"],
                    "text": "".join(b["text"] for b in blocks if b["type"] == "text"),
                    "reasoning": "".join(b["thinking"] for b in blocks if b["type"] == "thinking"),
                    "finish": answer["stop_reason"]}
        items = answer["output"]
        return {"calls": [(i["name"], json.loads(i["arguments"])) for i in items if i["type"] == "function_call"],
                "text": "".join(p["text"] for i in items if i["type"] == "message" for p in i["content"]),
                "reasoning": "".join(p["text"] for i in items if i["type"] == "reasoning" for p in i["content"]),
                "finish": answer.get("status")}

    def reassemble(self, events: list[tuple[str | None, dict]]) -> dict:
        """The answer the stream's events add up to, in the non-streamed shape."""
        if self.name == "openai":
            message, calls, finish = {"content": "", "reasoning_content": ""}, {}, None
            for _, e in events:
                for c in e.get("choices", []):
                    d = c.get("delta", {})
                    message["content"] += d.get("content") or ""
                    message["reasoning_content"] += d.get("reasoning_content") or ""
                    for t in d.get("tool_calls") or []:
                        call = calls.setdefault(t["index"], {"id": None, "function": {"name": "", "arguments": ""}})
                        call["id"] = t.get("id") or call["id"]
                        call["function"]["name"] += t.get("function", {}).get("name") or ""
                        call["function"]["arguments"] += t.get("function", {}).get("arguments") or ""
                    finish = c.get("finish_reason") or finish
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
            return {"choices": [{"message": message, "finish_reason": finish}]}
        if self.name == "anthropic":
            blocks, stop = {}, None
            for kind, e in events:
                if kind == "content_block_start":
                    blocks[e["index"]] = dict(e["content_block"]) | {"_json": ""}
                elif kind == "content_block_delta":
                    b, d = blocks[e["index"]], e["delta"]
                    if d["type"] == "text_delta":
                        b["text"] += d["text"]
                    elif d["type"] == "thinking_delta":
                        b["thinking"] += d["thinking"]
                    elif d["type"] == "input_json_delta":
                        b["_json"] += d["partial_json"]
                elif kind == "message_delta":
                    stop = e["delta"].get("stop_reason")
            for b in blocks.values():
                if b["type"] == "tool_use":
                    b["input"] = json.loads(b["_json"] or "{}")
                del b["_json"]
            return {"content": [blocks[i] for i in sorted(blocks)], "stop_reason": stop}
        items, status = {}, None
        for kind, e in events:
            if kind == "response.output_item.done":
                items[e["output_index"]] = e["item"]
            elif kind in ("response.completed", "response.incomplete", "response.failed"):
                status = e["response"]["status"]
        return {"output": [items[i] for i in sorted(items)], "status": status}


def post(client: httpx.Client, dialect: Dialect, body: dict, stream: bool) -> tuple[dict, float, float]:
    """(answer, TTFT s, total s); a streamed answer is reassembled from its events."""
    start = time.perf_counter()
    if not stream:
        r = client.post(dialect.path, json=body | {"temperature": 0.0})
        r.raise_for_status()
        return r.json(), float("nan"), time.perf_counter() - start
    events, first, kind = [], None, None
    with client.stream("POST", dialect.path, json=body | {"temperature": 0.0, "stream": True}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: ") and line != "data: [DONE]":
                first = first if first is not None else time.perf_counter()
                events.append((kind, json.loads(line[6:])))
    return dialect.reassemble(events), first - start, time.perf_counter() - start


def chat_build_ms(bundle, dialect: Dialect, bodies: list[dict]) -> float:
    """The median time this process takes to read, render and constrain the requests (`pipeline.Chat`)."""
    from holon_plugins.frontends.http import anthropic, responses, schemas
    from holon_plugins.frontends.http.pipeline import Chat
    convert = {"openai": lambda b: schemas.ChatRequest(**b).conversation(),
               "anthropic": lambda b: anthropic.conversation(anthropic.MessagesRequest(**b), True)[0],
               "responses": lambda b: responses.ResponsesRequest(**b).conversation()}[dialect.name]
    times = []
    for body in bodies * 5:
        start = time.perf_counter()
        Chat(bundle, convert(body))
        times.append((time.perf_counter() - start) * 1e3)
    return round(statistics.median(times), 2)


def _bundle(model: str):
    from holon.core.assembly import resolve_model
    from holon.core.engine import EngineConfig
    from holon.core.plugins import PluginSet, discover
    source, plugin = resolve_model(PluginSet(discover()), EngineConfig(model=model))
    return plugin.bundle(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", help="the served checkpoint: also time the render and constraint build here")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    results, ok = [], True
    bundle = _bundle(args.model) if args.model else None
    with httpx.Client(base_url=args.url, timeout=600) as client:
        served = client.get("/v1/models").json()["data"][0]["id"]
        for name in ("openai", "anthropic", "responses"):
            dialect = Dialect(name, args.max_tokens)
            bodies = []
            for choice in ("named", "required", "auto"):        # named first: its constraint is new to the server
                body = dialect.first(choice) | {"model": served}
                bodies.append(body)
                row = {"dialect": name, "choice": choice}
                for turn in ("call", "answer"):
                    answers = {}
                    # The first request is the server's first sight of this conversation (cold: no cached prefix,
                    # and a new constraint is vetted and compiled); the streamed and whole answers compared after it
                    # both reuse its prefix, so they are the same generation unless a parser disagrees.
                    for key, stream in (("cold", True), ("stream", True), ("whole", False)):
                        answer, ttft, total = post(client, dialect, body, stream)
                        answers[key] = (dialect.summary(answer), ttft, total, answer)
                    same = answers["stream"][0] == answers["whole"][0]
                    summary = answers["whole"][0]
                    row[turn] = {"summary": summary, "stream_equals_whole": same,
                                 "cold_generated_the_same": answers["cold"][0] == summary,
                                 **({} if same else {"variants": {k: v[0] for k, v in answers.items()}}),
                                 "ttft_ms_cold": round(answers["cold"][1] * 1e3, 1),
                                 "ttft_ms_warm": round(answers["stream"][1] * 1e3, 1),
                                 "total_ms": round(answers["whole"][2] * 1e3, 1)}
                    ok &= same
                    if turn == "call":
                        if not summary["calls"]:
                            row["answer"] = None
                            break
                        body = dialect.second(body, answers["whole"][3])
                results.append(row)
                print(json.dumps(row), flush=True)
            if bundle is not None:
                results.append({"dialect": name, "chat_build_ms_median": chat_build_ms(bundle, dialect, bodies)})
        metrics = client.get("/metrics").json()
    calls = [r for r in results if "call" in r]
    record = {"workload": f"round_trips_{args.name}", "date": datetime.date.today().isoformat(),
              "gpu_class": gpu_class(args.gpu), "served_model": served,
              **({"model": model_id(args.model)} if args.model else {}),
              "command": f"python bench/round_trips.py --name {args.name} --max-tokens {args.max_tokens}"
                         + (" --model <the served checkpoint>" if args.model else ""),
              "commit": git("rev-parse", "HEAD"), "method": __doc__.split("Usage")[0].strip(),
              "summary": {"requests": len(calls), "every_stream_equals_whole": ok,
                          "called": sum(bool(r["call"]["summary"]["calls"]) for r in calls),
                          "called_get_weather_valid": sum(
                              any(n == "get_weather" and set(a) == {"city", "unit"} and a["unit"] in ("celsius",
                                                                                                     "fahrenheit")
                                  for n, a in r["call"]["summary"]["calls"]) for r in calls),
                          "answered": sum(bool(r.get("answer") and r["answer"]["summary"]["text"]) for r in calls)},
              "metrics": {k: v for k, v in metrics.items() if isinstance(v, int)}, "results": results}
    out = ROOT / "evidence" / f"round_trips_{args.name}_{record['gpu_class']}_{record['date']}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"]), out)


if __name__ == "__main__":
    main()
