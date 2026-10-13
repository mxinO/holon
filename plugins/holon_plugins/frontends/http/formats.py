"""The reasoning and tool-call formats bundles name (`FrontendBundle.reasoning_format` / `tool_format`,
docs/m05-grammar.md §3.8, D6, D7), each implemented once: its streaming parser and the constraint of its output.

Parsers turn text into typed events. Each is a state machine over a buffer that holds back only what may still turn
out to be a marker (or a call not yet closed), so the events do not depend on how the text is cut: a streamed answer
parses as the whole text does (`final` marks the end of the text).

Constraints are `grammar.constraint@1` plain dicts (D6), the grammar plugin's resource: never xgrammar objects.
"""
from __future__ import annotations

import json
from functools import partial
from typing import Any

from holon.api import ParseEvent, Reasoning, Text, ToolArgsDelta, ToolCallEnd, ToolCallStart

THINK, END_THINK = "<think>", "</think>"
CALL, END_CALL = "<tool_call>", "</tool_call>"
FUNCTION, END_FUNCTION, PARAMETER, END_PARAMETER = "<function=", "</function>", "<parameter=", "</parameter>"


def _split(buf: str, markers: tuple[str, ...], final: bool) -> tuple[str, str | None, str]:
    """(text before the first marker in `buf`, that marker, what follows it); with no marker, the text that cannot
    be the start of one — all of it if `final` — and the held-back rest."""
    hits = [(i, m) for m in markers if (i := buf.find(m)) >= 0]
    if hits:
        i, m = min(hits)
        return buf[:i], m, buf[i + len(m):]
    keep = 0 if final else max((k for m in markers for k in range(1, len(m)) if buf.endswith(m[:k])), default=0)
    return buf[:len(buf) - keep], None, buf[len(buf) - keep:]


def _tag(begin: str, content: dict[str, Any], end: str) -> dict[str, Any]:
    return {"type": "tag", "begin": begin, "content": content, "end": end}


# -- reasoning ---------------------------------------------------------------------------------------------------------

class Think:
    """think@1 (Qwen3, 3.5, 3.8): reasoning between <think> and </think>. The prompt has opened it (Qwen3.5 / 3.8,
    thinking on), closed it (thinking off), or leaves it to the model, which may open it first (Qwen3). `ends` (the
    tool format's call marker) also end it and stay in the text: Qwen3.5+ starts a call without closing its
    reasoning (R §4.2)."""

    def __init__(self, prompt_tail: str, ends: tuple[str, ...] = ()) -> None:
        tail = prompt_tail.rstrip()
        self.state = "reasoning" if tail.endswith(THINK) else "text" if tail.endswith(END_THINK) else "start"
        self.ends = ends
        self.buf = ""

    def channel(self) -> str:
        return self.state

    def feed(self, text: str, final: bool = False) -> list[ParseEvent]:
        self.buf += text
        out: list[ParseEvent] = []
        if self.state == "start":
            head = self.buf.lstrip()
            if head.startswith(THINK):
                self.buf, self.state = head[len(THINK):], "reasoning"
            elif THINK.startswith(head) and not final:
                return out                                   # it may still open
            else:
                self.state = "text"
        if self.state == "reasoning":
            before, marker, rest = _split(self.buf, (END_THINK, *self.ends), final)
            if before:
                out.append(Reasoning(before))
            if marker is None:
                self.buf = rest
                return out
            self.buf, self.state = rest if marker == END_THINK else marker + rest, "text"
        if self.buf:
            out.append(Text(self.buf))
            self.buf = ""
        return out

    @staticmethod
    def constraint(prompt_tail: str, body: dict[str, Any], ends: tuple[str, ...], calls: bool) -> dict[str, Any]:
        """`body` after the reasoning the prompt leaves open (or lets the model open): reasoning is free text that
        holds neither </think> nor `ends`; it ends with </think> and whitespace — or, when `body` starts with a call,
        with that call (the parser's rule)."""
        state = Think(prompt_tail).state
        if state == "text":
            return body
        begin = "" if state == "reasoning" else THINK
        free = {"type": "any_text", "excludes": [END_THINK, *ends]}
        alternatives = [{"type": "sequence", "elements": [_tag(begin, free, END_THINK),
                                                          {"type": "regex", "pattern": r"\s*"}, body]}]
        if calls:
            opened = [{"type": "regex", "pattern": THINK}] if begin else []
            alternatives.append({"type": "sequence", "elements": [*opened, free, body]})
        if state == "start":
            alternatives.append(body)
        return {"type": "or", "elements": alternatives}


# -- tool calls --------------------------------------------------------------------------------------------------------

def _parameters(tool: dict[str, Any]) -> dict[str, Any]:
    return tool["function"].get("parameters") or {"type": "object"}


def _arguments(tool: dict[str, Any], style: str) -> dict[str, Any]:
    return {"type": "json_schema", "schema": _parameters(tool), "strict": bool(tool["function"].get("strict"))} | (
        {"style": style} if style != "json" else {})


class Hermes:
    """hermes@1 (Qwen3): `<tool_call>\\n{"name": …, "arguments": {…}}\\n</tool_call>`. A call is parsed when it
    closes (or the text ends): its arguments are then known to be a JSON object; a body that is not such a call
    stays text."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.buf, self.inside, self.calls = "", False, 0

    def channel(self) -> str:
        return "tool" if self.inside else "text"

    def feed(self, text: str, final: bool = False) -> list[ParseEvent]:
        self.buf += text
        out: list[ParseEvent] = []
        while True:
            if not self.inside:
                before, marker, self.buf = _split(self.buf, (CALL,), final)
                if before:
                    out.append(Text(before))
                if marker is None:
                    return out
                self.inside = True
            end = self.buf.find(END_CALL)
            if end < 0 and not final:
                return out
            body, self.buf = (self.buf[:end], self.buf[end + len(END_CALL):]) if end >= 0 else (self.buf, "")
            self.inside = False
            if call := _hermes(body):
                out += [ToolCallStart(self.calls, call[0]), ToolArgsDelta(self.calls, call[1]), ToolCallEnd(self.calls)]
                self.calls += 1
            else:
                out.append(Text(CALL + body + (END_CALL if end >= 0 else "")))

    @staticmethod
    def call(tool: dict[str, Any]) -> dict[str, Any]:
        """The constraint of one call to `tool`, as the template writes it (`Q3T:60-68`)."""
        name = json.dumps(tool["function"]["name"], ensure_ascii=False)
        return _tag(f'{CALL}\n{{"name": {name}, "arguments": ', _arguments(tool, "json"), f"}}\n{END_CALL}")


def _hermes(body: str) -> tuple[str, str] | None:
    """(name, arguments as JSON) of a hermes call's body, or None."""
    try:
        call = json.loads(body)
        args = call.get("arguments", {})
        args = json.loads(args) if isinstance(args, str) else args
    except (ValueError, AttributeError):
        return None
    if not isinstance(call.get("name"), str) or not isinstance(args, dict):
        return None
    return call["name"], json.dumps(args, ensure_ascii=False)


_PYTHON = {"True": True, "False": False, "None": None}


class QwenXML:
    """qwen_xml@1 (Qwen3.5) and @2 (Qwen3.6+, 3.8): `<tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n
    </parameter>\\n…</function>\\n</tool_call>`. The arguments stream as JSON a parameter at a time (`{` at the
    function, each `"key": value` when its parameter closes, `}` at the end), each value typed by the tool's schema.
    The template writes strings raw and other values as @1 Jinja's `| string` (Python's True / False / None; mappings
    and sequences as JSON, `Q35T:120-124`) or @2 JSON (`tojson`); a value that does not read as one of its schema's
    types stays a string."""

    def __init__(self, tools: list[dict[str, Any]], version: int) -> None:
        self.properties = {t["function"]["name"]: _parameters(t).get("properties") for t in tools}
        self.python = version == 1
        self.state, self.buf, self.calls = "text", "", 0
        self.function, self.n_args = "", 0

    def channel(self) -> str:
        return "text" if self.state == "text" else "tool"

    def feed(self, text: str, final: bool = False) -> list[ParseEvent]:
        self.buf += text
        out: list[ParseEvent] = []
        while True:
            if self.state == "text":
                before, marker, self.buf = _split(self.buf, (CALL,), final)
                if before:
                    out.append(Text(before))
                if marker is None:
                    return out
                self.state = "call"
            elif self.state == "call":                       # after <tool_call>: <function=NAME>
                head = self.buf.lstrip()
                if head.startswith(FUNCTION) and (gt := head.find(">")) >= 0:
                    self.function, self.n_args, self.buf = head[len(FUNCTION):gt], 0, head[gt + 1:]
                    out += [ToolCallStart(self.calls, self.function), ToolArgsDelta(self.calls, "{")]
                    self.state = "args"
                elif (FUNCTION.startswith(head) or head.startswith(FUNCTION)) and not final:
                    return out
                else:                                        # not a call after all
                    out.append(Text(CALL))
                    self.state = "text"
            elif self.state == "args":                       # parameters until </function> (or </tool_call>)
                hits = [(i, m) for m in (PARAMETER, END_FUNCTION, END_CALL) if (i := self.buf.find(m)) >= 0]
                i, marker = min(hits) if hits else (-1, None)
                if marker == PARAMETER:
                    rest = self.buf[i + len(PARAMETER):]
                    gt = rest.find(">")
                    end = rest.find(END_PARAMETER, gt) if gt >= 0 else -1
                    if end >= 0:
                        key, sep = rest[:gt], ", " if self.n_args else ""
                        pair = [json.dumps(v, ensure_ascii=False) for v in (key, self._value(key, rest[gt + 1:end]))]
                        out.append(ToolArgsDelta(self.calls, f"{sep}{pair[0]}: {pair[1]}"))
                        self.n_args, self.buf = self.n_args + 1, rest[end + len(END_PARAMETER):]
                        continue
                if marker is None or marker == PARAMETER:
                    if not final:
                        return out
                    self.buf = ""                            # the text ended inside the call: a cut parameter is lost
                else:
                    self.buf = self.buf[i + len(marker):]
                out += [ToolArgsDelta(self.calls, "}"), ToolCallEnd(self.calls)]
                self.calls += 1
                self.state = "close" if marker == END_FUNCTION else "text"
            else:                                            # after </function>: </tool_call>
                head = self.buf.lstrip()
                if head.startswith(END_CALL):
                    self.buf = head[len(END_CALL):]
                elif END_CALL.startswith(head) and not final:
                    return out
                self.state = "text"

    def _value(self, key: str, text: str) -> Any:
        text = text.removeprefix("\n").removesuffix("\n")   # the template's newlines around a value
        properties = self.properties.get(self.function)
        types = _types(properties.get(key) if isinstance(properties, dict) else None)
        if types == {"string"}:
            return text
        try:
            value = _PYTHON[text] if self.python and text in _PYTHON else json.loads(text)
        except ValueError:
            return text
        return value if not types or _kinds(value) & types else text

    @staticmethod
    def call(tool: dict[str, Any]) -> dict[str, Any]:
        """The constraint of one call to `tool`, as the template writes it (`Q35T:112-127`); xgrammar's `qwen_xml`
        style writes values as JSON (strings raw), which both versions read."""
        begin = f"{CALL}\n{FUNCTION}{tool['function']['name']}>\n"
        return _tag(begin, _arguments(tool, "qwen_xml"), f"\n{END_FUNCTION}\n{END_CALL}")


def _types(schema: Any) -> set[str]:
    """The JSON types a value of `schema` may have, as far as `type`, `enum`, `const` and `anyOf` / `oneOf` tell
    (none: any)."""
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    types = {t} if isinstance(t, str) else set(t) if isinstance(t, list) else set()
    lists = lambda *keys: [v for k in keys if isinstance(schema.get(k), list) for v in schema[k]]  # noqa: E731
    for value in lists("enum") + ([schema["const"]] if "const" in schema else []):
        types |= _kinds(value)
    for branch in lists("anyOf", "oneOf"):
        types |= _types(branch)
    return types


def _kinds(value: Any) -> set[str]:
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return {"integer", "number"}
    return {{float: "number", str: "string", list: "array", dict: "object", type(None): "null"}[type(value)]}


REASONING = {"think@1": Think}
TOOLS = {"hermes@1": Hermes, "qwen_xml@1": partial(QwenXML, version=1), "qwen_xml@2": partial(QwenXML, version=2)}
CALLS = {"hermes@1": Hermes.call, "qwen_xml@1": QwenXML.call, "qwen_xml@2": QwenXML.call}
