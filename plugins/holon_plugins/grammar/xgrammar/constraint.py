"""`grammar.constraint@1` (docs/m05-grammar.md D6): the plain dict a request carries in `SamplingParams.ext`, checked
and translated to an xgrammar grammar here — the frontends build these dicts and never import xgrammar.

    {"type": "json_schema", "schema": {...} | bool, "strict": False, "style": "json"}   (style "qwen_xml": the
                                                       arguments of a Qwen3.5 / 3.8 XML tool call)
    {"type": "json_object"}
    {"type": "regex", "pattern": "..."}
    {"type": "ebnf", "grammar": "..."}                 (its rule `root` starts it)
    {"type": "tag", "begin": "...", "content": <constraint>, "end": "..." | ["...", ...]}
    {"type": "triggered_tags", "triggers": ["..."], "tags": [<tag>, ...], "at_least_one": False,
     "stop_after_first": False}
    {"type": "any_text", "excludes": ["..."]}
    {"type": "sequence" | "or", "elements": [<constraint>, ...]}

JSON (schemas in the `json` style and `json_object`) is xgrammar's any-whitespace JSON grammar with every whitespace
run capped at WHITESPACE_CAP characters (D5: pretty-printed JSON and its drafts stay allowed, a model cannot pad
forever). The cap is a rewrite of the grammar's text rather than xgrammar's `max_whitespace_cnt`, which caps typed
rules only: the rewrite caps untyped subschemas too, at the same cost (≈ 13.5 MB and 30 ms more per compiled grammar
at a 152k vocabulary, docs/m05-grammar.md §11).

`strict` schemas are checked against the reject list (D9): features xgrammar ignores or gets wrong are refused, so an
accepted strict schema admits only valid instances (the JSON-Schema-Test-Suite oracle, plugins/tests/test_grammar.py).
"""
from __future__ import annotations

import json
import re
from typing import Any

import xgrammar as xgr

KEY = "grammar.constraint@1"
WHITESPACE_CAP = 64

# type -> (required fields, optional fields)
_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "json_schema": ({"schema"}, {"strict", "style"}), "json_object": (set(), set()), "regex": ({"pattern"}, set()),
    "ebnf": ({"grammar"}, set()), "tag": ({"begin", "content", "end"}, set()),
    "triggered_tags": ({"triggers", "tags"}, {"at_least_one", "stop_after_first"}), "any_text": (set(), {"excludes"}),
    "sequence": ({"elements"}, set()), "or": ({"elements"}, set())}


def check(c: Any, path: str = "") -> None:
    """Raise ValueError naming what is wrong with constraint `c` (its form, or a strict schema's unsupported
    feature)."""
    if not isinstance(c, dict) or c.get("type") not in _FIELDS:
        raise ValueError(f"{KEY}{path}: a dict whose type is one of {sorted(_FIELDS)}")
    required, optional = _FIELDS[c["type"]]
    if missing := required - set(c):
        raise ValueError(f"{KEY}{path}: {c['type']} needs {sorted(missing)}")
    if extra := set(c) - required - optional - {"type"}:
        raise ValueError(f"{KEY}{path}: {c['type']} has no {sorted(extra)}")
    strings = lambda v: isinstance(v, list) and all(isinstance(s, str) for s in v)      # noqa: E731
    flags = [c.get(f, False) for f in ("strict", "at_least_one", "stop_after_first")]
    ok = all(isinstance(f, bool) for f in flags) and {
        "json_schema": lambda: isinstance(c["schema"], dict | bool) and c.get("style", "json") in ("json", "qwen_xml"),
        "regex": lambda: isinstance(c["pattern"], str), "ebnf": lambda: isinstance(c["grammar"], str),
        "tag": lambda: isinstance(c["begin"], str) and (isinstance(c["end"], str) or strings(c["end"])),
        "triggered_tags": lambda: strings(c["triggers"]) and isinstance(c["tags"], list),
        "any_text": lambda: strings(c.get("excludes", [])),
        "sequence": lambda: isinstance(c["elements"], list), "or": lambda: isinstance(c["elements"], list),
    }.get(c["type"], lambda: True)()
    if not ok:
        raise ValueError(f"{KEY}{path}: malformed {c['type']}")
    if c["type"] == "json_schema" and c.get("strict") and (why := strict_violation(c["schema"])):
        raise ValueError(f"{KEY}{path}: strict schema: {why}")
    for i, sub in enumerate(c.get("elements", c.get("tags", []))):
        if c["type"] == "triggered_tags" and not (isinstance(sub, dict) and sub.get("type") == "tag"):
            raise ValueError(f"{KEY}{path}/tags/{i}: a tag")
        check(sub, f"{path}/{'tags' if c['type'] == 'triggered_tags' else 'elements'}/{i}")
    if c["type"] == "tag":
        check(c["content"], f"{path}/content")


def grammar(c: dict) -> xgr.Grammar:
    """The xgrammar grammar of a checked constraint."""
    kind = c["type"]
    if kind == "json_schema" and c.get("style", "json") == "json":
        return _capped(xgr.Grammar.from_json_schema(json.dumps(c["schema"]), any_whitespace=True, strict_mode=True))
    if kind == "json_object":
        return _capped(xgr.Grammar.builtin_json_grammar())
    if kind == "regex":
        return xgr.Grammar.from_regex(c["pattern"])
    if kind == "ebnf":
        return xgr.Grammar.from_ebnf(c["grammar"])
    return xgr.Grammar.from_structural_tag(json.dumps({"type": "structural_tag", "format": _format(c)}))


def _format(c: dict) -> dict:
    """xgrammar's structural-tag format of constraint `c`; a leaf is its grammar's text (one translation per form)."""
    kind = c["type"]
    if kind == "json_schema" and c.get("style") == "qwen_xml":
        return {"type": "json_schema", "json_schema": c["schema"], "style": "qwen_xml"}
    if kind in ("json_schema", "json_object", "regex", "ebnf"):
        return {"type": "grammar", "grammar": str(grammar(c))}
    if kind == "tag":
        return {"type": "tag", "begin": c["begin"], "content": _format(c["content"]), "end": c["end"]}
    if kind == "triggered_tags":
        return {"type": kind, "triggers": c["triggers"], "tags": [_format(t) for t in c["tags"]],
                "at_least_one": c.get("at_least_one", False), "stop_after_first": c.get("stop_after_first", False)}
    if kind == "any_text":
        return {"type": kind, "excludes": c.get("excludes", [])}
    return {"type": kind, "elements": [_format(e) for e in c["elements"]]}


# A string literal or a character class (with its `*`) of xgrammar's printed EBNF, so a quoted "[ \n\t]*" is skipped.
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\[(?:[^\]\\]|\\.)*\]\*?')
_RUN = re.compile(r"\[(?: |\\[nrt])+\]\*")         # a whitespace run, e.g. [ \n\r\t]*


def _capped(g: xgr.Grammar) -> xgr.Grammar:
    """`g` with every unbounded whitespace run capped at WHITESPACE_CAP characters (a pattern's own such class,
    inside a string, is capped too: stricter, never looser)."""
    cap = f"{{0,{WHITESPACE_CAP}}}"
    return xgr.Grammar.from_ebnf(_TOKEN.sub(lambda m: m[0][:-1] + cap if _RUN.fullmatch(m[0]) else m[0], str(g)))


# -- the strict reject list (D9) -------------------------------------------------------------------------------------

_DRAFTS = {"json-schema.org/draft-04/schema", "json-schema.org/draft-06/schema", "json-schema.org/draft-07/schema",
           "json-schema.org/draft/2019-09/schema", "json-schema.org/draft/2020-12/schema"}
# Ignored or mishandled by xgrammar wherever they appear (R §6.2 and the oracle).
_UNSUPPORTED = {"oneOf", "not", "if", "then", "else", "multipleOf", "dependencies", "dependentRequired",
                "dependentSchemas", "contains", "minContains", "maxContains", "additionalItems", "$dynamicRef",
                "$dynamicAnchor", "$recursiveRef", "$recursiveAnchor", "contentMediaType", "contentEncoding",
                "contentSchema"}
# Keywords that assert only for one JSON type: without `type`, xgrammar does not enforce them.
_TYPED = {"string": {"minLength", "maxLength", "pattern", "format"},
          "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
          "array": {"minItems", "maxItems", "prefixItems", "unevaluatedItems"},
          "object": {"required", "additionalProperties", "patternProperties", "propertyNames", "minProperties",
                     "maxProperties", "unevaluatedProperties"}}
_ASSERTIONS = {"type", "properties", "items", "enum", "const", "anyOf", "allOf", "$ref", "uniqueItems",
               *_UNSUPPORTED, *(k for ks in _TYPED.values() for k in ks)}
# Formats the oracle checks xgrammar enforces exactly (its date, date-time, time, ipv4 and hostname are syntax only:
# 2024-02-30, a leap second at 22:59; its ipv6 takes a leading zero in an embedded IPv4 octet).
FORMATS = {"email", "uri", "uuid", "duration"}
# Patterns xgrammar reads more loosely than ECMA 262: its \S and [^\s] exclude ASCII whitespace only.
_LOOSE_PATTERN = re.compile(r"(?<!\\)(?:\\\\)*\\S|\[\^(?:[^\]\\]|\\.)*\\s")
_PYTHON = {"string": str, "number": (int, float), "boolean": bool, "null": type(None), "array": list, "object": dict}
_SCHEMA_DICTS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_LISTS = ("anyOf", "allOf", "prefixItems")
_SCHEMAS = ("additionalProperties", "items", "propertyNames", "unevaluatedProperties", "unevaluatedItems")


def strict_violation(schema: Any) -> str | None:
    """Why a strict request cannot have `schema` (the first reason found, with its JSON pointer), or None."""
    if isinstance(schema, dict) and "$schema" in schema and _draft(schema["$schema"]) not in _DRAFTS:
        return f"$schema {schema['$schema']!r} is not a supported draft (04, 06, 07, 2019-09, 2020-12)"
    return _violation(schema, schema, "")


def _draft(uri: Any) -> str:
    return re.sub(r"^https?://|#$", "", uri) if isinstance(uri, str) else ""


def _violation(node: Any, root: Any, at: str) -> str | None:
    if not isinstance(node, dict):
        return None
    keys = set(node)
    why = next((f"{k} is not supported" for k in sorted(keys & _UNSUPPORTED)), None)
    if why is None and at and ("$schema" in node or "$id" in node):
        why = "an embedded schema resource ($schema or $id below the root)"
    if why is None:
        why = _local_violation(node, root, keys)
    if why is not None:
        return f"{at or '/'}: {why}"
    for k in _SCHEMA_DICTS:
        for name, sub in (node.get(k) or {}).items() if isinstance(node.get(k), dict) else ():
            if why := _violation(sub, root, f"{at}/{k}/{_escape(name)}"):
                return why
    for k in _SCHEMA_LISTS:
        for i, sub in enumerate(node.get(k) or []):
            if why := _violation(sub, root, f"{at}/{k}/{i}"):
                return why
    return next((why for k in _SCHEMAS if k in node and (why := _violation(node[k], root, f"{at}/{k}"))), None)


def _local_violation(node: dict, root: Any, keys: set[str]) -> str | None:
    if len(node.get("allOf") or []) >= 2:
        return "allOf with two or more branches is not supported"
    if node.get("uniqueItems") is True:
        return "uniqueItems is not supported"
    if isinstance(node.get("items"), list):
        return "items as an array (a tuple) is not supported; use prefixItems"
    if any(isinstance(node.get(k), bool) for k in ("exclusiveMinimum", "exclusiveMaximum")):
        return "boolean exclusiveMinimum / exclusiveMaximum (draft 4) are not supported"
    if "$ref" in node and (why := _ref_violation(node["$ref"], root)):
        return why
    for head in ("$ref", "enum", "const", "anyOf", "allOf"):
        if head in node and (siblings := (keys & _ASSERTIONS) - {head} - _allowed_beside(node, head)):
            return f"{sorted(siblings)} beside {head} are not supported"
    types = node.get("type", [])
    types = {types} if isinstance(types, str) else set(types) if isinstance(types, list) else set()
    for kind, words in _TYPED.items():
        if (untyped := keys & words) and not types & ({kind, "integer"} if kind == "number" else {kind}):
            return f"{sorted(untyped)} without type {kind!r} are not supported"
    if isinstance(node.get("required"), list) and (missing := set(node["required"]) - set(node.get("properties", {}))):
        return f"required {sorted(missing)} not in properties"
    patterns = [node["pattern"]] if isinstance(node.get("pattern"), str) else []
    patterns += node["patternProperties"] if isinstance(node.get("patternProperties"), dict) else []
    if loose := next((p for p in patterns if _LOOSE_PATTERN.search(p)), None):
        return f"pattern {loose!r}: \\S and [^\\s] are not supported (ECMA 262 whitespace is Unicode's)"
    if "pattern" in node and keys & {"format", "minLength", "maxLength"}:
        return f"pattern beside {sorted(keys & {'format', 'minLength', 'maxLength'})} is not supported"
    if "format" in node and node["format"] not in FORMATS:
        return f"format {node['format']!r} is not supported (supported: {sorted(FORMATS)})"
    return None


def _allowed_beside(node: dict, head: str) -> set[str]:
    """`type` beside enum / const when every value has that type (xgrammar keeps the values, drops the type)."""
    if head not in ("enum", "const") or "type" not in node:
        return set()
    values = node["enum"] if head == "enum" and isinstance(node["enum"], list) else [node.get(head)]
    types = [node["type"]] if isinstance(node["type"], str) else node["type"]
    return {"type"} if isinstance(types, list) and all(any(_is(v, t) for t in types) for v in values) else set()


def _is(value: Any, kind: str) -> bool:
    if isinstance(value, bool) and kind in ("number", "integer"):
        return False
    if kind == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    return isinstance(value, _PYTHON.get(kind, ()))


def _ref_violation(ref: Any, root: Any) -> str | None:
    """Only local JSON-pointer refs that resolve ('#' or '#/…'); remote and anchor refs are refused."""
    if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
        return f"$ref {ref!r}: only local refs ('#/…') are supported"
    node = root
    for part in ref[2:].split("/") if ref != "#" else ():
        part = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), part).replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return f"$ref {ref!r} does not resolve"
    return None


def _escape(name: str) -> str:
    return name.replace("~", "~0").replace("/", "~1")
