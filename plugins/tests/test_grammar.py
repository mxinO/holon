"""grammar.xgrammar (docs/m05-grammar.md D3–D9, §5).

- The oracle (model-free, D9): no strict schema the plugin accepts admits an invalid instance — every draft of the
  JSON-Schema-Test-Suite (fixtures/jsts.sh) and a typed probe table; `json_object` is xgrammar's JSON with whitespace
  runs capped; every form of `grammar.constraint@1` translates, malformed ones are refused.
- L0-mask (GPU): the Triton `apply` equals xgrammar's CPU `apply_token_bitmask_inplace`, eager and replayed in a graph;
  the plant `mask-bit-order` fails it.
- Masked L0-dist (M0.4 §5's method): speculative sampling through the real `fill` probe draws the first three tokens
  of a prefix-dependent toy grammar with exactly the enumerated masked law — greedy, T = 1, top-p, stop-token drafts;
  every output replays through a fresh matcher. The plants `mask-column-shift` and `no-rollback-after-probe` fail it
  (run once, recorded in docs/m05-grammar.md §11: `_law_holds` under each plant; the end-to-end checks below keep
  them failing permanently).
- End to end on the CPU (a tiny Qwen3 with Qwen3's tokenizer): constrained greedy outputs and finish reasons equal
  HF's logits masked by a fresh matcher, with and without speculation (the oracle drafter: right, wrong, mixed
  drafts); five plants change them; failing constraints (compile error, timeout, the memory watchdog) end only their
  request, with a positive control after the kill.
"""
import dataclasses
import json
import pathlib
import time

import numpy as np
import pytest
import torch
import xgrammar as xgr
from holon.api import MaskRow, SamplingParams
from holon.api.batch import Seq
from holon.core.engine import Engine, EngineConfig
from holon_judge.planted import PLANTS
from holon_plugins.grammar.xgrammar import kernel
from holon_plugins.grammar.xgrammar.constraint import KEY, check, grammar
from holon_plugins.grammar.xgrammar.plugin import XGrammar
from holon_plugins.grammar.xgrammar.worker import failure
from scipy import stats

from .fixtures.oracle import ORACLE
from .test_rejection import K, N, _sampler, _step, target  # noqa: F401 — fixture
from .test_spec import ORACLE_SLOTS, _plugins

JSTS = pathlib.Path(__file__).parent / "fixtures" / "jsts"
DRAFTS = {"draft3": "http://json-schema.org/draft-03/schema#", "draft4": "http://json-schema.org/draft-04/schema#",
          "draft6": "http://json-schema.org/draft-06/schema#", "draft7": "http://json-schema.org/draft-07/schema#",
          "draft2019-09": "https://json-schema.org/draft/2019-09/schema",
          "draft2020-12": "https://json-schema.org/draft/2020-12/schema"}


@pytest.fixture(scope="module")
def compiler():
    return xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=False)        # strings only, no vocabulary


def _compiled(compiler, c):
    """The compiled grammar of constraint `c`, or None where the plugin refuses it (a 400: `check`, a compile
    error)."""
    try:
        check(c)
        return compiler.compile_grammar(grammar(c))
    except (ValueError, RuntimeError):
        return None


def _admits(compiled, text: str) -> bool:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()


def _dump(value, indent=None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))


# -- the oracle ---------------------------------------------------------------------------------------------------

def _groups(draft: str):
    """(schema, tests) of every group of `draft`, the schema naming its draft; a format-assertion group again with
    `type: string` over its string instances (untyped, every format is refused under strict)."""
    for f in sorted((JSTS / draft).rglob("*.json")):
        for group in json.loads(f.read_text()):
            schema, tests = group["schema"], group["tests"]
            if isinstance(schema, dict):
                schema = {"$schema": DRAFTS[draft], **schema}
            yield schema, tests
            if f.parent.name == "format" and isinstance(schema, dict) and "type" not in schema:
                yield {**schema, "type": "string"}, [t for t in tests if isinstance(t["data"], str)]


def _obj(props, **kw):
    return {"type": "object", "properties": props, "required": list(props), **kw}



D4, D7 = DRAFTS["draft4"], DRAFTS["draft7"]
# R §6.2's typed probes: (schema, a valid instance, an invalid one) — accepted under strict, each enforced.
PROBES = [
    (_obj({"a": {"type": "integer"}}), {"a": 1}, {"a": "x"}),
    ({"enum": ["x", "y"]}, "x", "z"), ({"const": "k"}, "k", "j"), ({"type": ["string", "null"]}, None, 1),
    ({"type": "string", "enum": ["é", "日本", "🙂"]}, "日本", "日"),
    ({"type": "string", "minLength": 3}, "abc", "ab"), ({"type": "string", "maxLength": 3}, "abc", "abcd"),
    ({"type": "string", "pattern": "^[a-z]+\\d$"}, "ab1", "ab"),
    ({"type": "string", "format": "uuid"}, "123e4567-e89b-12d3-a456-426614174000", "nope"),
    ({"type": "string", "format": "uri"}, "http://a.b/c", "nope"), ({"type": "string", "format": "duration"}, "P1D",
                                                                     "nope"),
    ({"type": "integer", "minimum": 5}, 5, 4), ({"type": "integer", "exclusiveMaximum": 5}, 4, 5),
    ({"type": "number", "minimum": 1.5}, 1.5, 1.4), ({"type": "number", "exclusiveMinimum": 1.5}, 1.6, 1.5),
    ({"type": "array", "items": {"type": "integer"}, "minItems": 2}, [1, 2], [1]),
    ({"type": "array", "items": {"type": "integer"}, "maxItems": 2}, [1, 2], [1, 2, 3]),
    ({"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": False}, [1, "a"],
     [1, "a", 2]),
    (_obj({"a": {"type": "integer"}}, additionalProperties=False), {"a": 1}, {"a": 1, "b": 2}),
    (_obj({"a": {"type": "integer"}}), {"a": 1}, {"a": 1, "b": 2}),           # strict mode: no extra properties
    ({"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": {"type": "string"}},
     {"a": 1, "b": "s"}, {"a": 1, "b": 2}),
    ({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["b"]},
     {"b": 1}, {"a": 1}),
    ({"type": "object", "patternProperties": {"^x_[a-z]+$": {"type": "integer"}}, "additionalProperties": False},
     {"x_a": 1}, {"x_a": "s"}),
    ({"type": "object", "propertyNames": {"type": "string", "pattern": "^[a-z]+$"},
      "additionalProperties": {"type": "integer"}},
     {"ab": 1}, {"A1": 1}),
    ({"type": "object", "additionalProperties": {"type": "integer"}, "maxProperties": 1}, {"a": 1}, {"a": 1, "b": 2}),
    ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, 1, True),
    ({"$defs": {"n": {"type": "integer"}}, **_obj({"a": {"$ref": "#/$defs/n"}})}, {"a": 1}, {"a": "s"}),
    ({"definitions": {"n": {"type": "integer"}}, **_obj({"a": {"$ref": "#/definitions/n"}})}, {"a": 1}, {"a": "s"}),
    ({"$defs": {"t": _obj({"c": {"type": "array", "items": {"$ref": "#/$defs/t"}}})}, "$ref": "#/$defs/t"},
     {"c": [{"c": []}]}, {"c": [{}]}),
    ({"$schema": D7, **_obj({"c": {"type": "array", "items": {"$ref": "#"}}})}, {"c": [{"c": []}]}, {"c": [{}]}),
]


# Schemas xgrammar reads differently from JSON Schema, each with an instance it would admit (found by review):
# (schema, invalid instance). The reject list must refuse them.
HOLES = [
    ({**_obj({"a": {"$ref": "#/foo"}}), "foo": {"type": "string", "format": "date"}}, {"a": "2024-02-30"}),
    ({**_obj({"a": {"$ref": "#/foo"}}), "foo": {"oneOf": [{"type": "integer"}, {"type": "integer", "minimum": 0}]}},
     {"a": 5}),
    ({"type": "object", "properties": {"a": {"$ref": "#/properties/b/default"}, "b": {
        "type": "integer", "default": {"not": {"type": "integer"}}}}, "required": ["a"]}, {"a": 1}),
    ({"type": "object", "properties": {"a": {"type": "string"}},
      "patternProperties": {"^a": {"type": "string", "maxLength": 1}}}, {"a": "long"}),
    ({"type": "object", "properties": {"ab1": {"type": "integer"}},
      "propertyNames": {"type": "string", "maxLength": 2}}, {"ab1": 1}),
    ({"$schema": D7, "type": "array", "prefixItems": [{}], "items": {"type": "integer"}}, ["a"]),
    ({"$schema": D4, "type": "number", "minimum": 5, "exclusiveMinimum": 1}, 5),
    *(({**_obj({"a": {"$ref": f"#/foo/{ref}"}}), "foo": {key: {"type": "integer"}, ref: {"type": "string"}}},
       {"a": "x"}) for ref, key in (("%61", "a"), ("a~1b", "a/b"), ("a~0b", "a~b"))),   # xgrammar: no unescaping
    *(({"type": "string", "pattern": p}, w) for p, w in (("^[a^]x[^\\s]$", "ax\u3000"), ("^[\\[]x[^\\s]$", "[x\u3000"),
                                                         ("^[\\^]x[^\\s]$", "^x\u3000"))),   # ECMA: U+3000 is \s
    *(({"type": "string", "pattern": "^a.b$"}, f"a{end}b") for end in "\u2028\u2029"),  # ECMA: . ends a line there
    ({"$schema": D4, "definitions": {"s": {"type": "string"}}, **_obj({"a": {
        "id": "http://example.com/inner", "definitions": {"s": {"type": "integer"}}, **_obj({"b": {
            "$ref": "#/definitions/s"}})}})}, {"a": {"b": "x"}}),
]


@pytest.mark.parametrize("draft", [*DRAFTS, "probes", "holes"])
def test_accepted_strict_schemas_admit_only_valid_instances(compiler, draft):
    groups = {"probes": [(s, [{"data": ok, "valid": True}, {"data": bad, "valid": False}]) for s, ok, bad in PROBES],
              "holes": [(s, [{"data": bad, "valid": False}]) for s, bad in HOLES]}.get(draft) or _groups(draft)
    accepted, admitted = 0, []
    for schema, tests in groups:
        compiled = _compiled(compiler, {"type": "json_schema", "schema": schema, "strict": True})
        if compiled is None:
            continue
        accepted += 1
        admitted += [(schema, t["data"]) for t in tests if not t["valid"] and any(
            _admits(compiled, text) for text in {_dump(t["data"]), json.dumps(t["data"])})]     # also \u escapes
        if draft == "probes":
            assert _admits(compiled, _dump(tests[0]["data"])), schema
    assert admitted == []
    assert accepted == {"probes": len(PROBES), "holes": 0, "draft3": 0}.get(draft, accepted) and (
        accepted or draft in ("holes", "draft3"))


def _depth(value) -> int:
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return 1 + max(map(_depth, children), default=0) if isinstance(value, dict | list) else 0


MALFORMED = ['{"a":1,}', "{'a':1}", "{a:1}", '{"a" 1}', '{"a":01}', '{"a":1.}', '{"a":.5}', '{"a":+1}', '{"a":1e}',
             '{"a":-}', '{"a":NaN}', '{"a":Infinity}', '{"a":"\\q"}', '{"a":"\\u12G4"}', '{"a":"\\u12"}',
             '{"a":"x\ny"}', '{"a":tru}', "[1,]", '{"a":1}}', 'é{"a":1}', '{"a":1}é', '{"é" :1 é}']


def test_json_object_is_any_json_object(compiler):
    # D5: every JSON-Schema-Test-Suite object to depth 8 — compact, at indents 2 and 4, with a long whitespace run (no
    # cap) — and non-ASCII text (escaped surrogate pairs too) is admitted, malformed JSON refused, both as by
    # xgrammar's builtin JSON grammar; unlike it, json_object refuses anything but an object at the top.
    obj, builtin = _compiled(compiler, {"type": "json_object"}), compiler.compile_builtin_json_grammar()
    values = [t["data"] for _, tests in _groups("draft2020-12") for t in tests
              if isinstance(t["data"], dict) and _depth(t["data"]) <= 8]
    texts = [_dump(v, indent) for v in values for indent in (None, 2, 4)]
    texts += ['{"é":"日本🙂"}', '{"a":["\\ud83d\\ude42"]}', "{\n" + " " * 300 + '"a": 1}']
    assert len(values) > 300 and all(_admits(obj, t) and _admits(builtin, t) for t in texts)
    assert not any(_admits(obj, t) or _admits(builtin, t) for t in MALFORMED)
    assert not any(_admits(obj, t) for t in ("[1]", "[]", '"x"', "1")) and _admits(builtin, "[1]")


TAG = {"type": "tag", "begin": "<a>", "content": {"type": "json_object"}, "end": "</a>"}
FORMS = [   # (constraint, admitted, refused)
    ({"type": "regex", "pattern": "(yes|no)!"}, ["yes!", "no!"], ["maybe!"]),
    ({"type": "ebnf", "grammar": 'root ::= "x" [0-9]+'}, ["x12"], ["y1"]),
    ({"type": "json_schema", "schema": _obj({"n": {"type": "integer"}, "s": {"type": "string"}}), "style": "qwen_xml"},
     ["<parameter=n>\n3\n</parameter>\n<parameter=s>\nhi\n</parameter>\n"], ["<parameter=n>\nx\n</parameter>\n"]),
    (TAG, ['<a>{"k": [1]}</a>'], ["<a>{}</b>", "{}"]),
    ({"type": "triggered_tags", "triggers": ["<t"], "tags": [{**TAG, "begin": "<t1>", "end": "</t1>"}],
      "at_least_one": True, "stop_after_first": True}, ["<t1>{}</t1>"], ["hi", "<t1>x</t1>", "<t1>{}</t1> bye"]),
    ({"type": "triggered_tags", "triggers": ["<t"], "tags": [{**TAG, "begin": "<t1>", "end": "</t1>"}]},
     ["hi", "hi <t1>{}</t1> bye <t1>{}</t1>"], ["hi <t1>x</t1>"]),
    ({"type": "any_text", "excludes": ["STOP"]}, ["anything"], ["a STOP b"]),
    ({"type": "sequence", "elements": [{"type": "regex", "pattern": "a+"}, TAG]}, ["aa<a>{}</a>"],
     ["<a>{}</a>", "aa<a>[]</a>"]),
    ({"type": "or", "elements": [{"type": "regex", "pattern": "x"}, {"type": "json_object"}]}, ["x", "{}"], ["y"]),
]
REFUSED = [
    "not a dict", {"type": "json_schema"}, {"type": "json_object", "extra": 1}, {"type": "regex", "pattern": 1},
    {"type": "json_schema", "schema": {}, "style": "yaml"}, {**TAG, "content": {"type": "nope"}},
    {"type": "triggered_tags", "triggers": ["<"], "tags": [{"type": "regex", "pattern": "x"}]},
    {**TAG, "content": {"type": "json_schema", "schema": {"oneOf": [{}]}, "strict": True}},    # strict, nested
    {"type": "regex", "pattern": "("},                                                          # a compile error
    *({"type": "json_schema", "schema": s, "strict": True} for s in (                          # malformed schemas
        {"anyOf": 5}, {"type": "string", "format": ["x"]}, {"type": "object", "properties": 5, "required": ["a"]})),
    {"type": ["regex"]},
]



def test_constraint_forms(compiler):
    for c, admitted, refused in FORMS:
        compiled = _compiled(compiler, c)
        assert all(_admits(compiled, t) for t in admitted) and not any(_admits(compiled, t) for t in refused), c
    assert [c for c in REFUSED if _compiled(compiler, c) is not None] == []
    # A compile error is the constraint's fault (a 400), running out of memory the server's (a 503).
    assert [failure(e)["busy"] for e in (RuntimeError("[1:2:3] x.cc:5: Regex parsing error"), MemoryError(),
                                         RuntimeError("std::bad_alloc"))] == [False, True, True]
    assert failure(RuntimeError("[1:2:3] x.cc:5: Regex parsing error"))["error"] == "Regex parsing error"


# -- L0-mask ------------------------------------------------------------------------------------------------------

def _reference(logits, mask, rows):
    """xgrammar's CPU apply on the compact block scattered to full rows."""
    out, keep = logits.cpu().clone(), rows.cpu() < len(logits)
    full = torch.full((len(logits), mask.shape[1]), -1, dtype=torch.int32)
    full[rows.cpu()[keep]] = mask.cpu()[keep]
    xgr.apply_token_bitmask_inplace(out, full, indices=rows.cpu()[keep].tolist())
    return out


def _mask_case(vocab, device, seed):
    g = torch.Generator().manual_seed(seed)
    r, width = 13, -(-vocab // 32)
    logits = torch.randn(r, vocab, generator=g).to(device)
    mask = torch.randint(-(1 << 31), 1 << 31, (9, width), generator=g, dtype=torch.int64).to(torch.int32)
    mask[1], mask[2] = -1, 0                                     # all allowed, all refused
    rows = torch.tensor([0, 3, 5, 12, 7, r, r + 5, 2, 9])        # two padding rows
    return logits, mask.to(device), rows[torch.randperm(9, generator=g)].to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("vocab", [151936, 248320, 1000])
def test_apply_equals_xgrammars_cpu_apply(vocab):
    logits, mask, rows = _mask_case(vocab, "cuda", 0)
    out = logits.clone()
    kernel.apply(out, mask, rows)
    assert torch.equal(out.cpu(), _reference(logits, mask, rows))
    static = [logits.clone(), mask.clone(), rows.clone()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel.apply(*static)
    for seed in (1, 2):                                          # replays read the static buffers' new contents
        case = _mask_case(vocab, "cuda", seed)
        for buf, new in zip(static, case, strict=True):
            buf.copy_(new)
        graph.replay()
        assert torch.equal(static[0].cpu(), _reference(*case))
    with PLANTS["mask-bit-order"]():
        out = logits.clone()
        kernel.apply(out, mask, rows)
    assert not torch.equal(out.cpu(), _reference(logits, mask, rows))


# -- masked L0-dist -----------------------------------------------------------------------------------------------

# A prefix-dependent toy grammar over 16 tokens ("ab" is one token; 15 stops): each column's mask depends on the
# tokens before it, and stopping is allowed after two or three tokens on some paths.
TOY_VOCAB = [*"abcdefghijklm", "ab", "n", "<eos>"]
STOP, END = 15, 16                                  # END: no token (the sequence stopped earlier)
TOY = {"type": "ebnf", "grammar": 'root ::= "a" [b-f] [g-n]? | [b-e] [h-m] [h-m]? | "f" ("g" | "ab" "c")'}


@pytest.fixture(scope="module")
def toy():
    proc = XGrammar(xgr.TokenizerInfo(TOY_VOCAB, vocab_size=16, stop_token_ids=[STOP]), [STOP], workers=1)
    yield proc
    proc.shutdown()


def _filtered(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The law the sampler draws from masked `logits` [V]: greedy, temperature, then top-p (as test_rejection)."""
    if params.temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(), len(logits)).double()
    p = torch.softmax(logits.double() / params.temperature, -1)
    order = p.argsort(descending=True)
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[order] = (p[order].cumsum(0) - p[order]) <= params.top_p
    p = torch.where(keep, p, 0)
    return p / p.sum()


def _toy_law(target, params) -> np.ndarray:  # noqa: F811
    """P(x1 x2 x3) = ∏ F(softmax((z(x_t−1) + log 1[M(x<t)]) / T)), masks from a fresh matcher per prefix."""
    _, trans = target
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(TOY_VOCAB, vocab_size=16, stop_token_ids=[STOP])) \
        .compile_grammar(grammar(TOY))
    law = np.zeros(17 ** 3)

    def walk(prefix, p, prev):
        if len(prefix) == 3:
            law[(prefix[0] * 17 + prefix[1]) * 17 + prefix[2]] += p
            return
        if prefix and prefix[-1] in (STOP, END):
            return walk(prefix + (END,), p, prev)
        matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=[STOP])
        assert all(matcher.accept_token(t) for t in prefix)
        mask = xgr.allocate_token_bitmask(1, 16)
        matcher.fill_next_token_bitmask(mask)
        logits = trans[prev].clone()
        xgr.apply_token_bitmask_inplace(logits[None], mask)
        for x, q in enumerate(_filtered(logits, params).tolist()):
            if q > 0:
                walk(prefix + (x,), p * q, x)
    walk((), 1.0, 0)
    return law


def _toy_sequences(proc, target, params, stop_drafts) -> tuple[np.ndarray, int]:  # noqa: F811
    """The first three tokens (END after a stop) of N sequences speculating k = K through `proc`'s real `fill`,
    drafting (prev + 1 + j) mod 16 — or the stop token first, with `stop_drafts` — and the count of refusals: commits
    the processor refused, plus emitted sequences a fresh matcher refuses. Sequences with the same history share one
    representative in the processor (its matchers advance identically), rebuilt by replaying that history's fills and
    commits."""
    _, trans = target
    rid = f"law-{time.monotonic_ns()}"
    request = SamplingParams(ext={KEY: TOY})
    assert proc.open(rid, request, [0])
    while not proc.ready(rid):
        time.sleep(0.01)
    sampler = _sampler(params)
    history = {0: ((), ())}                     # id -> (fills and commits so far: ((drafts, chunk), ...), tokens)
    reps, refusals = {}, 0

    def rep(h):
        if h not in reps:
            seq = reps[h] = Seq(rid, [0], request, prompt_len=1, index=len(reps))
            for drafts, chunk in history[h][0]:
                proc.fill([MaskRow(seq, 0, K + 1, list(drafts))], np.empty((K + 1, proc.mask_width), np.int32))
                proc.commit(seq, list(chunk))
        return reps[h]

    hist, prev = np.zeros(N, np.int64), torch.zeros(N, dtype=torch.long)
    out, filled, done = np.full((N, 3), END), np.zeros(N, np.int64), np.zeros(N, bool)
    while not done.all():
        drafts = (prev[:, None] + 1 + torch.arange(K)) % 16
        if stop_drafts:
            drafts[:, 0] = STOP
        masks = torch.full((N, K + 1, proc.mask_width), -1, dtype=torch.int32)
        for h in np.unique(hist[~done]):
            rows = np.flatnonzero((hist == h) & ~done)
            buf = np.empty((K + 1, proc.mask_width), np.int32)
            proc.fill([MaskRow(rep(h), 0, K + 1, drafts[rows[0]].tolist())], buf)
            masks[rows] = torch.from_numpy(buf)
        logits = trans[torch.cat([prev[:, None], drafts], 1)]
        proc.apply(logits.view(-1, 16), masks.view(-1, proc.mask_width), torch.arange(N * (K + 1)))
        tokens, n_accept = _step(sampler, logits, drafts, torch.as_tensor(10 + filled))
        tokens, n = tokens.numpy(), n_accept.numpy() + 1
        stops = np.where(tokens == STOP, np.arange(K + 1), K + 1).min(1)
        n = np.minimum(n, stops + 1)                             # the engine stops at a stop token
        children = {}
        for i in np.flatnonzero(~done):
            key = (int(hist[i]), tuple(drafts[i].tolist()), tuple(tokens[i, : n[i]].tolist()))
            if key not in children:
                children[key] = len(history)
                fills, emitted = history[key[0]]
                history[children[key]] = (fills + (key[1:],), emitted + key[2])
            hist[i] = children[key]
            take = min(n[i], 3 - filled[i])
            out[i, filled[i]: filled[i] + take] = tokens[i, :take]
            filled[i] += take
            done[i] = filled[i] >= 3 or stops[i] < n[i]
            prev[i] = int(tokens[i, n[i] - 1])
        for child in children.values():                          # the processor commits each new history once
            try:
                rep(child)
            except Exception:
                refusals += 1
    proc.close(rid)
    compiled = xgr.GrammarCompiler(proc.info).compile_grammar(grammar(TOY))
    for h in np.unique(hist):
        matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=[STOP])
        refusals += not all(matcher.accept_token(t) for t in history[h][1])
    return out, refusals


def _pvalue(counts: np.ndarray, law: np.ndarray) -> float:
    """χ² p-value of `counts` against `law` (cells of expected count < 5 pooled); 0 for a draw outside the law."""
    if counts[law == 0].any():
        return 0.0
    if (law > 0).sum() == 1:
        return 1.0
    expected = law * counts.sum()
    small = expected < 5
    obs, exp = np.append(counts[~small], counts[small].sum()), np.append(expected[~small], expected[small].sum())
    keep = exp > 0
    return stats.chisquare(obs[keep], exp[keep] * obs[keep].sum() / exp[keep].sum()).pvalue


CASES = {"greedy": (SamplingParams(temperature=0.0), False), "t1": (SamplingParams(temperature=1.0), False),
         "top_p": (SamplingParams(temperature=1.0, top_p=0.9), False),
         "stop_draft": (SamplingParams(temperature=1.0), True)}


def _law_holds(toy, target, case) -> bool:  # noqa: F811
    params, stop_drafts = CASES[case]
    seqs, refusals = _toy_sequences(toy, target, params, stop_drafts)
    counts = np.bincount((seqs[:, 0] * 17 + seqs[:, 1]) * 17 + seqs[:, 2], minlength=17 ** 3)
    return refusals == 0 and _pvalue(counts, _toy_law(target, params)) > 1e-3


@pytest.mark.parametrize("case", CASES)
def test_masked_tokens_have_the_masked_law(toy, target, case):  # noqa: F811
    assert _law_holds(toy, target, case)


def test_a_checker_that_cannot_start_fails_busy(monkeypatch):
    # Workers that die before they are ready (here: an interpreter that exits at once) do not leave constraints
    # waiting forever: after STARTS of them in a row the waiting ones fail `busy`, and new ones are refused.
    from holon_plugins.grammar.xgrammar import vetting

    monkeypatch.setattr(vetting, "sys", type("Sys", (), {"executable": "/bin/false"}))
    checker = vetting.Vetting(xgr.TokenizerInfo(TOY_VOCAB, vocab_size=16, stop_token_ids=[STOP]).serialize_json(), 16,
                              [STOP], workers=1)
    try:
        checker.submit("k", json.dumps(TOY))
        deadline, done = time.monotonic() + 10, []
        while not done and time.monotonic() < deadline:
            time.sleep(0.05)
            done = checker.drain()
        assert [(k, o.kind) for k, o in done] == [("k", "busy")] and not checker.alive
        with pytest.raises(ValueError, match="down"):
            checker.submit("k2", json.dumps(TOY))
    finally:
        checker.shutdown()


# -- end to end on the CPU ----------------------------------------------------------------------------------------

QWEN3 = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 24
GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
REQUESTS = [   # (constraint or None, n)
    ({"type": "regex", "pattern": "(yes|no)( please)?"}, 2),
    ({"type": "json_schema", "schema": _obj({"name": {"type": "string", "maxLength": 6}, "n": {"type": "integer",
                                                "maximum": 999}, "on": {"type": "boolean"}}), "strict": True}, 1),
    ({"type": "tag", "begin": "<a>", "content": {"type": "json_schema", "schema": {"enum": ["x", "y"]}},
      "end": "</a>"}, 1),
    ({"type": "json_object"}, 1),
    (None, 1),
]


@pytest.fixture(scope="module")
def qwen_checkpoint(tmp_path_factory):
    """A tiny random Qwen3 over Qwen3's real vocabulary (151936 logits, 151669 tokens, byte-level BPE)."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained(QWEN3, local_files_only=True)
    except OSError:
        pytest.skip(f"needs {QWEN3}'s tokenizer in the local Hugging Face cache")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    path = tmp_path_factory.mktemp("tiny_qwen3_vocab")
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _qwen_engine(path, spec=False, **kw) -> Engine:
    slots = {"logits_processor": "grammar.xgrammar"} | (ORACLE_SLOTS if spec else {})
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=4, max_step_tokens=32,
               max_model_len=64, state_memory=4 << 20, max_draft_tokens=3 if spec else 0, slots=slots,
               plugin_options={"grammar.workers": 1}) | kw
    return Engine(EngineConfig(**cfg), plugins=_plugins())


@pytest.fixture(scope="module")
def e2e(qwen_checkpoint):
    """The requests (prompts, params) and their references — HF's greedy tokens with the logits masked by a fresh
    matcher, and finish reasons — plus a plain and a speculating engine."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(qwen_checkpoint, dtype=torch.float32).eval()
    plain = _qwen_engine(qwen_checkpoint)
    eos = sorted(plain.eos)
    compiler = xgr.GrammarCompiler(plain.processor.info, cache_enabled=False)
    g = torch.Generator().manual_seed(3)
    requests, refs = [], []
    for c, n in REQUESTS:
        prompt = torch.randint(0, 150000, (int(torch.randint(4, 12, (1,), generator=g)),), generator=g).tolist()
        params = dataclasses.replace(GREEDY, n=n, ext={KEY: c} if c else {})
        matcher = xgr.GrammarMatcher(compiler.compile_grammar(grammar(c)), override_stop_tokens=eos) if c else None
        ids, reason = list(prompt), "length"
        with torch.no_grad():
            for _ in range(MAX_TOKENS):
                logits = hf(torch.tensor([ids])).logits[0, -1]
                if matcher is not None:
                    mask = xgr.allocate_token_bitmask(1, len(logits))
                    matcher.fill_next_token_bitmask(mask)
                    xgr.apply_token_bitmask_inplace(logits[None], mask)
                    assert matcher.accept_token(int(logits.argmax()))
                ids.append(int(logits.argmax()))
                if ids[-1] in eos:
                    reason = "stop"
                    break
        requests.append((prompt, params))
        refs.append((ids[len(prompt):], reason))
    assert [r for _, r in refs[:4]] == ["stop", "length", "stop", "stop"]    # the finish-reason checks need stops
    spec = _qwen_engine(qwen_checkpoint, spec=True)
    yield requests, refs, {"plain": plain, "spec": spec}
    plain.close()
    spec.close()


def _outcomes(engine, requests):
    """(tokens, finish reason) of each sequence of each request; an error's reason names its kind."""
    rids = [engine.add(p, params) for p, params in requests]
    out = {(r, i): [[], None] for r, (_, params) in zip(rids, requests, strict=True) for i in range(params.n)}
    while engine.has_work():
        if wait := engine.idle_for():
            time.sleep(wait)
        for d in engine.step():
            out[d.rid, d.index][0] += d.token_ids
            if d.finish_reason:
                out[d.rid, d.index][1] = d.finish_reason + (f":{d.kind}" if d.kind else "")
    return [[tuple(out[r, i]) for i in range(params.n)] for r, (_, params) in zip(rids, requests, strict=True)]


def _expected(requests, refs):
    return [[tuple(ref)] * params.n for (_, params), ref in zip(requests, refs, strict=True)]


def _oracle(requests, refs, mode):
    ORACLE.sequences = {tuple(p): p + ref[0] for (p, _), ref in zip(requests, refs, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.written.clear()
    ORACLE.hidden.clear()
    ORACLE.hidden_slots.clear()
    ORACLE.k, ORACLE.vocab, ORACLE.mode, ORACLE.real = 3, 151936, mode, None


@pytest.mark.parametrize("mode", ["plain", "right", "wrong", "mixed"])
def test_constrained_outputs_are_the_masked_reference(e2e, mode):
    # Greedy constrained requests (a regex with n = 2; a strict schema, which the random model fills with capped
    # whitespace runs; a structural tag around a schema; json_object) beside a free one, plain and speculating —
    # drafts that cross the grammar's structure (right), that it refuses (wrong) and both: the same tokens and finish
    # reasons as HF masked by a fresh matcher.
    requests, refs, engines = e2e
    engine = engines["plain" if mode == "plain" else "spec"]
    _oracle(requests, refs, mode)
    refused = engine.processor.drafts_refused
    assert _outcomes(engine, requests) == _expected(requests, refs)
    if mode in ("wrong", "mixed"):
        assert engine.processor.drafts_refused > refused


@pytest.mark.parametrize("plant", ["mask-column-shift", "no-rollback-after-probe", "mask-bit-order",
                                   "stop-tokens-masked", "closing-tokens-masked"])
def test_plants_change_constrained_generation(e2e, plant):
    requests, refs, engines = e2e
    spec = plant in ("mask-column-shift", "no-rollback-after-probe")
    _oracle(requests, refs, "right")
    with PLANTS[plant]():
        got = _outcomes(engines["spec" if spec else "plain"], requests)
    assert got != _expected(requests, refs)


def test_masks_follow_the_tokenizer(e2e):
    # Byte-level tokens: a non-ASCII enum (multi-byte characters split across tokens) is admitted token by token and
    # a prefix of it is not complete; ids past the tokenizer's vocabulary are never allowed; a stop token only once
    # the constraint is complete.
    _, _, engines = e2e
    proc = engines["plain"].processor
    tok = engines["plain"].source.tokenizer()
    params = SamplingParams(ext={KEY: {"type": "json_schema", "schema": {"enum": ["日本", "🙂x", "é"]}}})
    assert proc.open("tok", params, [0])
    while not proc.ready("tok"):
        time.sleep(0.01)
    for i, value in enumerate(["日本", "🙂x", "é"]):
        seq = Seq("tok", [0], params, prompt_len=1, index=i)
        ids = tok.encode(json.dumps(value, ensure_ascii=False))
        buf = np.empty((len(ids) + 1, proc.mask_width), np.int32)
        proc.fill([MaskRow(seq, 0, len(ids) + 1, ids)], buf)
        bits = np.unpackbits(buf.view(np.uint8), bitorder="little").reshape(len(buf), -1)
        assert all(bits[j, t] for j, t in enumerate(ids)) and not bits[:, len(tok):].any()
        assert [bits[j, sorted(proc.stops)].any() for j in range(len(buf))] == [False] * len(ids) + [True]
        proc.commit(seq, ids)
    proc.close("tok")


def test_a_failing_constraint_ends_only_its_request(e2e, qwen_checkpoint):
    # D5's outcomes on the plain engine beside requests that finish unchanged: a compile error (client), a compile
    # past a lowered timeout (busy, then refused at once from the negative cache), a compile the memory watchdog
    # kills under a lowered budget (client) — the respawned worker then vets a small schema under the same budget
    # (the positive control). ignore_eos with a constraint, malformed constraints, stop tokens past the vocabulary,
    # and a constraint no processor serves, are 400s.
    requests, refs, engines = e2e
    engine, vetting = engines["plain"], engines["plain"].processor.vetting
    slow = {"type": "string", "maxLength": 2000}                                # ≈ 3 s to compile
    large = {"type": "object", "properties": {f"p{i}": {"type": "string", "pattern": f"^[a-z]{{{10 + i},{60 + i}}}$"}
                                              for i in range(30)}}             # ≈ 2 s, 53 MB compiled
    failing = [{"type": "regex", "pattern": "("}, {"type": "json_schema", "schema": slow},
               {"type": "json_schema", "schema": large}]
    reasons, defaults = [], (vetting.timeout, vetting.budget)
    try:
        for c, lowered in zip(failing, ({}, {"timeout": 0.3}, {"budget": 16 << 20}), strict=True):
            vetting.timeout, vetting.budget = lowered.get("timeout", defaults[0]), lowered.get("budget", defaults[1])
            got = _outcomes(engine, [(requests[0][0], dataclasses.replace(GREEDY, ext={KEY: c})), *requests[3:]])
            assert got[1:] == _expected(requests[3:], refs[3:])
            reasons.append(got[0][0][1])
        small = {"type": "json_schema", "schema": {"type": "string", "maxLength": 20}}       # 2 MB compiled
        [[(_, reason)]] = _outcomes(engine, [(requests[1][0], dataclasses.replace(GREEDY, ext={KEY: small}))])
    finally:
        vetting.timeout, vetting.budget = defaults
    assert reasons == ["error:client", "error:busy", "error:client"] and reason in ("stop", "length")
    assert len(vetting.starts) == 3                         # the first worker and two respawns
    with pytest.raises(ValueError, match="took over") as busy:                # the negative cache
        engine.add(requests[0][0], dataclasses.replace(GREEDY, ext={KEY: failing[1]}))
    assert busy.value.kind == "busy"
    with pytest.raises(ValueError, match="ignore_eos"):
        engine.add(requests[0][0], dataclasses.replace(GREEDY, ignore_eos=True, ext=requests[0][1].ext))
    deep = {"type": "json_object"}
    for _ in range(5000):
        deep = {"type": "sequence", "elements": [deep]}
    huge = {"type": "json_schema", "schema": {"enum": ["x" * 1000] * 2000}}
    for c, stops in [({"type": ["regex"]}, ()), ({"type": "json_schema", "schema": {"anyOf": 5}, "strict": True}, ()),
                     (deep, ()), (huge, ()), (requests[0][1].ext[KEY], (1 << 40,))]:
        with pytest.raises(ValueError) as malformed:   # a 400, never an exception that ends the engine
            engine.add(requests[0][0], dataclasses.replace(GREEDY, stop_token_ids=stops, ext={KEY: c}))
        assert malformed.value.kind == "client"
    with pytest.raises(ValueError, match="serves"):
        _qwen_engine(qwen_checkpoint, slots={}).add(*requests[0])
