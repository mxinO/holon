"""Qwen3.8-Flash-Next's model plugin on the CPU against HF's fixtures (docs/v1-flash-next.md §6.1): the tiny model
(8 layers, QSA at 3 and 7, PLE at 1, n = 4, vocabulary 248,320) from its GGUF as llama.cpp's converter writes it, run by
the engine in float32 with the testing CPU providers; the fixtures are HF's one-shot forward under D2's tie rule
(`holon_testing.qwen4exp_files`). Each prompt (100 tokens with an EOS) runs as its 124 fixture ids with prompt logprobs,
prefilled one-shot and in chunks of 7, 10 and 13 tokens, and as its 100-token prompt decoded greedily for 24 steps.

- Logits: each position's prompt logprob within 1e-4 of the fixture's (its output mixer's x through the GGUF's
  `lm_head`), the top-1 token identical; the greedy tokens are HF's.
- QSA: each (layer, position)'s selected tokens are HF's set (dense steps select every earlier token); the fixtures'
  boundary ties occur.
- PLE: the n-gram table rows are HF's, bitwise.
- Prefix hit vs recompute: the second request, hitting the first's checkpoint at 64, gives bitwise the prompt logprobs
  of a run without the prefix cache.
- MTP (D9): the head loaded from the tiny Strata-format pack (`drafters.mtp` with `mtp.path`): its output-mixer x
  within 1e-4 of the judge's pooled reference over the pack's own tensors (the fixture's `mtp_x`), one-shot and in
  chunks of 13 (the carried R); greedy speculation emits HF's tokens and, within 1e-5, the logprobs of a run without a
  drafter, its drafts replaced so that the first j are right and the next wrong, j = 0 … 3 in turn (a random head is
  rarely right): verify rows accept 0, 1, 2 and 3 drafts, and GDN, PLE and QSA rewind.
  `drafters.dflash` refuses the target (its residual is not hidden-size).
- Plants that must fail: `qsa_dense_limit_off_by_one` (a 20-token step runs dense), `ple_conv_dilation_1`,
  `ple_eos_cut_off`, and §6.3's (the judge's, for L2 on the full model): `qsa_select_shift`, `qsa_pool_rope_last`,
  `hc_inject_unscaled`, `ple_gate_no_sqrt`.

Needs the tiny GGUF: `FN_TINY_GGUF=<…/tiny-qwen4exp-f32.gguf>`, the MTP pack beside it (both built by
`python -m holon_testing.qwen4exp_files` in the golden venv; their sha256 are pinned), else skipped.
"""
import dataclasses
import hashlib
import os
import pathlib

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_testing import qwen4exp_files as q

GGUF = os.environ.get("FN_TINY_GGUF")
pytestmark = pytest.mark.skipif(GGUF is None, reason="needs FN_TINY_GGUF (the tiny model's GGUF)")
CHUNKS = (128, 7, 10, 13)                   # one-shot, then chunked prefill (10: a step ending at 20 tokens)
DECODE = SamplingParams(temperature=0.0, max_tokens=q.DECODE, ignore_eos=True)
PROMPT_LP = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=1)


@pytest.fixture(scope="module")
def tiny():
    path = pathlib.Path(GGUF)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == q.GGUF_SHA256
    fixtures, meta = q.load_fixtures()
    from holon_std.weights.gguf.source import GGUFSource

    head = dict(GGUFSource(path).tensors())["output.weight"]
    return path, fixtures, meta, head


def _engine(path, chunk: int, prefix: bool = False, mtp: pathlib.Path | None = None, drafter: str = "mtp",
            options: dict | None = None) -> Engine:
    spec = {} if mtp is None else {
        "max_draft_tokens": 3, "plugin_options": {"spec.k": "3", f"{drafter}.path": str(mtp)} | (options or {}),
        "slots": {"drafter": f"drafters.{drafter}", "draft_policy": "std.spec.fixed_k",
                  "verifier": "std.spec.greedy"}}
    return Engine(EngineConfig(str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=1,
                               max_step_tokens=chunk, max_model_len=256, state_memory=64 << 20, prefix_cache=prefix,
                               **spec))


def _record(engine: Engine) -> dict:
    """Wrap every QSA layer's selection and PLE's row hashing: per call, (positions, output)."""
    seen = {"ple": []}
    for i, layer in enumerate(engine.executor.model.layers):
        mixer = layer.mixer
        if hasattr(mixer, "_select"):
            seen[i] = []
            mixer._select = _spy(mixer._select, seen[i])
        if layer.pre is not None:
            layer.pre._rows = _spy(layer.pre._rows, seen["ple"], rows=True)
    return seen


def _spy(fn, into: list, rows: bool = False):
    def call(*args):
        out = fn(*args)
        positions = args[4] if not rows else None
        into.append((positions.clone() if positions is not None else None, out.clone(), args[0].clone()))
        return out
    return call


def _ask(engine: Engine, ids: list[int], params: SamplingParams):
    rid, tokens, lp = engine.add(ids, params), [], None
    while engine.has_work():
        for d in engine.step():
            if d.rid == rid:
                tokens += d.token_ids
                lp = d.prompt_logprobs if d.prompt_logprobs is not None else lp
    return tokens, lp


def _check(tiny, chunk: int) -> None:
    path, fixtures, meta, head = tiny
    for p, prompt in enumerate(q.prompts()):
        ids, x = fixtures[f"p{p}.ids"].tolist(), fixtures[f"p{p}.x"]
        engine = _engine(path, chunk)
        assert _ask(engine, prompt, DECODE)[0] == ids[len(prompt):]                       # greedy: HF's tokens
        seen = _record(engine)
        _, lp = _ask(engine, ids, PROMPT_LP)
        want = torch.log_softmax(x @ head.T, dim=-1)                                       # the fixture's logits
        got = torch.tensor(lp.token)
        ref = want[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
        assert (got - ref).abs().max() <= 1e-4, (chunk, p, (got - ref).abs().max())
        assert [t[0] for t in lp.top_ids] == want[:-1].argmax(-1).tolist()
        assert int(meta[f"p{p}.boundary_ties"]) > 0
        for layer in (3, 7):                                                               # QSA's token sets
            sets = fixtures[f"p{p}.sets.{layer}"]
            chosen = {}
            for positions, blocks, _ in seen[layer]:
                for t, row in zip(positions.tolist(), blocks.tolist(), strict=True):
                    chosen[t] = {4 * b + i for b in row if b >= 0 for i in range(4) if 4 * b + i <= t}
            for t in range(len(ids)):
                want_set = {s for s in sets[t].tolist() if s >= 0}
                assert chosen.get(t, set(range(t + 1))) == want_set, (chunk, p, layer, t)
        rows = torch.cat([out for _, out, _ in seen["ple"]])
        assert torch.equal(rows, fixtures[f"p{p}.ple_ids"])                               # PLE: bitwise
        engine.close()


@pytest.mark.parametrize("chunk", CHUNKS)
def test_the_engine_matches_hf(tiny, chunk):
    _check(tiny, chunk)


def test_a_prefix_hit_is_bitwise_a_recompute(tiny):
    """A request continuing a cached one resumes at its checkpoint (64, PLE's alignment: GDN's chunk): its greedy tokens
    and their logprobs are bitwise those of a run without the prefix cache. (Prompt-logprob requests never hit: they
    need every position.)"""
    from holon_std.mixers.gdn.units import CHUNK

    path, fixtures, _, _ = tiny
    ids = fixtures["p0.ids"].tolist()
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=1)
    plain = _decode(_engine(path, 128), ids[:110], params)
    engine = _engine(path, 128, prefix=True)
    ple = engine.spec.block.param("pre")
    assert ple.param("alignment") == ple.param("in_step") == CHUNK
    _ask(engine, ids[:100], DECODE)
    hit = _decode(engine, ids[:110], params)
    assert engine.stats()["prefix_tokens_saved"] == 64
    assert hit == plain


def _decode(engine: Engine, ids: list[int], params: SamplingParams) -> list:
    rid, out = engine.add(ids, params), []
    while engine.has_work():
        out += [(d.token_ids, d.logprobs.token) for d in engine.step() if d.rid == rid and d.token_ids]
    return out


@pytest.mark.parametrize("chunk", [128, 13])
def test_the_mtp_head_matches_the_pooled_reference(tiny, chunk):
    path, fixtures, _, _ = tiny
    pack = path.with_name(f"{q.NAME}-mtp-q2_0.gguf")
    assert hashlib.sha256(pack.read_bytes()).hexdigest() == q.MTP_PACK_SHA256
    engine = _engine(path, chunk, mtp=pack, options={"mtp.front": "off"})     # the head's output on the step's rows
    kv = ("state.mtp.kv", "state.mtp.idx")
    for name in kv:                                 # (pages no step writes compare equal too)
        engine.ledger.pools[name].buffer.zero_()
    head, rows = engine.executor.drafter.head, []
    forward = head.forward
    head.forward = lambda e, r, ctx: rows.append(out := forward(e, r, ctx)) or out
    ids = fixtures["p0.ids"].tolist()
    _ask(engine, ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    got = head.out(torch.cat([r for r in rows if len(r) > 1]))          # the step's rows, not its draft sub-steps
    assert got.shape[0] == len(ids) and (got - fixtures["p0.mtp_x"]).abs().max() <= 1e-4
    front = _engine(path, chunk, mtp=pack)          # its state writes only (write_kv): the same KV and index keys
    for name in kv:
        front.ledger.pools[name].buffer.zero_()
    _ask(front, ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    for name in kv:
        assert torch.equal(front.ledger.pools[name].buffer, engine.ledger.pools[name].buffer), name
    engine.close()
    engine = front                                  # the drafts below: the default, front-only
    drafter, steps = engine.executor.drafter, [0]
    run = drafter.run

    def right_for_j(ctx, h):                  # the first j drafts right and the next one wrong, j = 0, 1, 2, 3 in turn
        run(ctx, h)
        table, dev, k = ctx.seq_table, ctx.batch.dev, ctx.batch.plan.draft_k
        if k and len(dev.logits_rows):
            j, steps[0] = steps[0] % (k + 1), steps[0] + 1
            slot, p = int(dev.slots[0]), int(table.num_computed[dev.slots[0]])
            right = ids[p + 1: p + 1 + k]
            drafts = right[:j] + [(t + 1) % engine.spec.vocab for t in right[j:]]
            table.draft_tokens[slot, : len(drafts)] = torch.tensor(drafts)
    drafter.run = right_for_j
    params = SamplingParams(temperature=0.0, max_tokens=q.DECODE, ignore_eos=True, logprobs=1)
    plain = _engine(path, chunk)
    (tokens, lps), (want, want_lps) = (_flat(_decode(e, ids[: q.PROMPT], params)) for e in (engine, plain))
    assert tokens == want == ids[q.PROMPT:]                                              # lossless
    assert (torch.tensor(lps) - torch.tensor(want_lps)).abs().max() <= 1e-5             # vs no drafter
    stats = engine.stats()
    assert all(stats[f"spec_rows_k3_accepting_{n}"] > 0 for n in range(4)), stats
    engine.close()


def test_the_mtp_head_plant_fails(tiny):
    """§6.3's `mtp_stream0` (the head reading stream 0 for every stream): its output leaves the pooled reference."""
    from holon_judge.planted import PLANTS
    path, fixtures, _, _ = tiny
    pack = path.with_name(f"{q.NAME}-mtp-q2_0.gguf")
    with PLANTS["mtp_stream0"]():
        engine = _engine(path, 128, mtp=pack, options={"mtp.front": "off"})
        head, rows = engine.executor.drafter.head, []
        forward = head.forward
        head.forward = lambda e, r, ctx: rows.append(out := forward(e, r, ctx)) or out
        ids = fixtures["p0.ids"].tolist()
        _ask(engine, ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        got = head.out(torch.cat([r for r in rows if len(r) > 1]))
        engine.close()
    assert (got - fixtures["p0.mtp_x"]).abs().max() > 1e-4


def _flat(deltas: list) -> tuple[list, list]:
    return [t for ts, _ in deltas for t in ts], [x for _, xs in deltas for x in xs]


def test_dflash_refuses_a_wide_residual(tiny, tmp_path):
    from holon_testing import dflash_files

    draft = dflash_files.save(tmp_path / "draft", 64, 320, (0, 2), mask=7)
    with pytest.raises(ValueError, match="residual is 256 wide"):
        _engine(tiny[0], 128, mtp=draft, drafter="dflash")


JUDGE_PLANTS = ("qsa_select_shift", "qsa_pool_rope_last", "hc_inject_unscaled", "ple_gate_no_sqrt",   # §6.3's
                "ple_state_not_carried", "ple_state_bf16", "shexp_gate_off")


@pytest.mark.parametrize("plant", ["qsa_dense_limit_off_by_one", "ple_conv_dilation_1", "ple_eos_cut_off",
                                   *JUDGE_PLANTS])
def test_plants_fail(tiny, plant, monkeypatch):
    from holon_judge.planted import PLANTS
    from holon_plugins.models.qwen4_exp import plugin
    from holon_std.mixers.qsa import kernels as qsa_kernels
    from holon_std.mixers.qsa import units as qsa_units

    if plant in JUDGE_PLANTS:
        with PLANTS[plant](), pytest.raises(AssertionError):
            _check(tiny, 10)
        return
    if plant == "qsa_dense_limit_off_by_one":
        limit = qsa_kernels.dense_limit
        monkeypatch.setattr(qsa_units.kernels, "dense_limit", lambda budget, ratio: limit(budget, ratio) + 1)
    elif plant == "ple_conv_dilation_1":
        recipe = plugin.Plugin.recipe

        def dilated(c, source):
            spec = recipe(c, source)
            return dataclasses.replace(spec, block=spec.block.with_(pre=spec.block.param("pre").with_(dilation=1)))
        monkeypatch.setattr(plugin.Plugin, "recipe", staticmethod(dilated))
    else:                                             # the oracle the CPU provider runs, its window never cut
        from holon_plugins.models.qwen4_exp import reference

        def no_cut(ids, history, slots, query_start, has_init, multipliers, sizes, offsets, ngram, heads_per, eos):
            out = torch.empty(len(ids), (ngram - 1) * heads_per, dtype=torch.long)
            for r, (s, e) in enumerate(zip(query_start[:-1].tolist(), query_start[1:].tolist(), strict=True)):
                seq = (history[slots[r]].tolist() if has_init[r] else [eos, eos]) + ids[s:e].tolist()
                for i in range(2, len(seq)):
                    heads = []
                    for n in range(2, ngram + 1):
                        mixed = 0
                        for shift in range(n):
                            mixed ^= reference._wrap(seq[i - shift] * int(multipliers[shift]))
                        start = (n - 2) * heads_per
                        heads += [mixed % int(sizes[h]) + int(offsets[h]) for h in range(start, start + heads_per)]
                    out[s + i - 2] = torch.tensor(heads)
                if e > s:
                    history[slots[r]] = torch.tensor(seq[-2:])
            return out
        monkeypatch.setattr(reference, "ple_rows", no_cut)
    with pytest.raises(AssertionError):
        _check(tiny, 10)


def test_the_decode_gates_plant_and_control(tiny, capsys):
    """The decode-divergence gate's tools (judge: `qsa_decode_select_shift`, `qwen4exp.forced_dense`): the plant
    leaves prompt logprobs bitwise and changes decode's; the control never selects."""
    from holon_judge import planted, qwen4exp

    path, fixtures, _, _ = tiny
    ids = fixtures["p0.ids"].tolist()[: q.PROMPT]
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=1, prompt_logprobs=1)

    def run():
        engine = _engine(path, 128)
        engine.add(ids, params)
        prompt, decode = None, []
        while engine.has_work():
            for d in engine.step():
                prompt = d.prompt_logprobs.token if d.prompt_logprobs is not None else prompt
                decode += d.logprobs.token if d.logprobs is not None else []
        engine.close()
        return prompt, decode
    plain = run()
    with planted.PLANTS["qsa_decode_select_shift"]():
        shifted = run()
    assert shifted[0] == plain[0] and shifted[1] != plain[1] and "fired" in capsys.readouterr().err
    with qwen4exp.forced_dense():
        engine = _engine(path, 128)
        seen = _record(engine)
        _ask(engine, ids, DECODE)
        assert all(not seen[layer] for layer in (3, 7))                                   # dense: no selection
        engine.close()
