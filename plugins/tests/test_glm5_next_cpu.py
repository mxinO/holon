"""GLM-5.3-Flash's model plugin on the CPU against HF's fixtures (docs/v1-glm.md §6.1): the tiny model (8 layers, DSA
at 3 and 7, the dense FFN at 0, n = 4, the real vocabulary) from its GGUF as llama.cpp's converter writes it, run by the
engine in float32 with the testing CPU providers; the fixtures are HF's one-shot forward under D2's tie rule over the
GGUF's own weights (`holon_testing.glm5next_files`). Each prompt (100 tokens with an EOS) runs as its 124 fixture ids
with prompt logprobs, one-shot and in chunks of 7 and 13 tokens, and as its 100-token prompt decoded greedily.

- Logits: each position's prompt logprob within 1e-4 of the fixture's (its head input x through the GGUF's head),
  the top-1 token identical; the greedy tokens are HF's.
- DSA: each (layer, position)'s selected tokens are HF's set (a dense step attends every earlier token).
- The F32 tensors the release keeps (the selection bias, A_log, dt_bias, the convs, mHC base and scale) load as f32
  with the file's bits (A_log through its −exp, as the judge reads it).
- Prefix hit vs recompute: bitwise. A short row beside a long one attends every block it sees, as alone (D5).
  Sub-batched (`rows`, D9): bitwise the unsplit run; off a prefix hit, the same tokens, logprobs within 1e-5.
- The decode run's tokens are HF's, each one's logprob within 1e-4 of the fixture's, its DSA sets HF's (A-m1).
- mHC's comb against HF's formula in float64 (C-m1).
- §6.3's plants the tiny CPU runs (`mhc_comb_transposed`, `mhc_sinkhorn_short`, `dsa_pool_ape_off`,
  `router_bias_in_weights`, `kda_state_not_carried`) are each caught by a pre-registered detector, by at least its
  floor (C-m2).
- The quantized loads `llama-quantize` leaves f32 in the fixtures (A-m1): a GGUF with Q6_K `attn_k_b` / `attn_v_b` and
  Q8_0 indexer and KDA-gate tensors, loaded through the recipe, against gguf-py's dequantization, bitwise.

Needs the tiny GGUF: `GLM_TINY_GGUF=<…/tiny-glm5next-f32.gguf>` (built by `python -m holon_testing.glm5next_files` in
the golden venv; its sha256 is pinned), else skipped.
"""
import hashlib
import os
import pathlib
import warnings
from unittest import mock

import numpy as np
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_testing import glm5next_files as g

GGUF = os.environ.get("GLM_TINY_GGUF")
SKIP = ("GLM's tiny-model proof (§6.1) did not run: set GLM_TINY_GGUF to the tiny CPU GGUF that `python -m "
        "holon_testing.glm5next_files --out <dir> --llama <llama.cpp 27c7a76>` builds in the golden venv "
        "(docs/release.md §22)")
if GGUF is None:
    warnings.warn(SKIP, stacklevel=1)       # loud: a skipped proof is not a passed one
pytestmark = pytest.mark.skipif(GGUF is None, reason=SKIP)
CHUNKS = (128, 7, 13)
DECODE = SamplingParams(temperature=0.0, max_tokens=g.DECODE, ignore_eos=True)
PROMPT_LP = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=1)
DECODE_LP = SamplingParams(temperature=0.0, max_tokens=g.DECODE, ignore_eos=True, logprobs=1)


@pytest.fixture(scope="module")
def tiny():
    path = pathlib.Path(GGUF)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == g.GGUF_SHA256["cpu"]
    fixtures, meta = g.load_fixtures()
    from holon_std.weights.gguf.source import GGUFSource

    head = dict(GGUFSource(path).tensors())["output.weight"]
    return path, fixtures, meta, head


def _engine(path, chunk: int, prefix: bool = False, seqs: int = 1) -> Engine:
    return Engine(EngineConfig(str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=seqs,
                               max_step_tokens=chunk, max_model_len=256, state_memory=64 << 20, prefix_cache=prefix))


def _record(engine: Engine) -> dict:
    """Wrap every DSA layer's selection: per call, (positions, blocks)."""
    seen = {}
    for i, layer in enumerate(engine.executor.model.layers):
        if hasattr(layer.mixer, "_select"):
            seen[i], select = [], layer.mixer._select

            def spy(*args, select=select, into=seen[i]):
                out = select(*args)
                into.append((args[4].clone(), out.clone()))
                return out
            layer.mixer._select = spy
    return seen


def _ask(engine: Engine, ids: list[int], params: SamplingParams):
    """Run one request: its generated tokens, its prompt logprobs (or None), its generated tokens' logprobs."""
    rid, tokens, lp, decode = engine.add(ids, params), [], None, []
    while engine.has_work():
        for d in engine.step():
            if d.rid == rid:
                tokens += d.token_ids
                lp = d.prompt_logprobs if d.prompt_logprobs is not None else lp
                decode += d.logprobs.token if d.logprobs is not None else []
    return tokens, lp, decode


@pytest.mark.parametrize("chunk", CHUNKS)
def test_the_engine_matches_hf(tiny, chunk):
    stats = _check(tiny, chunk)
    assert stats["logprob"] <= 1e-4, stats                         # f32 against f32 HF: summation order only
    assert stats["top1"] == stats["sets"] == stats["tokens"] == 0, stats


# Each plant's pre-registered detector and floor (C-m2): the check that must catch it, and by how much at least.
# Measured at step 3's close (chunk 13): evidence/glm_step3_close_2026-10-25.json, `plants`.
PLANT_FLOORS = {"mhc_comb_transposed": ("logprob", 0.1), "router_bias_in_weights": ("logprob", 0.1),
                "kda_state_not_carried": ("logprob", 0.1), "dsa_pool_ape_off": ("sets", 50),
                "mhc_sinkhorn_short": ("comb", 1e-3)}


@pytest.mark.parametrize("plant", sorted(PLANT_FLOORS))
def test_plants_fail(tiny, plant):
    """§6.3's plants the tiny CPU model runs (chunked, 13): each is caught by its pre-registered detector by at least
    its floor — the clean run sits at ≤ 2e-6 in logprob, 0 sets and 1.7e-6 in comb (`test_the_engine_matches_hf`,
    `test_the_comb_is_a_float64_sinkhorns`)."""
    from holon_judge.planted import PLANTS

    detector, floor = PLANT_FLOORS[plant]
    with PLANTS[plant]():
        stats = _comb_error(tiny) if detector == "comb" else _check(tiny, 13)
    print(f"plant {plant}: {stats}")
    assert (stats if detector == "comb" else stats[detector]) >= floor, (plant, stats)


def _check(tiny, chunk: int) -> dict:
    """The engine against the fixtures, per prompt: the greedy run (its tokens and each one's logprob, its DSA sets),
    then the prompt-logprob run over the 124 ids (each position's logprob, top-1, DSA sets). Returns the largest
    |Δ logprob| and the counts of top-1 flips, set mismatches and token mismatches."""
    path, fixtures, _, head = tiny
    stats = {"logprob": 0.0, "top1": 0, "sets": 0, "tokens": 0}
    for p, prompt in enumerate(g.prompts()):
        ids, x = fixtures[f"p{p}.ids"].tolist(), fixtures[f"p{p}.x"]
        want = torch.log_softmax(x @ head.T, dim=-1)                                       # the fixture's logits
        ref = want[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
        engine = _engine(path, chunk)
        seen = _record(engine)
        tokens, _, decode = _ask(engine, prompt, DECODE_LP)
        n = len(prompt)
        stats["tokens"] += sum(a != b for a, b in zip(tokens, ids[n:], strict=True))
        if tokens == ids[n:]:                                  # decode logprobs: of HF's tokens, at their positions
            _worst(stats, torch.tensor(decode) - ref[n - 1:])
        _sets(stats, fixtures, p, seen, range(n, len(ids) - 1))     # the decode rows' (the last token is not fed)
        for calls in seen.values():
            calls.clear()
        _, lp, _ = _ask(engine, ids, PROMPT_LP)
        _worst(stats, torch.tensor(lp.token) - ref)
        stats["top1"] += sum(a[0] != b for a, b in zip(lp.top_ids, want[:-1].argmax(-1).tolist(), strict=True))
        _sets(stats, fixtures, p, seen, range(len(ids)))
        engine.close()
    return stats


def _worst(stats: dict, delta: torch.Tensor) -> None:
    stats["logprob"] = max(stats["logprob"], float(delta.abs().max()))


def _sets(stats: dict, fixtures: dict, p: int, seen: dict, positions: range) -> None:
    """Count the positions whose selected tokens (DSA layers 3 and 7) are not HF's (a dense step: every earlier one)."""
    for layer in (3, 7):
        sets, chosen = fixtures[f"p{p}.sets.{layer}"], {}
        for at, blocks in seen[layer]:
            for t, row in zip(at.tolist(), blocks.tolist(), strict=True):
                chosen[t] = {4 * b + i for b in row if b >= 0 for i in range(4) if 4 * b + i <= t}
        stats["sets"] += sum(chosen.get(t, set(range(t + 1))) != {s for s in sets[t].tolist() if s >= 0}
                             for t in positions)


def test_the_comb_is_a_float64_sinkhorns(tiny):
    """C-m1: every mHC mix's comb (the kernel's output, chunked at 13) against HF's formula in float64 from the same
    R and weights, the iteration count read from the GGUF: within 1e-5 (fp32 against f64: 1.7e-6 measured, its
    logits' matmul included); one iteration short is 7.3e-3 (`test_plants_fail`'s floor: 1e-3)."""
    assert _comb_error(tiny) <= 1e-5


def _comb_error(tiny) -> float:
    """The largest |comb − comb_f64| over every mHC mix of prompt 0's prompt-logprob run, chunked at 13."""
    import gguf
    from holon_std.hyper import units

    path, fixtures, _, _ = tiny
    fields = gguf.GGUFReader(path).fields
    iters = int(fields["glm5-next.hyper_connection.sinkhorn_iterations"].parts[-1][0])
    engine, worst = _engine(path, 13), [0.0]
    for m in engine.executor.model.modules():
        if isinstance(m, units._MHC):
            def spy(r, pending=None, m=m):
                out = type(m).mix(m, r, pending)                  # the class's (a plant patches it)
                k = m._mix
                want = _sinkhorn64(r, m.fn, m.base, m.scale, k.streams, k.eps, k.hc_eps, iters)
                worst[0] = max(worst[0], float((out[2].double() - want).abs().max()))
                return out
            m.mix = spy
    _ask(engine, fixtures["p0.ids"].tolist(), PROMPT_LP)
    engine.close()
    return worst[0]


def _sinkhorn64(r, fn, base, scale, n: int, eps: float, hc_eps: float, iters: int) -> torch.Tensor:
    """HF's mHC comb in float64, from the R the mix read (after any pending write-back)."""
    x = r.double()
    m = torch.nn.functional.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps), fn.double())
    logits = m[:, 2 * n:].view(-1, n, n) * scale[2].double() + base[2 * n:].double().view(n, n)
    comb = logits.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return comb


def test_the_release_f32_tensors_load_with_their_bits(tiny):
    """Step 2's review A-m1: the selection bias, A_log (as −exp then log reads it), dt_bias, the convs and mHC's base
    and scale are float32 parameters holding the file's values exactly."""
    from holon_judge import gguf

    path = tiny[0]
    engine = _engine(path, 128)
    state = dict(engine.executor.model.named_parameters())
    want = dict(gguf.hf_state(path))
    pairs = {"layers.3.ffn.parts.0.bias": "model.layers.3.mlp.gate.e_score_correction_bias",
             "layers.0.mixer.A_log": "model.layers.0.self_attn.forget_gate.A_log",
             "layers.0.mixer.dt_bias": "model.layers.0.self_attn.forget_gate.dt_bias",
             "layers.0.mixer.conv_weight": "model.layers.0.self_attn.conv1d.weight",
             "layers.2.attn_hc.base": "model.layers.2.attn_hc.base",
             "layers.2.ffn_hc.scale": "model.layers.2.ffn_hc.scale"}
    for ours, hf in pairs.items():
        t = engine.executor.model.get_parameter(ours) if ours in state else getattr(
            engine.executor.model.get_submodule(ours.rpartition(".")[0]), ours.rpartition(".")[2])
        assert t.dtype == torch.float32 and torch.equal(t.reshape(want[hf].shape), want[hf]), ours
        assert not torch.equal(t, t.bfloat16().float()), ours                 # (a value bf16 would round)
    engine.close()


def test_a_prefix_hit_is_bitwise_a_recompute(tiny):
    """A request continuing a cached one resumes at its checkpoint (64: KDA's chunk) with the recompute's greedy tokens
    and logprobs, bitwise. Both run 64-row prompt steps, so every step's matmuls see the same rows on both sides (the
    CPU's matmul bits follow a call's rows: an expert's product over 3 rows is not its product over 2)."""
    path, fixtures, _, _ = tiny
    ids = fixtures["p0.ids"].tolist()
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=1)
    plain = _decode(_engine(path, 64), ids[:110], params)
    engine = _engine(path, 64, prefix=True)
    _ask(engine, ids[:100], DECODE)
    hit = _decode(engine, ids[:110], params)
    assert engine.stats()["prefix_tokens_saved"] == 64
    assert hit == plain


def _decode(engine: Engine, ids: list[int], params: SamplingParams) -> list:
    rid, out = engine.add(ids, params), []
    while engine.has_work():
        out += [(d.token_ids, d.logprobs.token) for d in engine.step() if d.rid == rid and d.token_ids]
    return out


def test_a_short_row_beside_a_long_one_attends_every_block(tiny):
    """D5: the dense path is the sparse kernel over every pool, so a row's attention does not depend on its step's
    other rows: a short prompt in a step beside a long (sparse) one selects every block it sees — exactly what its
    dense step attends (`all_blocks`) — so its attention's inputs are the same. (Its bits beside the long row are
    the GPU kernels' test, step 4: CPU matmuls' bits depend on a call's rows.)"""
    from holon_std.mixers.qsa.reference import all_blocks

    path, fixtures, _, _ = tiny
    long, short = fixtures["p0.ids"].tolist(), fixtures["p1.ids"].tolist()[:9]
    engine = _engine(path, 256, seqs=2)
    seen = _record(engine)
    engine.add(long, PROMPT_LP)
    engine.add(short, PROMPT_LP)
    while engine.has_work():
        engine.step()
    positions, blocks = seen[3][0]                                 # the one step: long's rows, then short's
    mine = slice(len(long), len(long) + len(short))
    assert torch.equal(blocks[mine], all_blocks(positions[mine], 4, blocks.shape[1]))
    assert (blocks[:len(long)] != all_blocks(positions[:len(long)], 4, blocks.shape[1])).any()    # (long: sparse)


def test_sub_batched_steps_keep_the_tokens_and_the_checkpoints(tiny):
    """D9's `rows` (patched to 64 here, the KDA rule's chunk; the recipe's 2,048 needs longer prompts): a 124-token
    prompt runs as two sub-batches through every mixer and the dense / shared FFN — bitwise the unsplit run (the
    routed experts, whose CPU bits follow a call's rows, see the same rows on both sides: an off-by-one in a piece's
    bounds or a conv tail not carried cannot hide under a tolerance; on the GPU the bound is the L0's, C-m3); a prefix
    hit off a sub-batched prompt resumes at its checkpoint (64, a piece's end) with the recompute's tokens, logprobs
    within 1e-5 (its 46-row step against the recompute's 120: the experts' row-count bits, ≈ 1e-6)."""
    from holon_plugins.models.glm5_next import recipe

    path, fixtures, _, _ = tiny
    ids = fixtures["p0.ids"].tolist()
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=1)
    plain = _decode(_engine(path, 128), ids[:120], params)
    with mock.patch.object(recipe, "ROWS", 64):
        split = _decode(_engine(path, 128), ids[:120], params)
        engine = _engine(path, 128, prefix=True)
        assert engine.executor.model.layers[0].mixer.rows == 64
        _ask(engine, ids[:100], DECODE)
        hit = _decode(engine, ids[:120], params)
        recompute = _decode(_engine(path, 128), ids[:120], params)
    assert split == plain
    assert [t for t, _ in hit] == [t for t, _ in recompute]
    assert np.abs(np.array([x for _, x in hit]) - np.array([x for _, x in recompute])).max() <= 1e-5


def test_quantized_loads_are_gguf_pys(tiny, tmp_path):
    """§6.1 (A-m1): the tensors `llama-quantize` leaves f32 in the fixtures, quantized — Q6_K `attn_k_b` / `attn_v_b`
    (3-D, rows of 256 / 512 values where the real file's are) and Q8_0 indexer and KDA-gate tensors — load through the
    recipe as gguf-py dequantizes them (D5's BF16 rounding is the GPU's: the CPU model is f32), per head."""
    import gguf
    from holon_testing.glm5next_files import quantized_copy

    src, quantized = quantized_copy(tiny[0], tmp_path)
    engine = _engine(src, 128)
    reader = gguf.GGUFReader(src)
    model = engine.executor.model
    checked = 0
    for t in reader.tensors:
        if t.name not in quantized:
            continue
        want = torch.from_numpy(gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).copy())
        want = want.reshape([int(n) for n in reversed(t.shape)])
        got = _loaded(model, t.name)
        assert torch.equal(got.reshape(want.shape), want), t.name                 # every head, bitwise
        checked += 1
    assert checked == len(quantized) and checked >= 8
    engine.close()


def _loaded(model, name: str) -> torch.Tensor:
    """The engine's dequantized value of GGUF tensor `name` (a parameter, or a linear's weight as it computes)."""
    from holon_plugins.models.glm5_next.recipe import RULES

    layer, part = name.split(".", 2)[1], name.split(".", 2)[2]
    target = next(r.target for r in RULES if r.pattern == f"blk.{{i}}.{part}").replace("{i}", layer)
    module, _, attr = target.rpartition(".")
    m = model.get_submodule(module)
    if attr == "weight" and hasattr(m, "in_features"):                 # a linear: what it computes, W (x = I)
        with torch.inference_mode():
            return m(torch.eye(m.in_features)).T.float()
    return getattr(m, attr).float()
