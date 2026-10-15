"""GDN rollback by masked replay (docs/m04-spec.md D6), on CPU with the reference kernels, end to end and per unit.

- Speculative greedy decoding of a tiny Qwen3.5 with GDN layers emits exactly what plain greedy decoding does, for
  right, wrong and mixed drafts, at overlap depth 0 and 1, with prefix hits and preemption (the `test_spec` matrix).
- The GDN units, bitwise: a verify step over every GDN layer followed by the model's rewind (n accepted of k, n < k
  and n = k) leaves each layer's conv window and delta-rule state exactly as 1 + n one-token calls of the same kernels
  over the same inputs would, and rows outside the step untouched — the step's glue runs once for every layer
  (docs/m07-perf.md D5). On a GPU (`kernels.fla`, `kernels.gdn_triton`) the delta rule is held to one per-token call
  over the accepted tokens (its decode specialization rounds differently, docs/m04-spec.md §1.3), and the verify step
  and rewind replayed from a CUDA graph captured at another acceptance equal the eager run.
"""
import pathlib

import numpy as np
import pytest
import torch
from holon.api import Batch, DeviceBatch, Manifest, SamplingParams, rewind_modules
from holon.core.batching import host_plan
from holon.core.build import StepContext
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import Installed, PluginSet, discover
from holon_testing.qwen3_5_files import save, tiny_hf

from .fixtures.oracle import ORACLE

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
GREEDY = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    model = tiny_hf(4, seed=1)
    path = save(model, tmp_path_factory.mktemp("tiny_qwen3_5_spec"))
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 320, (n,), generator=g).tolist() for n in (5, 70, 131, 30)]
    plain = _engine(path, max_draft_tokens=0).generate(prompts, GREEDY)
    return path, prompts, plain


def _engine(path, **kw) -> Engine:
    found = discover()
    for name in ("oracle_policy", "oracle_drafter"):
        manifest = Manifest.load(FIXTURES / name / "plugin.toml")
        found[manifest.name] = Installed(manifest, f"tests.fixtures.{name}", FIXTURES / name)
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=48,
               max_model_len=256, state_memory=8 << 20, max_draft_tokens=3,
               slots={"drafter": "tests.oracle.drafter", "draft_policy": "tests.oracle.policy",
                      "verifier": "std.spec.greedy"}) | kw
    return Engine(EngineConfig(**cfg), plugins=PluginSet(found))


def _oracle(prompts, plain, mode):
    ORACLE.sequences = {tuple(p): p + e for p, e in zip(prompts, plain, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.steps, ORACLE.k, ORACLE.mode, ORACLE.vocab, ORACLE.ahead = 0, 3, mode, 320, False
    ORACLE.hidden.clear()
    ORACLE.hidden_slots.clear()


@pytest.mark.parametrize("memory", [8 << 20, 320 << 10], ids=["roomy", "preempting"])
@pytest.mark.parametrize("overlap", [False, True], ids=["depth0", "depth1"])
@pytest.mark.parametrize("mode", ["right", "wrong", "mixed"])
def test_speculative_greedy_on_gdn_is_plain_greedy(tiny, mode, overlap, memory):
    path, prompts, plain = tiny
    _oracle(prompts, plain, mode)
    engine = _engine(path, overlap=overlap, state_memory=memory, max_model_len=160)
    assert engine.generate(prompts, GREEDY) == plain
    assert engine.generate(prompts, GREEDY) == plain and engine.state.tokens_saved > 0      # prefix hits
    assert memory == 8 << 20 or engine.sched.preemptions > 0


def _ctx(engine, slots, starts, n, n_logits, device="cpu") -> StepContext:
    """A step over request slots `slots` (n tokens each from `starts`), its GDN metadata planned and prepared."""
    plan = host_plan(0, np.array(slots, np.int32), np.array(starts, np.int32), np.array(n, np.int32),
                     np.array(n_logits, np.int32), np.zeros(sum(n), np.int64))
    qs = torch.tensor([0, *np.cumsum(n).tolist()])
    dev = DeviceBatch(input_ids=torch.zeros(sum(n), dtype=torch.long),
                      positions=torch.cat([torch.arange(s, s + k) for s, k in zip(starts, n, strict=True)]),
                      query_start=qs, slots=torch.tensor(slots), logits_index=torch.arange(sum(n)),
                      logits_rows=torch.repeat_interleave(torch.arange(len(n)), torch.tensor(n)))
    dev = DeviceBatch(**{f: getattr(dev, f).to(device) for f in ("input_ids", "positions", "query_start", "slots",
                                                                    "logits_index", "logits_rows")})
    ctx = StepContext(Batch(plan, dev), {}, seq_table=engine.executor.seq_table)
    planner = engine.executor.planners["recurrent"]
    ctx.metas["recurrent"] = meta = planner.plan_bounds(ctx.batch, None)
    planner.prepare_device(meta, ctx)
    return ctx


@pytest.fixture(scope="module")
def gpu_engine(tmp_path_factory):
    """A tiny Qwen3.5 of 64-wide heads on the GPU kernels, bf16, without graphs (the test captures its own)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    path = save(tiny_hf(4, seed=1, dim=64), tmp_path_factory.mktemp("tiny_qwen3_5_gpu"))
    engine = _engine(path, device="cuda", dtype=torch.bfloat16, cuda_graphs=False, page_size=16)
    yield engine
    engine.close()


def _pools(gdns) -> list[torch.Tensor]:
    return [t for gdn in gdns for t in gdn._states()]


def _reference(engine, gdn, slots, starts, q, accepted, device) -> None:
    """From the states in the pools: 1 + n one-token calls per row of the same kernels over the inputs the verify step
    saved; on a GPU, the delta rule in one per-token call over the accepted tokens instead."""
    conv, ssm = gdn._states()
    saved_x, saved_a, saved_b, _ = gdn._replay
    mixed = {}
    for j in range(1 + max(accepted)):
        rows = [r for r, n in enumerate(accepted) if j <= n]
        meta = _ctx(engine, [slots[r] for r in rows], [starts[r] + j for r in rows], [1] * len(rows),
                    [1] * len(rows), device).metas["recurrent"]
        at = torch.tensor([r * q + j for r in rows], device=device)
        out = gdn._conv(saved_x[at], gdn.conv_weight, conv, meta.slots, meta.query_start, meta.has_init)
        if device == "cpu":
            gdn._rule_call(meta.rule, meta.handle, out, saved_a[at], saved_b[at], ssm, meta.slots, meta.has_init)
        mixed |= {(r, j): out[i] for i, r in enumerate(rows)}
    if device != "cpu":
        keys = sorted(mixed)
        at = torch.tensor([r * q + j for r, j in keys], device=device)
        lens = [1 + n for n in accepted]
        meta = _ctx(engine, slots, starts, lens, lens, device).metas["recurrent"]
        gdn._rule_call(meta.rule, meta.handle, torch.stack([mixed[k] for k in keys]), saved_a[at], saved_b[at], ssm,
                       meta.slots, meta.has_init, out_slots=meta.slots)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("accepted", [(0, 3), (1, 2), (3, 3)], ids=["none-all", "some", "all"])
def test_rewind_is_the_decode_steps_bitwise(tiny, request, accepted, device):
    engine = request.getfixturevalue("gpu_engine") if device == "cuda" else _engine(tiny[0])
    model = engine.executor.model
    gdns = [layer.mixer for layer in model.layers if hasattr(layer.mixer, "rewind")]
    assert len(gdns) > 1
    g = torch.Generator().manual_seed(4)
    for conv, ssm in zip(*[iter(_pools(gdns))] * 2, strict=True):
        conv.copy_(torch.randn(conv.shape, generator=g))
        ssm.copy_(torch.randn(ssm.shape, generator=g) * 0.1)
    before = [t.clone() for t in _pools(gdns)]
    slots, starts, q = [0, 1], [7, 12], 4                  # slot 2 is outside the step: untouched
    dtype = gdns[0].conv_weight.dtype
    xs = [torch.randn(len(slots) * q, engine.spec.hidden, generator=g).to(device, dtype) for _ in gdns]
    ctx = _ctx(engine, slots, starts, [q] * len(slots), [q] * len(slots), device)
    n_accept = engine.executor.seq_table.n_accept

    def step() -> None:
        for gdn, x in zip(gdns, xs, strict=True):
            gdn(x, ctx)
        rewind_modules(model, ctx)

    n_accept[torch.tensor(slots)] = torch.tensor(accepted, device=n_accept.device)
    step()
    got = [t.clone() for t in _pools(gdns)]
    for t, b in zip(_pools(gdns), before, strict=True):
        t.copy_(b)
    for gdn in gdns:                                       # each layer over its own saved inputs
        _reference(engine, gdn, slots, starts, q, accepted, device)
    for got_t, want_t, before_t in zip(got, _pools(gdns), before, strict=True):
        assert torch.equal(got_t[slots], want_t[slots])
        assert torch.equal(got_t[2], before_t[2])        # another request (unit 3, the padding slot, is the sink)
    if device == "cuda":                                   # graphed: captured at another acceptance, then replayed
        for t, b in zip(_pools(gdns), before, strict=True):
            t.copy_(b)
        n_accept[torch.tensor(slots)] = torch.tensor(accepted[::-1], device=n_accept.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        n_accept[torch.tensor(slots)] = torch.tensor(accepted, device=n_accept.device)
        graph.replay()
        sink = gdns[0]._sink                               # both rows' verify states race there
        assert all(torch.equal(a[:sink], b[:sink]) for a, b in zip(_pools(gdns), got, strict=True))
