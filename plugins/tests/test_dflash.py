"""DFlash2 (`drafters.dflash`, docs/dev/m04-spec.md D4, D5) on a tiny Qwen3.5 target with a tiny random draft, on CPU.

One engine run per configuration (overlap depth 0 and 1, roomy and preempting pools, chunked prefill), three greedy
requests and a sampled one in the same batches:
- greedy output is plain greedy's (`std.spec.rejection`);
- every block drafted for a greedy request equals the published forward (`reference.py`) run on the target features
  the taps saw for that request's committed positions — the top-k candidates and the greedy walk. A random draft is
  rarely right, so on alternate steps the drafts are replaced by the right tokens after the drafter ran: accepted
  drafts move the next block's anchor and the context it attends to;
- the sampled request's published laws are the contract's: real rows sum to 1 and hold their draft with q > 0;
  greedy rows' are one-hot at their draft.

On a GPU (Qwen3-0.6B with a random draft of its geometry, quantized by llama.cpp's converter): verify graphs.
"""
import pathlib

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_plugins.drafters.dflash import plugin as dflash
from holon_plugins.drafters.dflash.reference import Reference
from holon_testing import dflash_files
from holon_testing.qwen3_5_files import save, tiny_hf
from safetensors.torch import load_file

GREEDY = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True, seed=1)
SAMPLED = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=20, ignore_eos=True, seed=2)
K = 7


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("dflash")
    (root / "target").mkdir()
    target = save(tiny_hf(4, seed=1), root / "target")
    draft = dflash_files.save(root / "draft", 64, 320, (0, 2), mask=7)
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 320, (n,), generator=g).tolist() for n in (5, 30, 17, 11)]
    plain = _engine(target).generate(prompts[:3], GREEDY)
    return target, draft, prompts, plain


def _engine(target, draft=None, **kw) -> Engine:
    cfg = dict(model=str(target), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=16,
               max_model_len=96, state_memory=8 << 20) | kw
    if draft is not None:
        cfg |= dict(max_draft_tokens=K, plugin_options={"spec.k": str(K), "dflash.path": str(draft)},
                    slots={"drafter": "drafters.dflash", "draft_policy": "std.spec.fixed_k",
                           "verifier": "std.spec.rejection"})
    return Engine(EngineConfig(**cfg))


class Watch:
    """Wraps the drafter's `run`: keeps the tapped features at each (slot, position) a step computed, checks each
    greedy block against the reference, checks published laws, and on alternate steps replaces the drafts by the right
    tokens (`expected`: each greedy request's whole sequence, by request id)."""

    def __init__(self, engine, draft, expected: dict[str, list[int]]) -> None:
        self.engine, self.expected = engine, expected
        self.drafter = drafter = engine.executor.drafter
        model = engine.executor.model
        cfg = dflash.checkpoint(engine.ctx.weight_source(str(draft)))
        self.reference = Reference(load_file(draft / "model.safetensors"), cfg, model.embed.weight.detach(),
                                   model.lm_head.weight.detach())
        self.features: dict[tuple[int, int], torch.Tensor] = {}
        self.blocks = self.forced = self.laws = 0
        run = drafter.run

        def watched(ctx, h):
            dev = ctx.batch.dev
            row = torch.searchsorted(dev.query_start[1:], torch.arange(len(h)), right=True)
            for t, (slot, pos) in enumerate(zip(dev.slots[row].tolist(), dev.positions.tolist(), strict=True)):
                self.features[slot, pos] = drafter.features[t].clone()
            run(ctx, h)
            if ctx.batch.plan.draft_k and len(dev.logits_rows):
                self.check(ctx)

        drafter.run = watched

    def check(self, ctx) -> None:
        table, plan = ctx.seq_table, ctx.batch.plan
        slots = plan.slots[plan.n_logits > 0]
        owner = {seq.slot: seq.rid for seq in self.engine.sched.running}
        for slot in slots.tolist():
            p, anchor = int(table.num_computed[slot]), int(table.last_token[slot])
            drafts, ids, q = table.draft_tokens[slot, :K], table.draft_q_ids[slot, :K], table.draft_q[slot, :K]
            assert torch.allclose(q.sum(-1), torch.ones(K), atol=1e-5)
            assert (q.gather(1, (ids == drafts[:, None]).int().argmax(1, keepdim=True)) > 0).all()
            assert ((ids == drafts[:, None]) & (q > 0)).any(1).all()
            self.laws += 1
            right = self.expected.get(owner.get(slot))
            if right is None:                                         # the sampled request
                continue
            assert (q.max(-1).values == 1).all()                      # greedy: one-hot at the draft
            hidden = self.reference.block(torch.stack([self.features[slot, i] for i in range(p)]), anchor)
            _, cand = self.reference.logits(hidden).topk(self.drafter.top_k)
            assert torch.equal(ids, cand) and torch.equal(drafts, self.reference.greedy(hidden, anchor,
                                                                                         self.drafter.top_k))
            self.blocks += 1
            if self.blocks % 2:                                       # right drafts, under one-hot laws
                right = torch.tensor(right[p + 1: p + 1 + K])
                if len(right) == K:
                    table.draft_tokens[slot, :K] = right
                    table.draft_q_ids[slot, :K, 0], table.draft_q[slot, :K] = right, torch.eye(4)[0]
                    self.forced += 1


@pytest.mark.parametrize(("overlap", "memory"), [(False, 300_000), (True, 8 << 20)],
                         ids=["depth0-preempting", "depth1-roomy"])
def test_dflash_drafts_like_its_reference_and_is_lossless(case, overlap, memory):
    target, draft, prompts, plain = case
    engine = _engine(target, draft, overlap=overlap, state_memory=memory)
    rids = [engine.add(p, GREEDY) for p in prompts[:3]] + [engine.add(prompts[3], SAMPLED)]
    watch = Watch(engine, draft, {r: p + e for r, p, e in zip(rids, prompts, plain, strict=False)})
    out = {r: [] for r in rids}
    while engine.has_work():
        for d in engine.step():
            out[d.rid] += d.token_ids
    assert [out[r] for r in rids[:3]] == plain and len(out[rids[3]]) == SAMPLED.max_tokens
    stats = engine.stats()
    accepted = sum(n for key, n in stats.items() if key.startswith("spec_rows_k7_accepting_") and key[-1] != "0")
    assert watch.blocks > 10 and watch.forced and accepted and watch.laws > watch.blocks
    assert memory == 8 << 20 or engine.sched.preemptions > 0


SNAP = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


needs_gpu = pytest.mark.skipif(SNAP is None or not torch.cuda.is_available() or not dflash_files.can_convert(),
                               reason="needs Qwen3-0.6B, a GPU and llama.cpp's converter")


@pytest.fixture(scope="module")
def quantized_draft(tmp_path_factory):
    """A random DFlash2 of Qwen3-0.6B's geometry (head_dim 128, 16 candidates), converted to q8_0 by llama.cpp."""
    return dflash_files.convert(dflash_files.save(tmp_path_factory.mktemp("draft"), 1024, 151936, (3, 20),
                                                  mask=151643, layers=1, window=64, heads=(8, 4, 128), inter=512,
                                                  rank=256, top_k=16), SNAP)


def _gpu_config(draft, **kw) -> EngineConfig:
    return EngineConfig(model=str(SNAP), dtype=torch.bfloat16, max_seqs=8, max_model_len=256, max_draft_tokens=K,
                        plugin_options={"spec.k": str(K), "dflash.path": str(draft)},
                        slots={"drafter": "drafters.dflash", "draft_policy": "std.spec.fixed_k",
                               "verifier": "std.spec.rejection"}, **kw)


@pytest.mark.slow
@pytest.mark.model
@needs_gpu
def test_the_startup_profile_runs_the_draft_block(quantized_draft, monkeypatch):
    # The pools are sized after the worst step's peak: eager verify steps (logprobs) run the drafter's block over every
    # slot, so the profile does too — without graphs, which would hide it in their capture.
    blocks = []
    walk = dflash.DFlashDrafter.walk
    monkeypatch.setattr(dflash.DFlashDrafter, "walk", lambda self, *a: blocks.append(a[1].shape) or walk(self, *a))
    Engine(_gpu_config(quantized_draft, cuda_graphs=False)).close()
    assert blocks and blocks[0][0] == 8                       # every slot's block


@pytest.mark.slow
@pytest.mark.model
@needs_gpu
def test_a_quantized_draft_runs_in_verify_graphs(quantized_draft):
    # Qwen3-0.6B with a random DFlash2 of its geometry (head_dim 128, 16 candidates), converted to q8_0 by llama.cpp:
    # quantized linears and codebook rows, the taps, the block sub-step and the sparse laws inside captured verify
    # graphs. Graphed and eager runs of the same padded plans (the eager runner calls the step instead of replaying
    # it) give the same tokens and acceptances, greedy and sampled; every verify step replays a graph.
    from transformers import AutoTokenizer

    class Eager:
        def run(self, key, fn):
            return fn()

        def __len__(self):
            return 0

    tok = AutoTokenizer.from_pretrained(SNAP)
    prompts = [tok(t)["input_ids"] for t in ("1, 2, 3, 4, 5, 6, 7, 8,", "def add(a, b):\n    return a + b\n",
                                             "The capital of France is", "a b c a b c a b")]
    runs = [SamplingParams(temperature=0.0, seed=5, max_tokens=24, ignore_eos=True),
            SamplingParams(temperature=0.8, top_p=0.9, seed=3, max_tokens=24, ignore_eos=True)]
    outs = []
    for graphed in (True, False):
        engine = Engine(_gpu_config(quantized_draft))
        assert engine.executor.drafter.model.fc._q is not None                  # the GGUF's q8_0, adopted
        if not graphed:
            engine.executor.graphs = Eager()
        tokens = [engine.generate(prompts, params) for params in runs]
        stats = engine.stats()
        assert stats["spec_verify_steps"] > 0 and stats["spec_verify_steps_graphed"] == stats["spec_verify_steps"]
        outs.append((tokens, {k: v for k, v in stats.items() if k.startswith("spec_") and "graphed" not in k}))
        engine.close()                                           # its memory back before the next engine sizes pools
    assert outs[0] == outs[1]
