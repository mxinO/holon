"""DFlash2 (`drafters.dflash`, docs/m04-spec.md D4, D5) on a tiny Qwen3.5 target with a tiny random draft, on CPU.

One engine run per configuration (overlap depth 0 and 1, roomy and preempting pools, chunked prefill), three greedy
requests and a sampled one in the same batches:
- greedy output is plain greedy's (`std.spec.rejection`);
- every block drafted for a greedy request equals the published forward (`reference.py`) run on the target features
  the taps saw for that request's committed positions — the top-k candidates and the greedy walk. A random draft is
  rarely right, so on alternate steps the drafts are replaced by the right tokens after the drafter ran: accepted
  drafts move the next block's anchor and the context it attends to;
- the sampled request's published laws are the contract's: real rows sum to 1 and hold their draft with q > 0;
  greedy rows' are one-hot at their draft.
"""

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


@pytest.mark.parametrize("memory", [8 << 20, 300_000], ids=["roomy", "preempting"])
@pytest.mark.parametrize("overlap", [False, True], ids=["depth0", "depth1"])
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
