"""The PLE unit's state on the GPU (docs/v1-flash-next.md acceptance 2, the replay half of its GPU unit test; the
chunked half is test_hc_ple_kernels): a verify row of 1 + 3 tokens (its real state left as before the step, the ops run
on a scratch copy) rewound to n accepted drafts leaves bitwise the state — and gives bitwise the residual updates of the
accepted tokens — of 1 + n one-token decode steps, for n = 0 … 3; and an in-step checkpoint inside a prefill holds
bitwise the state a step ending there leaves. The unit runs on its own: its two Slot specs bound to ledger pools, its
planner's metadata made per step, with Qwen3.8-Flash-Next's sizes (4 streams of 2,560; PLE 16 heads of 160). As the
contract states, both paths get the same K / V projection rows: the test projects one row at a time."""
import types

import numpy as np
import pytest
import torch
from holon.api import Batch, DeviceBatch, InStepPublish, Payload, Ref
from holon.core.batching import host_plan
from holon.core.build import BuildContext, StepContext
from holon.core.memory import Ledger
from holon.core.plugins import PluginSet, discover
from holon_plugins.models.qwen4_exp import units

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
N, H, NGRAM, PER, DIM, SLOTS = 4, 2560, 3, 8, 160, 4
SIZES = (100003, 100019, 100043, 100049, 100057, 100069, 100103, 100109,
         100129, 100151, 100153, 100169, 100183, 100189, 100193, 100207)


def _unit():
    ref = Ref("qwen4_exp.ple@1", n=N, width=N * H, embed_dim=2 * PER * DIM, ngram=NGRAM, heads_per_ngram=PER,
              eos=248044, multipliers=(2 ** 61 + 1, 2 ** 59 + 3, 2 ** 57 + 5), sizes=SIZES,
              offsets=tuple(int(x) for x in np.cumsum((0,) + SIZES[:-1])), table_rows=-(-sum(SIZES) // 128) * 128,
              dilation=NGRAM, kernel=4)
    device = torch.device("cuda")
    ledger = Ledger(device)
    ctx = BuildContext(PluginSet(discover()), device, torch.bfloat16,
                       {"max_seqs": SLOTS, "max_step_tokens": 256, "max_draft_tokens": 3, "page_size": 64}, ledger)
    specs = units.PLE.state_specs(dict(ref.params), (1,), ctx)
    pools = {s.name: ledger.request(f"state.{s.name}", unit_bytes=s.shape.unit_bytes, min_units=SLOTS + 1,
                                    pref_units=SLOTS + 1) for s in specs}
    payloads = ledger.plugin_pool("ple.checkpoints", unit_bytes=specs[1].shape.unit_bytes, min_units=2, pref_units=2)
    history_payloads = ledger.plugin_pool("ple.history.checkpoints", unit_bytes=16, min_units=2, pref_units=2)
    ledger.materialize(None)
    for s in specs:
        ctx.bind_state(s, pools[s.name])
    ple = ctx.unit(ref, layer=1, hidden=H)
    g = torch.Generator().manual_seed(0)
    for p in ple.parameters():
        p.data = (torch.randn(p.shape, generator=g) * (0.1 if p.dim() == 1 else p.shape[-1] ** -0.5)).to(
            device, p.dtype)
    for proj in (ple.key_proj, ple.value_proj):      # projected one row at a time: the same rows in on both paths
        forward = proj.forward                       # (a GEMM's bits may depend on its rows; the contract's
        proj.forward = lambda e, forward=forward: torch.cat([forward(row[None]) for row in e])    # precondition)
    return ple, ledger, (payloads, history_payloads)


def _step(ple, ledger, slot: int, start: int, ids: list[int], verify: bool = False, publishes=(), r=None):
    """One step of one row (its residual `r`, else random): the planner's metadata, then the unit; returns the
    residual it updated."""
    n = len(ids)
    plan = host_plan(0, np.array([slot], np.int32), np.array([start], np.int32), np.array([n], np.int32),
                     np.array([n if verify else 1], np.int32), np.array(ids, np.int64))
    plan.publishes = list(publishes)
    dev = DeviceBatch(input_ids=torch.tensor(ids, device="cuda"),
                      positions=torch.arange(start, start + n, device="cuda"),
                      query_start=torch.tensor([0, n], device="cuda", dtype=torch.int32),
                      slots=torch.tensor([slot], device="cuda", dtype=torch.int32),
                      logits_index=torch.tensor([n - 1], device="cuda"), logits_rows=torch.tensor([0], device="cuda"))
    ctx = StepContext(Batch(plan, dev), pools=ledger.pools)
    planner = units.PLEPlanner()
    ctx.metas[units.GROUP] = planner.plan_bounds(ctx.batch, None)
    planner.prepare_device(ctx.metas[units.GROUP], ctx)
    if r is None:
        r = torch.randn(n, N * H, generator=torch.Generator().manual_seed(start), dtype=torch.float32).cuda()
    ple(r, ctx)
    return r, ctx


def _state(ple, slot):
    history, conv = ple._views()
    return history[slot].clone(), conv[slot].clone()


@torch.no_grad()
def test_replay_rewind_is_bitwise_one_token_steps():
    ple, ledger, _ = _unit()
    g = torch.Generator().manual_seed(1)
    prompt, row = torch.randint(0, 248000, (37,), generator=g).tolist(), torch.randint(0, 248000, (4,),
                                                                                         generator=g).tolist()
    for accepted in range(4):
        _step(ple, ledger, 0, 0, prompt)                        # slot 0: verify, then rewind
        _step(ple, ledger, 1, 0, prompt)                        # slot 1: decode steps
        before = _state(ple, 0)
        r = torch.randn(4, N * H, generator=g).cuda()
        r_verify, ctx = _step(ple, ledger, 0, len(prompt), row, verify=True, r=r.clone())
        assert all(torch.equal(a, b) for a, b in zip(_state(ple, 0), before, strict=True))     # S0 kept
        ctx.seq_table = types.SimpleNamespace(n_accept=torch.full((SLOTS + 1,), accepted, device="cuda"))
        ple.rewind(ctx)
        steps = [_step(ple, ledger, 1, len(prompt) + j, [row[j]], r=r[j: j + 1].clone())[0]
                 for j in range(1 + accepted)]
        assert all(torch.equal(a, b) for a, b in zip(_state(ple, 0), _state(ple, 1), strict=True)), accepted
        assert torch.equal(r_verify[: 1 + accepted], torch.cat(steps))


@torch.no_grad()
def test_an_in_step_checkpoint_is_the_state_a_step_ending_there_leaves():
    ple, ledger, (conv_payloads, history_payloads) = _unit()
    ids = torch.randint(0, 248000, (150,), generator=torch.Generator().manual_seed(2)).tolist()
    publishes = [InStepPublish(0, units.CONV, 64, Payload((0,), conv_payloads.name)),
                 InStepPublish(0, units.HISTORY, 64, Payload((0,), history_payloads.name))]
    _step(ple, ledger, 0, 0, ids, publishes=publishes)
    _step(ple, ledger, 1, 0, ids[:64])                          # a step ending at 64
    history, conv = ple._views(history_payloads, conv_payloads)
    want = _state(ple, 1)
    assert torch.equal(history[0], want[0]) and torch.equal(conv[0], want[1])
