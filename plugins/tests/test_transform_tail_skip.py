"""transforms.tail_skip (A1, docs/dev/v1-offload.md D1, D3, §6: Strata's route tail skip, `approx`): its contract
`moe_route_transform.tail_skip@1` from both providers (the oracle on the CPU, one Triton program on CUDA) against the
oracle, exactly; then on the GPU end to end: bound by the explicit option only, with `tail_skip.rank` at the router's
width nothing is dropped (the outputs are the resident store's, bit for bit), below it missed routes are dropped (fewer
counted), decode steps graphed."""
import pytest
import torch
from holon.core.build import BuildContext
from holon_plugins.transforms.tail_skip import reference

from .test_expert_io import DEVICES, ROLES, E, K, plugins  # noqa: F401
from .test_moe_engine import KEY, _gpu_engine, _served, gpu_moe  # noqa: F401

PREBUILT = ("expert_io",)  # CUDA extensions built if missing (conftest)


OP = "moe_route_transform.tail_skip@1"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tokens, rank", [(1, 7), (6, 7), (6, 4), (300, 7), (4096, 6)])
def test_tail_skip_matches_the_oracle(plugins, device, tokens, rank):  # noqa: F811
    """Unique ids per row, some dropped already; about half the experts missing in some role. Only a missed expert
    every row routes at `rank` or later is dropped, all its entries; weights are returned untouched."""
    g = torch.Generator().manual_seed(tokens + rank)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(tokens)]).int()
    ids[torch.rand(tokens, K, generator=g) < 0.05] = -1
    residency = torch.where(torch.rand(E, ROLES, generator=g) < 0.4, -1, 100).int()
    weights = torch.rand(tokens, K, generator=g)
    want, same = reference.tail_skip(ids, weights, residency, rank)
    ctx = BuildContext(plugins, torch.device(device), options={"plugins": {"tail_skip.rank": rank}})
    spec = ctx.select_kernel(OP, site="test", experts=E, top_k=K)
    assert spec.provider == "transforms.tail_skip"
    got, w = spec.impl(ctx, experts=E, top_k=K)(ids.to(device), weights.to(device), residency.to(device))
    assert torch.equal(got.cpu(), want) and w is not None and torch.equal(w.cpu(), weights) and same is weights
    # the contract's own terms, on the oracle
    dropped = (want < 0) & (ids >= 0)
    missed = (residency < 0).any(-1)
    assert torch.equal(want[~dropped], ids[~dropped]) and missed[ids[dropped].long()].all()
    for e in ids[dropped].unique().tolist():               # every entry of a dropped expert, none at a better rank
        at = (ids == e).nonzero()[:, 1]
        assert at.min() >= rank and ((want == -1) | (ids != e)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tail_skip_on_the_gpu(gpu_moe):  # noqa: F811
    """The route transform on the native path at a quarter of the experts cached: at `tail_skip.rank` 2 (top-2: no
    rank reaches it) the streams and prompt logprobs are the resident store's, bit for bit; at rank 1 the missed
    experts only ever routed second are dropped — fewer routes counted, the outputs no longer the resident store's —
    in graphed decode steps."""
    file, prompts, want = gpu_moe
    options = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.route_transform": "tail_skip"}
    counted = {}
    for rank in (2, 1):
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=options | {"tail_skip.rank": rank})
        store, n = engine.ctx.shared_objects[KEY], []
        store.taps.append(lambda step, record, n=n: n.append(int(record["counts"].sum())))
        got = _served(engine, prompts)
        assert (got == want) == (rank == 2) and engine.stats()["graphs"] > 0
        assert {v["provider"] for k, v in engine.ctx.kernels.items() if k.endswith(OP)} == {"transforms.tail_skip"}
        counted[rank] = sum(n)
        engine.close()
    assert counted[1] < counted[2]
