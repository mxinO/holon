"""L0 for the offload store's in-graph contracts (docs/v1-offload.md §5; std.moe.kernels): `moe_resolve@1` and
`expert_fetch@1` from kernels.expert_io (CUDA) and testing.ref_kernels (CPU) against the std oracles, exactly.

Resolve at E = 128, k = 8, two roles, 1 to 2049 tokens: decode, prefill and padding rows mixed, dropped routes
(negative ids), experts resident in one role and missing in the other, the null slots shared by every padding row and
dropped route of a rank; and past its staging slots (the miss count raised, overflowing misses on the last staging
slot). Fetch: units gathered from a mapped host pool's device view (zero-copy) and from device memory, only the first
n of the list, nothing else written."""
import pytest
import torch
from holon.core import platform
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_std.moe import reference
from holon_std.moe.kernels import DECODE, EXPERT_FETCH, MOE_RESOLVE, PADDING, PREFILL

E, K, ROLES = 128, 8, 2
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]


@pytest.fixture(scope="module")
def plugins():
    return PluginSet(discover())


def kernel(plugins, device: str, op: str, provider: str, **facts):
    ctx = BuildContext(plugins, torch.device(device))
    spec = ctx.select_kernel(op, site="test", **facts)
    assert spec.provider == provider
    return spec.impl(ctx, **facts)


def routing(tokens: int, seed: int) -> dict[str, torch.Tensor]:
    """Unique ids per row (some dropped), row kinds of every sort, a map with about half the pieces resident (in
    distinct slots past the null and staging slots, per role) and host units a permutation."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(tokens)]).int()
    ids[torch.rand(tokens, K, generator=g) < 0.05] = -1                      # dropped routes
    kinds = torch.tensor([DECODE, PREFILL, PADDING], dtype=torch.uint8)[torch.randint(0, 3, (tokens,), generator=g)]
    residency = torch.full((E, ROLES), -1, dtype=torch.int32)
    for r in range(ROLES):
        resident = torch.rand(E, generator=g) < 0.5
        residency[resident, r] = (K + 200 + torch.randperm(1000, generator=g)[:int(resident.sum())]).int()
    host = torch.stack([torch.randperm(E, generator=g) for _ in range(ROLES)], 1).int()
    counts = torch.randint(0, 50, (E,), generator=g).int()
    return {"ids": ids, "kinds": kinds, "residency": residency, "host": host, "counts": counts}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tokens, staging", [(1, 8), (5, 40), (64, E), (513, E), (2049, E), (64, 16)])
def test_resolve_matches_the_oracle(plugins, device, tokens, staging):
    provider = "testing.ref_kernels" if device == "cpu" else "kernels.expert_io"
    resolve = kernel(plugins, device, MOE_RESOLVE, provider, experts=E, top_k=K, roles=ROLES, staging=staging)
    case = routing(tokens, seed=tokens + staging)
    slots, fetch, n_fetch, increments = reference.moe_resolve(case["ids"], case["kinds"], case["residency"],
                                                              case["host"], staging)
    dev = {name: t.clone().to(device) for name, t in case.items()}
    misses = torch.full((1,), 3, dtype=torch.int32, device=device)
    got = resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["counts"], misses)
    assert torch.equal(got[0].cpu(), slots) and torch.equal(got[2].cpu(), n_fetch)
    for r in range(ROLES):
        n = min(int(n_fetch[r]), staging)
        assert torch.equal(got[1][r, :n].cpu(), fetch[r, :n])
    assert torch.equal(dev["counts"].cpu(), case["counts"] + increments)
    assert int(misses[0]) == max(3, int(n_fetch.max()))
    # the contract's own terms, on the oracle: null slot j for padding rows and dropped routes, staging past them
    null = (case["kinds"][:, None] == PADDING) | (case["ids"] < 0)
    assert (slots[:, null] == torch.arange(K).expand(tokens, K)[null]).all()
    assert ((slots[:, ~null] >= K) & ((slots[:, ~null] < K + staging) | (slots[:, ~null] >= K + 200))).all()
    assert (n_fetch > staging).any() == (staging == 16)            # the overflow case overflows


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mapped", [True, False])
def test_fetch_gathers_the_listed_units(plugins, device, mapped):
    if mapped and device == "cpu":
        pytest.skip("a mapped host pool is the CUDA case")
    provider = "testing.ref_kernels" if device == "cpu" else "kernels.expert_io"
    unit, staging, first, n = 4096 + 48, 6, K, 4
    fetch_ = kernel(plugins, device, EXPERT_FETCH, provider, unit_bytes=unit, staging=staging)
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (10, unit), dtype=torch.uint8, generator=g)
    if mapped:
        host = platform.host_memory(data.numel(), torch.device(device), mapped=True)
        host.view(10, unit).copy_(data)
        src = platform.device_alias(host, torch.device(device)).view(10, unit)
    else:
        src = data.to(device)
    dst = torch.randint(0, 256, (K + staging + 3, unit), dtype=torch.uint8, generator=g).to(device)
    units = torch.tensor([7, 2, 9, 2, 0, 5], dtype=torch.int32)
    want = reference.expert_fetch(units, n, data, dst.cpu(), first)
    fetch_(units.to(device), torch.tensor([n], dtype=torch.int32, device=device), src, dst, first)
    assert torch.equal(dst.cpu(), want)
    assert torch.equal(want[first + 1], data[2]) and not torch.equal(want[first + n], data[0])
