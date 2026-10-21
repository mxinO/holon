"""kernels.state_copy (L0): std.state_tier's `state_copy@1` moves rows of the K chunk sizes between a device pool and a
mapped host pool's device view, both directions and both paths, scattered rows with a consecutive run, against the
oracle bitwise, and an empty list."""
import pytest
import torch
from holon.core import platform
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_std.state_tier import reference
from holon_std.state_tier.kernels import STATE_COPY

pytestmark = pytest.mark.l0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("chunk", [34_816, 65_536, 61_440, 3_145_728])   # 27B q8_0 KV, 8B bf16 KV, GDN conv, GDN SSM
@pytest.mark.parametrize("direction", ["h2d", "d2h"])
@pytest.mark.parametrize("copy", ["sm", "dma"])
def test_state_copy_moves_the_listed_rows_exactly(direction, chunk, copy):
    """Between a device pool and a mapped host pool (`[segments · capacity, chunk]` rows), on a stream of its own as
    the tier's: scattered rows plus a run consecutive in both pools, then an empty list; the destination equals the
    oracle's bitwise, nothing else written."""
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    facts = {"direction": direction, "chunk": chunk, "copy": copy}
    spec = ctx.select_kernel(STATE_COPY, site="test", **facts)
    assert spec.provider == "kernels.state_copy"
    copier = spec.impl(ctx, **facts)
    rows = 2 * (8 if chunk > 1 << 20 else 64)                   # two segments of a pool
    g = torch.Generator().manual_seed(chunk)
    host = platform.host_memory(rows * chunk, torch.device("cuda"), mapped=True)
    host.view(rows, chunk).copy_(torch.randint(0, 256, (rows, chunk), dtype=torch.uint8, generator=g))
    mapped = platform.device_alias(host, torch.device("cuda")).view(rows, chunk)
    device = torch.randint(0, 256, (rows, chunk), dtype=torch.uint8, generator=g).cuda()
    src, dst = (mapped, device) if direction == "h2d" else (device, mapped)
    n = rows // 2
    src_rows = torch.cat([torch.randperm(rows, generator=g)[:n - 3], torch.arange(3)])
    dst_rows = torch.cat([torch.randperm(rows - 3, generator=g)[:n - 3] + 3, torch.arange(3)])   # a run of 3 rows
    before = dst.cpu()
    want = before.clone()
    reference.state_copy(src.cpu(), src_rows, want, dst_rows)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        copier(src, src_rows, dst, dst_rows)
        copier(src, src_rows[:0], dst, dst_rows[:0])
    stream.synchronize()
    assert torch.equal(dst.cpu(), want) and not torch.equal(want, before)
