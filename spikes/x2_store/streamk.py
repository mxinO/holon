"""Is the unpatched expert-id mmq placement-invariant at the vehicle's shapes on this GPU? Q4_K gate/up (2048 →
2×768, SwiGLU) and Q6_K down (768 → 2048) stacks of 128 random experts (realistic scales), placed at random slots of
arenas of 128–6,280 slots: per token count, whether a stream-k fixup kernel ran and whether the experts' rows equal
the 128-slot stack's, bit for bit.

    CUDA_VISIBLE_DEVICES=0 python spikes/x2_store/streamk.py
"""
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover

SHAPES = (("q4_k", 2048, 1536, "swiglu", 1152, 144, (0, 4), (0, 0x20, 0, 0x10)),     # d, dmin halves
          ("q6_k", 768, 2048, "none", 630, 210, (208, 210), (0, 0x20)))               # d half
ARENAS, TOKENS = (128, 136, 904, 1672, 6280), (9, 64, 300, 512)


def main() -> None:
    dev = torch.device("cuda")
    ctx = BuildContext(PluginSet(discover()), dev)
    for name, k, out, glu, row, block, (a, b), scale in SHAPES:
        g = torch.Generator().manual_seed(0)
        experts = torch.randint(0, 256, (128, out, row), dtype=torch.uint8, generator=g)
        experts.view(128, out, -1, block)[..., a:b] = torch.tensor(scale, dtype=torch.uint8)
        for tokens in TOKENS:
            h = torch.Generator().manual_seed(1)
            ids = torch.stack([torch.randperm(128, generator=h)[:8] for _ in range(tokens)])
            swiglu = glu == "swiglu"
            x = torch.randn(tokens, 1 if swiglu else 8, k, generator=h)
            x = x.to(dev, torch.bfloat16 if swiglu else torch.float32)
            got = {}
            for slots in ARENAS:
                facts = dict(layout=f"ggml.{name}/gguf.native", experts=slots, out=out, top_k=8, glu=glu,
                             rows_per_token=1 if swiglu else 8, act_dtype=str(x.dtype).removeprefix("torch."),
                             out_dtype="float32", act_quant="a8", **{"in": k})
                kernel = ctx.select_kernel("linear_experts@2", site=f"s{slots}", **facts).impl(ctx, **facts)
                where = torch.randperm(slots - 8, generator=g)[:128] + 8 if slots > 128 else torch.arange(128)
                arena = torch.zeros(slots, out, row, dtype=torch.uint8)
                arena[where] = experts
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    y = kernel(x, {"blocks": arena.to(dev)}, where[ids].int().to(dev))
                    torch.cuda.synchronize()
                got[slots] = any("fixup" in e.name for e in prof.events()), y.cpu()
            print(name, tokens, {s: {"stream_k": f, "equal": torch.equal(y, got[128][1])} for s, (f, y) in got.items()},
                  flush=True)


if __name__ == "__main__":
    main()
