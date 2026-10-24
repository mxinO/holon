"""No Gated DeltaNet kernel (kernels.fla, kernels.gdn_triton) reads or writes past its allocations (M0.3 L0, run once
per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
        python -m tests.memcheck_gdn_kernels

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so an access past a state
pool (a row on the last unit), the inputs, the chunk indices or the outputs is outside any allocation. Qwen3.5's
shape (Hv 32) and the small one (head dim 64) in both v-head orders: the mixed batch of the L0 (lengths 1 to 1000,
a zero-token row, slot 0 and the last slot) on the chunked path and a decode batch on the recurrent path, each with
has_init mixed and then all set (so the last unit is read too); the convolution on the same batches; the gated norm
with a strided z; the recurrent path on column views of wider rows (q, k, v, a and b read in place).
"""
import torch

from tests import test_gdn_kernels as t


def main() -> None:
    torch.manual_seed(0)
    for shape in ("qwen3.5", "small"):
        hk, hv, d = t.SHAPES[shape]
        c, g = t.channels(hk, hv, d), t.gates(hv)
        batches = [(t.LENS, t.INIT, t.SLOTS), ([1] * len(t.DECODE_SLOTS), t.DECODE_INIT, t.DECODE_SLOTS)]
        for lens, init, slots in batches + [(lens, [True] * len(lens), slots) for lens, _, slots in batches]:
            for order in ("grouped", "tiled"):
                pool = torch.zeros(t.UNITS, hv, d, d, device="cuda")
                t.run_batch(t.gdn(shape, order), t.tokens(sum(lens), hk, hv, d), g, pool, lens, init, slots)
            x = torch.randn(sum(lens), c, device="cuda").bfloat16()
            w = torch.randn(c, t.KERNEL, device="cuda").bfloat16()
            state = torch.zeros(t.UNITS, c, t.KERNEL - 1, device="cuda").bfloat16()
            t.conv(c)(x, w, state, t.dev(slots), t.query_start(lens), t.dev(init, torch.bool))
        norm = t.select("gated_rmsnorm@1", dim=d, eps=t.EPS, dtype="bfloat16")
        zw = torch.randn(37, hv + 1, d, device="cuda").bfloat16()
        norm(torch.randn(37, hv, d, device="cuda").bfloat16(), zw[:, 1:], torch.ones(d, device="cuda").bfloat16())
        x = t.tokens(len(t.DECODE_SLOTS), hk, hv, d)                 # the recurrent path's in-place reads
        wide = {n: torch.cat([v.reshape(len(v), -1)] * 2, 1) for n, v in x.items()}
        views = {n: wide[n][:, :x[n][0].numel()].view(x[n].shape) for n in x}
        pool = torch.zeros(t.UNITS, hv, d, d, device="cuda")
        t.run_batch(t.gdn(shape, "tiled"), views, g, pool, [1] * len(t.DECODE_SLOTS), t.DECODE_INIT, t.DECODE_SLOTS)
    torch.cuda.synchronize()
    print("memcheck run: gated_delta_rule (contiguous and column views), causal_conv1d, gated_rmsnorm on qwen3.5 and "
          "small")


if __name__ == "__main__":
    main()
