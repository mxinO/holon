"""No kernel of kernels.triton_fp8 reads or writes past its allocations (docs/m06-lowp.md acceptance 12, once per
class):

    cd plugins
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m tests.memcheck_triton_fp8

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so an access past the
weight, its block scales, the activations, their e4m3 copy and scales, the split-K partials or the output is outside
any allocation. Every shape of the L0 (the 8B's with their pinned configurations, both decode row kinds with and
without splits), rows across the decode / prefill switch and partial prefill tiles; `linear_rows@1` at 2 to 16 rows
(partial row and column tiles).
"""
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.triton_fp8.plugin import LinearRows, QLinear

from tests.test_triton_fp8 import SHAPES, activations, weight


def main() -> None:
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    for out, k in SHAPES:
        parts = weight(out, k, out + k)
        lin = QLinear(ctx, layout="fp8_e4m3/hf", **{"in": k, "out": out}, act_dtype="bfloat16",
                      out_dtype="bfloat16", act_quant="a8")
        for t in (1, 9, 16, 17, 100):
            lin(activations(t, k, t), parts)
    rows = LinearRows(ctx)
    for out in (16, 96, 250, 256):
        w = torch.randn(out, 2560, device="cuda").bfloat16()
        for t in (2, 3, 4, 9, 16):
            rows(torch.randn(t, 2560, device="cuda").bfloat16(), w)
    torch.cuda.synchronize()
    print(f"memcheck run: decode and prefill, {len(SHAPES)} shapes, linear_rows, on sm {ctx.fingerprint.sm}")


if __name__ == "__main__":
    main()
