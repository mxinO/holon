"""No kernel of kernels.ggml_cuda reads past its allocations (M0.1 L0, run once per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m tests.memcheck_ggml_cuda

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so a read past the weight
blocks (or the activations, or the workspace) is outside any allocation. Every type, `in` 768 and 2304 (not
multiples of 512), mmvq at 1, 2 and the type's threshold, mmq above it and at 513 rows, and get_rows.
"""
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.ggml_cuda.plugin import TYPES

from tests.test_ggml_cuda import activations, qlinear, random_blocks


def main() -> None:
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    for name in sorted(TYPES):
        for k in (768, 2304):
            kernel = qlinear(ctx, name, k)
            w = torch.from_numpy(random_blocks(name, 300, k, seed=k)).cuda()
            for rows in (1, 2, kernel.mmvq_rows, kernel.mmvq_rows + 1, 513):
                kernel(activations(rows, k, seed=rows), w)
            facts = {"layout": f"ggml.{name}/gguf.native", "vocab": 300, "hidden": k, "dtype": "bfloat16"}
            ctx.select_kernel("qembedding@1", site="memcheck", **facts).impl(ctx, **facts)(
                torch.tensor([0, 299, 7], device="cuda"), w)
    torch.cuda.synchronize()
    print(f"memcheck run: {len(TYPES)} types x in 768, 2304 on sm {ctx.fingerprint.sm}")


if __name__ == "__main__":
    main()
