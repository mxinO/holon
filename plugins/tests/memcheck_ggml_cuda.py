"""No kernel of kernels.ggml_cuda reads past its allocations (M0.1 L0, run once per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m tests.memcheck_ggml_cuda

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so a read past the weight
blocks (or the activations, or the workspace) is outside any allocation. Every type, `in` 768 and 2304 (not
multiples of 512), mmvq at 1, 2 and the type's threshold, mmq above it and at 513 rows (bf16 out: the stream-k
fixup's staged tiles, docs/dev/m07-perf.md D4), and get_rows; the fused SwiGLU at one row and inputs prepared once for
mmvq (D6); the HF FP4 layouts through `prepare` (NVFP4's scale2 on every path) at `act_quant` a8 and a4 — native
W4A4 on sm_120, which reads the zeroed tail `prepare` appends and nothing past it.
"""
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.ggml_cuda.plugin import FP4, TYPES

from tests.test_ggml_cuda import activations, hf_fp4, prepared, qlinear, random_blocks


def run(ctx, name: str, k: int, parts: dict, **facts) -> None:
    """Every path of a `qlinear` (bf16 and f32 out), its fused SwiGLU and its prepared inputs."""
    for out_dtype in ("bfloat16", "float32"):
        kernel = qlinear(ctx, name, k, out_dtype, **facts)
        for rows in (1, 2, kernel.mmvq_rows, kernel.mmvq_rows + 1, 513):
            kernel(activations(rows, k, seed=rows), parts)
    for rows in (1, 2):
        kernel(kernel.prepare_input(activations(rows, k, seed=rows)), parts)
    glu = {r: p[:296] if r == "blocks" else p for r, p in parts.items()}          # halves of 148 rows
    qlinear(ctx, name, k, out=296, op="qlinear_glu@1", **facts)(activations(1, k, seed=1), glu)


def main() -> None:
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    for name in sorted(TYPES):
        for k in (768, 2304):
            w = torch.from_numpy(random_blocks(name, 300, k, seed=k)).cuda()
            run(ctx, name, k, {"blocks": w})
            if name == "nvfp4":                   # get_rows has no NVFP4
                continue
            facts = {"layout": f"ggml.{name}/gguf.native", "vocab": 300, "hidden": k, "dtype": "bfloat16"}
            ctx.select_kernel("qembedding@1", site="memcheck", **facts).impl(ctx, **facts)(
                torch.tensor([0, 299, 7], device="cuda"), w)
    for elem in FP4:
        for k in (768, 2304):
            for act_quant in ("a8", "a4"):
                w = prepared(qlinear(ctx, f"{elem}/hf", k, act_quant=act_quant), elem, hf_fp4(elem, 300, k, seed=k))
                run(ctx, f"{elem}/hf", k, w, act_quant=act_quant)
    torch.cuda.synchronize()
    print(f"memcheck run: {len(TYPES)} types x in 768, 2304, HF FP4 at a8 and a4 on sm {ctx.fingerprint.sm}")


if __name__ == "__main__":
    main()
