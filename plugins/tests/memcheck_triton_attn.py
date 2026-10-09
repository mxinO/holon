"""No kernel of kernels.triton_attn reads or writes past its allocations with quantized KV (M0.3 L0, once per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m tests.memcheck_triton_attn

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so an access past the last
unit's q8_0 scales (the end of the cache's byte segments), the block table or q is outside any allocation. Every
head_dim, q8_0 and fp8 (Triton's bit decode), kv_write, prefill, decode with and without splits, and verify.
"""
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.triton_attn import plugin

from tests.test_attention_kernels import GEOMETRIES, MAX_SEQS, PAGE
from tests.test_attention_kv_quant import FP8, Q8, ROWS, quant_case, step_slots, writer


def main() -> None:
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    for heads, kv_heads, head_dim in GEOMETRIES:
        facts = {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "page": PAGE, "q_dtype": "bfloat16",
                 "layout": Q8, "kv_dtype": "q8_0", "max_seqs": MAX_SEQS}
        for layout, kernels in ((Q8, (plugin.Prefill, plugin.Decode, plugin.Verify)), (FP8, (plugin.Decode,
                                                                                               plugin.Verify))):
            for cls, op in zip(kernels, ("attention.paged_prefill", "attention.paged_decode",
                                         "attention.paged_verify")[-len(kernels):], strict=True):
                rows = ROWS[op]
                q, raw, cache, batch, pages = quant_case(rows, heads, kv_heads, head_dim, layout)
                unit, off = step_slots(rows, pages, batch)
                writer(layout, kv_heads, head_dim)(raw[unit, 0, :, off], raw[unit, 1, :, off], cache, unit, off)
                kernel = cls(ctx, **facts)
                handle = kernel.plan(batch, pages)
                if hasattr(kernel, "prepare"):
                    kernel.prepare(handle)
                kernel.run(q, cache, handle)
    torch.cuda.synchronize()
    print(f"memcheck run: q8_0 and fp8 KV, head_dim 64/128/256, on sm {ctx.fingerprint.sm}")


if __name__ == "__main__":
    main()
