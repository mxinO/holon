"""No MoE kernel of kernels.ggml_cuda reads or writes past its allocations (M0.2 L0, run once per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
        python -m tests.memcheck_ggml_cuda_moe

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so an access past an expert
stack (expert E−1's last row), the ids, the activations, the outputs or the workspace is outside any allocation.
Every type at E = 16, k = 8, with experts 0 and E−1 in every row: the merged gate/up stack (`in` 768 and 2304, glu
none and swiglu) and down (one input per slot) on mmvq (1, 2 and the limit) and mmq (the limit + 1, 513 tokens:
a 512-token mmq chunk and a 1-token mmvq one); `moe_topk` at E 16, 128 and 512; `moe_combine`.
"""
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.ggml_cuda.plugin import TYPES

from tests.test_ggml_cuda import random_blocks

E, K, INTER = 16, 8, 256


def main() -> None:
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    gen = torch.Generator().manual_seed(0)

    def select(op: str, **facts):
        return ctx.select_kernel(op, site="memcheck", **facts).impl(ctx, **facts)

    def ids(tokens: int) -> torch.Tensor:
        rest = 1 + torch.rand(tokens, E - 2, generator=gen).argsort(-1)[:, :K - 2]
        return torch.cat([torch.tensor([[0, E - 1]] * tokens), rest], 1).int().cuda()

    def experts(name: str, k: int, out: int, rows_per_token: int, glu: str = "none"):
        return select("linear_experts@1", layout=f"ggml.{name}/gguf.native", experts=E, top_k=K, glu=glu,
                      rows_per_token=rows_per_token, act_dtype="bfloat16", out_dtype="float32", act_quant="a8",
                      out=out, **{"in": k})

    for name in sorted(TYPES):
        for hidden in (768, 2304):
            gate_up = [experts(name, hidden, 2 * INTER, 1, glu) for glu in ("none", "swiglu")]
            down = experts(name, INTER, hidden, K)
            gu = random_blocks(name, E * 2 * INTER, hidden, seed=hidden)
            dn = random_blocks(name, E * hidden, INTER, seed=hidden + 1)
            w_gu, w_dn = (torch.from_numpy(b).cuda().unflatten(0, (E, -1)) for b in (gu, dn))     # per expert
            t = down.mmvq_tokens
            for tokens in (1, 2, t, t + 1, 513):
                route = ids(tokens)
                x = (1e-3 * torch.randn(tokens, 1, hidden, generator=gen)).bfloat16().cuda()
                for kern in gate_up:
                    kern(x, {"blocks": w_gu}, route)
                down(torch.randn(tokens, K, INTER, generator=gen).cuda(), {"blocks": w_dn}, route)
    for n in (16, 128, 512):
        select("moe_topk@1", experts=n, top_k=K, renormalize=True)(torch.randn(13, n, generator=gen).cuda())
    out = torch.empty(13, 2048, dtype=torch.bfloat16, device="cuda")
    select("moe_combine@1", hidden=2048, top_k=K, dtype="bfloat16", y_dtype="float32")(
        torch.randn(13, K, 2048, generator=gen).cuda(), torch.rand(13, K, generator=gen).cuda(), out)
    torch.cuda.synchronize()
    print(f"memcheck run: {len(TYPES)} types x in 768, 2304, E {E}, k {K}, topk, combine on sm {ctx.fingerprint.sm}")


if __name__ == "__main__":
    main()
