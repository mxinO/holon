"""No MoE kernel of kernels.ggml_cuda reads or writes past its allocations (M0.2 L0, run once per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
        python -m tests.memcheck_ggml_cuda_moe

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so an access past an expert
stack (expert E−1's last row), the ids, the activations, the outputs or the workspace is outside any allocation.
Every type at E = 16, k = 8, with experts 0 and E−1 in every row: the merged gate/up stack (`in` 768 and 2304, glu
none and swiglu) and down (one input per slot) on mmvq (1, 2 and the limit) and mmq (the limit + 1, 513 tokens:
a 512-token mmq chunk and a 1-token mmvq one; where `pass_rows` is above a chunk, docs/dev/v1-moe-prefill.md D0, also
2,048 and 4,608: the compact and device-wide id helpers, a 4,096-row call then a chunk's), each call also with
passed buffers of exactly their sizes (`out`, and `scratch` of `scratch_bytes`: docs/dev/v1-moe-prefill.md D10); the
FP4 types also at `act_quant` a4 through `prepare` (native W4A4 on sm_120 reads the zeroed tail after expert E−1),
NVFP4 with per-expert scale2 ([2, E] gate/up, [E] down); `moe_topk` at E 16, 128 and 512; `moe_combine`.
"""
import torch
from holon.api import Layout, QTensor, Scheme
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.ggml_cuda.plugin import FP4, TYPES

from tests.test_ggml_cuda import hf_fp4, random_blocks

E, K, INTER = 16, 8, 256


def main() -> None:
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    gen = torch.Generator().manual_seed(0)

    def select(op: str, **facts):
        return ctx.select_kernel(op, site="memcheck", **facts).impl(ctx, **facts)

    def ids(tokens: int) -> torch.Tensor:
        rest = 1 + torch.rand(tokens, E - 2, generator=gen).argsort(-1)[:, :K - 2]
        return torch.cat([torch.tensor([[0, E - 1]] * tokens), rest], 1).int().cuda()

    def experts(layout: str, k: int, out: int, rows_per_token: int, glu: str = "none", act_quant: str = "a8"):
        return select("linear_experts@2", layout=layout, experts=E, top_k=K, glu=glu, rows_per_token=rows_per_token,
                      act_dtype="bfloat16", out_dtype="float32", act_quant=act_quant, out=out, **{"in": k})

    def stack(kernel, layout: str, parts: dict, out: int, k: int) -> dict:
        """The parts as the kernel reads them (`prepare`: an HF repack, the native spec's tail)."""
        elem, name = layout.split("/")
        q = QTensor(Layout(Scheme(elem), name), (E, out, k), {r: p.cuda() for r, p in parts.items()})
        return kernel.prepare(q).parts if hasattr(kernel, "prepare") else q.parts

    cases = [(f"ggml.{name}/gguf.native", "a8") for name in sorted(TYPES)] + \
        [(f"ggml.{name}/gguf.native", "a4") for name in FP4] + [("nvfp4/hf", "a8"), ("nvfp4/hf", "a4")]
    for layout, act_quant in cases:
        name = layout.split("/")[0].removeprefix("ggml.")
        for hidden in (768, 2304):
            gate_up = [experts(layout, hidden, 2 * INTER, 1, glu, act_quant) for glu in ("none", "swiglu")]
            down = experts(layout, INTER, hidden, K, act_quant=act_quant)
            if layout.endswith("/hf"):
                gu, dn = (hf_fp4(name, out, k, seed=hidden + i, experts=E) for i, (out, k) in
                          enumerate(((2 * INTER, hidden), (hidden, INTER))))
                gu["scale2"] = torch.stack([gu["scale2"], gu["scale2"].flip(0)])          # [2, E]
            else:
                gu, dn = ({"blocks": torch.from_numpy(random_blocks(name, E * out, k, seed=hidden + i)).unflatten(
                    0, (E, -1))} for i, (out, k) in enumerate(((2 * INTER, hidden), (hidden, INTER))))
            w_gu = [stack(kern, layout, gu if kern.swiglu else {r: p for r, p in gu.items() if r != "scale2"},
                          2 * INTER, hidden) for kern in gate_up]
            w_dn = stack(down, layout, dn, hidden, INTER)
            t = down.mmvq_tokens
            for tokens in (1, 2, t, t + 1, 513) + ((2048, 4608) if down.pass_rows > 512 else ()):
                route = ids(tokens)
                x = (1e-3 * torch.randn(tokens, 1, hidden, generator=gen)).bfloat16().cuda()
                h = torch.randn(tokens, K, INTER, generator=gen).cuda()
                for kern, w, xs in (*((k, w, x) for k, w in zip(gate_up, w_gu, strict=True)), (down, w_dn, h)):
                    y = kern(xs, w, route)
                    # D10's passed buffers, each its own allocation of exactly its size
                    kern(xs, w, route, out=torch.empty_like(y),
                         scratch=torch.empty(kern.scratch_bytes(tokens), dtype=torch.uint8, device="cuda"))
    # T6: `in` 640 (FN's down projections), weights in buffers of exactly their size: mmq's last K step stays inside
    for name in ("q2_0", "iq4_nl", "q4_0", "q8_0"):
        layout = f"ggml.{name}/gguf.native"
        down = experts(layout, 640, 1024, K)
        w = {"blocks": torch.from_numpy(random_blocks(name, E * 1024, 640, seed=7)).unflatten(0, (E, -1)).cuda()}
        dense = select("qlinear@2", layout=layout, out=1024, act_dtype="bfloat16", out_dtype="bfloat16",
                       act_quant="a8", **{"in": 640})
        wd = {"blocks": torch.from_numpy(random_blocks(name, 1024, 640, seed=8)).cuda()}
        for tokens in (1, down.mmvq_tokens + 1, 513):
            down(torch.randn(tokens, K, 640, generator=gen).cuda(), w, ids(tokens))
            dense((1e-3 * torch.randn(tokens, 640, generator=gen)).bfloat16().cuda(), wd)
    for n in (16, 128, 512):
        select("moe_topk@1", experts=n, top_k=K, renormalize=True)(torch.randn(13, n, generator=gen).cuda())
    out = torch.empty(13, 2048, dtype=torch.bfloat16, device="cuda")
    select("moe_combine@1", hidden=2048, top_k=K, dtype="bfloat16", y_dtype="float32")(
        torch.randn(13, K, 2048, generator=gen).cuda(), torch.rand(13, K, generator=gen).cuda(), out)
    torch.cuda.synchronize()
    print(f"memcheck run: {len(cases)} layouts and modes x in 768, 2304, E {E}, k {K}; in 640 (T6), topk, combine on "
          f"sm {ctx.fingerprint.sm}")


if __name__ == "__main__":
    main()
