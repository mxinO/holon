from holon.api import Constraint, KernelSpec

from .kernels import BY_OP, PagedAttention

KV_LAYOUTS = ("kv.hnd", "kv.hnd.fp8", "kv.hnd.q8_0")    # every KV layout the paged reference reads
# Weight layouts the references read, by scheme: ggml blocks as stored (dequantized by gguf-py), HF quantized weights
# (std.weights.hf's reference), and dense stacks.
HF = {"fp8_e4m3", "nvfp4", "mxfp4"}
WEIGHT_SCHEMES = {"qlinear": {"ggml", *HF}, "qembedding": {"ggml"}, "linear_experts": {"ggml", "dense", *HF}}


def _reads(schemes: set[str]):
    return lambda facts: facts["layout"].split("/")[0].split(".")[0] in schemes


class Plugin:
    def register(self, regs) -> None:
        for op, impl in BY_OP.items():
            paged = impl is PagedAttention
            regs.kernels.add(op, KernelSpec(
                op=op, provider="testing.ref_kernels", impl=impl, requires=Constraint(devices=frozenset({"cpu"})),
                supports=(lambda f: f["layout"] in KV_LAYOUTS) if paged else
                _reads(WEIGHT_SCHEMES[op]) if op in WEIGHT_SCHEMES else None,
                props=frozenset({"deterministic"} if paged else {"deterministic", "graph_safe"})))
