"""What a resolve variant costs per layer call (docs/v1-offload.md §13's F11 pick-and-decay term): `moe_resolve@1`
against `moe_resolve.lfu@1` and `moe_resolve.lru@1` (kernels.expert_io) at E = 128, k = 8, two roles, all decode
rows, 48 calls per CUDA graph (a step's layers), replayed 20 times:

    CUDA_VISIBLE_DEVICES=<gpu> python spikes/x3_policies/bench_resolve.py
"""
import sys

import torch
from holon.core.plugins import PluginSet, discover
from holon_std.moe.kernels import DECODE

sys.path.insert(0, "plugins/tests")
from test_expert_io import ROLES, E, K, _state, kernel, routing  # noqa: E402

LAYERS, REPLAYS = 48, 20


def main() -> None:
    plugins = PluginSet(discover())
    for tokens in (1, 8, 64):                        # decode rows per step
        for op in ("moe_resolve@1", "moe_resolve.lfu@1", "moe_resolve.lru@1"):
            resolve = kernel(plugins, "cuda", op, "kernels.expert_io", experts=E, top_k=K, roles=ROLES, staging=E)
            case = routing(tokens, 1, whole=True)
            case["kinds"][:] = DECODE
            state = {} if op == "moe_resolve@1" else _state(op.split(".")[1].split("@")[0], 1)
            dev = {k: v.cuda() for k, v in (case | state).items()}
            words = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(3)]
            delta = torch.zeros(10000, 3, dtype=torch.int32, device="cuda")
            args = (dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"], dev["counts"], words[0],
                    words[1], delta, words[2], 0)
            kw = {n: dev[n] for n in state}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    resolve(*args, **kw)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(LAYERS):
                    words[2].zero_()
                    resolve(*args, **kw)
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPLAYS):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            print(tokens, op, f"{start.elapsed_time(end) / REPLAYS / LAYERS * 1000:.1f} us per call", flush=True)


if __name__ == "__main__":
    main()
