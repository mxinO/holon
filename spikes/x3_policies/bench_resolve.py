"""What a resolve variant costs per layer call (docs/v1-offload.md §13's F11 pick-and-decay term): `moe_resolve@1`
against `moe_resolve.lfu@1` and `moe_resolve.lru@1` (kernels.expert_io) at E = 128, k = 8, two roles, all decode
rows, 48 calls per CUDA graph (a step's layers), replayed 20 times. Every call starts from the same map and state
(four small copies before it in every arm: the map, the counts, the state, padding), so each variant call has its
candidates and victims to pair; the copies' cost is the `copies` row:

    CUDA_VISIBLE_DEVICES=<gpu> python spikes/x3_policies/bench_resolve.py
"""
import sys

import torch
from holon.core.plugins import PluginSet, discover
from holon_std.moe.kernels import DECODE

sys.path.insert(0, "plugins/tests")
from test_expert_io import ROLES, E, K, _state, kernel, routing  # noqa: E402

LAYERS, REPLAYS = 48, 20


def per_call_us(call) -> float:
    """`call()`'s device time, µs per call, replayed from a graph of LAYERS calls."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAYERS):
            call()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPLAYS / LAYERS * 1000


def arm(plugins, op: str, tokens: int):
    """One resolve call of `op` on a fresh map and state, with its restoring copies."""
    resolve = kernel(plugins, "cuda", op, "kernels.expert_io", experts=E, top_k=K, roles=ROLES, staging=E)
    case = routing(tokens, 1, whole=True)
    case["kinds"][:] = DECODE
    state = {} if op == "moe_resolve@1" else _state(op.split(".")[1].split("@")[0], 1)
    dev = {k: v.cuda() for k, v in (case | state).items()}
    words = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(3)]
    routed = torch.zeros(E, dtype=torch.int32, device="cuda")
    delta = torch.zeros(10000, 3, dtype=torch.int32, device="cuda")
    kw = {n: dev[n] for n in state}
    restore = [dev["residency"], dev["counts"], *kw.values()]
    restore += [torch.zeros(1, device="cuda") for _ in range(4 - len(restore))]
    saved = [t.clone() for t in restore]

    def call():
        for t, s in zip(restore, saved, strict=True):     # the call's starting map and state
            t.copy_(s)
        words[2].zero_()
        resolve(dev["ids"], dev["kinds"], dev["residency"], dev["host"], dev["limits"], dev["counts"], routed,
                words[0], words[1], delta, words[2], 0, **kw)

    def copies():
        for t, s in zip(restore, saved, strict=True):
            t.copy_(s)
        words[2].zero_()
    return call, copies


def main() -> None:
    plugins = PluginSet(discover())
    for tokens in (1, 8, 64):                        # decode rows per step
        for op in ("moe_resolve@1", "moe_resolve.lfu@1", "moe_resolve.lru@1"):
            call, copies = arm(plugins, op, tokens)
            print(tokens, op, f"{per_call_us(call):.1f} us per call", flush=True)
        print(tokens, "copies", f"{per_call_us(copies):.1f} us per call", flush=True)


if __name__ == "__main__":
    main()
