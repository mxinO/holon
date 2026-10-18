"""moe_prefill's per-call GEMM bench: one layer's expert GEMMs of a GGUF on one GPU, through kernels.ggml_cuda's own
module, against two bounds of the same work, CUDA-event timed (median of `--iters` after warm-up).

- `mmq_id`: the merged gate/up stack (Q4_K here, [E, 2·inter, ·]) with bf16 activations [T, 1, hidden], and the down
  stack with f32 activations [T, k, inter], as `LinearExperts` calls them (slots = E + k), at T tokens per call.
- `dense_mmq`: the same MACs on one expert's matrix — mmq without ids at T·k rows: the GEMM a perfect grouping would
  run (no empty tiles, every tile full), so `mmq_id / dense_mmq` is the grouping's overhead.
- `bf16_mm`: cuBLAS bf16 (f32 accumulation) at T·k rows on one dequantized expert matrix — a dequant + tensor-core
  path's GEMM alone; `dequant` the whole stack's dequantization to bf16 (that path's per-pass cost).
- `--freetoken` (run in FreeToken's venv): its `mul_mat_id` instead, on the same stacks and routing (bf16 in and out,
  as its MoE calls it: its down input is bf16) — the per-call time of its grouped launch at equal work; with `--f32`
  its activations are f32, so it stores f32 as Holon's call does (equal I/O: docs/v1-moe-prefill.md row 2's split).
- Past `ID_PAIRS` (T·k > 32,768: 8,192 tokens at top-8) Holon's call is not run (its id helper's bound); the bounds
  and the dequant still are (D7's break-even at 8,192-token passes).
- `--j-check`: whether mmq's column-tile width J changes a row's bits — the first n tokens' rows of one call of T
  tokens (J = 128) against a call of those n tokens alone (J chosen against n: 16 → 16, 32 → 32, 64 → 64), both stacks.
- Routing: per token k distinct experts drawn without replacement with popularity ∝ rank^-s (s = 0.5: the top 10% of
  experts take ≈ 27% of routes, M6's measured range 23–36%), a fixed seed.

    python spikes/moe_prefill/gemm.py --model <gguf> [--layer 0] [--tokens 512 1024 2048 4096] [--freetoken | --j-check]
        [--json out]
"""
from __future__ import annotations

import argparse
import functools
import json
import statistics

import gguf
import torch


def timed(fn, iters: int) -> float:
    """Median ms per call."""
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def routing(tokens: int, experts: int, k: int, s: float, gen: torch.Generator) -> torch.Tensor:
    logp = -s * torch.log(torch.arange(1, experts + 1, dtype=torch.float64))
    logp = logp[torch.randperm(experts, generator=gen)]
    gumbel = -torch.log(-torch.log(torch.rand(tokens, experts, generator=gen, dtype=torch.float64)))
    return (logp + gumbel).topk(k, dim=1).indices.to(torch.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--skew", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--freetoken", action="store_true")
    ap.add_argument("--f32", action="store_true", help="--freetoken with f32 activations (and so f32 out)")
    ap.add_argument("--j-check", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tensors = {t.name: t for t in gguf.GGUFReader(a.model).tensors}
    blk = f"blk.{a.layer}."
    gate, up, down = (tensors[blk + f"ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
    e, inter, _ = gate.data.shape
    hidden = down.data.shape[1]
    gate_up = torch.cat([torch.from_numpy(gate.data.copy()), torch.from_numpy(up.data.copy())], 1).to(dev)
    stacks = {"gate_up": (int(gate.tensor_type), gate_up, hidden, 2 * inter, 1),     # ggml's type ids
              "down": (int(down.tensor_type), torch.from_numpy(down.data.copy()).to(dev), inter, hidden, a.top_k)}
    gen = torch.Generator().manual_seed(0)
    rows = (freetoken if a.freetoken else j_check if a.j_check else holon)(a, stacks, e, gen, dev)
    out = {"gpu": torch.cuda.get_device_name(), "layer": a.layer, "experts": e, "top_k": a.top_k, "inter": inter,
           "hidden": hidden, "skew": a.skew, "engine": "freetoken" if a.freetoken else "holon",
           "activations": "f32" if a.f32 else "as the engine calls it", "rows": rows}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


def freetoken(a, stacks: dict, e: int, gen: torch.Generator, dev) -> list[dict]:
    from freetoken.kernel import ggml
    rows = []
    for name, (typ, blocks, n_in, n_out, r) in stacks.items():
        padded = torch.zeros(blocks.numel() + 4096, dtype=torch.uint8, device=dev)     # mmq's zero tail past the
        padded[:blocks.numel()] = blocks.view(-1)                                     # last row (FreeToken checks)
        blocks = padded[:blocks.numel()].view(blocks.shape)
        for t in a.tokens:
            ids = routing(t, e, a.top_k, a.skew, gen).to(dev)
            x = torch.randn(t, r, n_in, device=dev, dtype=torch.float32 if a.f32 else torch.bfloat16)
            ms = timed(functools.partial(ggml.mul_mat_id, blocks, typ, x if r > 1 else x[:, 0], ids), a.iters)
            gmac = t * a.top_k * n_in * n_out / 1e9
            rows.append({"stack": name, "tokens": t, "mul_mat_id_ms": round(ms, 4), "tops": round(2 * gmac / ms, 1)})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def j_check(a, stacks: dict, e: int, gen: torch.Generator, dev) -> list[dict]:
    from holon_plugins.kernels.ggml_cuda import extension
    from holon_plugins.kernels.ggml_cuda import plugin as ggml
    lib = extension.load(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])
    k, slots, prec = a.top_k, e + a.top_k, ggml.PREC["q8_1"]
    errors = torch.zeros(1, dtype=torch.int32, device=dev)
    rows = []
    for name, (typ, blocks, n_in, n_out, r) in stacks.items():
        flat = blocks.view(-1, blocks.shape[-1])

        def call(x, ids, typ=typ, flat=flat, n_in=n_in, n_out=n_out):
            y = torch.empty(len(ids), k, n_out, device=dev)
            ws = torch.empty(lib.workspace_bytes_id(0, n_in, len(ids), k, slots), dtype=torch.uint8, device=dev)
            lib.mmq_id(typ, prec, flat, x, ids, y, ws, slots, errors)
            return y
        for t in a.tokens:
            ids = routing(t, e, k, a.skew, gen).to(dev)
            x = torch.randn(t, r, n_in, device=dev, dtype=torch.bfloat16 if r == 1 else torch.float32)
            whole = call(x, ids)
            for n in (16, 32, 64):
                part = call(x[:n].contiguous(), ids[:n].contiguous())
                rows.append({"stack": name, "tokens": t, "rows": n, "bitwise": bool(torch.equal(part, whole[:n])),
                             "max_abs_diff": float((part - whole[:n]).abs().max())})
                print(json.dumps(rows[-1]), flush=True)
    assert int(errors.item()) == 0, int(errors.item())
    return rows


def holon(a, stacks: dict, e: int, gen: torch.Generator, dev) -> list[dict]:
    from holon_plugins.kernels.ggml_cuda import extension
    from holon_plugins.kernels.ggml_cuda import plugin as ggml
    lib = extension.load(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])
    k, slots, prec = a.top_k, e + a.top_k, ggml.PREC["q8_1"]
    errors = torch.zeros(1, dtype=torch.int32, device=dev)
    rows = []
    for name, (typ, blocks, n_in, n_out, r) in stacks.items():
        flat = blocks.view(-1, blocks.shape[-1])
        w_bf16 = torch.empty(e * n_out, n_in, dtype=torch.bfloat16, device=dev)
        deq = timed(functools.partial(lib.dequantize, typ, flat, w_bf16), a.iters)
        one, one_bf16 = blocks[0], w_bf16[:n_out].t()
        for t in a.tokens:
            ids = routing(t, e, k, a.skew, gen).to(dev)
            x = torch.randn(t, r, n_in, device=dev, dtype=torch.bfloat16 if r == 1 else torch.float32)
            mmq_id = None
            if t * k <= ggml.ID_PAIRS:
                y = torch.empty(t, k, n_out, device=dev)
                ws = torch.empty(lib.workspace_bytes_id(0, n_in, t, k, slots), dtype=torch.uint8, device=dev)
                mmq_id = timed(functools.partial(lib.mmq_id, typ, prec, flat, x, ids, y, ws, slots, errors), a.iters)
                del y, ws
            xd = torch.randn(t * k, n_in, device=dev, dtype=torch.bfloat16)
            yd = torch.empty(t * k, n_out, device=dev)
            wsd = torch.empty(lib.workspace_bytes(0, n_in, t * k), dtype=torch.uint8, device=dev)
            dense = timed(functools.partial(lib.mmq, typ, prec, one, xd, yd, None, wsd), a.iters)
            bf16 = timed(functools.partial(torch.mm, xd, one_bf16), a.iters)
            gmac = t * k * n_in * n_out / 1e9
            per = torch.bincount(ids.view(-1).long().cpu(), minlength=e)
            tiles = int(((per + 127) // 128).sum())         # J = 128 column tiles holding tokens, of ceil(T/128)·slots
            row = {"stack": name, "type": typ, "tokens": t, "mmq_id_ms": mmq_id and round(mmq_id, 4),
                   "dense_mmq_ms": round(dense, 4), "bf16_mm_ms": round(bf16, 4), "dequant_stack_ms": round(deq, 4),
                   "mmq_id_tops": mmq_id and round(2 * gmac / mmq_id, 1), "dense_mmq_tops": round(2 * gmac / dense, 1),
                   "bf16_mm_tflops": round(2 * gmac / bf16, 1),
                   "grouping_overhead": mmq_id and round(mmq_id / dense, 2),
                   "tiles_used": tiles, "tiles_launched": -(-t // 128) * min(slots, t * k),
                   "column_fill": round(t * k / (tiles * 128), 3)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        assert int(errors.item()) == 0, int(errors.item())
    return rows


if __name__ == "__main__":
    main()
