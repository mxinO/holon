"""M0.7 step 0 (docs/m07-perf.md D1, D2, D8, D9): the ggml quantized matmul's time per call against its row count,
graph-timed, for every (weight type, shape, activation mode) of the `qlinear` sites in the given locks — mmvq (rows
1-8), mmq with today's f32 -> bf16 cast, mmq writing f32 (no cast: what D4 would cost at most), and llama.cpp's
choice — and a read-only stream over 2 GiB (D8's bandwidth reference).

Each variant is today's `QLinear` with its dispatch forced (`mmvq_rows`), so a call costs what the engine's does
(the activation's q8_1 quantization, native FP4's f32 activation copy, NVFP4's `scale2`). Weights are the rows of
the lock's GGUF file's tensors of that type and input size (a call on random bytes runs up to 5% slower); HF FP4
layouts get random codes with valid scales. They come in enough copies that the calls cycle through more than three
times the L2 cache, as a step reads its weights from DRAM.

    python bench/qmm_sweep.py --locks plans/.../N.lock ... [--rows 1 2 ... 9] [--json out]
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import os
import tomllib

import gguf
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.ggml_cuda import plugin as ggml

MODES = {"q8_1": ggml.Q8_1, "fp4": ggml.NATIVE_FP4}


HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def sites(locks: list[str]) -> dict[tuple, dict]:
    """(layout, in, out, mode) -> the locks naming it (and whether each runs that mode) and their GGUF files, from the
    locks' ggml qlinear sites."""
    out: dict[tuple, dict] = {}
    for path in locks:
        with open(path, "rb") as f:
            lock = tomllib.load(f)
        for key, site in lock.get("sites", {}).items():
            if not key.endswith("/qlinear@2") or site["impl"] != "kernels.ggml_cuda":
                continue
            facts, mode = site["config"]["facts"], site["config"]["launch"]["act"]["mmq"]
            name = ggml.weight_type(facts["layout"])
            for m in ("q8_1", "fp4") if name in ggml.FP4 else ("q8_1",):
                entry = out.setdefault((facts["layout"], facts["in"], facts["out"], m),
                                       {"locks": {}, "facts": facts, "files": []})
                entry["locks"][lock["model_id"] + "/" + lock["fingerprint_class"].split("/")[0]] = m == mode
                entry["files"] += gguf_file(lock["model_id"])
    return out


def gguf_file(model_id: str) -> list[str]:
    """The cached GGUF file of a lock's model id ("<org>/<repo>/<file stem>@<digest>"), if any."""
    parts = model_id.split("@")[0].split("/")
    if len(parts) != 3:
        return []
    return glob.glob(f"{HUB}/models--{parts[0]}--{parts[1]}/snapshots/*/{parts[2]}.gguf")[:1]


@functools.cache
def tensors(path: str) -> list:
    return list(gguf.GGUFReader(path).tensors)


def weights(files: list[str], name: str, k: int, n: int, copies: int, tail: int) -> tuple[list[torch.Tensor], str]:
    """`copies` weights [n, row bytes] (each followed by `tail` zeroed bytes): the rows of the first file's 2-D
    tensors of this type and input size, in file order (wrapping), else random bytes with valid FP4 scales."""
    rb = row_bytes(name, k)
    for path in files:
        found = [t.data.reshape(int(t.shape[1]), rb) for t in tensors(path) if len(t.shape) == 2
                 and t.tensor_type.name.lower() == name and int(t.shape[0]) == k]
        if not found:
            continue
        out, at = [], 0
        source = [(a, i) for a in found for i in range(0, len(a), 4096)]       # 4096-row chunks, in order
        for _ in range(copies):
            w = torch.empty(n, rb, dtype=torch.uint8, device="cuda")
            filled = 0
            while filled < n:
                a, i = source[at % len(source)]
                chunk = a[i:i + min(4096, n - filled)]
                w[filled:filled + len(chunk)] = torch.from_numpy(chunk.copy()).cuda()
                filled, at = filled + len(chunk), at + 1
            out.append(ggml.with_tail(w, tail))
        return out, os.path.basename(path)
    out = []
    for _ in range(copies):
        w = torch.randint(0, 256, (n, rb), dtype=torch.uint8, device="cuda")
        if name in ggml.BLOCKS:                     # FP4 block scales: e4m3 x4 (NVFP4) or e8m0 (MXFP4), not NaN
            group, block = ggml.BLOCKS[name]
            blocks = w.view(n, -1, block)
            blocks[..., :4 if name == "nvfp4" else 1] = torch.randint(0x30, 0x40 if name == "nvfp4" else 0x80,
                                                                      blocks[..., :4 if name == "nvfp4" else 1].shape,
                                                                      dtype=torch.uint8, device="cuda")
        out.append(ggml.with_tail(w, tail))
    return out, "random"


def row_bytes(name: str, k: int) -> int:
    values, nbytes = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[name.upper()]]
    return k // values * nbytes


def per_call_us(fn, copies: int, calls: int, reps: int = 5) -> float:
    """One graph of `calls` calls cycling through the weight copies, replayed `reps` times: µs per call."""
    for i in range(copies):
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i % copies)
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    del graph
    return start.elapsed_time(end) / (calls * reps) * 1e3


def stream_gbs(gib: int = 2, reps: int = 10) -> dict:
    """A read-only reduction over `gib` GiB of f32: the best of `reps` (GB/s)."""
    x = torch.zeros(gib << 28, device="cuda")
    times = []
    for _ in range(reps + 1):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        x.sum()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    nbytes = x.numel() * 4
    return {"bytes": nbytes, "gb_s_best": round(nbytes / min(times[1:]) / 1e9, 1),
            "gb_s_median": round(nbytes / sorted(times[1:])[reps // 2] / 1e9, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locks", nargs="+", required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--json")
    args = ap.parse_args()
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    sm, ext = ctx.fingerprint.sm, ggml.Extension()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    result = {"gpu": torch.cuda.get_device_name(), "sm": sm, "l2_mb": round(l2 / 2**20), "stream": stream_gbs(),
              "shapes": []}
    print(result["gpu"], "stream", result["stream"], flush=True)
    for (layout, k, n, mode), entry in sorted(sites(args.locks).items()):
        name = ggml.weight_type(layout)
        if mode == "fp4" and sm != 120:
            continue
        facts = dict(entry["facts"], act_quant="a4" if mode == "fp4" else entry["facts"]["act_quant"])
        kernels = {dtype: ggml.QLinear(ext, MODES[mode], ctx, **(facts | {"out_dtype": dtype}))
                   for dtype in ("bfloat16", "float32")}
        llama_rows = kernels["bfloat16"].mmvq_rows
        weight = n * row_bytes(name, k)
        copies = max(2, math.ceil(3 * l2 / weight))
        blocks, source = weights(entry["files"], name, k, n, copies, kernels["bfloat16"].tail)
        scale2 = torch.ones(1, device="cuda") if name == "nvfp4" else None   # NVFP4's per-tensor scale (HF)
        parts = [{"blocks": b} | ({"scale2": scale2} if scale2 is not None else {}) for b in blocks]
        row = {"layout": layout, "type": name, "in": k, "out": n, "mode": mode, "locks": entry["locks"],
               "weight_mb": round(weight / 1e6, 2), "copies": copies, "weights": source,
               "llama_mmvq_rows": llama_rows, "us": {}}
        for rows in args.rows:
            x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
            us = {}
            for variant, dtype, force in (("mmvq", "bfloat16", 8), ("mmq_cast", "bfloat16", 0),
                                          ("mmq_f32", "float32", 0)):
                if variant == "mmvq" and rows > 8:
                    continue
                kernel = kernels[dtype]
                kernel.mmvq_rows = force
                call = functools.partial(lambda i, f, x, ps: f(x, ps[i]), f=kernel, x=x, ps=parts)
                us[variant] = round(per_call_us(call, copies, max(40, copies)), 2)
            us["llama"] = us["mmvq"] if rows <= llama_rows else us["mmq_cast"]
            row["us"][rows] = us
        blocks = parts = None                       # freed before the next shape's weights
        torch.cuda.empty_cache()
        result["shapes"].append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
