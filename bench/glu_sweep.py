"""M0.7 step 2 (docs/m07-perf.md D6): the fused gate/up + SwiGLU at one row against its composition, per merged gate/up
shape of the given locks, graph-timed: `qlinear_glu` (mmvq's fused gate, `silu(gate) · up` from the f32 sums) vs
`qlinear` writing [1, 2·inter] in bf16, then FlashInfer's `silu_and_mul`.

A lock's merged gate/up sites are its `qlinear` sites whose `out` is twice another site's `in` (down's input is the
intermediate size). Weights as bench/qmm_sweep.py's: the GGUF file's rows of that type and input size (else random
bytes with valid FP4 scales), copies cycled beyond three times the L2 cache.

    python bench/glu_sweep.py --locks plans/.../N.lock ... [--json out]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import flashinfer
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.ggml_cuda import plugin as ggml
from qmm_sweep import MODES, per_call_us, row_bytes, sites, weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locks", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    ggml.GLU_TYPES = frozenset(ggml.TYPES)            # time the fused kernel for every type: the sweep decides the set
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    ext, l2 = ggml.Extension(), torch.cuda.get_device_properties(0).L2_cache_size
    found = {key: e for key, e in sites(a.locks).items() if key[3] == "q8_1"}
    inters = {k for (_, k, _, _) in found}
    result = {"gpu": torch.cuda.get_device_name(), "sm": ctx.fingerprint.sm, "shapes": []}
    for (layout, k, n, mode), entry in sorted(found.items()):
        if n // 2 not in inters or n % 8:
            continue
        name, facts = ggml.weight_type(layout), dict(entry["facts"], out_dtype="bfloat16")
        fused = ggml.QLinearGlu(ext, MODES[mode], ctx, **facts, glu="swiglu")
        plain = ggml.QLinear(ext, MODES[mode], ctx, **facts)
        copies = max(2, math.ceil(3 * l2 / (n * row_bytes(name, k))))
        blocks, source = weights(entry["files"], name, k, n, copies, 0)
        scale2 = torch.ones(1, device="cuda") if name == "nvfp4" else None
        parts = [{"blocks": b} | ({"scale2": scale2} if scale2 is not None else {}) for b in blocks]
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        us = {"fused": per_call_us(lambda i, f=fused, x=x, p=parts: f(x, p[i]), copies, 200),
              "composed": per_call_us(lambda i, f=plain, x=x, p=parts: flashinfer.silu_and_mul(f(x, p[i])), copies,
                                      200)}
        row = {"type": name, "in": k, "out": n, "locks": entry["locks"], "weights": source,
               "us": {v: round(t, 3) for v, t in us.items()},
               "saving_pct": round((us["composed"] - us["fused"]) / us["composed"] * 100, 2)}
        result["shapes"].append(row)
        print(json.dumps(row), flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
