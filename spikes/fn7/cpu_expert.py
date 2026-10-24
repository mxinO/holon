"""FN step 7 item 4: the F5b CPU path's time per expert (`cpu_experts@1`, ggml-cpu's dot products) at FN IQ2_XS's
expert shapes (IQ2_S / IQ2_XXS / IQ1_M gate/up 1280×2560 as 2·640 rows of 2560, Q2_0 down 2560×640), by job count,
tokens per job and threads; synchronous runs (the worker's wake-up not included). Usage: cpu_expert.py [tier]"""
from __future__ import annotations

import json
import sys
import time

import torch
from holon_plugins.kernels.ggml_cpu import extension
from holon_plugins.kernels.ggml_cpu.plugin import TYPES

HIDDEN, INTER, TOP_K, TOKENS, JOBS, UNITS = 2560, 640, 10, 4, 32, 64


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "x86-avx2"
    lib = extension.load(tier)
    out = []
    for gu in ("iq2_s", "iq2_xxs", "iq1_m"):
        e = lib.Experts(TYPES[gu], TYPES["q2_0"], HIDDEN, INTER, TOP_K, TOKENS, JOBS)
        gu_bytes = {"iq2_s": 82, "iq2_xxs": 66, "iq1_m": 56}[gu] * HIDDEN // 256 * 2 * INTER
        dn_bytes = 10 * INTER // 32 * HIDDEN
        g = torch.Generator().manual_seed(0)
        # random bytes are valid blocks for these types' dot products (any grid index, sign and scale bits)
        gate_up = torch.randint(0, 256, (UNITS, gu_bytes), dtype=torch.uint8, generator=g)
        down = torch.randint(0, 256, (UNITS, dn_bytes), dtype=torch.uint8, generator=g)
        down.view(UNITS, -1, 10)[..., :2] = torch.tensor([0, 0x2c], dtype=torch.uint8)    # f16 scales: small
        x = torch.randn(TOKENS, HIDDEN, generator=g)
        rows = torch.zeros(TOKENS * TOP_K, HIDDEN)
        for jobs, tokens in ((1, 1), (2, 1), (4, 1), (8, 1), (4, 4)):
            table = torch.zeros(1 + JOBS, 4 + TOKENS, dtype=torch.int32)
            table[0, 0] = jobs
            for j in range(jobs):
                table[1 + j, :4] = torch.tensor([j, j, j, tokens])
                table[1 + j, 4:4 + tokens] = torch.arange(tokens) * TOP_K + j % TOP_K
            task = lib.Task(e, table, x, gate_up, down, rows, 0)
            for threads in (1, 2, 4, 6, 8):
                task.run(threads)
                t0, n = time.perf_counter(), 50
                for _ in range(n):
                    task.run(threads)
                us = (time.perf_counter() - t0) / n * 1e6
                row = {"gate_up": gu, "jobs": jobs, "tokens": tokens, "threads": threads, "us": round(us, 1),
                       "us_per_expert": round(us / jobs, 1),
                       "gbs": round(jobs * (gu_bytes + dn_bytes) / us / 1e3, 2)}
                out.append(row)
                print(json.dumps(row), flush=True)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
