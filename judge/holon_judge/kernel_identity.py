"""Speculation off, nothing changes (docs/dev/m04-spec.md §1.6): `kernels.triton_attn`'s paged attention as it is now,
called with its default facts, against the same kernel at an earlier revision — bitwise, on decode rows, verify rows
and blocks of 8 up to 3000 tokens, model-dtype and q8_0 KV, every head_dim, one split and several.

    python -m holon_judge.kernel_identity --rev 71e6fd3 --out local/evidence/<record>.json

The earlier kernel is read with `git show` into a module of its own; both run on the same inputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

import torch
from holon_std.mixers.gqa.reference import quantize_q8_0

KERNEL = "plugins/holon_plugins/kernels/triton_attn/kernel.py"
GEOMETRIES = [(32, 8, 128), (16, 4, 256), (24, 4, 256), (16, 8, 64)]
ROWS = {"decode": [(0, 1), (15, 1), (99, 1), (4096, 1)], "verify": [(0, 3), (13, 3), (998, 3), (4093, 3)],
        "block": [(5, 8), (2040, 8), (3000, 8)]}
PAGE = 16


def _revision(rev: str, root: pathlib.Path):
    """The kernel module at git revision `rev`."""
    source = subprocess.run(["git", "show", f"{rev}:{KERNEL}"], cwd=root, check=True, capture_output=True,
                            text=True).stdout
    path = pathlib.Path(tempfile.mkdtemp()) / "triton_attn_kernel_at_rev.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("triton_attn_kernel_at_rev", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                      # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def _case(rows, heads: int, kv_heads: int, head_dim: int, q8: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pages = -(-max(s + n for s, n in rows) // PAGE)
    units = len(rows) * pages + 3
    table = torch.randperm(units, generator=g)[: len(rows) * pages].view(len(rows), pages).int().cuda()
    raw = torch.randn(units, 2, kv_heads, PAGE, head_dim, generator=g)
    kv = tuple(x.cuda() for x in quantize_q8_0(raw)) if q8 else raw.to(torch.bfloat16).cuda()
    q = torch.randn(sum(n for _, n in rows), heads, head_dim, generator=g).to(torch.bfloat16).cuda()
    lens = torch.tensor([s + n for s, n in rows], dtype=torch.int32).cuda()
    return q, kv, table, torch.arange(len(rows), dtype=torch.int32).cuda(), lens, rows[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", required=True, help="the revision whose kernel the current one must equal")
    parser.add_argument("--out", help="write the record here (JSON)")
    a = parser.parse_args()
    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True,
                                       text=True).stdout.strip())
    from holon_plugins.kernels.triton_attn import kernel as now
    then = _revision(a.rev, root)
    cases, equal = [], 0
    for geometry in GEOMETRIES:
        for kind, rows in ROWS.items():
            for q8 in (False, True):
                q, kv, table, slots, lens, nq = _case(rows, *geometry, q8)
                for splits in (1, 3):
                    args = (q, kv, table, slots, lens)
                    same = torch.equal(then.paged_attention(*args, nq=nq, splits=splits, cfg=then.Config(64, 4, 1)),
                                       now.paged_attention(*args, nq=nq, splits=splits, cfg=now.Config(64, 4, 1)))
                    cases.append({"geometry": list(geometry), "rows": kind, "kv": "q8_0" if q8 else "bf16",
                                  "splits": splits, "bitwise_equal": same})
                    equal += same
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, check=True, capture_output=True,
                            text=True).stdout.strip()
    record = {"gpu": torch.cuda.get_device_name(), "command": f"python -m holon_judge.kernel_identity --rev {a.rev}",
              "kernel": KERNEL, "against": a.rev, "commit": commit, "cases": len(cases), "bitwise_equal": equal,
              "results": cases}
    print(f"{equal}/{len(cases)} cases bitwise equal to {a.rev}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(record, indent=1) + "\n")
    return 0 if equal == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
