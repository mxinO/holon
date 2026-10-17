"""Build the promote module ahead of time (the engine never compiles):

    python -m holon_plugins.kernels.promote.prebuild [--sm 86 89 120]

Without --sm, builds for the SMs of the visible GPUs; under a minute per SM; needs nvcc of torch's CUDA major.
"""
from __future__ import annotations

import argparse

import torch

from . import extension


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m holon_plugins.kernels.promote.prebuild", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sm", type=int, nargs="+", choices=extension.ARCHS)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    sms = args.sm or sorted({10 * a + b for a, b in map(torch.cuda.get_device_capability,
                                                        range(torch.cuda.device_count()))})
    if not sms or not set(sms) <= set(extension.ARCHS):
        p.error(f"pass --sm from {list(extension.ARCHS)} (visible GPUs: {sms or 'none'})")
    for sm in sms:
        print(f"sm {sm}: {extension.build(sm, args.verbose)}", flush=True)


if __name__ == "__main__":
    main()
