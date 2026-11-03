"""Build the state_copy module ahead of time (the engine never compiles):

    python -m holon_plugins.kernels.state_copy.prebuild [--sm 86 89 120]

Without --sm, builds for the SMs of the visible GPUs; under a minute per SM; needs nvcc of torch's CUDA major.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from . import extension


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m holon_plugins.kernels.state_copy.prebuild", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sm", type=int, nargs="+", choices=extension.ARCHS)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    sms = args.sm or sorted({10 * a + b for a, b in map(torch.cuda.get_device_capability,
                                                        range(torch.cuda.device_count()))})
    if not sms or not set(sms) <= set(extension.ARCHS):
        p.error(f"pass --sm from {list(extension.ARCHS)} (visible GPUs: {sms or 'none'})")
    if len(sms) > 1:   # one process per SM: torch's loader renames a module built twice in one process (v1-pip F8)
        for sm in sms:
            subprocess.run([sys.executable, "-m", __spec__.name, "--sm", str(sm), *(["--verbose"] * args.verbose)],
                           check=True)
        return
    print(f"sm {sms[0]}: {extension.build(sms[0], args.verbose)}", flush=True)


if __name__ == "__main__":
    main()
