"""Build the ggml_cpu modules ahead of time (the engine never compiles):

    python -m holon_plugins.kernels.ggml_cpu.prebuild [--tier x86-avx2 x86-avx512] [--sm N]

Without --tier, builds every tier (a module loads on any x86-64 host: only its kernels use the tier's instructions),
each in its own process: torch names a module linked a second time in one process with other arguments `<name>_v1`,
which `extension.load` never finds (an AVX-512 host's F5b then failed with NotBuilt after `holon prebuild`).
`--sm` is accepted and ignored (`holon prebuild` passes it to every plugin; CPU code does not depend on the GPU), so
the second SM finds the modules built. Under a minute per tier.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from . import extension


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m holon_plugins.kernels.ggml_cpu.prebuild", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", nargs="+", choices=sorted(extension.TIERS))
    p.add_argument("--sm", type=int, nargs="*", help="ignored")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    tiers = args.tier or sorted(extension.TIERS)
    for tier in tiers:
        if extension.module_path(tier).is_file():
            print(f"{tier}: {extension.module_path(tier)} (built)", flush=True)
        elif len(tiers) > 1:                       # one process per tier (above)
            subprocess.run([sys.executable, "-m", __spec__.name, "--tier", tier, *(["--verbose"] * args.verbose)],
                           check=True)
        else:
            print(f"{tier}: {extension.build(tier, args.verbose)}", flush=True)


if __name__ == "__main__":
    main()
