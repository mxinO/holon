"""Build the ggml_cpu modules ahead of time (the engine never compiles):

    python -m holon_plugins.kernels.ggml_cpu.prebuild [--tier x86-avx2 x86-avx512] [--sm N]

Without --tier, builds every tier (a module loads on any x86-64 host: only its kernels use the tier's instructions).
`--sm` is accepted and ignored (`holon prebuild` passes it to every plugin; CPU code does not depend on the GPU), so
the second SM finds the modules built. Under a minute per tier.
"""
from __future__ import annotations

import argparse

from . import extension


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m holon_plugins.kernels.ggml_cpu.prebuild", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", nargs="+", choices=sorted(extension.TIERS))
    p.add_argument("--sm", type=int, nargs="*", help="ignored")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    for tier in args.tier or sorted(extension.TIERS):
        if extension.module_path(tier).is_file():
            print(f"{tier}: {extension.module_path(tier)} (built)", flush=True)
        else:
            print(f"{tier}: {extension.build(tier, args.verbose)}", flush=True)


if __name__ == "__main__":
    main()
