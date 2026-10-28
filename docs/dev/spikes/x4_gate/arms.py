"""The gate's arms (spikes/x4_gap/ctx.py's format): Holon under the offload lock's settings and kernel pins, and the
x4_refs reference arms at the same expert share. Writes arms_natural.json and arms_capped.json beside this file.

    python spikes/x4_gate/arms.py --pin '<moe.cache of the natural fit>' --kernels op=provider ... [--ring-copy dma]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
EXPERT_MIB = 16740                      # the experts' bytes (x4_refs: 6,144 x 2.86 MB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True, help="the natural fit's moe.cache (the lock's)")
    ap.add_argument("--share", type=float, required=True, help="its share of the experts' bytes")
    ap.add_argument("--kernels", nargs="*", default=[])
    ap.add_argument("--ring-copy", default="dma")
    ap.add_argument("--profile", default="profile_x4_gate.json")
    a = ap.parse_args()

    def holon(name: str, cache: str, policy: str = "placement.lfu@1") -> dict:
        opts = ["moe.store=std.moe.offload@1", f"moe.placement={policy}", f"moe.placement.profile={a.profile}",
                f"moe.cache={cache}", f"moe.ring_copy={a.ring_copy}"]
        kernels = f" --kernels {' '.join(a.kernels)}" if a.kernels else ""
        return {"name": name, "engine": "holon",
                "flags": f"--host-memory 21474836480 --plugin-options {' '.join(opts)}{kernels}"}

    def cache(name: str, share: float, pinned: bool) -> dict:
        mib = round(share * EXPERT_MIB)
        return {"name": name, "engine": "llama",
                "flags": f"-cmoe --moe-cache-mib {mib}{' --load-mode none' if pinned else ''} -lv 4"}

    def ft(name: str, share: float) -> dict:
        return {"name": name, "engine": "freetoken", "flags": f"--moe-cache-rate {share:.3f}"}

    natural = [holon("holon_nat", a.pin), {"name": "llama_fit", "engine": "llama", "flags": " -lv 4"},
               {"name": "llama_ncmoe16", "engine": "llama", "flags": "--n-cpu-moe 16 -lv 4"},
               cache("cache_nat", a.share, False), cache("cache_nat_pinned", a.share, True), ft("ft_nat", a.share),
               holon("holon_47", "*=0.47"), cache("cache_47", 0.4719, False), cache("cache_47_pinned", 0.4719, True),
               ft("ft_47", 0.47)]
    capped = [holon("holon_25", "*=0.25"), {"name": "cap_fit", "engine": "llama", "flags": "-fitt 7150 -lv 4"},
              {"name": "cap_ncmoe36", "engine": "llama", "flags": "--n-cpu-moe 36 -lv 4"},
              cache("cap_cache", 0.2509, False), cache("cap_cache_pinned", 0.2509, True), ft("cap_ft", 0.25)]
    for name, arms in (("natural", natural), ("capped", capped)):
        (HERE / f"arms_{name}.json").write_text(json.dumps(arms, indent=1) + "\n")


if __name__ == "__main__":
    main()
