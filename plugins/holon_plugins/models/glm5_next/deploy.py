"""What fits GLM-5.3-Flash's lock to a host short of its ≈ 92 GiB (docs/dev/v1-auto-config.md D1), asked by `holon
serve` after a start refused for host memory: `spill`, mode (ii) as measured on the lock's card (docs/dev/v1-glm.md
§13 step 7) — RAM holds the share of the experts the host budget fits (`moe.ram=fit`, std's: docs/dev/v1-glm-ram.md
D1), the rest read from the checkpoint's file (exclusive tiers), with the store's settings that mode was measured at.
On a card it was not measured on it offers nothing: std's generic spill (the tiers and `fit` alone) applies."""
from __future__ import annotations

from collections.abc import Mapping

# Mode (ii) by the lock's card (its fingerprint class's GPU), as measured on an RTX 5090 with a 64 GiB host (the
# evidence `glm_step7_mode2_d8x_2026-10-30.json`): RAM's share sized to the host; F10 with D8t's budget; split calls;
# the read-ahead rows; 8,192-row passes; the loan's rule as measured there
MODE2 = {"rtx-5090": {"moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.ram": "fit",
                      "moe.placement": "placement.promote@1", "moe.split": "on", "moe.read_ahead": "135",
                      "moe.stream_tokens": "65536", "ggml_cuda.pass_rows": "8192", "moe.loan_fixed_mib": "1024",
                      "moe.loan_margin": "1.12", "moe.loan_taper": "8192,32768", "moe.loan_largest": "0.55"}}
# The knee by card (docs/dev/v1-glm-ram.md D4): below this host RAM (GiB), 32K decode falls under this rate (tok/s).
KNEE = {"rtx-5090": (72, 24)}     # measured: 32K decode 24.2 tok/s at 72 GiB, 19.5 at 64 (docs/dev/v1-glm-ram.md §10)
# The least host RAM (GiB: MemTotal, or a lower cgroup limit) the spill serves on, on every card (the owner: refuse
# rather than serve slowly; docs/dev/v1-glm-ram.md §12): a 64 GB machine's MemTotal is ≈ 62-63 GiB, so 60 takes it
# and refuses 56 GiB and less, where 1K decode measured 17 tok/s and below
MIN_RAM = 60
LOCK_RAM = 88                     # GiB: the RTX 5090 lock starts there (docs/dev/v1-glm-ram.md §10)


def spill(options: Mapping[str, object], checkpoint: str, *, card: str) -> tuple[dict[str, str], str] | None:
    """Mode (ii) on the card it was measured on (`card`, the lock's fingerprint class); None on any other, or for
    options already setting RAM's share or the tiers."""
    gpu = card.partition("/")[0]
    if options.get("moe.store") != "std.moe.offload@1" or "moe.ram" in options or "moe.tiers" in options:
        return None
    floor = (f"below {MIN_RAM} GiB of RAM it refuses to start (its lock pins every expert: about {LOCK_RAM} GiB "
             "for full speed)")
    if gpu not in MODE2:                       # std's generic spill (the tiers and fit) applies, with the minimum
        return {"moe.ram_min": str(MIN_RAM)}, f"mode (ii) unmeasured on this card: {floor}"
    ram, rate = KNEE[gpu]
    return dict(MODE2[gpu]) | {"moe.ram_min": str(MIN_RAM)}, (
        "mode (ii): the experts this host's RAM holds pinned, the rest read from the checkpoint's file (keep it on a "
        "local NVMe drive); with 64 GiB of RAM 1K-4K prefill ≈ 2.5–2.8×, decode ≈ 1.7–1.9× and 32K prefill ≈ 1.1× "
        f"slower than the lock, faster with more (docs/dev/v1-glm.md); below ≈ {ram} GiB 32K decode falls under ≈ "
        f"{rate} tok/s; {floor} (docs/dev/v1-glm-ram.md)")
