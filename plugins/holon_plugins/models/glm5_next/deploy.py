"""What fits GLM-5.3-Flash's RTX 5090 lock to a host short of its ≈ 92 GiB (docs/dev/v1-auto-config.md D1), asked by
`holon serve` after a start refused for host memory: `spill`, mode (ii) as measured (docs/dev/v1-glm.md §13 step 7) —
RAM holds a share of the experts, the rest read from the checkpoint's file (exclusive tiers), with the store's settings
that mode was measured at."""
from __future__ import annotations

from collections.abc import Mapping

# Mode (ii) as measured on an RTX 5090 with a 64 GiB host (the evidence `glm_step7_mode2_d8x_2026-10-30.json`): RAM's
# share of the experts sized for that host; F10 with D8t's budget; split calls; the read-ahead rows; 8,192-row passes;
# the loan's rule as measured there
MODE2 = {"moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.ram": "*=0.6238",
         "moe.placement": "placement.promote@1", "moe.split": "on", "moe.read_ahead": "135",
         "moe.stream_tokens": "65536", "ggml_cuda.pass_rows": "8192", "moe.loan_fixed_mib": "1024",
         "moe.loan_margin": "1.12", "moe.loan_taper": "8192,32768", "moe.loan_largest": "0.55"}


def spill(options: Mapping[str, object], checkpoint: str) -> tuple[dict[str, str], str] | None:
    """Mode (ii) for a lock that pins every expert and lends its 32K steps their activations (the RTX 5090's: mode
    (ii) was measured there only); None for any other, or one already setting RAM's share."""
    # `require`: the RTX 5090's lock, the only GLM lock that lends its steps (the hook is not given the lock's class;
    # plan §10's host-budget follow-up passes it, and MODE2 is then keyed by class)
    if options.get("moe.store") != "std.moe.offload@1" or "moe.ram" in options or "moe.tiers" in options or \
            options.get("moe.lend_peak") != "require":
        return None
    return dict(MODE2), ("mode (ii): about 62% of the experts in pinned RAM (sized for a 64 GB host), the rest read "
                         "from the checkpoint's file (keep it on a local NVMe drive); 1K-4K prefill ≈ 2.5–2.8×, "
                         "decode ≈ 1.7–1.9× and 32K prefill ≈ 1.1× slower than the lock on an RTX 5090 "
                         "(docs/dev/v1-glm.md); if it does not fit either, keep these options with a smaller moe.ram "
                         "share")
