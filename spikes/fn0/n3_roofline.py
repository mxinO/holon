"""FN step 0, N3 (docs/v1-flash-next.md section 8): the proxy roofline — every acceptance-5 row predicted from
measured inputs, before any FN kernel exists. Ratios are Holon / Strata on D11's contexts, Strata's denominators the
medians of its gated runs (N1). Fixed before the comparison and recorded with the inputs:

**Decode** (per token): t = (dense bytes + GDN state + hit bytes) / BW + miss bytes / PCIe + per-step overhead.
- dense bytes per token: every non-expert tensor of the file but the embedding and the PLE table (GGUF headers);
  GDN state: 36 layers x 48 x 128 x 128 fp32, read and written;
- hit rate: Holon's `placement.lfu@1` per layer, seeded from the expert profile, replayed on Strata's own decode
  routing (`--dump-routing`, tail skip off) at §7's cache (`slots_per_layer`; checked against Strata's measured hit
  rate at Strata's own cache size);
- BW: 85% of the card's DRAM bandwidth; PCIe and the overhead from X's two Holon rows on the RTX 5080 (Qwen3-30B-A3B
  lfu and static at 25%: the slope of time over miss bytes gives the F5a fetch rate, the rest the per-step overhead,
  scaled from X's ~11 to FN's 16 launches per layer);
- MTP: Holon's verify passes accept as Strata's do (the same pack, full head, min-p 0), so the MTP ratio is the
  drafting-off ratio; A1 (tail skip) gains what Strata's tail skip gains between its no-approx and defaults rows.

**Prefill** (per request): Strata's phase timers (`STRATA_PREFILL_TIMING`, St prefill.cpp:2000-2012) give each
phase's share of its prompt time; Holon's time is Strata's gated time x the sum of share x f over phases, f being
Holon's cost per phase relative to Strata's: 1.0 for the ggml mmq expert and dense GEMMs (assumes the MoE prefill
track's D1-D5 and IQ1_M's native loader, D1'), and bands for the parts FN writes in Triton (HC in fp32 ieee, GDN,
QSA). A step is also bound below by streaming the non-resident experts its tokens route to over PCIe: at 1K the
routed share of them comes from windows of the decode routing trace (1,024 consecutive tokens; 1.0 beyond the
trace's length).

    python spikes/fn0/n3_roofline.py --card rtx-5080 --out <record.json>
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import routing  # noqa: E402  (spikes/fn0/routing.py, beside this file)
import run  # noqa: E402

CARDS = {   # file, dense GB, expert MB per (layer, expert), DRAM GB/s x 0.85, §7's cache (GB, 8K steps), Holon PCIe
    "rtx-5080": {"file": "IQ2_XS", "dense_gb": 3.4222, "expert_mb": 1.44267, "bw": 960 * 0.85, "cache_gb": 6.7},
    "rtx-5090": {"file": "IQ3_S", "dense_gb": 4.1765, "expert_mb": 2.04640, "bw": 1792 * 0.85, "cache_gb": 21.45},
}
LAYERS, TOPK, GDN_STATE_GB = 48, 10, 36 * 48 * 128 * 128 * 4 * 2 / 1e9
# X's Holon rows, RTX 5080, Qwen3-30B-A3B Q4_K_M at 25% (evidence/x8_close_rtx-5080_2026-10-18.json, 1K): lfu and F1
X_ROWS = {"lfu": {"ms": 1e3 / 119.1672, "miss_mb": 169.7687, "hit": 0.8478},
          "static": {"ms": 1e3 / 83.6541, "miss_mb": 321.0333, "hit": 0.7072}}
X_DENSE_GB, X_ROUTED, X_EXPERT_MB, X_BW = 1.0, 384, 2.857, 960 * 0.85
X_LAUNCHES, FN_LAUNCHES = 11, 16
# Holon's cost per Strata prefill phase (central, low, high)
F = {"hc read": (1.5, 1.0, 2.0), "gdn": (1.1, 0.9, 1.5), "gdn conv+gates": (1.1, 0.9, 1.5),
     "gdn recurrence": (1.1, 0.9, 1.5), "gdn out proj": (1.0, 1.0, 1.0), "qsa attn": (1.2, 1.0, 1.6),
     "qsa select": (1.2, 1.0, 1.6), "qsa indexer": (1.2, 1.0, 1.6), "dequant": (0.5, 0.25, 1.0),
     "gemm gate/up": (1.0, 1.0, 1.3), "gemm down": (1.0, 1.0, 1.3)}
PCIE_STREAM = 47.5                                      # GB/s, the prefill ring's copies (X's M1)
PHASE = re.compile(r" ([a-z+/ ]+?) (\d+) \([\d.]+%\)")


def x_fit() -> tuple[float, float]:
    """(F5a fetch GB/s, per-step overhead ms) from X's two rows."""
    a, b = X_ROWS["lfu"], X_ROWS["static"]
    hit = lambda r: X_ROUTED * r["hit"] * X_EXPERT_MB / 1e3 / X_BW * 1e3          # noqa: E731  ms
    slope = (b["ms"] - a["ms"] - (hit(b) - hit(a))) / (b["miss_mb"] - a["miss_mb"])  # ms per MB
    over = a["ms"] - X_DENSE_GB / X_BW * 1e3 - hit(a) - a["miss_mb"] * slope
    return 1 / slope, over


def decode_ms(card: dict, hit: float, pcie: float, over: float, context: int) -> float:
    routed = LAYERS * TOPK * card["expert_mb"] / 1e3                     # GB per token
    qsa = 0.24 if context > 2051 else 0.03                               # ms: sparse reads + selection at 32K
    return ((card["dense_gb"] + GDN_STATE_GB + hit * routed) / card["bw"] * 1e3 + (1 - hit) * routed * 1e3 / pcie
            + over + qsa)


def phases(log: pathlib.Path) -> dict[int, list[dict[str, float]]]:
    """Each prefill's phase times (ms) by prompt length, from a Strata server log with STRATA_PREFILL_TIMING."""
    out: dict[int, list[dict[str, float]]] = {}
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"strata prefill timing: (\d+) tokens, GPU timeline", line)
        if m:
            out.setdefault(int(m[1]) + 1, []).append({k.strip(): float(v) for k, v in PHASE.findall(line)})
    return out


def factor(ph: dict[str, float], which: int) -> float:
    total = sum(ph.values())
    return sum(v * F.get(k, (1.0, 1.0, 1.0))[which] for k, v in ph.items()) / total


def routed_share(trace: str, profile: str, slots: int, window: int) -> float:
    """The mean share of a layer's non-resident experts (the profile's first `slots` resident) that `window`
    consecutive tokens of the trace route to; 1.0 when the trace is shorter than the window."""
    tokens, order = routing.read(trace), routing.profile(profile)
    if window > len(tokens):
        return 1.0
    shares = []
    for a in range(0, len(tokens) - window + 1, 256):
        hit = sum(len(set(tokens[a:a + window, layer].ravel().tolist()) - set(order[layer][:slots]))
                  for layer in range(LAYERS))
        shares.append(hit / (LAYERS * (512 - slots)))
    return statistics.mean(shares)


def lfu_hit(table: dict[str, float], slots: float) -> float:
    pts = sorted((int(k), v) for k, v in table.items())
    for (s0, h0), (s1, h1) in zip(pts, pts[1:], strict=False):
        if s0 <= slots <= s1:
            return h0 + (h1 - h0) * (slots - s0) / (s1 - s0)
    raise ValueError(f"{slots} slots outside the replayed range")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", choices=sorted(CARDS), required=True)
    ap.add_argument("--strata", required=True, help="the gated Strata run's record.json (bench/offload_ctx.py)")
    ap.add_argument("--diag", required=True, help="the diagnostic run's server log (prefill timers)")
    ap.add_argument("--lfu", required=True, help="spikes/fn0/routing.py's output for this card's trace")
    ap.add_argument("--trace", required=True, help="the card's Strata decode routing trace (--dump-routing)")
    ap.add_argument("--profile", required=True, help="Strata's data/expert-profile.bin")
    ap.add_argument("--peak-kb", type=float, default=200.0, help="the measured non-lendable peak per token (N3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    card = CARDS[a.card]
    pcie, over_x = x_fit()
    over = over_x * FN_LAUNCHES / X_LAUNCHES
    lfu = json.loads(pathlib.Path(a.lfu).read_text())["hit_rate"]
    cache_gb = card["cache_gb"] - 8192 * (a.peak_kb - 200.0) / 1e6      # §7's 8K-step cache with the measured peak
    slots = cache_gb * 1e3 / card["expert_mb"] / LAYERS
    hit = lfu_hit(lfu, slots)
    strata = json.loads(pathlib.Path(a.strata).read_text())["summary"]
    rows = {}
    ph = phases(pathlib.Path(a.diag))
    for c in (1024, 4096, 32768):
        def med(arm: str, k: str, c: int = c) -> float:
            return statistics.median(v[k]["median"] for key, v in strata[arm].items() if key.startswith(f"{c}/"))
        s_off, s_mtp = med("strata_na_off", "decode_tps"), med("strata_na_mtp", "decode_tps")
        d_off, d_mtp = med("strata_def_off", "decode_tps"), med("strata_def_mtp", "decode_tps")
        a1 = d_off / s_off                                               # Strata's own approximation gain
        dec = {w: 1e3 / decode_ms(card, hit, pcie, over * k, c) for w, k in (("central", 1.0), ("low", 1.3),
                                                                               ("high", 0.7))}
        pf = statistics.median(factor(p, 0) for p in ph[c])
        pf_lo = statistics.median(factor(p, 2) for p in ph[c])               # the slow bound: high f
        pf_hi = statistics.median(factor(p, 1) for p in ph[c])
        s_pf = med("strata_na_off", "prefill_tps")
        share = routed_share(a.trace, a.profile, round(slots), min(c, 8192))
        stream_s = share * (LAYERS * 512 - cache_gb * 1e3 / card["expert_mb"]) * card["expert_mb"] / 1e3 / PCIE_STREAM
        cap = min(c, 8192) / stream_s
        h_pf = {w: min(s_pf / f, cap) for w, f in (("central", pf), ("low", pf_lo), ("high", pf_hi))}
        rows[str(c)] = {
            "strata": {"na_off_decode": round(s_off, 2), "na_mtp_decode": round(s_mtp, 2),
                       "def_off_decode": round(d_off, 2), "def_mtp_decode": round(d_mtp, 2),
                       "na_prefill": round(s_pf, 1), "def_prefill": round(med("strata_def_off", "prefill_tps"), 1)},
            "holon": {"decode_off": {k: round(v, 1) for k, v in dec.items()},
                      "prefill": {k: round(v, 1) for k, v in h_pf.items()}, "prefill_stream_cap": round(cap, 1),
                      "routed_nonresident_share": round(share, 3)},
            "ratios": {
                "noapprox_off_decode": {k: round(v / s_off, 3) for k, v in dec.items()},
                "noapprox_mtp_decode": {k: round(v / s_off, 3) for k, v in dec.items()},     # same acceptance
                "defaults_off_decode": {k: round(v * a1 / d_off, 3) for k, v in dec.items()},
                "defaults_mtp_decode": {k: round(v * a1 * s_mtp / s_off / d_mtp, 3) for k, v in dec.items()},
                "noapprox_prefill": {k: round(v / s_pf, 3) for k, v in h_pf.items()},
                "defaults_prefill": {k: round(v / med("strata_def_off", "prefill_tps"), 3) for k, v in h_pf.items()}},
            "strata_phase_shares": {k: round(statistics.median(p.get(k, 0) for p in ph[c]) /
                                             statistics.median(sum(p.values()) for p in ph[c]), 4)
                                    for k in sorted({k for p in ph[c] for k in p})}}
    below = sorted({f"{c} {r}" for c, row in rows.items() for r, v in row["ratios"].items() if v["central"] < 0.9})
    rec = {"name": "fn0_n3_roofline", "date": datetime.date.today().isoformat(), "gpu": a.card,
           "commit": run.git("rev-parse", "HEAD"), "command": f"python spikes/fn0/n3_roofline.py --card {a.card} ...",
           "method": __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0],
           "inputs": {"x_fit": {"f5a_fetch_gbps": round(pcie, 2), "overhead_ms_x": round(over_x, 3),
                                "overhead_ms_fn": round(over, 3)}, "card": card, "cache_gb": round(cache_gb, 3),
                      "slots_per_layer": round(slots, 1), "lfu_hit_rate": round(hit, 4), "lfu_replay": lfu,
                      "non_lendable_peak_kb": a.peak_kb, "phase_factors": F, "prefill_stream_gbps": PCIE_STREAM},
           "rows": rows, "rows_below_0_9_central": below}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({c: r["ratios"] for c, r in rows.items()}, indent=1))
    print("below 0.9 (central):", below)


if __name__ == "__main__":
    main()
