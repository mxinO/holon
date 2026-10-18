"""FN step 0, N3 (docs/v1-flash-next.md section 8): the proxy roofline — every acceptance-5 row predicted from
measured inputs, before any FN kernel exists. Ratios are Holon / Strata on D11's contexts, Strata's denominators the
medians of its gated runs (N1). Rev 2 takes panel C's step-0 review (§13): per-row caches, an MTP pass model, q8_0's
gain, Strata's gather phase. The inputs:

**The cache per row**: §7's budget at the card's lock step (the RTX 5080's lock steps 4K, lead decision; the 5090's 8K)
less the measured non-lendable peak over §7's 200 KB per token; the drafting-off rows deploy without the 0.89 GB MTP
pack, the MTP rows with it; and, for the RTX 5080, step 7's target (≥ 8.2 GB with the pack, §13).

**Decode, drafting off** (per token): t = (dense bytes + GDN state + hit bytes) / BW + miss bytes / PCIe + overhead.
- dense bytes per token: every non-expert tensor of the file but the embedding and the PLE table (GGUF headers);
  GDN state: 36 layers x 48 x 128 x 128 fp32, read and written;
- hit rate: Holon's `placement.lfu@1` per layer, seeded from the expert profile, replayed on Strata's own decode
  routing (`--dump-routing`, tail skip off; it matches Strata's measured rate at Strata's cache size);
- BW: 85% of the card's DRAM bandwidth; PCIe and the overhead from X's two Holon rows on the RTX 5080 (Qwen3-30B-A3B
  lfu and static at 25%: the slope of time over miss bytes gives the F5a fetch rate, the rest the per-step overhead,
  scaled from X's ~11 to FN's 16 launches per layer).

**Decode, MTP** (per verify pass of 4 rows, 3 drafts): the pass reads the dense bytes once and each distinct expert of
its 4 tokens once — lfu replayed on 4-token window unions (one decay per pass) — plus 3 draft steps of the MTP head
(its 0.18 GB of dense parts, the full-vocabulary head, 10 of its Q2_0 experts, one layer's overhead each) and the MTP
head over the pass's verify rows (0.275 ms, panel C's 0.25-0.3); a pass
yields Strata's own recorded accepted tokens per pass in the matched row (the same pack, full head, min-p 0).
A1 (tail skip) gains what Strata's tail skip gains between its no-approx and defaults drafting-off rows; the defaults
MTP rows also carry Strata's suffix drafter and calibrated draft floor, which Holon does not have (a non-goal).

**Prefill** (per request): Strata's phase timers (`STRATA_PREFILL_TIMING`, St prefill.cpp:2000-2012) give each phase's
share of its prompt time; Holon's time is Strata's gated time x the sum of share x f, f being Holon's cost per phase
relative to Strata's: 1.0 for the ggml mmq expert and dense GEMMs (assumes the MoE prefill track's D1-D5 and IQ1_M's
native loader, D1'); bands for the parts FN writes in Triton (HC in fp32 ieee, GDN, QSA); and Strata's "dequant"
phase — the per-expert gather and its issue gaps, on IQ3_S too, not IQ1_M's dequantization — at 0.3 [0.1, 0.5]:
Holon's arena mmq reads the slots in place (no gather), leaving the issue gaps. A step
is bound below by streaming the non-resident experts its tokens route to over PCIe (at 1K the routed share from 1,024-
token windows of the trace; 1.0 beyond the trace's length). The defaults rows give Holon's q8_0 KV (read in-kernel by
`qsa_attend@1`) Strata's own int8 gain (its defaults over its no-approx prefill, where above 1).

    python spikes/fn0/n3_roofline.py --card rtx-5080 --strata <N1 record> --diag <diag server log> --trace <routing> \
        --profile <expert-profile.bin> --peak-kb 222 --out <record.json>
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

CARDS = {   # file, dense GB, expert MB per (layer, expert), DRAM GB/s x 0.85, the head's GB, the lock's step and §7's
    # cache at that step (GB, with the MTP pack resident), step 7's target (GB)
    "rtx-5080": {"file": "IQ2_XS", "dense_gb": 3.4222, "expert_mb": 1.44267, "bw": 960 * 0.85, "head_gb": 0.338,
                 "step": 4096, "cache_gb": 7.5, "target_gb": 8.2},
    "rtx-5090": {"file": "IQ3_S", "dense_gb": 4.1765, "expert_mb": 2.04640, "bw": 1792 * 0.85, "head_gb": 0.521,
                 "step": 8192, "cache_gb": 21.45, "target_gb": None},
}
MTP_PACK_GB, MTP_DENSE_GB, MTP_EXPERT_MB, DRAFTS, WINDOW = 0.889, 0.181, 1.382, 3, 4
MTP_HEAD_MS = 0.275            # the MTP head over a pass's verify rows (panel C: 0.25-0.3 ms)
LAYERS, TOPK, GDN_STATE_GB = 48, 10, 36 * 48 * 128 * 128 * 4 * 2 / 1e9
# X's Holon rows, RTX 5080, Qwen3-30B-A3B Q4_K_M at 25% (evidence/x8_close_rtx-5080_2026-10-18.json, 1K): lfu and F1
X_ROWS = {"lfu": {"ms": 1e3 / 119.1672, "miss_mb": 169.7687, "hit": 0.8478},
          "static": {"ms": 1e3 / 83.6541, "miss_mb": 321.0333, "hit": 0.7072}}
X_DENSE_GB, X_ROUTED, X_EXPERT_MB, X_BW = 1.0, 384, 2.857, 960 * 0.85
X_LAUNCHES, FN_LAUNCHES = 11, 16
# Holon's cost per Strata prefill phase (central, low, high)
F = {"hc read": (1.5, 1.0, 2.0), "gdn": (1.1, 0.9, 1.5), "gdn conv+gates": (1.1, 0.9, 1.5),
     "gdn recurrence": (1.1, 0.9, 1.5), "gdn out proj": (1.0, 1.0, 1.0), "qsa attn": (1.2, 1.0, 1.6),
     "qsa select": (1.2, 1.0, 1.6), "qsa indexer": (1.2, 1.0, 1.6), "dequant": (0.3, 0.1, 0.5),
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


def mtp_pass_ms(card: dict, hit: float, union: float, pcie: float, over: float, context: int) -> float:
    """A verify pass of WINDOW rows (`union` distinct experts per layer, `hit` of them resident), and its drafts."""
    experts = LAYERS * union * card["expert_mb"] / 1e3
    qsa = 0.24 if context > 2051 else 0.03
    verify = ((card["dense_gb"] + GDN_STATE_GB + hit * experts) / card["bw"] * 1e3 + (1 - hit) * experts * 1e3 / pcie
              + over + qsa)
    draft = (MTP_DENSE_GB + card["head_gb"] + TOPK * MTP_EXPERT_MB / 1e3) / card["bw"] * 1e3 + over / LAYERS
    return verify + DRAFTS * draft + MTP_HEAD_MS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", choices=sorted(CARDS), required=True)
    ap.add_argument("--strata", required=True, help="the gated Strata run's record.json (bench/offload_ctx.py)")
    ap.add_argument("--diag", required=True, help="the diagnostic run's server log (prefill timers)")
    ap.add_argument("--trace", required=True, help="the card's Strata decode routing trace (--dump-routing)")
    ap.add_argument("--profile", required=True, help="Strata's data/expert-profile.bin")
    ap.add_argument("--peak-kb", type=float, default=200.0, help="the measured non-lendable peak per token (N3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    card = CARDS[a.card]
    pcie, over_x = x_fit()
    over = over_x * FN_LAUNCHES / X_LAUNCHES
    tokens, seed = routing.read(a.trace), routing.profile(a.profile)
    mtp_gb = card["cache_gb"] - card["step"] * (a.peak_kb - 200.0) / 1e6
    caches = {"off": mtp_gb + MTP_PACK_GB, "mtp": mtp_gb}
    if card["target_gb"]:
        caches |= {"target_off": card["target_gb"] + MTP_PACK_GB, "target_mtp": card["target_gb"]}
    slots = {k: v * 1e3 / card["expert_mb"] / LAYERS for k, v in caches.items()}
    hits = {k: routing.lfu_hits(tokens, round(n), seed, WINDOW if k.endswith("mtp") else 1) for k, n in slots.items()}
    union = routing.union_size(tokens, WINDOW)
    strata = json.loads(pathlib.Path(a.strata).read_text())["summary"]
    rows = {}
    ph = phases(pathlib.Path(a.diag))
    for c in (1024, 4096, 32768):
        def med(arm: str, k: str, c: int = c) -> float:
            return statistics.median(v[k]["median"] for key, v in strata[arm].items() if key.startswith(f"{c}/"))
        s_off, s_mtp = med("strata_na_off", "decode_tps"), med("strata_na_mtp", "decode_tps")
        d_off, d_mtp = med("strata_def_off", "decode_tps"), med("strata_def_mtp", "decode_tps")
        accepted = med("strata_na_mtp", "tokens_per_pass")
        a1 = d_off / s_off                                               # Strata's own approximation gain
        bands = (("central", 1.0), ("low", 1.3), ("high", 0.7))
        dec = {w: 1e3 / decode_ms(card, hits["off"], pcie, over * k, c) for w, k in bands}
        mtp = {w: accepted * 1e3 / mtp_pass_ms(card, hits["mtp"], union, pcie, over * k, c) for w, k in bands}
        pf = {w: statistics.median(factor(p, i) for p in ph[c]) for w, i in (("central", 0), ("low", 2), ("high", 1))}
        s_pf, d_pf = med("strata_na_off", "prefill_tps"), med("strata_def_off", "prefill_tps")
        step = min(c, card["step"])
        share = routed_share(a.trace, a.profile, round(slots["mtp"]), step)
        stream_s = share * (LAYERS * 512 - caches["mtp"] * 1e3 / card["expert_mb"]) * card["expert_mb"] / 1e3 / \
            PCIE_STREAM
        cap = step / stream_s
        h_pf = {w: min(s_pf / f, cap) for w, f in pf.items()}
        kv = max(1.0, d_pf / s_pf)                                       # q8_0 KV gains what Strata's int8 gains
        ratios = {
            "noapprox_off_decode": {k: v / s_off for k, v in dec.items()},
            "noapprox_mtp_decode": {k: v / s_mtp for k, v in mtp.items()},
            "noapprox_prefill": {k: v / s_pf for k, v in h_pf.items()},
            "defaults_off_decode": {k: v * a1 / d_off for k, v in dec.items()},
            "defaults_mtp_decode": {k: v * a1 / d_mtp for k, v in mtp.items()},
            "defaults_prefill": {k: min(v * kv, cap) / d_pf for k, v in h_pf.items()}}
        if card["target_gb"]:
            ratios |= {"target_noapprox_off_decode": {
                w: 1e3 / decode_ms(card, hits["target_off"], pcie, over * k, c) / s_off for w, k in bands},
                "target_noapprox_mtp_decode": {w: accepted * 1e3 / mtp_pass_ms(
                    card, hits["target_mtp"], union, pcie, over * k, c) / s_mtp for w, k in bands}}
        rows[str(c)] = {
            "strata": {"na_off_decode": round(s_off, 2), "na_mtp_decode": round(s_mtp, 2), "accepted_per_pass":
                       round(accepted, 3), "def_off_decode": round(d_off, 2), "def_mtp_decode": round(d_mtp, 2),
                       "na_prefill": round(s_pf, 1), "def_prefill": round(d_pf, 1)},
            "holon": {"decode_off": {k: round(v, 1) for k, v in dec.items()},
                      "decode_mtp": {k: round(v, 1) for k, v in mtp.items()},
                      "prefill": {k: round(v, 1) for k, v in h_pf.items()}, "prefill_stream_cap": round(cap, 1),
                      "routed_nonresident_share": round(share, 3), "q8_0_kv_gain": round(kv, 3)},
            "ratios": {r: {k: round(v, 3) for k, v in b.items()} for r, b in ratios.items()},
            "strata_phase_shares": {k: round(statistics.median(p.get(k, 0) for p in ph[c]) /
                                             statistics.median(sum(p.values()) for p in ph[c]), 4)
                                    for k in sorted({k for p in ph[c] for k in p})}}
    below = sorted({f"{c} {r}" for c, row in rows.items() for r, v in row["ratios"].items()
                    if v["central"] < (0.9 if r.endswith("prefill") else 1.0)})       # each cell's bar (acceptance 5)
    rec = {"name": "fn0_n3_roofline", "date": datetime.date.today().isoformat(), "gpu": a.card,
           "commit": run.git("rev-parse", "HEAD"), "command": f"python spikes/fn0/n3_roofline.py --card {a.card} ...",
           "method": __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0],
           "inputs": {"x_fit": {"f5a_fetch_gbps": round(pcie, 2), "overhead_ms_x": round(over_x, 3),
                                "overhead_ms_fn": round(over, 3)}, "card": card,
                      "cache_gb": {k: round(v, 3) for k, v in caches.items()},
                      "slots_per_layer": {k: round(v, 1) for k, v in slots.items()},
                      "lfu_hit_rate": {k: round(v, 4) for k, v in hits.items()},
                      "union_experts_per_layer_per_pass": round(union, 2), "non_lendable_peak_kb": a.peak_kb,
                      "phase_factors": F, "prefill_stream_gbps": PCIE_STREAM},
           "rows": rows, "bars": {"decode": 1.0, "prefill": 0.9},
           "rows_below_bar_central": below}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({c: {r: v["central"] for r, v in row["ratios"].items()} for c, row in rows.items()}, indent=1))
    print("below their bar (central):", below)


if __name__ == "__main__":
    main()
