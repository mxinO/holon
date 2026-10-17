"""x4_gap's evidence records: the runs' medians, the nsys breakdowns (recomputed from the SQLite exports), the policy
replays, and the derived gap tables (README). One record per GPU class.

    python spikes/x4_gap/record.py --runs <dir with q1 q1nsys q2 q2nsys q3 q3nsys q3b q3bnsys, replay.json> \\
        --date 2026-10-17 --commit <measured commit>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import statistics

import nsys_breakdown as nb

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf"
EXPERT_MB = 2.857            # mean bytes of one expert (gate + up + down), M6 (the file's tensor table)
LAYERS = 48
ENGINES = {"llama.cpp": "f74e2fa6a31c (llama-server, --moe-cache-mib)", "FreeToken": "21d9578bde22",
           "Holon": "the v1-offload branch at b492595 (step 2: std.moe.offload@1, static placement, F5a)"}


def rows(d: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in (d / "raw.jsonl").read_text().splitlines()]


def scrub(arm: dict) -> dict:
    """An arm as run, without the local report paths of its nsys wrap."""
    return {k: re.sub(r"-o \S+", "-o <report>", v) if k in ("wrap", "server_wrap") else v for k, v in arm.items()}


def medians(rs: list[dict]) -> dict:
    out: dict = {}
    for r in rs:
        for c, m in r.get("results", {}).items():
            for k, v in m.items():
                out.setdefault(r["arm"], {}).setdefault(c, {}).setdefault(k, []).append(v)
    return {a: {c: {k: {"median": round(statistics.median(v), 4), "min": min(v), "max": max(v), "n": len(v)}
                    for k, v in m.items()} for c, m in cs.items()} for a, cs in out.items()}


def breakdown(path: pathlib.Path, window: str = "nvtx", prompt: int = 0) -> dict:
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    w = nb.windows(db, window, strings)
    tokens = 255 if window == "nvtx" else int(window.split(":")[1])
    out = {"decode": nb.breakdown(db, w["decode"], tokens, strings, 15)}
    if prompt and "prefill" in w:
        out["prefill"] = nb.breakdown(db, w["prefill"], prompt, strings, 15)
        out["prefill"]["window_ms"] = round((w["prefill"][1] - w["prefill"][0]) / 1e6, 1)
    return out


def us(b: dict, cat: str) -> float:
    return b["categories"].get(cat, {}).get("us_per_token", 0.0)


def gap(holon: dict, ref: dict) -> list[dict]:
    """Holon − reference, per category (µs per token, profiled), plus the overlap and idle terms: window = Σ categories
    − overlap + idle."""
    cats = sorted(set(holon["categories"]) | set(ref["categories"]), key=lambda c: -abs(us(holon, c) - us(ref, c)))
    out = [{"component": c, "holon_us": us(holon, c), "ref_us": us(ref, c),
            "delta_us": round(us(holon, c) - us(ref, c), 1)} for c in cats]
    for b in (holon, ref):
        b["overlap_us"] = round(sum(v["us_per_token"] for v in b["categories"].values()) - b["busy_us_per_token"], 1)
    out.append({"component": "overlap (concurrent streams, subtracted)", "holon_us": -holon["overlap_us"],
                "ref_us": -ref["overlap_us"], "delta_us": round(ref["overlap_us"] - holon["overlap_us"], 1)})
    out.append({"component": "idle (no kernel or copy on the GPU)", "holon_us": holon["idle_us_per_token"],
                "ref_us": ref["idle_us_per_token"],
                "delta_us": round(holon["idle_us_per_token"] - ref["idle_us_per_token"], 1)})
    return sorted(out, key=lambda r: -abs(r["delta_us"]))


def rtx5090(runs: pathlib.Path) -> dict:
    med = medians(rows(runs / "q1"))
    nsys = {a: breakdown(runs / "q1" / f"nsys_{a}.sqlite", "logits:200" if a.startswith("llama") else "nvtx",
                         0 if a.startswith("llama") else 1024)
            for a in ("holon_res", "llama_res", "ft_fused", "holon_off100", "ft_off100")}
    tps = {a: {c: med[a][c]["decode_tps"]["median"] for c in med[a]} for a in med}
    ratios = {c: {"holon/llama-server": round(tps["holon_res"][c] / tps["llama_res"][c], 3),
                  "holon/freetoken-fused": round(tps["holon_res"][c] / tps["ft_fused"][c], 3),
                  "holon offload store, all cached − resident (ms per token)":
                      round(1e3 / tps["holon_off100"][c] - 1e3 / tps["holon_res"][c], 3),
                  "freetoken offload, rate 1.0 − fused (ms per token)":
                      round(1e3 / tps["ft_off100"][c] - 1e3 / tps["ft_fused"][c], 3)} for c in ("1024", "4096")}
    return {"question": "1: all-resident decode, batch 1", "summary": med, "ratios": ratios,
            "nsys_1k": nsys, "gap_1k_vs_llama": gap(nsys["holon_res"]["decode"], nsys["llama_res"]["decode"]),
            "gap_1k_vs_freetoken": gap(nsys["holon_res"]["decode"], nsys["ft_fused"]["decode"]),
            "arms": [scrub(a) for a in json.loads((runs / "q1" / "arms.json").read_text())],
            "nsys_arms": [scrub(a) for a in json.loads((runs / "q1" / "arms_nsys.json").read_text())]}


def rtx5080(runs: pathlib.Path) -> dict:
    med = medians(rows(runs / "q2") + rows(runs / "q3") + rows(runs / "q3b"))
    replay = json.loads((runs / "replay.json").read_text())
    nsys = {a: breakdown(runs / "q2" / f"nsys_{a}.sqlite", prompt=4096)
            for a in ("holon_static", "holon_prof", "ft_nat")}
    for q, a in (("q3", "holon_prof_c4096"), ("q3b", "holon_prof_c4096_k2048")):
        nsys[a] = breakdown(runs / q / f"nsys_{a}.sqlite", prompt=4096)

    def m(arm, c, k):
        return med[arm][c][k]["median"]

    fit, at = {}, {}
    for c in ("1024", "4096"):
        # Holon's decode time against its miss bytes: two measured placements of one cache size (static, profiled)
        (x0, y0), (x1, y1) = [(m(a, c, "miss_mb_per_token"), 1e3 / m(a, c, "decode_tps")) for a in ("holon_static",
                                                                                                   "holon_prof")]
        slope = (y1 - y0) / (x1 - x0)
        base = y1 - slope * x1
        ft_mb = m("ft_nat", c, "ft_missing_per_layer") * LAYERS * EXPERT_MB
        ft_cap_mb = m("ft_cap", c, "ft_missing_per_layer") * LAYERS * EXPERT_MB
        fit[c] = {"ms_per_token_at_0_mb": round(base, 3), "ms_per_mb": round(slope, 5),
                  "effective_gather_gb_s": round(1 / slope, 1), "freetoken_miss_mb_per_token": round(ft_mb, 1),
                  "freetoken_capped_miss_mb_per_token": round(ft_cap_mb, 1)}
        rows_ = {"freetoken's measured misses": ft_mb}
        for p in ("lru", "lfu", "belady"):
            rows_[f"{p} replay (profile seed)"] = replay[f"r0_holon_prof@{c}"]["policies"][p]["fetch_mb_per_token"]
        rows_["static, profile seed (measured)"] = m("holon_prof", c, "miss_mb_per_token")
        rows_["static, id-order seed (as built, measured)"] = m("holon_static", c, "miss_mb_per_token")
        ref = {"freetoken": m("ft_nat", c, "decode_tps"), "llama.cpp pinned cache": m("llama_cache_pinned", c,
                                                                                     "decode_tps")}
        at[c] = {k: {"miss_mb_per_token": round(v, 1), "tps": round(1e3 / (base + slope * v), 1),
                     **{f"vs {r}": round(1e3 / (base + slope * v) / t, 3) for r, t in ref.items()}}
                 for k, v in rows_.items()}
        cap = {"freetoken's measured misses (25%)": ft_cap_mb} | {
            f"{p} replay (25%, profile seed)": replay[f"r0_holon_prof25@{c}"]["policies"][p]["fetch_mb_per_token"]
            for p in ("lru", "lfu", "belady")}
        tps = {k: 1e3 / (base + slope * v) for k, v in cap.items()}
        at[c + " capped"] = {k: {"miss_mb_per_token": round(v, 1), "tps": round(tps[k], 1),
                                 "vs freetoken (25%)": round(tps[k] / m("ft_cap", c, "decode_tps"), 3)}
                             for k, v in cap.items()}
    fetch = {a: round(us(nsys[a]["decode"], "fetch (expert gather / copy)") / mb, 2) for a, mb in
             (("holon_static", m("holon_static", "4096", "miss_mb_per_token")),
              ("holon_prof", m("holon_prof", "4096", "miss_mb_per_token")),
              ("ft_nat", fit["4096"]["freetoken_miss_mb_per_token"]))}
    return {"question": "2: offload decode at the natural fit (47%); 3: prefill at 4K", "summary": med,
            "replay": replay, "decode_fit": fit, "holon_at_other_miss_bytes": at,
            "fetch_us_per_mb_4k": fetch, "nsys_4k": nsys,
            "gap_4k_decode_static_vs_freetoken": gap(nsys["holon_static"]["decode"], nsys["ft_nat"]["decode"]),
            "gap_4k_decode_profiled_vs_freetoken": gap(nsys["holon_prof"]["decode"], nsys["ft_nat"]["decode"]),
            "arms": [scrub(a) for q in ("q2", "q3", "q3b") for a in json.loads((runs / q / "arms.json").read_text())],
            "nsys_arms": [scrub(a) for q in ("q2", "q3", "q3b")
                          for a in json.loads((runs / q / "arms_nsys.json").read_text())]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    runs = pathlib.Path(a.runs)
    common = {"date": a.date, "model": MODEL, "commit": a.commit, "engines": ENGINES,
              "method": "spikes/x4_gap/README.md: bench/offload_ctx.py's prompts and timing (fresh process per arm, "
                        "512 + 64 warm-up, one request per context generating 256 tokens, greedy, ignore_eos, "
                        "a 33,792-token context allocated), interleaved rounds, medians; nsys 2025.3 with "
                        "--cuda-graph-trace=node on one extra run per arm",
              "commands": ["python spikes/x4_gap/ctx.py run --name <q> --arms <arms.json> --rounds <n> "
                           "--model <gguf> --contexts 1024 4096 --ctx 33792",
                           "nsys export --type sqlite <report>; python spikes/x4_gap/nsys_breakdown.py <sqlite>",
                           "python spikes/x4_gap/routing_profile.py <profile.json> <M6 agent.npz> <M6 mixed.npz>",
                           "python spikes/x4_gap/replay.py <holon trace.npz> ...",
                           "python spikes/x4_gap/record.py --runs <dir> --date <date> --commit <commit>"]}
    for gpu, fn in (("rtx-5090", rtx5090), ("rtx-5080", rtx5080)):
        record = {"name": "x4_gap", "gpu": gpu} | common | fn(runs)
        out = ROOT / "evidence" / f"x4_gap_{gpu}_{a.date}.json"
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(out)


if __name__ == "__main__":
    main()
