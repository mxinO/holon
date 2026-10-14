"""Markdown tables from benchmark records (release design §5.4): the README's and the release record's numbers are
rendered from evidence, never copied by hand. Each table links its record; losses and failures stay in.

- `bars.py` records: per metric the gate statistic of Holon and of each llama.cpp build and their ratio; a ratio within
  2 sd of its per-round ratios of 1 is printed "≈"; one below the gate (0.8) is marked.
- `run.py` records, the short form (the record holds p99s, throughput and KV tokens): per record and backend,
  TTFT / TPOT p50 and the max inter-token gap's p99 per trace (with the paired TTFT ratio to the first backend
  where the record has it); a backend that failed is "—" with its reason, one that failed some rounds says so.

    python bench/report.py evidence/<record>.json ... [--relative-to docs]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics

GATE = 0.8


def _ratio(value: float, per_round: list[float], gated: bool) -> str:
    """The ratio, "≈" when within 2 sd of its per-round values of 1, bold when a gated one is under the gate."""
    text = f"{value:.2f}"
    if len(per_round) > 1 and abs(value - 1) < 2 * statistics.stdev(per_round):
        text = "≈ " + text
    return f"**{text} (< {GATE})**" if gated and value < GATE else text


def bars(records: list[tuple[str, dict]]) -> str:
    """One table per GPU class."""
    tables = []
    for cls in sorted({r["gpu_class"] for _, r in records}):
        group = [(link, r) for link, r in records if r["gpu_class"] == cls]
        builds = sorted({b for _, r in group for b in r["llama_cpp"]}, key=lambda b: b != "gate")
        commit = {b: next((r["llama_cpp"][b]["build_commit"] for _, r in group
                           if r["llama_cpp"][b].get("build_commit")), "?") for b in builds}
        head = ["file (record)", "lock", "metric", "Holon t/s"]
        for b in builds:
            head += [f"llama.cpp {b} ({commit[b]}) t/s", f"Holon / {b}"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for link, r in group:
            lock = r["lock"]["fingerprint_class"].split("/")[0] + (" (borrowed)" if r["lock"]["borrowed"] else "")
            for metric, res in r["results"].items():
                cells = [f"[{r['cell']}]({link})", lock, metric, _num(res["holon"]["statistic"])]
                for b in builds:
                    ratio = res.get(f"ratio_{b}")
                    cells += [_num(res[b]["statistic"]), "—" if ratio is None else
                              _ratio(ratio, res[f"ratio_{b}_per_round"], gated=b == "gate")]
                lines.append("| " + " | ".join(cells) + " |")
        gpu = group[0][1]["gpu"]
        ggml = ", ".join(sorted({r["ggml"][:8] for _, r in group}))
        tables.append(f"{gpu}; Holon's ggml at {ggml}\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


def serving(records: list[tuple[str, dict]]) -> str:
    """One table over the records: a row per (record, backend), a column per trace (TTFT / TPOT p50, max gap p99 ms;
    "×r": the paired TTFT ratio to the record's first backend)."""
    traces = list(dict.fromkeys(t for _, r in records for t in r["traces_sha256"]))
    lines = ["| record | model | GPU | backend | "
             + " | ".join(f"{t}: TTFT / TPOT p50, gap p99 ms" for t in traces) + " |",
             "|" + "---|" * (4 + len(traces))]
    for link, r in records:
        failed = {run["backend"]: run["error"].replace("|", "\\|") for run in r["runs"] if "error" in run}
        temperature = r.get("matched", {}).get("temperature")
        for b in r["order"]:
            done = len({run["round"] for run in r["runs"] if run["backend"] == b and "error" not in run})
            label = b + (f", T={temperature}" if temperature else "") + (
                f" ({done}/{r['rounds']} rounds)" if 0 < done < r["rounds"] else "")
            cells = []
            for t in traces:
                s = r["summary"].get(t, {}).get(b)
                if not s:
                    cells.append("—")
                    continue
                gap = s.get("max_gap_ms", {}).get("p99")
                paired = s.get("paired_ttft", {}).get("median")
                cells.append(f"{s['ttft_ms']['p50']} / {s['tpot_ms']['p50']}, {'—' if gap is None else gap}"
                             + ("" if paired is None else f" (×{paired:.2f})"))
            if not done:
                cells = [f"— ({failed.get(b, 'not run')})"] + [""] * (len(traces) - 1)
            lines.append(f"| [{r['workload']}]({link}) | {r['model']} | {r['gpu_class']} | {label} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines)


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--relative-to", default=".", help="directory of the document the links are written into")
    args = ap.parse_args()
    loaded = [(os.path.relpath(p, args.relative_to), json.loads(pathlib.Path(p).read_text())) for p in args.records]
    bar_records = [(link, r) for link, r in loaded if "cell" in r]
    parts = [bars(bar_records)] if bar_records else []
    run_records = [(link, r) for link, r in loaded if "workload" in r]
    parts += [serving(run_records)] if run_records else []
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
