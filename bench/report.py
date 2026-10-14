"""Markdown tables from benchmark records (release design §5.4): the README's and the release record's numbers are
rendered from evidence, never copied by hand. Each table links its record; losses and failures stay in.

- `bars.py` records: per metric the gate statistic of Holon and of each llama.cpp build and their ratio; a ratio within
  2 sd of its per-round ratios of 1 is printed "≈"; one below the gate (0.8) is marked.
- `run.py` records: per trace and backend TTFT / TPOT p50 / p99 and output tok/s; a backend that failed is "—" with
  its error.

    python bench/report.py evidence/<record>.json ... [--relative-to docs]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics

GATE = 0.8


def _ratio(holon: list[float], base: list[float], value: float) -> str:
    """The ratio, "≈" when within 2 sd of its per-round values of 1, bold when under the gate."""
    per_round = [h / b for h, b in zip(holon, base, strict=False)]
    text = f"{value:.2f}"
    if len(per_round) > 1 and abs(value - 1) < 2 * statistics.stdev(per_round):
        text = "≈ " + text
    return f"**{text} (< {GATE})**" if value < GATE else text


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
                              _ratio(res["holon"]["per_round"], res[b]["per_round"], ratio)]
                lines.append("| " + " | ".join(cells) + " |")
        gpu = group[0][1]["gpu"]
        ggml = ", ".join(sorted({r["ggml"][:8] for _, r in group}))
        tables.append(f"{gpu}; Holon's ggml at {ggml}\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


def serving(link: str, r: dict) -> str:
    failed = {run["backend"]: run["error"] for run in r["runs"] if "error" in run}
    lines = [f"[{r['workload']}]({link}), {r['model']}, {r['gpu_class']}, {r['rounds']} round(s)", "",
             "| trace | backend | TTFT p50 / p99 ms | TPOT p50 / p99 ms | out tok/s |", "|---|---|---|---|---|"]
    for trace, by_backend in r["summary"].items():
        for b in r["order"]:
            if b in by_backend:
                s = by_backend[b]
                lines.append(f"| {trace} | {b} | {s['ttft_ms']['p50']} / {s['ttft_ms']['p99']} | "
                             f"{s['tpot_ms']['p50']} / {s['tpot_ms']['p99']} | {s['throughput_tok_s']} |")
            else:
                lines.append(f"| {trace} | {b} | — | — | — ({failed.get(b, 'not run')}) |")
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
    parts += [serving(link, r) for link, r in loaded if "workload" in r]
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
