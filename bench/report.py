"""Markdown tables from benchmark records (release design §5.4): the README's and the release record's numbers are
rendered from evidence, never copied by hand. Each table links its record; losses and failures stay in. Tables per
GPU class (absolute numbers are per board; ratios are the claim).

- `bars.py` records (`--tables bars`): per metric the gate statistic of Holon and of each llama.cpp build and their
  ratio; a ratio within 2 sd of its per-round ratios of 1 is printed "≈"; one below the gate (0.8) is marked.
- `run.py` records:
  - `ratios`: Holon served (`holon_http`, or an arm of it) against every baseline of its record, per trace — TTFT p50,
    TPOT p50, the max inter-token gap's p99 and output throughput, each as Holon's advantage (> 1: Holon faster; the
    baseline's latency over Holon's, Holon's throughput over the baseline's), "≈" as above from the per-round ratios;
    a baseline that failed is "—" with its reason.
  - `absolute`: per record, trace and backend, TTFT p50 / p99, TPOT p50 / p99, the max gap p50 / p99 / max (ms),
    output tok/s and the rounds that ran; KV tokens per record.
  - `spec`: the speculating backends (drafts in their counters), labelled with trace and sampling: TPOT p50, the
    speedup over the same engine plain (the same trace and sampling, its record first, else a record of the same
    model and class),
    accepted / drafted, tokens per verify row and the depth-1 share of verify steps (Holon's counters), the share of
    output tokens repeating their context.
  - `aa`: A/A arms (`holon_http@aa`, no flags) against `holon_http`: the per-round ratios and their mean and sd, the
    noise floor of the ratios above.

    python bench/report.py evidence/<record>.json ... [--relative-to docs] [--tables bars ratios absolute spec aa]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
from collections.abc import Callable

GATE = 0.8
SERVED = "holon_http"
# (column, value from a summary, higher is better)
METRICS: list[tuple[str, Callable[[dict], float | None], bool]] = [
    ("TTFT p50", lambda s: s["ttft_ms"]["p50"], False),
    ("TPOT p50", lambda s: s["tpot_ms"]["p50"], False),
    ("max gap p99", lambda s: s["max_gap_ms"]["p99"], False),
    ("out tok/s", lambda s: s["throughput_tok_s"], True),
]


def _ratio(value: float, per_round: list[float], gated: bool = False) -> str:
    """The ratio, "≈" when within 2 sd of its per-round values of 1, bold when a gated one is under the gate."""
    text = f"{value:.2f}"
    if len(per_round) > 1 and abs(value - 1) < 2 * statistics.stdev(per_round):
        text = "≈ " + text
    return f"**{text} (< {GATE})**" if gated and value < GATE else text


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def _cls(r: dict) -> str:
    return r["gpu_class"]


def _by_class(records: list[tuple[str, dict]]) -> list[tuple[str, list[tuple[str, dict]]]]:
    return [(c, [(link, r) for link, r in records if _cls(r) == c]) for c in sorted({_cls(r) for _, r in records})]


def bars(records: list[tuple[str, dict]]) -> str:
    """One table per GPU class."""
    tables = []
    for _, group in _by_class(records):
        builds = sorted({b for _, r in group for b in r["llama_cpp"]}, key=lambda b: b != "gate")
        commit = {b: next((r["llama_cpp"][b]["build_commit"] for _, r in group
                           if r["llama_cpp"].get(b, {}).get("build_commit")), "?") for b in builds}
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
                    cells += [_num(res.get(b, {}).get("statistic")), "—" if ratio is None else
                              _ratio(ratio, res[f"ratio_{b}_per_round"], gated=b == "gate")]
                lines.append("| " + " | ".join(cells) + " |")
        gpu = next((r["gpu"] for _, r in group if r.get("gpu")), group[0][1]["gpu_class"])
        ggml = ", ".join(sorted({r["ggml"][:8] for _, r in group}))
        tables.append(f"{gpu}; Holon's ggml at {ggml}\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


def _per_round(r: dict, backend: str, trace: str) -> dict[int, dict]:
    return {run["round"]: run["summary"] for run in r["runs"]
            if run["backend"] == backend and run.get("trace") == trace and "error" not in run}


def _failed(r: dict, backend: str) -> str | None:
    """A backend's failure reason if it never ran; else None."""
    if any(run["backend"] == backend and "error" not in run for run in r["runs"]):
        return None
    return next((run["error"] for run in r["runs"] if run["backend"] == backend), "not run").replace("|", "\\|")


def _advantage(r: dict, holon: str, base: str, trace: str) -> list[str]:
    """Holon's advantage over `base` per metric: pooled, with "≈" from the per-round ratios."""
    pooled = (r["summary"][trace][holon], r["summary"][trace][base])
    mine, theirs = _per_round(r, holon, trace), _per_round(r, base, trace)
    cells = []
    for _, get, higher in METRICS:
        h, b = get(pooled[0]), get(pooled[1])
        if not h or not b:
            cells.append("—")
            continue
        per_round = [(get(mine[k]) / get(theirs[k])) if higher else (get(theirs[k]) / get(mine[k]))
                     for k in mine.keys() & theirs.keys() if get(mine[k]) and get(theirs[k])]
        cells.append(_ratio(h / b if higher else b / h, per_round))
    return cells


def _holon_served(r: dict) -> str | None:
    return next((b for b in r["order"] if b.partition("@")[0] == SERVED and not b.endswith("@aa")), None)


def _label(r: dict) -> str:
    t = r.get("matched", {}).get("temperature")
    corpus = ", ".join(r.get("corpus", {})) or "uniform ids"
    return f"{'T=' + str(t) if t else 'greedy'}, {corpus}"


def ratios(records: list[tuple[str, dict]]) -> str:
    tables = []
    for cls, group in _by_class(records):
        head = ["record", "model", "trace (sampling, prompts)", "Holon served vs", *(m[0] for m in METRICS)]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for link, r in group:
            holon = _holon_served(r)
            if holon is None:
                continue
            bases = [b for b in r["order"] if not b.startswith("holon")]
            for trace in r["traces_sha256"]:
                for b in bases:
                    lead = [f"[{r['workload']}]({link})", r["model"], f"{trace} ({_label(r)})", b]
                    why = _failed(r, b) or _failed(r, holon)
                    if why or b not in r["summary"].get(trace, {}) or holon not in r["summary"].get(trace, {}):
                        cells = [f"— ({why or 'no result on this trace'})"] + [""] * (len(METRICS) - 1)
                    else:
                        cells = _advantage(r, holon, b, trace)
                    lines.append("| " + " | ".join(lead + cells) + " |")
        if len(lines) > 2:
            tables.append(f"{cls}: Holon's advantage (> 1: Holon faster or higher)\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


def absolute(records: list[tuple[str, dict]]) -> str:
    tables = []
    for cls, group in _by_class(records):
        head = ["record", "trace", "backend", "TTFT p50 / p99 ms", "TPOT p50 / p99 ms", "max gap p50 / p99 / max ms",
                "out tok/s", "rounds"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for link, r in group:
            kv = r.get("matched", {}).get("kv_tokens")
            name = f"[{r['workload']}]({link})" + (f" (KV {kv} tokens)" if kv else "")
            for trace in r["traces_sha256"]:
                for b in r["order"]:
                    s = r["summary"].get(trace, {}).get(b)
                    if not s:
                        lines.append(f"| {name} | {trace} | {b} | — ({_failed(r, b) or 'no result'}) |  |  |  |  |")
                        continue
                    t, p, g = s["ttft_ms"], s["tpot_ms"], s["max_gap_ms"]
                    done = len(_per_round(r, b, trace))
                    lines.append(f"| {name} | {trace} | {b} | {t['p50']} / {t['p99']} | {p['p50']} / {p['p99']} | "
                                 f"{g['p50']} / {g['p99']} / {g['max']} | {s['throughput_tok_s']} | "
                                 f"{done}/{r['rounds']} |")
        tables.append(f"{cls}\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


def _spec(stats: dict) -> tuple[str, str, str]:
    """Accepted / drafted, tokens per verify row and the depth-1 share of verify steps ("—" where not counted)."""
    drafted = stats.get("spec_drafted", 0)
    rows = {k: n for k, n in stats.items() if k.startswith("spec_rows_k")}
    accepted = stats.get("spec_accepted", sum(int(k.rsplit("_", 1)[1]) * n for k, n in rows.items()))
    per_row = sum((int(k.rsplit("_", 1)[1]) + 1) * n for k, n in rows.items()) / sum(rows.values()) if rows else None
    steps = stats.get("spec_verify_steps")
    return (f"{accepted / drafted:.3f}" if drafted else "—", "—" if per_row is None else f"{per_row:.2f}",
            f"{stats.get('spec_verify_steps_overlapped', 0) / steps:.3f}" if steps else "—")


def spec(records: list[tuple[str, dict]]) -> str:
    tables = []
    for cls, group in _by_class(records):
        head = ["record", "trace (sampling, prompts)", "backend", "TPOT p50 ms", "speedup over its plain (TPOT p50)",
                "accepted / drafted", "tokens per verify row", "depth-1 share", "repeated output"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for link, r in group:
            for trace, by_backend in r["summary"].items():
                for b, s in by_backend.items():
                    if not s["stats"].get("spec_drafted"):
                        continue
                    base = b.partition("@")[0]           # the same engine without drafts, this record first
                    plain = next((p["summary"][trace][base] for _, p in sorted(group, key=lambda g: g[1] is not r)
                                  if p["model"] == r["model"] and _label(p) == _label(r)
                                  and base in p["summary"].get(trace, {})
                                  and not p["summary"][trace][base]["stats"].get("spec_drafted")), None)
                    tpot = s["tpot_ms"]["p50"]
                    speedup = f"{plain['tpot_ms']['p50'] / tpot:.2f}" if plain else "—"
                    rep = s.get("repeated_share")
                    lines.append(f"| [{r['workload']}]({link}) | {trace} ({_label(r)}) | {b} | {tpot} | {speedup} | "
                                 + " | ".join(_spec(s["stats"])) + f" | {'—' if rep is None else f'{rep:.3f}'} |")
        if len(lines) > 2:
            tables.append(f"{cls}: speculation\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


def aa(records: list[tuple[str, dict]]) -> str:
    head = ["record", "GPU", "trace", *(f"{m[0]}: rounds; mean ± sd" for m in METRICS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for link, r in records:
        if f"{SERVED}@aa" not in r["order"]:
            continue
        for trace in r["traces_sha256"]:
            mine, ref = _per_round(r, f"{SERVED}@aa", trace), _per_round(r, SERVED, trace)
            cells = []
            for _, get, _higher in METRICS:
                xs = [get(mine[k]) / get(ref[k]) for k in sorted(mine.keys() & ref.keys())
                      if get(mine[k]) and get(ref[k])]
                spread = f"; {statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else ""
                cells.append(", ".join(f"{x:.3f}" for x in xs) + spread)
            lines.append(f"| [{r['workload']}]({link}) | {r['gpu_class']} | {trace} | " + " | ".join(cells) + " |")
    return "\n".join(lines) if len(lines) > 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--relative-to", default=".", help="directory of the document the links are written into")
    ap.add_argument("--tables", nargs="+", default=["bars", "ratios", "spec", "aa", "absolute"],
                    choices=["bars", "ratios", "absolute", "spec", "aa"])
    args = ap.parse_args()
    loaded = [(os.path.relpath(p, args.relative_to), json.loads(pathlib.Path(p).read_text())) for p in args.records]
    bar_records = [(link, r) for link, r in loaded if "cell" in r]
    run_records = [(link, r) for link, r in loaded if "workload" in r]
    render = {"bars": lambda: bars(bar_records) if bar_records else "", "ratios": lambda: ratios(run_records),
              "absolute": lambda: absolute(run_records), "spec": lambda: spec(run_records),
              "aa": lambda: aa(run_records)}
    print("\n\n".join(filter(None, (render[t]() if (bar_records if t == "bars" else run_records) else ""
                                    for t in args.tables))))


if __name__ == "__main__":
    main()
