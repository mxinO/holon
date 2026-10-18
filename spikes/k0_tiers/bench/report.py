"""Markdown tables of K's step-0 records (README's results):

    python report.py <k0_m2_copy_*.json> <k0_m4_nvme_*.json> …
"""
import json
import sys
from pathlib import Path


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def m2(tag: str, r: dict) -> str:
    m = r["m2"]
    blob = m["blob"]
    parts = [f"#### K-M2 on the {tag}", "",
             f"Blob (one copy, S2 host memory): H2D {f(blob['h2d_gbps'])} GB/s, D2H {f(blob['d2h_gbps'])} GB/s; "
             f"torch's pinned allocator: " + ", ".join(f"{x['direction']} {f(x['gbps'])}" for x in blob["rows"]
                                                     if x["alloc"] == "torch_pinned")
             + "; SM copy of one 1 GiB blob: " + ", ".join(f"{x['direction']} {f(x['gbps'])} ({x['config']})"
                                                         for x in blob["rows"] if x.get("path") == "sm")
             + f"; page-locking {f(blob['pin_s_per_gib'], 2)} s per GiB.", ""]
    rows = []
    for name, s in m["shapes"].items():
        for x in s["rows"]:
            if x["frag"] != "random":
                continue
            b = blob[f"{x['direction']}_gbps"]
            loop = x.get("loop")
            rows.append([name, x["layout"], x["direction"], x["load"], x["chunks"],
                         f"{f(loop['gbps'])} ({f(loop['gbps'] / b, 2)}×; {f(loop['host_ms'])} ms, "
                         f"{f(loop['us_per_chunk'], 2)} µs/chunk)" if loop else "n/a",
                         f"{f(x['batch']['gbps'])} ({f(x['batch']['gbps'] / b, 2)}×; "
                         f"{f(x['batch']['host_ms'], 2)} + {f(x['batch']['build_ms'], 2)} ms)",
                         f"{f(x['sm']['gbps'])} ({f(x['sm']['gbps'] / b, 2)}×; {x['sm']['config']}, "
                         f"≥95% from {x.get('sm_saturating_ctas', '–')} CTAs)",
                         all(x[k]["verified"] for k in ("loop", "batch", "sm") if k in x)])
    parts += ["Fragmented (every unit its own run), GB/s (× blob; host time of one call):", "",
              table(["shape", "host layout", "dir", "load", "chunks", "per-run loop (submit)",
                     "BatchAsync (submit + build)", "SM copy (best config)", "verified"], rows), ""]
    rows = []
    for name, s in m["shapes"].items():
        for x in s["rows"]:
            if x["frag"] == "random" or x["load"] != "32K" and not (name == "gdn_ssm_27b" and x["load"] == "16K"):
                continue
            rows.append([name, x["layout"], x["direction"], x["frag"], x["load"], x["chunks"],
                         f(x["loop"]["gbps"]) if "loop" in x else "n/a", f(x["batch"]["gbps"]), f(x["sm"]["gbps"])])
    parts += ["Runs of 4 / 16 units consecutive in both pools, GB/s:", "",
              table(["shape", "host layout", "dir", "runs", "load", "chunks", "loop", "batch", "SM"], rows), ""]
    return "\n".join(parts)


def m3(tag: str, r: dict) -> str:
    m = r["m3"]
    c = m["arms"]["control"]["step_ms"]
    rows = [["control", f(c["p50"], 2), f(c["p90"], 2), f(c["mean"], 2), "1", "1", "1", "–", "–"]]
    for arm, x in m["arms"].items():
        if arm == "control":
            continue
        s = x["step_ms"]
        rows.append([arm, f(s["p50"], 2), f(s["p90"], 2), f(s["mean"], 2), f(x["median_ratio"], 3),
                     f(x["p90_ratio"], 3), f(x["mean_ratio"], 3),
                     " / ".join(f(v, 3) for v in x["round_mean_ratios"]), f(x.get("copy_gbps"))])
    probe = m["small_copy_probe_ms"]
    prow = [[k, *(f"{f(v[s]['p50'], 3)} / {f(v[s]['p90'], 3)}" if s in v else "–"
                  for s in ("4096", "65536", "1048576")), f(v["feeder"]["gbps"]) if "feeder" in v else "–"]
            for k, v in probe.items()]
    screen = r.get("clock_screen", {})
    return "\n".join([
        f"#### K-M3 on the {tag}: {m['model']}, lock `{Path(m['lock']).name}`, 4 rows from {m['ctx_start']} to "
        f"{m['ctx_end']} tokens, {m['rounds']} ABBA rounds × {m['steps_per_segment']} steps per segment; clock "
        f"screen steady: {screen.get('steady')}, during the arms: {m['clock_steady']}", "",
        table(["arm", "step p50 ms", "p90", "mean", "median ×", "p90 ×", "mean ×", "mean × per round",
               "copies GB/s"], rows), "",
        "A small copy on another stream while the arm's copies run (p50 / p90 ms), and the copies' rate alone:", "",
        table(["copies in flight", "4 KiB", "64 KiB", "1 MiB", "GB/s alone"], prow), ""])


def m4(tag: str, r: dict) -> str:
    m = r["m4"]
    rows = [[x["kind"], x["mode"], x["shape"], x["threads"], f(x["gbps"], 2), f(x["lat_p50_ms"], 2),
             f(x["lat_p99_ms"], 2)] for x in m["rows"]]
    parts = [f"#### K-M4 on the {tag} ({m['storage']})", "",
             f"Fill of the read file {f(m['fill_gbps'], 2)} GB/s; O_DIRECT probe: {m['alignment']['direct_probe']}",
             "", table(["kind", "mode", "shape", "threads", "GB/s", "p50 ms", "p99 ms"], rows), "",
             "C reference (no GIL): " + ", ".join(f"{x['kind']} ×{x['threads']} {f(x['gbps'], 2)} GB/s"
                                                  for x in m["c_reference"]), ""]
    rows = [[x.get("threads_each", 2), "free-running" if x["read_share_target"] is None else x["read_share_target"],
             f(x["read_share"], 2), f(x["read"]["gbps"], 2), f(x["write"]["gbps"], 2), f(x["measured_gbps"], 2),
             f(x["model_gbps"], 2), f(x["measured_over_model"], 3),
             " / ".join(f(v, 2) for v in x["model_inputs"].values())] for x in m["mixed"]]
    parts += ["Mixed reads and writes (O_DIRECT 557,056 B units):", "",
              table(["threads per kind", "read share held", "measured share", "reads GB/s", "writes GB/s",
                     "measured GB/s", "modelled GB/s", "measured / model", "pure read / write GB/s"], rows), ""]
    rows = [[x["mode"], x["io_threads"], x["io_kind"], f(x["alone_ms"]["p50"], 2), f(x["body_ms"]["p50"], 2),
             f(100 * x["slowdown_p50"], 1), f(100 * x["slowdown_mean"], 1), f(x["io_gbps_during"], 2)]
            for x in m["gil"]["rows"]]
    parts += ["Engine-loop stand-in beside the I/O threads:", "",
              table(["loop", "I/O threads", "kind", "body alone p50 ms", "beside p50 ms", "p50 %", "mean %",
                     "I/O GB/s during"], rows), ""]
    s = m["sustained"]
    if "skipped" in s:
        parts.append(f"Sustained writes: skipped ({s['skipped']}).")
    else:
        parts.append(f"Sustained writes ({s['threads']} threads, O_DIRECT 557,056 B): {f(s['written_gb'], 0)} GB in "
                     f"{f(s['seconds'], 0)} s, mean {f(s['mean_gbps'], 2)} GB/s; start {f(s['start_gbps'], 2)} GB/s; "
                     f"cliff at {f(s['cliff_at_gb'], 0)} GB; after it {f(s['after_cliff_gbps'], 2)} GB/s.")
    parts += ["", f"Test files removed: {m['test_file_removed']}.", ""]
    return "\n".join(parts)


def main():
    for path in sys.argv[1:]:
        r = json.loads(Path(path).read_text())
        tag = r["tag"]
        if "m2" in r:
            print(m2(tag, r))
        if "m3" in r:
            print(m3(tag, r))
        if "m4" in r:
            print(m4(tag, r))
        if "verdicts" in r:
            print(f"Verdicts ({tag}): `{json.dumps(r['verdicts'], default=float)}`\n")


if __name__ == "__main__":
    main()
