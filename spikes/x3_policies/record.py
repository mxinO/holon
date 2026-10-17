"""The evidence record of `run.py`'s runs: each run's summary, its snapshot against all-resident's (`resident_0`;
with the n-gram drafter `resident_ngram_0`) and against its own configuration's first run (run-identical), by
`holon_judge snapshot.diff`, and the plants' firing reports from the runs' logs:

    python spikes/x3_policies/record.py <runs dir> <evidence.json> <commit> <command> ...
"""
import json
import pathlib
import sys

from holon_judge.snapshot import Snapshot, diff


def main() -> None:
    runs, out, commit, commands = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    fired = [line.strip() for log in sorted(runs.glob("*.log")) for line in log.read_text().splitlines()
             if line.startswith("plant ")]
    rows = []
    for meta_path in sorted(runs.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        snap = runs / f"{meta['name']}_{meta['run']}.json"
        reference = runs / ("resident_ngram_0.json" if meta["spec"] else "resident_0.json")
        if snap.is_file() and not meta["name"].startswith("resident"):
            mine = Snapshot.load(snap)
            meta["vs_resident"] = diff(Snapshot.load(reference), mine) or "identical"
            if meta["run"]:
                meta["vs_run_0"] = diff(Snapshot.load(runs / f"{meta['name']}_0.json"), mine) or "identical"
        rows.append(meta)
    record = {"gpu": "rtx-a5000", "model": "unsloth/Qwen3-30B-A3B-GGUF Q4_K_M", "commit": commit,
              "commands": commands, "runs": rows, "plants_fired": fired}
    out.write_text(json.dumps(record, indent=1) + "\n")
    for r in rows:
        print(r["name"], r["run"], r.get("vs_resident"), r.get("vs_run_0", ""), r.get("error", "")[:120])


if __name__ == "__main__":
    main()
