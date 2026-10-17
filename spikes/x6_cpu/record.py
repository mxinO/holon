"""The evidence record of steps 6 and 5 (docs/v1-offload.md §14) from the runs' outputs:

    python spikes/x6_cpu/record.py <runs dir> <out.json> <commit>

`<runs dir>` holds `tune_split_25.json` (tune --cpu-split), `step6/` (snapshots `snap/`, the L2 judgements `l2.json`,
`l2_plant.json`, the floors' `plain_floor_vs_golden.json`, `split_floor_vs_golden.json`, the plant's log) and
`step5/` (snapshots `snap/`, `decode.json`). Snapshots are compared with `holon_judge.snapshot.diff`."""
import json
import pathlib
import re
import sys

import torch
from holon_judge.snapshot import Snapshot, diff


def _load(path: pathlib.Path):
    """A JSON output, absolute paths reduced to their file names (records carry no host path)."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return pathlib.PurePath(x).name if isinstance(x, str) and x.startswith("/") else x
    return clean(json.loads(path.read_text())) if path.is_file() else None


def _snapshots(snap: pathlib.Path, reference: Snapshot) -> dict:
    out = {}
    for meta in sorted(snap.glob("*.meta.json")):
        m = json.loads(meta.read_text())
        run = snap / meta.name.replace(".meta.json", ".json")
        s = Snapshot.load(run) if run.is_file() else None
        out[meta.name.removesuffix(".meta.json")] = m | {"vs_resident": "identical" if s and not diff(reference, s)
                                                          else (diff(reference, s) if s else None)}
    return out


def _l2(path: pathlib.Path) -> dict | None:
    r = _load(path)
    return None if r is None else {k: r.get(k) for k in ("overall", "floor", "budgets", "failures", "passed")}


def main() -> None:
    runs, out, commit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    s6, s5 = runs / "step6", runs / "step5"
    reference = Snapshot.load(s6 / "snap" / "resident_0.json")
    snaps6 = _snapshots(s6 / "snap", reference)
    f5b = sorted(k for k in snaps6 if k.startswith("f5b_"))
    plant = sorted(k for k in snaps6 if k.startswith("plant_wait"))
    first = Snapshot.load(s6 / "snap" / f"{f5b[0]}.json")
    identical = {k: not diff(first, Snapshot.load(s6 / "snap" / f"{k}.json")) for k in f5b + plant}
    fired = re.findall(r"plant cpu-wait-skipped fired: (\{.*\})", (s6 / "snap.log").read_text())
    record = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "commit": commit,
        "step6": {
            "tune_cpu_split_25pct_batch1": _load(runs / "tune_split_25.json"),
            "snapshots": snaps6, "f5b_run_identical_to_run0": identical,
            "plant_cpu_wait_skipped": {"fired": fired, "run_identical": not diff(
                Snapshot.load(s6 / "snap" / f"{plant[0]}.json"), Snapshot.load(s6 / "snap" / f"{plant[1]}.json"))
                if len(plant) > 1 else None},
            "tune_first_run_without_share_1": _load(runs / "tune_split_25_first.json"),
            "decode_ab_share_half_4_threads": _load(runs / "smoke.json"),
            "l2_f5b": _l2(s6 / "l2.json"), "l2_plant_cpu_gate_up_order": _l2(s6 / "l2_plant.json"),
            "floors_vs_golden": {"plain_mmvq": _l2(s6 / "plain_floor_vs_golden.json"),
                                 "path_aware_split": _l2(s6 / "split_floor_vs_golden.json")},
        },
        "step5": {
            "snapshots": _snapshots(s5 / "snap", reference) if (s5 / "snap").is_dir() else None,
            "decode": _load(s5 / "decode.json"),
            "decode_128_tokens_superseded": _load(s5 / "decode_128.json"),
        },
        "commands": {
            "tune": "python -m holon_tools tune <gguf> --cpu-split --no-lock --batch 1 --tokens 128 --ops rmsnorm "
                    "--max-seqs 1 --max-model-len 4096 --host-memory 21474836480 --loop-core 2 --plugin-options "
                    "moe.store=std.moe.offload@1 moe.cache=*=0.25 cpu_task.cores=4-9",
            "snapshots": "python spikes/x6_cpu/run.py <gguf> <dir> <name|runs|plant|options> ...",
            "l2": "python spikes/x6_cpu/l2.py <gguf> cand split.json <options>; holon_judge golden --q8-1 mmvq "
                  "--cpu-split split.json ...; holon_judge compare --golden v1-fp32 --candidate cand --floor floor",
            "decode": "python spikes/x6_cpu/decode.py <gguf> out.json <arm|options> ... --rounds 3 --tokens 512",
        },
    }
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
