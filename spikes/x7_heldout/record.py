"""Step 7's evidence record from the runs' outputs (docs/v1-offload.md acceptance 1):

    python spikes/x7_heldout/record.py <snapshot dir> <a1 dir> <page-cache dir> <out.json> <commit>

<snapshot dir>: run.py's metas and snapshots (F10 at 25% ×3, its decode-trace parity, the page-cache tier RAM-capped);
<a1 dir>: `compare.txt` (holon_judge compare of A1's L2 candidate) and `decode_log.txt` (spikes/x6_cpu/decode.py's
lines); <page-cache dir>: `decode_log.txt` (gguf_direct against the page-cache tier, RAM-capped)."""
import json
import pathlib
import statistics
import sys

import torch
from holon_judge import snapshot


def arms(path: pathlib.Path) -> dict:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            (name, r), = json.loads(line).items()
            out.setdefault(name.split("#")[0], []).extend(r["tok_s"])
    return {k: {"tok_s": v, "median": statistics.median(v)} for k, v in out.items()}


def main() -> None:
    snaps, a1, pc, out = map(pathlib.Path, sys.argv[1:5])
    commit = sys.argv[5]
    metas = {p.name.removesuffix(".meta.json"): json.loads(p.read_text()) for p in sorted(snaps.glob("*.meta.json"))}
    for name in metas:                                          # identity against all-resident, re-checked here
        ref = "resident_0.json" if metas[name]["trace"] == "snapshot" else "resident-parity_0.json"
        if not name.startswith("resident") and (snaps / f"{name}.json").exists():
            metas[name]["diff"] = snapshot.diff(snapshot.Snapshot.load(snaps / ref),
                                                snapshot.Snapshot.load(snaps / f"{name}.json"))
    record = {"what": "docs/v1-offload.md step 7: held-out plugins (acceptance 1)", "gpu": "RTX A5000",
              "gpu_class": "rtx-a5000", "model": "Qwen3-30B-A3B Q4_K_M (unsloth GGUF)", "commit": commit,
              "torch": torch.__version__, "commands": [
                  "python spikes/x7_heldout/run.py <model.gguf> <dir> '<name>|<runs>|<trace>|<options>' ...",
                  "python spikes/x7_heldout/l2.py <model.gguf> cand.safetensors '<options>'",
                  "python -m holon_judge compare --golden judge/goldens/qwen3-30b-a3b-q4_k_m/v1-fp32.safetensors "
                  "--candidate cand.safetensors --floor judge/goldens/qwen3-30b-a3b-q4_k_m/v1-bfloat16-mmvq-floor"
                  ".safetensors",
                  "python spikes/x6_cpu/decode.py <model.gguf> out.json '<arm>|<options>' ... --batch 1 --tokens 256 "
                  "--depth 512 --rounds 3 --engine-args '--host-memory <bytes> --max-model-len 4096'"],
              "runs": metas, "a1_l2_compare": (a1 / "compare.txt").read_text(),
              "a1_decode": arms(a1 / "decode_log.txt"), "pagecache_decode": arms(pc / "decode_log.txt")}
    out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
