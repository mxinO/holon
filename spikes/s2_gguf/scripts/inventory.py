"""Per-tensor ggml type inventory of the S2 files -> results/inventory.json."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import gguf  # noqa: E402

from models import FILES, path  # noqa: E402

out = {}
for key in FILES:
    r = gguf.GGUFReader(path(key))
    by_type, by_role = Counter(), defaultdict(Counter)
    nbytes = Counter()
    for t in r.tensors:
        ty = t.tensor_type.name
        by_type[ty] += 1
        nbytes[ty] += int(t.n_bytes)
        by_role[re.sub(r"^blk\.\d+\.", "blk.N.", t.name)][ty] += 1
    out[key] = {
        "file": FILES[key][1],
        "tensors_by_type": dict(by_type),
        "gib_by_type": {k: round(v / 2**30, 3) for k, v in nbytes.items()},
        "types_by_role": {k: dict(v) for k, v in sorted(by_role.items())},
    }
    print(key, dict(by_type))
    for role, c in sorted(by_role.items()):
        print(f"   {role:28s} {dict(c)}")
(HERE / "results" / "inventory.json").write_text(json.dumps(out, indent=1))
