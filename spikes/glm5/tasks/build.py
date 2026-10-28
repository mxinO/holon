"""GLM's task-accuracy set (docs/dev/v1-glm.md acceptance 1(d)): FN's 72 synthetic items (`spikes/fn7/tasks/build.py`,
seed 7: math, code and reasoning as FN's, the same draws), the recall filler sized with GLM-5.3-Flash's tokenizer to
about 8K and 32K tokens (so the items after it draw on: the multi-turn ones are GLM's own). Synthetic, as FN's
(Apache-2.0, this repository). `run.py` renders each chat with GLM's template.

    python spikes/glm5/tasks/build.py --tokenizer <Maya-S ...-00001-of-00003.gguf> \\
        --out spikes/glm5/tasks/glm_tasks.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = "synthetic: spikes/glm5/tasks/build.py over spikes/fn7/tasks/build.py, seed 7 (Apache-2.0, this repository)"


def fn_build():
    """FN's builder module (its item generators, unchanged)."""
    spec = importlib.util.spec_from_file_location("fn_build", ROOT / "spikes" / "fn7" / "tasks" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE = SOURCE
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="a GLM-5.3-Flash GGUF (its tokenizer sizes the filler)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from holon.core.assembly import resolve_model
    from holon.core.engine import EngineConfig
    from holon.core.plugins import PluginSet, discover
    source, plugin = resolve_model(PluginSet(discover()), EngineConfig(a.tokenizer))
    tokenizer = plugin.bundle(source).tokenizer
    fn = fn_build()
    r = random.Random(fn.SEED)
    items = [*fn.math_items(r), *fn.code_items(), *fn.reasoning_items(r), *fn.recall_items(r, tokenizer),
             *fn.multiturn_items(r)]
    meta = {"name": "glm_tasks", "built": datetime.date.today().isoformat(), "items": len(items), "source": SOURCE}
    pathlib.Path(a.out).write_text("\n".join(json.dumps(x) for x in [meta, *items]) + "\n")
    print(json.dumps(meta), {c: sum(x["category"] == c for x in items) for c in dict.fromkeys(x["category"]
                                                                                              for x in items)},
          [x.get("context_tokens") for x in items if x["category"] == "recall"])


if __name__ == "__main__":
    main()
