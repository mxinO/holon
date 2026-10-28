"""Build the extension for the visible GPU (or a given SM): python scripts/build_ext.py [sm]."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ggml_ops  # noqa: E402

t = time.time()
ext = ggml_ops.load_ext(int(sys.argv[1]) if len(sys.argv) > 1 else None)
print(ext.__file__, f"{time.time() - t:.0f}s")
