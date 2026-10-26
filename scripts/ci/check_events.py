"""Fences: a file that waits on generic streams (`torch.accelerator.current_stream`, `torch.Stream`) never makes a
CUDA-typed event (`torch.cuda.Event`): a generic stream's `wait_event` silently ignores one (torch 2.11), so the fence
orders nothing (GLM step 7's moves). std and plugins use `torch.Event` with the accelerator's streams; a kernel plugin
that keeps CUDA-typed events waits them on CUDA-typed streams only."""
from __future__ import annotations

from _common import CORE, ROOT, fail, py_files

GENERIC = ("torch.accelerator.current_stream", "torch.Stream(")
msgs = []
for p in py_files(CORE, ROOT / "std", ROOT / "plugins", ROOT / "judge"):
    if "tests" in p.parts:
        continue
    text = p.read_text()
    if "torch.cuda.Event" in text and any(g in text for g in GENERIC):
        msgs += [f"{p.relative_to(ROOT)}:{i}" for i, line in enumerate(text.splitlines(), 1)
                 if "torch.cuda.Event" in line]
fail(msgs, "no CUDA-typed event beside generic streams")
