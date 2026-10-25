"""Step 4's review C m4, measured on the real model (report only; the kernel is unchanged): `mla_attend`'s dots a
32-token tile — today's 4 (q̃ and p each as two bf16 parts), 3 (p in one part, FlashAttention's practice) and 2 (q̃ in
one part too) — judged by what the model's output sees. One engine scores FN's four `v1-long` prompts (GLM's ids) with
each variant swapped into `kernels.qsa_triton.mla` (the same kernel source with fewer parts); KL(4 dots || variant)
over the prompts' positions, beside gate (c)'s clean chunked pair for scale.

Usage: attend_dots.py <model> '<engine flags>' [--step 16384] [--out <json>]"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import pathlib
import sys
import textwrap

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spikes/glm5"))


_ORIGINAL: dict = {}      # the kernel as shipped (its source the variants are cut from)


def a_dir() -> pathlib.Path:
    """Where the variants' sources go (`GLM_SCRATCH`, else the working directory)."""
    return pathlib.Path(os.environ.get("GLM_SCRATCH", "."))


def variant(dots: int):
    """The attend kernel with `dots` dots a tile (3: p one part; 2: q̃ and p one part each)."""
    from holon_plugins.kernels.qsa_triton import mla
    src = textwrap.dedent(inspect.getsource(_ORIGINAL.setdefault("attend", mla._attend).fn))
    one_p = "acc = acc * alpha[:, None] + tl.dot(p_hi, cv) + tl.dot(p_lo, cv)"
    one_q = "scores = tl.dot(q_hi, ct) + tl.dot(q_lo, ct)"
    assert one_p in src and one_q in src
    src = src.replace(one_p, "acc = acc * alpha[:, None] + tl.dot(pw.to(tl.bfloat16), cv)")
    if dots == 2:
        src = src.replace(one_q, "scores = tl.dot(q_hi, ct)")
    if not src.startswith("@"):
        src = "@triton.jit\n" + src
    path = pathlib.Path(a_dir(), f"attend_{dots}_dots.py")
    path.write_text("import triton\nimport triton.language as tl\nfrom holon_plugins.kernels.qsa_triton.mla import "
                    "_split2  # noqa: F401\n\n\n" + src)
    spec = importlib.util.spec_from_file_location(f"glm_attend_{dots}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._attend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--step", type=int, default=16384)
    ap.add_argument("--out")
    a = ap.parse_args()
    import consistency as c
    from holon_judge.compare import kl_topk
    from holon_judge.engine import score
    from holon_judge.recording import PromptSet
    from holon_plugins.kernels.qsa_triton import mla
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import engine
    full = PromptSet.load(c.SET)
    keep = {p.id: p for p in full.prompts}
    prompts = PromptSet(full.name, full.sha256, [keep[p] for p in c.PROMPTS])
    e = engine(a.model, a.flags, max_step_tokens=a.step, max_seqs=1, prefix_cache=False,
               max_model_len=max(len(p.ids) for p in prompts.prompts) + 64)
    base, out = mla._attend, {"step": a.step, "gpu": torch.cuda.get_device_name()}
    try:
        ref = score(e, prompts, c.K)
        for dots in (3, 2):
            mla._attend = variant(dots)
            kl = kl_topk(ref, score(e, prompts, c.K))[0].double()
            out[f"{dots}_dots"] = {"kl_mean": kl.mean().item(), "kl_p50": kl.median().item(),
                                   "kl_p99": kl.quantile(0.99).item()}
            mla._attend = base
    finally:
        mla._attend = base
        e.close()
    print(json.dumps(out, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
