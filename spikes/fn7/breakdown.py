"""FN step 7, item 0 (docs/v1-flash-next.md §9): where a step's GPU time goes, by component, on the full model.

`run` builds one eager engine (graphs and overlap off, so every kernel sits under the host's NVTX ranges) and runs one
phase on a `v1-long` prompt cut to `--ctx` tokens: `decode` (one-token steps), `verify` (MTP on: each step is a verify
pass of 1 + 3 rows, its rewind and the head's three draft sub-steps) or `prefill` (the prompt in `--max-step-tokens`
steps). The measured steps run between cudaProfilerStart / Stop under `nsys profile --capture-range=cudaProfilerApi`,
each in a `step` range, its components in theirs: `hc.mix` / `hc.combine` (the gated-residual kernels, the output
mixer's one mix included), `gdn` and `qsa` (the mixers), `moe` (router, top-k and the routed experts), `shared` (the
gated shared expert), `ple` (layer 1's pre unit), `lm_head`, and `mtp` (the drafter's run: its own components are
counted under `mtp.*`, not the trunk's). The host's per-step wall time (synchronized) is kept for the prefill phase's
timers. `summarize` attributes every GPU operation of the trace to its component: milliseconds and kernels per step, and
the rest of the step (sampling, planning copies, embedding, glue) as `other`; each with its largest kernels.

    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx --sample=none -o <rep> \\
        python spikes/fn7/breakdown.py run --model <...-00001-of-00002.gguf> --pack <mtp-q2_0.gguf> --phase decode \\
        --ctx 4096 --out <json>
    python spikes/fn7/breakdown.py summarize --rep <rep>.nsys-rep --run <json> --out <json>
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DRAFTS = 3
COMPONENTS = ("hc.mix", "hc.combine", "gdn", "qsa", "moe", "shared", "ple", "lm_head")


def prompt(ctx: int) -> list[int]:
    """The longest `v1-long` chat prompt's first `ctx` ids."""
    rows = [json.loads(line) for line in (ROOT / "judge/prompts/v1-long.jsonl").read_text().splitlines() if line]
    ids = max((r["ids"][:r["gen_start"]] for r in rows), key=len)
    if len(ids) < ctx:
        raise ValueError(f"no v1-long prompt has {ctx} tokens")
    return ids[:ctx]


def instrument(engine) -> None:
    """NVTX ranges around each component (hooks and wrappers only: the kernels are unchanged)."""
    import torch
    from holon_std.hyper import units as hyper

    nvtx, state = torch.cuda.nvtx, {"mtp": False}

    def name(base: str) -> str:
        return f"mtp.{base}" if state["mtp"] else base

    def ranged(base: str, fn):
        def wrapped(*a, **k):
            nvtx.range_push(name(base))
            try:
                return fn(*a, **k)
            finally:
                nvtx.range_pop()
        return wrapped

    hyper._HC.mix = ranged("hc.mix", hyper._HC.mix)
    hyper._HC.combine = ranged("hc.combine", hyper._HC.combine)

    seen = set()

    def hook(module, base: str) -> None:
        if id(module) in seen:           # a module the drafter shares with the trunk (e.g. lm_head): one range
            return
        seen.add(id(module))
        def push(m, args) -> None:           # hooks return None: a value would replace the args or the output
            nvtx.range_push(name(base))

        def pop(m, args, out) -> None:
            nvtx.range_pop()
        module.register_forward_pre_hook(push)
        module.register_forward_hook(pop)

    ex = engine.executor
    roots = [ex.model] + [m for m in vars(ex.drafter).values() if isinstance(m, torch.nn.Module)] if ex.drafter else [
        ex.model]
    for root in roots:
        for path, m in root.named_modules():
            leaf = path.rsplit(".", 1)[-1]
            if leaf == "mixer":
                hook(m, "qsa" if "qsa" in type(m).__module__.lower() or "QSA" in type(m).__name__ else "gdn")
            elif path.endswith("ffn.parts.0"):
                hook(m, "moe")
                if hasattr(m, "add_residual"):         # std.ffn.sum calls it in place of forward
                    m.add_residual = ranged("moe", m.add_residual)
            elif path.endswith("ffn.parts.1"):
                hook(m, "shared")
            elif leaf == "pre":
                hook(m, "ple")
            elif leaf == "lm_head":
                hook(m, "lm_head")
    if ex.drafter is not None:
        run = ex._run_drafter

        def drafter(ctx, h):
            state["mtp"] = True
            nvtx.range_push("mtp")
            try:
                run(ctx, h)
            finally:
                nvtx.range_pop()
                state["mtp"] = False
        ex._run_drafter = drafter


def run(a) -> dict:
    import torch
    from holon.api import SamplingParams
    from holon.core.engine import Engine, EngineConfig

    ids = prompt(a.ctx)
    kw = {}
    if a.phase == "verify":
        kw = {"max_draft_tokens": DRAFTS, "plugin_options": {"spec.k": str(DRAFTS), "mtp.path": a.pack},
              "slots": {"drafter": "drafters.mtp", "draft_policy": "std.spec.fixed_k", "verifier": "std.spec.greedy"}}
    engine = Engine(EngineConfig(a.model, max_model_len=a.ctx + 256, max_step_tokens=a.max_step_tokens, max_seqs=1,
                                 cuda_graphs=False, overlap=False, **kw))
    instrument(engine)
    prefill = a.phase == "prefill"
    if prefill:          # a warm-up prompt first (kernel autotuning, allocations), then the measured one
        engine.add(ids[-a.max_step_tokens:][::-1], SamplingParams(temperature=0.0, max_tokens=1))
        while engine.has_work():
            engine.step()
    engine.add(ids, SamplingParams(temperature=0.0, max_tokens=1 if prefill else 4 + a.steps, ignore_eos=True))
    if not prefill:      # the prefill (until the first token) and two steps, unmeasured
        while engine.has_work() and not any(d.token_ids for d in engine.step()):
            pass
        for _ in range(2):
            engine.step()
    walls = []
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    while engine.has_work() and len(walls) < (10**6 if prefill else a.steps):
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push("step")
        engine.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    torch.cuda.profiler.stop()
    stats = {k: v for k, v in engine.stats().items() if k.startswith("spec")}
    engine.close()
    return {"steps": len(walls), "wall_ms": [round(w * 1e3, 3) for w in walls], "spec": stats}


def summarize(a) -> dict:
    """Each kernel, memcpy and memset of the measured steps attributed to the innermost component range open on its
    launching thread when it was launched (the trace's sqlite export: runtime calls by correlation id, NVTX push / pop
    ranges): GPU milliseconds and kernels per step by component, and each one's largest kernels ([ms, launches] per
    step)."""
    import bisect
    import collections
    import sqlite3

    rep = pathlib.Path(a.rep)
    db = rep.with_suffix(".sqlite")
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(db), str(rep)],
                   check=True, capture_output=True)
    c = sqlite3.connect(db)
    names = dict(c.execute("SELECT id, value FROM StringIds"))
    ranges = collections.defaultdict(list)                      # tid -> [(start, end, name)]
    for start, end, text, tid, text_id in c.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL"):
        ranges[tid].append((start, end, text if text is not None else names.get(text_id, "")))
    for v in ranges.values():
        v.sort()
    launch = {corr: (tid, start) for corr, tid, start in c.execute(
        "SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    ops = [(*r, "kernel") for r in c.execute(
        "SELECT k.start, k.end, k.correlationId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName")]
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        with contextlib.suppress(sqlite3.OperationalError):
            ops += [(*r, kind, kind) for r in c.execute(f"SELECT start, end, correlationId FROM {table}")]
    ms = collections.Counter()
    kernels = collections.Counter()
    memops = collections.Counter()
    by_kernel = collections.defaultdict(collections.Counter)    # component -> kernel name -> ms
    count = collections.defaultdict(collections.Counter)        # component -> kernel name -> launches
    steps = set()
    for start, end, corr, name, kind in ops:
        tid, t = launch.get(corr, (None, None))
        if tid is None:
            continue
        rs = ranges[tid]
        open_ = [r for r in rs[:bisect.bisect_right(rs, (t, float("inf"), ""))] if r[1] >= t]
        step = [r for r in open_ if r[2] == "step"]
        if not step:
            continue
        steps.add(step[0][0])
        inner = [r for r in open_ if r[2] in COMPONENTS or r[2] == "mtp" or r[2].startswith("mtp.")]
        comp = max(inner)[2] if inner else "other"            # the latest-opened (innermost) range
        if any(r[2] == "mtp" for r in open_):
            comp = comp if comp.startswith("mtp") else "mtp"
        ms[comp] += (end - start) / 1e6
        (kernels if kind == "kernel" else memops)[comp] += 1
        by_kernel[comp][name] += (end - start) / 1e6
        count[comp][name] += 1
    n = len(steps)
    total = sum(ms.values()) / n
    table = {k: {"ms": round(ms[k] / n, 4), "share": round(ms[k] / n / total, 4), "kernels": kernels[k] / n,
                 "memops": memops[k] / n, "top": {name: [round(t / n, 4), count[k][name] / n]
                                                  for name, t in by_kernel[k].most_common(8)}}
             for k in (*COMPONENTS, "mtp", *sorted(x for x in ms if x.startswith("mtp.")), "other") if k in ms}
    info = json.loads(pathlib.Path(a.run).read_text())
    walls = sorted(info["wall_ms"])
    return info | {"measured_steps": n, "gpu_ms_per_step": round(total, 4),
                   "kernels_per_step": sum(kernels.values()) / n, "memops_per_step": sum(memops.values()) / n,
                   "wall_ms_median": walls[len(walls) // 2] if walls else None, "components": table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--pack", help="Strata's MTP pack (verify)")
    r.add_argument("--phase", choices=("decode", "verify", "prefill"), required=True)
    r.add_argument("--ctx", type=int, default=4096)
    r.add_argument("--steps", type=int, default=8, help="measured decode or verify steps")
    r.add_argument("--max-step-tokens", type=int, default=8192)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--rep", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        import torch
        result = {"name": "fn7_breakdown", "date": datetime.date.today().isoformat(),
                  "gpu": torch.cuda.get_device_name(), "model": pathlib.Path(a.model).name,
                  "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           cwd=ROOT).stdout.strip(),
                  "args": {k: v for k, v in vars(a).items() if k not in ("model", "pack", "out", "cmd")}} | run(a)
    else:
        result = summarize(a)
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "wall_ms"}, indent=1))


if __name__ == "__main__":
    main()
