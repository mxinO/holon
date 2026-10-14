"""Judge command line.

    python -m holon_judge golden   --model <hf dir or id | .gguf> [--config <hf dir or id>] [--q8-1 mmq|mmvq
                                   [--act q8_1|fp4|fp4-static]] --prompts <jsonl> --out <file> [--threads N] [--k 20]
                                   [--dtype bfloat16 --device cuda]          # the floor: HF in the activation dtype
                                   [--kv q8_0|fp8]                           # ... reading K/V in that cache format
                                   [--stream]                                # a dense checkpoint one layer at a time
    python -m holon_judge gate     --model <hf dir or id | .gguf> --original <hf dir or id> [--json <file>]
                                   # per-tensor NMSE against the bf16 original (docs/m06-lowp.md §4.8)
    python -m holon_judge compare  --golden <file> --candidate <file> [--floor <file>] [--json <file>]
    python -m holon_judge l2       --model <hf dir or id | .gguf> --prompts <jsonl> --golden <file> --floor <file>
                                   [--source gguf_dense] [--max-step-tokens N] [--kv q8_0|fp8] [--plant <bug>]
                                   [--dtype bfloat16] [--k 20] [--save <file>] [--json <file>] [--ceiling 1e-4]
    python -m holon_judge retokenize --prompts <jsonl> --model <hf dir or id> --out <jsonl>
                                   # the same texts in another model's tokenizer (each must decode back exactly)
    python -m holon_judge snapshot-diff <a.json> <b.json>
    python -m holon_judge snapshot --model <hf dir or id> --prompts <jsonl> [--runs 3] [--json <file>]
                                   # the snapshot trace on fresh engines: run-identical?
    python -m holon_judge bounded  --model <hf dir or id | .gguf> --prompts <jsonl> [--json <file>] [--plant <bug>]
                                   [--sessions N] [--graphs-vs-eager]
                                   # prefix cache on/off x overlap depth 1/0 on a multi-turn trace; or decode
                                   # graphs on vs off at a batch of N sessions
    python -m holon_judge constrained --model <.gguf> --jsb <JSONSchemaBench dir> --engine-args "<speculating>"
                                   [--calibration <jsonl>] [--plant <bug>] [--json <file>]
                                   # grammar + speculation vs grammar (M0.5 §5)

`compare`, `l2`, `gate`, `snapshot-diff`, `snapshot`, `bounded` and `constrained` exit with status 1 on a failed
check.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib
import re
import shlex
import sys

import torch

from . import golden, modes
from . import kv as kv_format
from .compare import Result, compare, compare_drafter
from .planted import PLANTS
from .recording import PromptSet, Recording


def _golden(a: argparse.Namespace) -> int:
    if a.threads:
        torch.set_num_threads(a.threads)
    rec = golden.record(a.model, PromptSet.load(a.prompts), golden.DTYPES[a.dtype], a.device, a.k, a.config,
                        a.q8_1, a.pure_hf, a.kv, a.act, a.stream)
    rec.save(a.out)
    print(json.dumps(rec.meta | {"bytes": pathlib.Path(a.out).stat().st_size}, indent=1))
    return 0


def _report(result: Result, json_out: str | None) -> int:
    print(result.table())
    if json_out:
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps(result.to_json(), indent=1) + "\n")
    return 1 if result.passed is False else 0


def _compare(a: argparse.Namespace) -> int:
    floor = Recording.load(a.floor) if a.floor else None
    return _report(compare(Recording.load(a.golden), Recording.load(a.candidate), floor), a.json)


def _l2(a: argparse.Namespace) -> int:
    from .engine import score_model

    if a.threads:
        torch.set_num_threads(a.threads)
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        cand = score_model(a.model, PromptSet.load(a.prompts), golden.DTYPES[a.dtype], a.device, a.k, a.source,
                           a.max_step_tokens, a.state_memory_gib and int(a.state_memory_gib * 2**30), a.max_seqs,
                           a.kv)
    if a.save:
        cand.save(a.save)
    return _report(compare(Recording.load(a.golden), cand, Recording.load(a.floor), a.ceiling), a.json)


def _gate(a: argparse.Namespace) -> int:
    """§4.8's gates: NMSE per tensor (per expert) against the bf16 original, by the tensor's format; a summary per
    role (the last part of the name, experts' projections apart), and the tensors over their gate."""
    from . import gguf, hf_quant

    path, original = golden.resolve_model(a.model), golden.resolve_model(a.original)
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        if path.suffix == ".gguf":
            types = gguf.hf_types(path)
            nmse = {n: (types[re.sub(r"\[\d+\]$", "", n)], v) for n, v in gguf.gate(path, original).items()}
            bound, folded = {n: gguf.GATES.get(t, gguf.GATE) for n, (t, _) in nmse.items()}, []
        else:
            nmse, folded = hf_quant.gate(path, original)
            bound = {n: hf_quant.GATES[elem] for n, (elem, _) in nmse.items()}
    by_role: dict[str, float] = {}
    for n, (_, v) in nmse.items():
        role = re.sub(r"\[\d+\]$", "", n).removesuffix(".weight").rsplit(".", 1)[-1]
        by_role[role] = max(by_role.get(role, 0.0), v)
    over = sorted(n for n, (_, v) in nmse.items() if not v <= bound[n])
    meta = gguf.file_meta(path) if path.suffix == ".gguf" else golden.model_meta(path)
    report = {"model": meta["model"], "original": golden.model_meta(original)["model"], "plant": a.plant,
              "tensors": len(nmse), "max": max(v for _, v in nmse.values()), "max_by_role": by_role,
              "gates": sorted(set(bound.values())), "norms_folded": len(folded), "over": len(over),
              "first_over": over[:8]}
    print(json.dumps(report, indent=1))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
    return 1 if over else 0


def _structure(a: argparse.Namespace) -> int:
    from holon.core.engine import Engine, EngineConfig

    from .structure import mismatches

    path = golden.resolve_model(a.model)
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        engine = Engine(EngineConfig(model=str(path), dtype=golden.DTYPES[a.dtype], device=a.device,
                                     max_model_len=1024, state_memory=1 << 28 if a.device == "cpu" else None))
    bad = mismatches(engine, path)
    print(json.dumps({"model": golden.model_meta(path)["model"], "plant": a.plant, "mismatches": len(bad),
                      "first": bad[:8]}))
    return 1 if bad else 0


def _hf_quant_check(a: argparse.Namespace) -> int:
    from .hf_quant import owner_check, repack_check

    path = golden.resolve_model(a.model)
    report = owner_check(path, a.llama_cpp, a.limit)
    out = {"model": golden.model_meta(path)["model"], "llama_cpp": bool(a.llama_cpp),
           "owners": {k: v | {"over": len(v["over"]), "first_over": v["over"][:8]} for k, v in report.items()}}
    failed = not report or any(v["over"] for v in report.values())
    if a.repack:
        r = repack_check(path, a.llama_cpp, a.repack, a.limit)
        out["repack"] = r | {"over": len(r["over"]), "first_over": r["over"][:8]} | (
            {"bytes_differ": len(r["bytes_differ"]), "first_bytes_differ": r["bytes_differ"][:8]}
            if r["bytes_differ"] is not None else {})
        failed |= not r["modules"] or bool(r["over"]) or bool(r["bytes_differ"])
    print(json.dumps(out))
    return 1 if failed else 0


def _retokenize(a: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(golden.resolve_model(a.model) / "tokenizer.json"))
    rows = [json.loads(line) for line in pathlib.Path(a.prompts).read_text().splitlines() if line]
    old = Tokenizer.from_file(str(golden.resolve_model(a.source_model) / "tokenizer.json")) if a.source_model else None
    for row in rows:
        ids = tok.encode(row["text"], add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != row["text"]:
            raise ValueError(f"{row['id']}: the text does not round-trip through the tokenizer")
        if old is not None:          # where generation started: the same text prefix, in the new tokens
            prefix = old.decode(row["ids"][:row["gen_start"]], skip_special_tokens=False)
            row["gen_start"] = len(tok.encode(prefix, add_special_tokens=False).ids)
        row["ids"], row["n_tokens"] = ids, len(ids)
    out = pathlib.Path(a.out)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} prompts, {sum(r['n_tokens'] for r in rows)} tokens -> {out}")
    return 0


def _snapshot_diff(a: argparse.Namespace) -> int:
    from .snapshot import Snapshot, diff

    where = diff(Snapshot.load(a.a), Snapshot.load(a.b))
    print(where or "identical")
    return 1 if where else 0


def _snapshot(a: argparse.Namespace) -> int:

    from holon.api import canonical_hash
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    from . import snapshot

    path = golden.resolve_model(a.model)
    requests = snapshot.trace(PromptSet.load(a.prompts))
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    config = dataclasses.replace(engine_config(str(path), flags.parse_args(shlex.split(a.engine_args))),
                                 dtype=golden.DTYPES[a.dtype])
    runs, pools = snapshot.repeat(lambda memory: Engine(dataclasses.replace(
        config, state_memory=config.state_memory or memory)), requests, a.runs)
    diffs = [f"pools differ: {p} vs {pools[0]}" if p != pools[0] else snapshot.diff(runs[0], run)
             for run, p in zip(runs[1:], pools[1:], strict=True)]
    report = {"model": golden.model_meta(path)["model"], "dtype": a.dtype, "runs": a.runs,
              "hardware": golden.hardware(torch.device("cuda")), "torch": torch.__version__,
              "trace": canonical_hash([[r.ids, r.params, r.after] for r in requests]),
              "steps": len(runs[0].steps), "pools": pools[0], "diffs": diffs, "run_identical": not any(diffs),
              **runs[0].meta}
    print(json.dumps(report, indent=1, default=str))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0 if report["run_identical"] else 1


def _drafter(a: argparse.Namespace) -> int:
    """L0-drafter: the engine's draft head against its fp32 reference, budgets from the bf16 reference (the floor)."""
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    from . import drafter

    if a.head == "dflash" or str(a.engine_model or a.model).endswith(".gguf"):
        return _drafter_cuts(a)
    prompts = PromptSet.load(a.prompts)
    recordings = {}
    for name, dtype in (("golden", torch.float32), ("floor", golden.DTYPES[a.dtype])):
        cached = pathlib.Path(a.cache) / f"mtp-{name}.safetensors" if a.cache else None
        if cached is not None and cached.exists():
            recordings[name] = Recording.load(cached)
            continue
        recordings[name] = drafter.reference(a.model, prompts, dtype, a.k)
        if cached is not None:
            recordings[name].save(cached)
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    config = engine_config(str(golden.resolve_model(a.engine_model or a.model)), flags.parse_args(shlex.split(
        a.engine_args)))
    longest = max(len(p.ids) for p in prompts.prompts)
    sizes = {"max_model_len": max(config.max_model_len, longest + 8), "max_step_tokens": max(config.max_step_tokens,
                                                                                          longest)}
    engine = Engine(dataclasses.replace(config, prefix_cache=False, **sizes))
    cand = drafter.candidate(engine, prompts, a.k)
    cand.meta |= {"model": golden.model_meta(golden.resolve_model(a.model))["model"],
                  "dtype": str(config.dtype).removeprefix("torch.")}
    engine.close()
    return _report(compare_drafter(recordings["golden"], cand, recordings["floor"]), a.json)


def _drafter_cuts(a: argparse.Namespace) -> int:
    """A head judged per cut of each prompt (DFlash2; an MTP head stored in a GGUF): the engine's draft after
    prefilling the cut, against the reference on the target's streams the judge records itself; GGUF weights
    dequantized, the floor rounding their matmuls to q8_1 as the kernel path of their row count does."""
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    from . import drafter

    flags = argparse.ArgumentParser()
    engine_flags(flags)
    model = str(golden.resolve_model(a.engine_model or a.model))
    config = engine_config(model, flags.parse_args(shlex.split(a.engine_args)))
    steps = 7 if a.head == "dflash" else config.max_draft_tokens            # drafts per cut (the policy's k)
    prompts = drafter.cut_prompts(PromptSet.load(a.prompts), a.cuts, steps + 1)
    longest = max(len(p.ids) for p in prompts.prompts)
    config = dataclasses.replace(config, prefix_cache=False, max_model_len=max(config.max_model_len, longest + 16),
                                 max_step_tokens=max(config.max_step_tokens, longest))
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        engine = Engine(config)
        record = drafter.dflash_candidate if a.head == "dflash" else (
            lambda e, p, k: drafter.mtp_candidate(e, p, steps, k))
        cand, inputs = record(engine, prompts, a.k)
    cand.meta |= {"model": golden.model_meta(pathlib.Path(model))["model"], "dtype": str(config.dtype).removeprefix(
        "torch.")}
    engine.close()
    del engine
    torch.cuda.empty_cache()
    if a.head == "dflash":
        draft = str(config.plugin_options["dflash.path"])
        quantized = draft.endswith(".gguf") or model.endswith(".gguf")
        reference = lambda dtype, q8_1: drafter.dflash_reference(draft, model, prompts, inputs, dtype, a.k,  # noqa: E731
                                                                 q8_1=q8_1 and quantized)
    else:
        import gguf

        reader = gguf.GGUFReader(model)
        reference = lambda dtype, q8_1: drafter.MTPReference(model, dtype, q8_1, reader).record(  # noqa: E731
            prompts, inputs, a.k)
    golden_rec = reference(torch.float32, False)
    return _report(compare_drafter(golden_rec, cand, reference(golden.DTYPES[a.dtype], True)), a.json)


def _bounded(a: argparse.Namespace) -> int:
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    from . import bounded, gguf

    path = golden.resolve_model(a.model)
    trace = bounded.Trace.from_prompts(PromptSet.load(a.prompts), sessions=a.sessions)
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    base = dataclasses.replace(engine_config(str(path), flags.parse_args(shlex.split(a.engine_args))),
                               dtype=golden.DTYPES[a.dtype], max_model_len=trace.max_len)

    def build(**overrides):
        return Engine(dataclasses.replace(base, **overrides))

    reference = None
    if a.graphs_vs_eager:                                # every decode step a batch of `sessions` rows, padded
        baseline = bounded.BASELINE | {"cuda_graphs": False}
        configs = {"graphs on": bounded.BASELINE | {"cuda_graphs": True}}
    elif a.spec_vs_plain:                                # the engine args speculate; the baseline does not
        if not base.max_draft_tokens:
            raise SystemExit("--spec-vs-plain: the engine args must speculate (--max-draft-tokens, --slots)")
        reference = {"max_draft_tokens": 0}
        baseline = bounded.BASELINE | reference
        configs = {"spec, prefix on, depth 1": {"prefix_cache": True, "overlap": True},
                   "spec, depth 0": dict(bounded.BASELINE)}
    else:
        baseline = bounded.BASELINE
        configs = {f"prefix {'on' if p else 'off'}, depth {int(o)}": {"prefix_cache": p, "overlap": o}
                   for p, o in ((True, True), (True, False), (False, True))}   # the baseline is off, 0
        if a.chunked:                                    # prefill in chunks of `chunked` tokens vs one shot
            configs[f"chunked {a.chunked}"] = bounded.BASELINE | {"max_step_tokens": a.chunked}
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        report = bounded.compare_configs(build, configs, trace, a.k, baseline, reference)
    meta = gguf.file_meta(path) if path.suffix == ".gguf" else golden.model_meta(path)
    report |= {"model": meta["model"], "dtype": a.dtype, "graphs_vs_eager": a.graphs_vs_eager,
               "spec_vs_plain": a.spec_vs_plain, "engine_args": a.engine_args,
               "hardware": golden.hardware(torch.device("cuda")), "torch": torch.__version__, "plant": a.plant}
    print(bounded.table(report))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["passed"] else 1


def _constrained(a: argparse.Namespace) -> int:
    from holon.core.assembly import resolve_model
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from holon.core.plugins import PluginSet, discover

    from . import constrained, gguf

    path = golden.resolve_model(a.model)
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    base = engine_config(str(path), flags.parse_args(shlex.split(a.engine_args)))
    if not base.max_draft_tokens:
        raise SystemExit("constrained: the engine args must speculate (--max-draft-tokens, --slots)")
    source, plugin = resolve_model(PluginSet(discover()), base)
    bundle = plugin.bundle(source)
    grammar = base.slots | {"logits_processor": "grammar.xgrammar"}
    configs = {"plain": {"slots": grammar, "max_draft_tokens": 0}, "spec": {"slots": grammar},
               # One sequence at a time: the GGUF kernels are not batch-invariant, so the reference's rows would
               # otherwise depend on which other sequences a run asked it to score.
               "reference": {"max_draft_tokens": 0, "prefix_cache": False, "overlap": False, "cuda_graphs": False,
                             "max_step_tokens": max(base.max_step_tokens, a.max_len), "max_seqs": 1}}

    def build(name: str):
        return Engine(dataclasses.replace(base, max_model_len=a.max_len, **configs[name]))

    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        calibration = [p.ids[:512] for p in PromptSet.load(a.calibration).prompts[:a.prompts]] if a.calibration else []
        report = constrained.judge(build, source.tokenizer(), lambda m: bundle.render_chat(m, enable_thinking=False),
                                   pathlib.Path(a.jsb), a.prompts, a.max_tokens, a.k, calibration)
    meta = gguf.file_meta(path) if path.suffix == ".gguf" else golden.model_meta(path)
    report |= {"model": meta["model"], "engine_args": a.engine_args, "plant": a.plant,
               "hardware": golden.hardware(torch.device("cuda")), "torch": torch.__version__}
    print(constrained.table(report))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m holon_judge")
    sub = ap.add_subparsers(required=True)

    g = sub.add_parser("golden", help="record HF teacher-forced top-k logprobs (golden or floor)")
    g.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    g.add_argument("--config", help="for a .gguf: the HF checkpoint (dir or hub id) it was made from")
    g.add_argument("--q8-1", choices=["mmq", "mmvq"], help="for a .gguf: Tier B's floor on that kernel path")
    g.add_argument("--pure-hf", action="store_true", help="an MoE floor with HF's own MoE block (information only)")
    g.add_argument("--kv", choices=kv_format.FORMATS, help="a floor whose attention reads K/V in this cache format")
    g.add_argument("--act", choices=modes.MODES, help="with --q8-1, a quantized HF checkpoint's activation mode on "
                   "that path (default q8_1: W4A8; fp4: W4A4-dynamic; fp4-static: W4A4-static, docs/m06-lowp.md §4.8)")
    g.add_argument("--stream", action="store_true", help="a dense HF checkpoint one decoder layer at a time (GGUF "
                   "files and quantized checkpoints always are)")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--threads", type=int, default=0, help="CPU threads; pin it to re-record a golden bitwise")
    g.add_argument("--dtype", default="float32", choices=golden.DTYPES)
    g.add_argument("--device", default="cpu")
    g.add_argument("--k", type=int, default=20)
    g.set_defaults(run=_golden)

    c = sub.add_parser("compare", help="compare a recording with the golden; with --floor, pass/fail")
    c.add_argument("--golden", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--floor")
    c.add_argument("--json", help="write the machine-readable result here")
    c.set_defaults(run=_compare)

    e = sub.add_parser("l2", help="score the prompt set through a Holon engine and judge it")
    e.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    e.add_argument("--source", help="weight source (default: detected); gguf_dense for GGUF Tier A")
    e.add_argument("--max-step-tokens", type=int, help="tokens per engine step (default: whole prompts)")
    e.add_argument("--max-seqs", type=int, default=64)
    e.add_argument("--kv", choices=kv_format.FORMATS, help="the engine's KV cache format (default: the model dtype)")
    e.add_argument("--state-memory-gib", type=float, help="state pools on the CPU (Tier A), which cannot be measured")
    e.add_argument("--threads", type=int, default=0, help="CPU threads (Tier A)")
    e.add_argument("--prompts", required=True)
    e.add_argument("--golden", required=True)
    e.add_argument("--floor", required=True)
    e.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    e.add_argument("--device", default="cuda")
    e.add_argument("--k", type=int, default=20)
    e.add_argument("--save", help="also save the candidate recording")
    e.add_argument("--json", help="write the machine-readable result here")
    e.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    e.add_argument("--ceiling", type=float, help="Tier A's sanity ceiling on KL p99 (fp32 throughout: 1e-4)")
    e.set_defaults(run=_l2)

    ga = sub.add_parser("gate", help="per-tensor NMSE against the bf16 original (docs/m06-lowp.md §4.8)")
    ga.add_argument("--model", required=True, help="a quantized HF checkpoint (dir or hub id) or a .gguf file")
    ga.add_argument("--original", required=True, help="the bf16 HF checkpoint it was made from")
    ga.add_argument("--json", help="write the report here")
    ga.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    ga.set_defaults(run=_gate)

    st = sub.add_parser("structure", help="every stacked expert of an HF MoE load equals the checkpoint's tensor")
    st.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    st.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    st.add_argument("--device", default="cuda", help="cpu: the reference kernels (e.g. a stack no GPU kernel reads)")
    st.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    st.set_defaults(run=_structure)

    hq = sub.add_parser("hf-quant-check", help="the reference dequantization of every quantized tensor of an HF "
                        "checkpoint against the format owners' decoders (docs/m06-lowp.md D10)")
    hq.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    hq.add_argument("--llama-cpp", help="a llama.cpp checkout at the ggml pin: also gguf-py on its FP4 repacks")
    hq.add_argument("--limit", type=int, help="the first N quantized tensors only")
    hq.add_argument("--repack", metavar="DEVICE", help="also acceptance 4 on DEVICE (e.g. cuda): kernels.ggml_cuda's "
                    "FP4 blocks by gguf-py × scale2 against the reference, and against the converter's bytes")
    hq.set_defaults(run=_hf_quant_check)

    rt = sub.add_parser("retokenize", help="a prompt set's texts in another model's tokenizer")
    rt.add_argument("--prompts", required=True)
    rt.add_argument("--model", required=True, help="HF checkpoint directory or hub id with tokenizer.json")
    rt.add_argument("--source-model", help="the tokenizer the set was built with (to carry gen_start over)")
    rt.add_argument("--out", required=True)
    rt.set_defaults(run=_retokenize)

    s = sub.add_parser("snapshot-diff", help="check two snapshots for byte identity")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(run=_snapshot_diff)

    t = sub.add_parser("snapshot", help="run the snapshot trace on fresh engines and check it is run-identical")
    t.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    t.add_argument("--prompts", required=True)
    t.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    t.add_argument("--runs", type=int, default=3)
    t.add_argument("--engine-args", default="", help="EngineConfig flags, as for `holon serve` (a class's settings)")
    t.add_argument("--json", help="write the machine-readable report here")
    t.set_defaults(run=_snapshot)

    d = sub.add_parser("drafter", help="L0-drafter: a draft head (MTP, DFlash2) against its fp32 reference")
    d.add_argument("--head", default="mtp", choices=("mtp", "dflash"), help="dflash: the engine args' dflash.path")
    d.add_argument("--cuts", type=int, default=4, help="dflash, a GGUF's MTP: drafts per prompt, at cuts of it")
    d.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    d.add_argument("--model", required=True, help="HF checkpoint with the head (the reference reads it)")
    d.add_argument("--engine-model", help="what the engine serves, if not --model (e.g. a .gguf of it)")
    d.add_argument("--prompts", required=True)
    d.add_argument("--engine-args", default="", help="EngineConfig flags that speculate with the head")
    d.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES, help="the floor's dtype")
    d.add_argument("--k", type=int, default=20)
    d.add_argument("--cache", help="directory keeping the references' recordings")
    d.add_argument("--json", help="write the machine-readable result here")
    d.set_defaults(run=_drafter)

    b = sub.add_parser("bounded", help="judge engine configurations against the reference on a multi-turn trace")
    b.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    b.add_argument("--prompts", required=True)
    b.add_argument("--sessions", type=int, default=48, help="concurrent sessions: the decode batch")
    b.add_argument("--graphs-vs-eager", action="store_true", help="judge graphed decode against an eager baseline")
    b.add_argument("--chunked", type=int, help="also judge prefill in chunks of this many tokens")
    b.add_argument("--spec-vs-plain", action="store_true",
                   help="judge speculative decoding (the engine args' drafter, policy, verifier) against plain")
    b.add_argument("--engine-args", default="", help="EngineConfig flags, as for `holon serve`")
    b.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    b.add_argument("--k", type=int, default=20)
    b.add_argument("--json", help="write the machine-readable report here")
    b.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    b.set_defaults(run=_bounded)

    cs = sub.add_parser("constrained", help="grammar with speculation vs without on JSON-schema prompts (M0.5 §5)")
    cs.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    cs.add_argument("--jsb", required=True, help="JSONSchemaBench data directory (test parquets)")
    cs.add_argument("--engine-args", default="", help="EngineConfig flags that speculate")
    cs.add_argument("--prompts", type=int, default=64)
    cs.add_argument("--calibration", help="a prompt set (natural text) whose unconstrained divergences calibrate")
    cs.add_argument("--max-tokens", type=int, default=512)
    cs.add_argument("--max-len", type=int, default=4096, help="max_model_len: prompt and output")
    cs.add_argument("--k", type=int, default=20)
    cs.add_argument("--json", help="write the machine-readable report here")
    cs.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    cs.set_defaults(run=_constrained)

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
