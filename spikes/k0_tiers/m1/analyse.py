"""K-M1: re-render divergence of a pi agent run (README.md here). CPU only.

Reads the raw transcript of `bench/capture_proxy.py --raw` (request bodies, the engine's generated token ids, the
reported usage), renders every prompt with Holon's own chat frontend for the served GGUF, and writes counts,
positions and statistics, never text, into the evidence record (its predictions and run facts are kept).

Usage:
  python spikes/k0_tiers/m1/analyse.py --raw <raw.jsonl> --model <Qwen3.8-27B-UD-Q4_K_M.gguf> \
      --record <pi_record.json> --out evidence/k0_m1_rerender_rtx-5090_<date>.json \
      [--clock-before <json>] [--clock-after <json>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import numpy as np

BLOCK = 64                                   # the 27B's recurrent checkpoint alignment
KS = (1, 2, 3, 4)
# The template's history branch that keeps an assistant turn's reasoning; the stripping copy never takes it.
KEEP_REASONING = ("{%- if preserve_thinking is undefined or preserve_thinking is true or "
                  "loop.index0 > ns.last_query_index %}")


def load_bundle(model: pathlib.Path):
    from holon_plugins.models.qwen3_5.bundle import Bundle
    from holon_std.weights.gguf.source import GGUFSource
    return Bundle(GGUFSource(model))


def render(bundle, body: dict, **kwargs) -> list[int]:
    """The prompt ids the server prefilled: the chat route's own path (`ChatRequest` → `Chat`), with `kwargs` added to
    the request's `chat_template_kwargs` (None removes a key)."""
    from holon_plugins.frontends.http.pipeline import Chat
    from holon_plugins.frontends.http.schemas import ChatRequest
    template = {k: v for k, v in ((body.get("chat_template_kwargs") or {}) | kwargs).items() if v is not None}
    return list(Chat(bundle, ChatRequest(**(body | {"chat_template_kwargs": template})).conversation()).ids)


def lcp(a: list[int] | np.ndarray, b: list[int] | np.ndarray) -> int:
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n


def floor(x: int) -> int:
    return x // BLOCK * BLOCK


def answer_checkpoints(p: int, g: int, k: int) -> list[int]:
    """`answer:<k>`'s kept positions: the last k multiples of 64 the decode rows reach, in (P, P + G − 1] (the last
    generated token is sampled, never fed)."""
    top = floor(p + g - 1)
    return [c for c in range(top, p, -BLOCK)][:k][::-1]


def site(d: int, gen: list[int], ids: dict[str, int]) -> str:
    """Where the next prompt leaves the answer, `d` tokens into it."""
    g = len(gen)
    if d < 0:
        return "before the answer"
    if d >= g:
        return "nowhere (verbatim to the end)"
    end = gen.index(ids["</think>"]) if ids["</think>"] in gen else None
    call = gen.index(ids["<tool_call>"]) if ids["<tool_call>"] in gen else None
    if d == g - 1 and gen[d] == ids["<|im_end|>"]:
        return "at the end token"
    if end is not None and d < end:
        return "in the reasoning"
    if end is not None and d == end:
        return "at </think>"
    if call is not None and d == call:
        return "at a <tool_call> start"
    if call is not None and d > call:
        return "at a <tool_call> start" if gen[d] == ids["<tool_call>"] else "inside a tool call"
    return "between </think> and the first tool call (content or whitespace)"


def sessions(raw: list[dict], record: dict) -> list[dict]:
    """The chat requests split into the record's runs, in order; each run's requests continue one conversation."""
    chats = [r for r in raw if r["path"].split("?")[0] == "/v1/chat/completions"]
    out, at = [], 0
    for task, series in record["series"].items():
        for i, run in enumerate(series["runs"]):
            n = run["served"]["chat_requests"]
            reqs = chats[at:at + n]
            at += n
            for a, b in zip(reqs, reqs[1:], strict=False):
                assert b["body"]["messages"][:len(a["body"]["messages"])] == a["body"]["messages"], "not one session"
            out.append({"task": task, "run": i + 1, "success": run["success"], "requests": reqs})
    assert at == len(chats), f"{len(chats) - at} chat requests outside the record's runs"
    return out


def replay(seqs: list[tuple[list[int], list[int]]], policy: int | None) -> list[int]:
    """Each request's hit (prefix tokens served) when every earlier request left its checkpoints: `prompt_end`
    (⌊P/64⌋·64 on its prompt) and, with `policy` = k, `answer:<k>`'s on prompt + generated tokens. KV is taken as
    resident; a hit is the deepest checkpoint within the shared prefix, short of the whole prompt."""
    hits, left = [], []                      # left: (sequence, positions)
    for prompt, gen in seqs:
        best = 0
        for seq, positions in left:
            m = min(lcp(seq, prompt), len(prompt) - 1)
            best = max([best, *(c for c in positions if c <= m)])
        hits.append(best)
        p, g = len(prompt), len(gen)
        left.append((np.asarray(prompt + gen), [floor(p)] + (answer_checkpoints(p, g, policy) if policy else [])))
    return hits


def analyse(raw_path: pathlib.Path, model: pathlib.Path, record: dict) -> dict:
    bundle = load_bundle(model)
    tok = bundle.tokenizer._tok
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("</think>", "<tool_call>", "<|im_end|>")}
    template = tok.chat_template
    assert template.count(KEEP_REASONING) == 1, "the template's reasoning branch moved"
    stripping = template.replace(KEEP_REASONING, "{%- if false %}")
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    runs = sessions(raw, record)

    checks = {"requests": 0, "prompt_len_equal_server": 0, "gen_len_equal_server": 0, "default_template_differs": 0}
    flat = []                                 # (prompt, gen, measured cached, run index, position in run)
    for r_i, run in enumerate(runs):
        for j, req in enumerate(run["requests"]):
            body, usage = req["body"], req.get("usage") or {}
            prompt, gen = render(bundle, body), req["token_ids"]
            checks["requests"] += 1
            checks["prompt_len_equal_server"] += len(prompt) == usage.get("input_tokens")
            checks["gen_len_equal_server"] += len(gen) == usage.get("output_tokens")
            default = render(bundle, body, preserve_thinking=None)
            checks["default_template_differs"] += default != prompt
            req["ids"] = {"served": prompt, "strip": render(bundle, body, chat_template=stripping)}
            flat.append((prompt, gen, usage.get("cached_tokens"), r_i, j))

    rows = []
    for run in runs:
        reqs = run["requests"]
        for j, (a, b) in enumerate(zip(reqs, reqs[1:], strict=False)):
            gen = a["token_ids"]
            p, g = len(a["ids"]["served"]), len(gen)
            seq = a["ids"]["served"] + gen
            m = lcp(seq, b["ids"]["served"])
            end = gen.index(ids["</think>"]) if ids["</think>"] in gen else None
            call = gen.index(ids["<tool_call>"]) if ids["<tool_call>"] in gen else None
            row = {"task": run["task"], "run": run["run"], "turn": j + 1, "prompt_len": p, "generated": g,
                   "answer_end": p + g, "next_prompt_len": len(b["ids"]["served"]), "lcp": m,
                   "lcp_minus_prompt": m - p, "answer_end_minus_lcp": p + g - m, "prompt_end": floor(p),
                   "think_end_offset": end, "first_tool_call_offset": call,
                   "tool_calls": gen.count(ids["<tool_call>"]), "divergence": site(m - p, gen, ids)}
            reach = min(m, p + g - 1)
            for k in KS:
                kept = answer_checkpoints(p, g, k)
                usable = [c for c in kept if c <= m]
                row[f"answer_{k}"] = {"kept": kept, "deepest_usable": max(usable, default=None),
                                      "within_one_block": bool(usable) and reach - max(usable) < BLOCK}
            sp = len(a["ids"]["strip"])
            sm = lcp(a["ids"]["strip"] + gen, b["ids"]["strip"])
            row["strip"] = {"prompt_len": sp, "lcp": sm, "lcp_minus_prompt": sm - sp,
                            "answer_end_minus_lcp": sp + g - sm, "divergence": site(sm - sp, gen, ids),
                            "within_one_block": {k: bool(u := [c for c in answer_checkpoints(sp, g, k) if c <= sm])
                                                 and min(sm, sp + g - 1) - max(u) < BLOCK for k in KS}}
            rows.append(row)

    seqs = [(f[0], f[1]) for f in flat]
    first = {(f[3], f[4]) for f in flat if f[4] == 0}
    policies = {"prompt_end": None} | {f"prompt_end+answer:{k}": k for k in KS}
    uncached = {}
    for name, k in policies.items():
        hits = replay(seqs, k)
        total = own = 0
        for i, ((prompt, _), hit) in enumerate(zip(seqs, hits, strict=True)):
            total += len(prompt) - hit
            if (flat[i][3], flat[i][4]) not in first:
                pp, pg = len(seqs[i - 1][0]), len(seqs[i - 1][1])
                own += pg - min(max(hit - pp, 0), pg)
        uncached[name] = {"uncached_input": total, "previous_output_uncached": own,
                          "previous_output_share": round(own / total, 4) if total else None}
        if k is None:
            measured = [f[2] for f in flat]
            uncached[name]["replay_equals_measured_cached"] = sum(h == c for h, c in zip(hits, measured, strict=True))
            uncached[name]["requests"] = len(hits)
            uncached["measured"] = {
                "uncached_input": sum(len(f[0]) - (f[2] or 0) for f in flat),
                "previous_output": sum(len(seqs[i - 1][1]) for i, f in enumerate(flat) if (f[3], f[4]) not in first)}
            uncached["measured"]["previous_output_share"] = round(
                uncached["measured"]["previous_output"] / uncached["measured"]["uncached_input"], 4)
    return {"checks": checks, "rows": rows, "uncached": uncached, "summary": summarize(rows, uncached)}


def summarize(rows: list[dict], uncached: dict) -> dict:
    n = len(rows)
    eligible = [r for r in rows if r["answer_1"]["kept"]]

    def share(sel: list[dict], test) -> dict:
        hit = sum(1 for r in sel if test(r))
        return {"count": hit, "of": len(sel), "share": round(hit / len(sel), 4) if sel else None}
    within = {k: {"all_answers": share(rows, lambda r, k=k: r[f"answer_{k}"]["within_one_block"]),
                  "answers_crossing_a_block": share(eligible, lambda r, k=k: r[f"answer_{k}"]["within_one_block"]),
                  "strip_template_all_answers": share(rows, lambda r, k=k: r["strip"]["within_one_block"][k])}
              for k in KS}
    sites, strip_sites = {}, {}
    for r in rows:
        sites[r["divergence"]] = sites.get(r["divergence"], 0) + 1
        strip_sites[r["strip"]["divergence"]] = strip_sites.get(r["strip"]["divergence"], 0) + 1
    gaps = [r["answer_end_minus_lcp"] for r in rows]
    calls = [r["tool_calls"] for r in rows]
    shapes = {"generated": {"min": min(r["generated"] for r in rows),
                            "p50": statistics.median(r["generated"] for r in rows),
                            "max": max(r["generated"] for r in rows)},
              "tool_calls_per_answer": {str(c): calls.count(c) for c in sorted(set(calls))},
              "with_reasoning": sum(r["think_end_offset"] is not None for r in rows),
              "with_text_before_the_first_call": sum(   # more than `</think>` and its "\n\n" before `<tool_call>`
                  r["first_tool_call_offset"] is not None and r["think_end_offset"] is not None
                  and r["first_tool_call_offset"] - r["think_end_offset"] > 2 for r in rows)}
    strip_offsets = [r["strip"]["lcp_minus_prompt"] for r in rows]
    verdict_k = next((k for k in KS if within[k]["all_answers"]["share"] >= 0.5), None)
    return {
        "answers": n, "answers_crossing_a_block": len(eligible),
        "answer_end_minus_lcp": {"min": min(gaps), "p50": statistics.median(gaps), "max": max(gaps),
                                 "le_0": sum(g <= 0 for g in gaps)},
        "answer_shapes": shapes,
        "strip_answer_end_minus_lcp_p50": statistics.median(r["strip"]["answer_end_minus_lcp"] for r in rows),
        "strip_lcp_minus_prompt": {str(d): strip_offsets.count(d) for d in sorted(set(strip_offsets))},
        "within_one_block_of_a_kept_checkpoint": within,
        "divergence_sites": sites, "divergence_sites_strip_template": strip_sites,
        "uncached": uncached,
        "decision_rule": {"rule": ">= 50% of answers re-rendered verbatim to within one block of a kept checkpoint",
                          "answer_on_by_default": verdict_k is not None, "smallest_k_meeting_it": verdict_k}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--record", type=pathlib.Path, required=True, help="pi_run.py's record of the same runs")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--clock-before", type=pathlib.Path)
    parser.add_argument("--clock-after", type=pathlib.Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    result = analyse(args.raw, args.model, record)
    out |= {"commit": record["commit"], "model": record["model"], "served": record["served"],
            "client": {k: record["client"][k] for k in ("pi", "node", "packages_digest", "command", "tools")},
            "series": {t: {"runs": len(s["runs"]), "compat": s["compat"], "thinking": s["thinking"],
                           "models_json": s["models_json"],
                           "outcomes": [{"success": r["success"], "turns": r["turns"], "wall_s": r["wall_s"],
                                         "requests": r["served"]["chat_requests"]} for r in s["runs"]]}
                       for t, s in record["series"].items()},
            "clock_screens": {name: json.loads(p.read_text()) for name, p in (("before", args.clock_before),
                                                                               ("after", args.clock_after)) if p},
            **result}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(result["checks"] | result["summary"], indent=1))


if __name__ == "__main__":
    main()
