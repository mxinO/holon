"""Deterministic benchmark workloads, stored as JSONL token-length specs (one session per line).

A session is a list of turns. Turn 0's prompt is `system` tokens shared by every session of the trace followed by
`append` tokens of its own; turn i > 0 is released `gap` seconds after turn i-1 finished and its prompt is turn
i-1's prompt + the tokens turn i-1 generated + `append` new tokens (a tool result and/or user message). Every turn
generates exactly `output` tokens (ignore_eos). A plain request is a one-turn session without a system prompt.

Specs hold lengths only; token ids are drawn at replay time from a seeded stream (`tokens`), so every backend sees
the same prompts: uniform ids by default, or spans of a natural-text corpus (`corpus`: a judge prompt set's token
ids for the model's tokenizer) where the text matters, e.g. speculative decoding, whose acceptance on uniform ids
(greedy continuations of token soup repeat) is far above natural text's.
This module is stdlib-only: it runs in every backend's environment.

Usage: python bench/workloads.py [--out-dir bench/traces] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random

VOCAB = (1000, 30000)   # plain-text token ids, clear of special tokens in Qwen-family vocabularies


def tokens(salt: str, key: str, n: int, corpus: list[int] | None = None) -> list[int]:
    """`n` token ids determined by (salt, key): uniform, or the corpus from a seeded offset (wrapping); str seeds hash
    with SHA-512, so they are stable across processes."""
    rng = random.Random(f"{salt}:{key}")
    if corpus:
        at = rng.randrange(len(corpus))
        return [corpus[(at + i) % len(corpus)] for i in range(n)]
    return [rng.randrange(*VOCAB) for _ in range(n)]


def corpus(path: str | pathlib.Path) -> list[int]:
    """A JSONL prompt set's token ids (`ids` per line), concatenated."""
    return [t for line in pathlib.Path(path).read_text().splitlines() if line for t in json.loads(line)["ids"]]


def turn(append: int, output: int, gap: float = 0.0) -> dict:
    return {"append": append, "output": output, "gap": round(gap, 4)}


def decode(batch: int, prompt: int, output: int) -> list[dict]:
    """`batch` requests of fixed lengths, all arriving at t=0: steady decode at batch size `batch`."""
    return [{"id": i, "arrival": 0.0, "system": 0, "turns": [turn(prompt, output)]} for i in range(batch)]


def long_prompt(decoders: int, prompt: int, output: int, long: int, at: float) -> list[dict]:
    """`decoders` streams decoding from t=0, then one `long`-token prompt arriving at `at` s: how long the streams
    stall behind its prefill (their max inter-token gap) and the long prompt's TTFT behind them."""
    return decode(decoders, prompt, output) + [{"id": decoders, "arrival": at, "system": 0, "turns": [turn(long, 32)]}]


def _lognormal(rng: random.Random, median: float, sigma: float, lo: int, hi: int) -> int:
    return min(hi, max(lo, round(rng.lognormvariate(math.log(median), sigma))))


def sharegpt(n: int, rate: float, seed: int) -> list[dict]:
    """Chat requests with ShareGPT-like length distributions (log-normal, prompts median ~200 and outputs ~180
    tokens) and Poisson arrivals at `rate` requests/s. Synthesized: no dataset is needed or shipped."""
    rng, t, out = random.Random(seed), 0.0, []
    for i in range(n):
        out.append({"id": i, "arrival": round(t, 4), "system": 0,
                    "turns": [turn(_lognormal(rng, 200, 1.0, 4, 3072), _lognormal(rng, 180, 0.9, 4, 1024))]})
        t += rng.expovariate(rate)
    return out


def agent(sessions: int, rate: float, seed: int, system: int = 1536) -> list[dict]:
    """Mixed agent multi-turn tool trace: sessions start as a Poisson process at `rate`/s and share a `system`
    prompt (instructions + tool schemas). The first turn adds a user task; each later turn adds a tool result
    (log-normal, median 256 tokens), sometimes followed by a user message, after a tool latency gap (exponential,
    mean 0.5 s). Generations are short tool calls or answers (24-192 tokens). Prefixes are long and shared."""
    rng, t, out = random.Random(seed), 0.0, []
    for i in range(sessions):
        turns = [turn(_lognormal(rng, 150, 0.6, 16, 1024), rng.randint(24, 192))]
        for _ in range(rng.randint(2, 7)):
            append = _lognormal(rng, 256, 0.9, 16, 2048) + (rng.randint(8, 96) if rng.random() < 0.3 else 0)
            turns.append(turn(append, rng.randint(24, 192), rng.expovariate(2.0)))
        out.append({"id": i, "arrival": round(t, 4), "system": system, "turns": turns})
        t += rng.expovariate(rate)
    return out


def context_len(session: dict) -> int:
    """Tokens in the session's last turn once generated: the longest context it reaches."""
    return session["system"] + sum(t["append"] + t["output"] for t in session["turns"])


def load(path: str | pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def write(path: pathlib.Path, sessions: list[dict]) -> None:
    path.write_text("".join(json.dumps(s, separators=(",", ":")) + "\n" for s in sessions))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent / "traces"))
    ap.add_argument("--seed", type=int, default=0)
    # 256 + 256 tokens keeps batch 64 of Qwen3-8B inside the ~5 GiB of KV a 24 GB card has left after weights.
    ap.add_argument("--decode-prompt", type=int, default=256)
    ap.add_argument("--decode-output", type=int, default=256)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = {f"decode_b{b}": decode(b, args.decode_prompt, args.decode_output) for b in (1, 8, 64, 256)}
    traces["sharegpt"] = sharegpt(200, 2.0, args.seed)
    traces["agent"] = agent(24, 0.2, args.seed)
    # Three streams fit every lock (max_seqs >= 4); 30k tokens fit a 32k context (M0.5's 36 s stall, M0.7 D1).
    traces["long_prompt"] = long_prompt(3, 256, 1024, 30000, 2.0)
    for name, sessions in traces.items():
        write(out / f"{name}.jsonl", sessions)
        turns = sum(len(s["turns"]) for s in sessions)
        print(f"{name}: {len(sessions)} sessions, {turns} requests, max context "
              f"{max(map(context_len, sessions))} tokens")


if __name__ == "__main__":
    main()
