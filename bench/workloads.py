"""Deterministic benchmark workloads, stored as JSONL token-length specs (one session per line).

A session is a list of turns, released at `arrival` seconds, or with `after` that many seconds after session `after`'s
last turn finished (a closed loop: one client running sessions one after another). Sessions with a `client` are served
round-robin by that client, one request at a time: its first session starts at `arrival`, and when a request of it
finishes, its next session in turn (by id, skipping finished ones) sends its next turn `gap` seconds later. Turn 0's
prompt is `system` tokens shared by every session of the trace followed by `append` tokens of its own; turn i > 0 is
released `gap` seconds after turn i-1 finished and its prompt is turn i-1's prompt + the tokens turn i-1 generated +
`append` new tokens (a tool result and/or user message). Every turn generates exactly `output` tokens (ignore_eos).
A plain request is a one-turn session without a system prompt.

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


def appends(salt: str, session: dict, corpus: list[int] | None = None) -> list[list[int]]:
    """Each turn's new tokens. Uniform ids per (session, turn); on a corpus, one walk from a seeded offset through the
    text outside the system prompt's span (`tokens(salt, "system", …)`), so no context repeats a span of itself, as a
    conversation's own text would not (overlapping draws made half of a 25K-token context verbatim repeats, which
    inflates speculative acceptance)."""
    turns, sid = session["turns"], session["id"]
    if not corpus:
        return [tokens(salt, f"{sid}.{i}", t["append"]) for i, t in enumerate(turns)]
    start = random.Random(f"{salt}:system").randrange(len(corpus))     # the system prompt's offset, as in `tokens`
    rest = (corpus[start:] + corpus[:start])[session["system"]:] if session["system"] else corpus
    need = sum(t["append"] for t in turns)
    if need > len(rest):
        raise ValueError(f"session {sid} appends {need} tokens; the corpus has {len(rest)} outside its system prompt")
    at, out = random.Random(f"{salt}:{sid}").randrange(len(rest)), []
    for t in turns:
        out.append([rest[(at + j) % len(rest)] for j in range(t["append"])])
        at += t["append"]
    return out


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


def agent(sessions: int, rate: float, seed: int, system: int = 1536, later: tuple[int, int] = (2, 7)) -> list[dict]:
    """Mixed agent multi-turn tool trace: sessions start as a Poisson process at `rate`/s and share a `system`
    prompt (instructions + tool schemas). The first turn adds a user task; each of the `later` (a range) turns adds a
    tool result (log-normal, median 256 tokens), sometimes followed by a user message, after a tool latency gap
    (exponential, mean 0.5 s). Generations are short tool calls or answers (24-192 tokens). Prefixes are long and
    shared."""
    rng, t, out = random.Random(seed), 0.0, []
    for i in range(sessions):
        turns = [turn(_lognormal(rng, 150, 0.6, 16, 1024), rng.randint(24, 192))]
        for _ in range(rng.randint(*later)):
            append = _lognormal(rng, 256, 0.9, 16, 2048) + (rng.randint(8, 96) if rng.random() < 0.3 else 0)
            turns.append(turn(append, rng.randint(24, 192), rng.expovariate(2.0)))
        out.append({"id": i, "arrival": round(t, 4), "system": system, "turns": turns})
        t += rng.expovariate(rate)
    return out


def agent_park(clients: int, per_client: int, seed: int, system: int = 4096) -> list[dict]:
    """Parked agent conversations (docs/v1-kv-tiers.md §8): `clients` closed-loop clients each serve `per_client`
    sessions round-robin, so a session waits while its client's other sessions take a turn and its state is parked.
    The sessions share a `system`-token prompt and open with a 512-token task; then 10–14 turns each add a tool result
    (log-normal, median 384, σ 0.6, at most 2K) to an output of reasoning plus a tool call (median 300, σ 0.6, at most
    2K, resent verbatim), after a gap of 0.5–3 s. Contexts end at ≈ 13–19K tokens."""
    rng, out = random.Random(seed), []
    for c in range(clients):
        for k in range(per_client):
            turns = [turn(512, _lognormal(rng, 300, 0.6, 16, 2048))]
            for _ in range(rng.randint(10, 14)):
                turns.append(turn(_lognormal(rng, 384, 0.6, 16, 2048), _lognormal(rng, 300, 0.6, 16, 2048),
                                  rng.uniform(0.5, 3.0)))
            out.append({"id": c * per_client + k, "arrival": round(2.0 * c, 4), "system": system, "client": c,
                        "turns": turns})
    return out


def lanes(sessions: list[dict], n: int, seed: int) -> list[dict]:
    """`n` clients each running its share of `sessions` one after another (closed loop): session i starts after
    session i - n has finished and a pause (exponential, mean 2 s); the first n start 5 s apart. However fast the
    engine, n sessions overlap."""
    rng = random.Random(seed)
    return [{**s, "arrival": round(5.0 * i if i < n else rng.expovariate(0.5), 4),
             **({"after": i - n} if i >= n else {})} for i, s in enumerate(sessions)]


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
    # Claude-Code-shaped (release D3's MTP decision): its ≈ 16K-token system prompt and tool schemas shared by 16
    # sessions of 11-17 turns (≥ 200 requests), run by two clients one after another, so two overlap on every card;
    # replayed on natural text (--corpus) at Claude Code's temperature.
    traces["claude_code"] = lanes(agent(16, 0.025, args.seed, system=16384, later=(10, 16)), 2, args.seed)
    # Three streams fit every lock (max_seqs >= 4); 30k tokens fit a 32k context (M0.5's 36 s stall, M0.7 D1).
    traces["long_prompt"] = long_prompt(3, 256, 1024, 30000, 2.0)
    # Milestone K's gate trace: 12 parked conversations over 4 clients (docs/v1-kv-tiers.md §8).
    traces["agent_park"] = agent_park(4, 3, args.seed)
    for name, sessions in traces.items():
        write(out / f"{name}.jsonl", sessions)
        turns = sum(len(s["turns"]) for s in sessions)
        print(f"{name}: {len(sessions)} sessions, {turns} requests, max context "
              f"{max(map(context_len, sessions))} tokens")


if __name__ == "__main__":
    main()
