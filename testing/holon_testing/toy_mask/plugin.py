"""A toy logits processor that proves the core's processor seams (docs/dev/m05-grammar.md §3) without a grammar: once a
request's marker token has been seen (in its prompt or output), only even tokens. Its masks depend on the committed
tokens and, along a verify row, on the drafts before each column, so a mask built from a stale or shifted state lets
a wrong token through; and it checks what the core promises it — every row's tokens committed before the row is
launched, the tokens each commit appends.

A request opts in with `ext["toy.even_after@1"] = {"marker": t}`, plus test knobs: `polls` (ready after that many
polls), `busy` (`ready` fails: a compile that timed out), `reject` (`open` fails), `fail_at` ((index, n): `commit`
fails once sequence `index` has n output tokens). Its `apply` is a dense masked fill (capturable)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from holon.api import RequestError

KEY = "toy.even_after@1"
EVEN, ALL = 0x55555555, -1              # mask words: bit t of word w allows token 32·w + t


class Plugin:
    def register(self, regs: Any) -> None:
        pass

    def build(self, ctx: Any, vocab: Any) -> ToyMask:
        return ToyMask(vocab)


class ToyMask:
    def __init__(self, vocab: Any) -> None:
        self.mask_width = -(-vocab.size // 32)
        self.opts: dict[str, dict[str, Any]] = {}      # rid -> its options
        self.seen: dict[tuple[str, int], bool] = {}    # (rid, index) -> marker committed (or in the prompt)
        self.out: dict[tuple[str, int], list[int]] = {}   # (rid, index) -> every token committed, in order
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.rows: list[tuple[str, int, int, list[int]]] = []   # filled rows: rid, index, columns, drafts

    def open(self, rid: str, params: Any, prompt: list[int]) -> bool:
        opts = params.ext.get(KEY)
        if opts is None:
            return False
        if opts.get("reject"):
            raise RequestError("client", "the toy rejects this request")
        self.opts[rid] = {**opts, "prompt_seen": opts["marker"] in prompt, "polls": opts.get("polls", 0)}
        self.opened.append(rid)
        return True

    def ready(self, rid: str) -> bool:
        opts = self.opts[rid]
        if opts.get("busy"):
            raise RequestError("busy", "the toy's compile timed out")
        opts["polls"] -= 1
        return opts["polls"] < 0

    def commit(self, seq: Any, out: list[int]) -> None:
        key, opts = (seq.rid, seq.index), self.opts[seq.rid]
        done = self.out.setdefault(key, [])
        if (fail := opts.get("fail_at")) and fail[0] == seq.index and len(done) + len(out) >= fail[1]:
            raise RequestError("engine", "the toy refuses a token")
        seen = self.seen.get(key, opts["prompt_seen"])
        for t in out:
            if seen and t % 2:
                raise RequestError("engine", f"token {t} after the marker")
            seen |= t == opts["marker"]
        self.seen[key] = seen
        done += out

    def close(self, rid: str) -> None:
        self.closed.append(rid)

    def fill(self, rows: list[Any], buf: np.ndarray) -> None:
        for row in rows:
            seq, opts = row.seq, self.opts[row.seq.rid]
            assert seq.pending == 0 and seq.inexact == 0, "a constrained row launched before its tokens committed"
            seen = self.seen.get((seq.rid, seq.index), opts["prompt_seen"])
            for j in range(row.n):              # column j masks the token after drafts[:j]
                if 0 < j <= len(row.drafts):
                    seen |= row.drafts[j - 1] == opts["marker"]
                buf[row.start + j] = EVEN if seen else ALL
            self.rows.append((seq.rid, seq.index, row.n, list(row.drafts)))

    def stats(self) -> dict[str, int]:
        return {"rows_filled": len(self.rows)}

    def apply(self, logits: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor) -> None:
        r, v = logits.shape
        bits = mask[:, :, None] >> torch.arange(32, dtype=torch.int32, device=mask.device)
        allowed = torch.ones(r + 1, v, dtype=torch.bool, device=logits.device)   # row r takes the padding rows
        allowed[rows.clamp(max=r)] = (bits & 1).bool().flatten(1)[:, :v]
        logits.masked_fill_(~allowed[:r], float("-inf"))
