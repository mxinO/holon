"""A fixed number of drafts per step by how many requests run (docs/dev/m04-spec.md D7), from the deployment's plugin
options:

- `spec.k`: `"<k>"`, or a table `"<rows>:<k>,<rows>:<k>,…"` read as "from this many running requests on, k" (e.g.
  `"1:3,3:2"`: k = 3 for one or two requests, 2 from three on);
- `spec.fallback`: `n` > 0 makes a step plain decoding (k = 0) when no running request has had a real draft in its last
  n steps (a drafter that often finds nothing, such as prompt lookup); the drafter keeps drafting on those steps, so
  the first hit turns speculation back on. Only for such drafters: one that drafts on verify steps alone (a draft
  head, MTP) would never turn it back on.
"""
from __future__ import annotations


def _table(spec: str) -> list[tuple[int, int]]:
    if ":" not in spec:
        return [(1, int(spec))]
    pairs = sorted((int(rows), int(k)) for rows, k in (item.split(":") for item in spec.split(",")))
    if pairs[0][0] != 1 or any(k < 0 for _, k in pairs):
        raise ValueError(f"spec.k {spec!r}: a table from 1 running request on, k >= 0")
    return pairs


class Plugin:
    def register(self, regs) -> None:
        pass

    def build(self, ctx):
        opts = ctx.options.get("plugins", {})
        return FixedK(_table(str(opts.get("spec.k", "3"))), int(opts.get("spec.fallback", 0)))


class FixedK:
    def __init__(self, table: list[tuple[int, int]], fallback: int) -> None:
        self.table, self.fallback = table, fallback
        self.dry: dict[int, int] = {}          # per request (by id): consecutive steps without a real draft

    def k(self, seqs) -> int:
        k = next((k for rows, k in reversed(self.table) if len(seqs) >= rows), 0)
        if self.fallback:
            self.dry = {id(s): self.dry.get(id(s), 0) for s in seqs}          # finished requests drop out
            if seqs and all(n >= self.fallback for n in self.dry.values()):
                return 0
        return k

    def ks(self) -> set[int]:
        """The k values this policy may answer (the engine captures a graph per verify width)."""
        return {k for _, k in self.table}

    def observe(self, seq, k: int, n_draft: int, n_accept: int) -> None:
        if self.fallback:
            self.dry[id(seq)] = 0 if n_draft else self.dry.get(id(seq), 0) + 1
