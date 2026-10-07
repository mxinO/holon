"""Holon's in-process engine (the checkout the bench belongs to; `run.py` puts it first on the path)."""
from __future__ import annotations

import dataclasses

from driver import Delta
from holon.api import SamplingParams
from holon.core import platform
from holon.core.engine import Engine, EngineConfig

PACKAGES = ("torch", "triton", "flashinfer-python")


class Backend:
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        self.engine = Engine(EngineConfig(model=model, max_seqs=max_batch, max_model_len=max_model_len,
                                          max_step_tokens=max_step_tokens))
        self._rids: dict[str, int] = {}

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        self._rids[self.engine.add(prompt, params)] = rid

    def step(self) -> list[Delta]:
        out = []
        for d in self.engine.step():
            done = d.finish_reason is not None
            out.append((self._rids.pop(d.rid) if done else self._rids[d.rid], d.token_ids, done))
        return out

    def has_work(self) -> bool:
        return self.engine.has_work()

    def info(self) -> dict:
        pools = [p for p in self.engine.ledger.pools.values() if p.tokens_per_unit]
        cfg = {k: v for k, v in dataclasses.asdict(self.engine.cfg).items() if k != "model"}
        return {"config": cfg,
                "fingerprint": platform.fingerprint(self.engine.device).cls,
                "kv_tokens": min(p.capacity * p.tokens_per_unit for p in pools) if pools else None}
