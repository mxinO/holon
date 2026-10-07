"""nano-vllm in-process: its LLMEngine (CUDA graphs for decode, prefix caching), one `step` at a time."""
from __future__ import annotations

from driver import Delta
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

PACKAGES = ("nano-vllm", "torch", "flash-attn", "triton")
GREEDY = 1e-5    # nano-vllm rejects temperature 0; this temperature makes its Gumbel-max sampler argmax in practice


class Backend:
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        self.config = {"max_num_seqs": max_batch, "max_model_len": max_model_len,
                       "max_num_batched_tokens": max_step_tokens}
        self.engine = LLMEngine(model, **self.config)
        self._rids: dict[int, int] = {}

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        seq = Sequence(prompt, SamplingParams(temperature=GREEDY, max_tokens=max_tokens, ignore_eos=True))
        self._rids[seq.seq_id] = rid
        self.engine.scheduler.add(seq)

    def step(self) -> list[Delta]:
        # LLMEngine.step() returns finished sequences only; these are its three calls, keeping the scheduled
        # sequences to report each new token.
        sched = self.engine.scheduler
        seqs, is_prefill = sched.schedule()
        before = [s.num_tokens for s in seqs]
        token_ids = self.engine.model_runner.call("run", seqs, is_prefill)
        sched.postprocess(seqs, token_ids, is_prefill)
        return [(self._rids[s.seq_id], [s.last_token], s.is_finished) for s, n in zip(seqs, before, strict=True)
                if s.num_tokens > n]

    def has_work(self) -> bool:
        return not self.engine.is_finished()

    def info(self) -> dict:
        cfg = self.engine.model_runner.config
        return {"config": self.config | {"kvcache_block_size": cfg.kvcache_block_size},
                "kv_tokens": cfg.num_kvcache_blocks * cfg.kvcache_block_size}
