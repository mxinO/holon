"""mini-sglang in-process: its offline scheduler (overlap scheduling, CUDA graphs, radix cache — the defaults of
`python -m minisgl`), driven one `overlap_loop` iteration per step instead of `run_forever`."""
from __future__ import annotations

import torch
from driver import Delta
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.message import DetokenizeMsg, UserMsg

PACKAGES = ("minisgl", "torch", "flashinfer-python", "sgl-kernel")


class _Offline(LLM):
    """The offline scheduler with its I/O hooks pointed at in-memory queues (no ZMQ, no tokenizer process)."""

    inbox: list[UserMsg]
    outbox: list[DetokenizeMsg]

    def offline_receive_msg(self, blocking: bool = False) -> list[UserMsg]:
        msgs, self.inbox = self.inbox, []        # never block: the driver owns waiting for arrivals
        return msgs

    def offline_send_result(self, reply: list[DetokenizeMsg]) -> None:
        self.outbox += reply


class Backend:
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        self.config = {"max_running_req": max_batch, "cuda_graph_max_bs": max_batch,
                       "max_seq_len_override": max_model_len, "max_extend_tokens": max_step_tokens}
        self.llm = _Offline(model, **self.config)
        self.llm.inbox, self.llm.outbox = [], []
        self._last = None                         # the batch in flight (overlap loop state)
        self._left: dict[int, int] = {}           # tokens still due per request

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        self._left[rid] = max_tokens
        params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)
        self.llm.inbox.append(UserMsg(uid=rid, input_ids=torch.tensor(prompt, dtype=torch.int32),
                                      sampling_params=params))

    @torch.inference_mode()
    def step(self) -> list[Delta]:
        self._last = self.llm.overlap_loop(self._last)
        out, self.llm.outbox = self.llm.outbox, []
        # Under overlap scheduling mini-sglang flags `finished` one token early: the last token comes from the batch
        # already in flight, and its own offline generate() keeps it. With ignore_eos the length alone ends a request.
        deltas = []
        for m in out:
            if m.uid in self._left:
                self._left[m.uid] -= 1
                done = not self._left[m.uid]
                if done:
                    del self._left[m.uid]
                deltas.append((m.uid, [m.next_token], done))
        return deltas

    def has_work(self) -> bool:
        s = self.llm
        return bool(s.inbox) or self._last is not None or s.prefill_manager.runnable or s.decode_manager.runnable

    def info(self) -> dict:
        engine = self.llm.engine
        return {"config": self.config | {"attention": type(engine.attn_backend).__name__,
                                         "page_size": engine.ctx.page_size},
                "kv_tokens": engine.num_pages * engine.ctx.page_size}
