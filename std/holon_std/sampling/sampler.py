"""Default sampler: vectorized over rows, with per-slot parameters held in device tensors.

Random draws are counter-based: one uniform per row, a hash of (seed, position), picks the token by inverse CDF
over the row's (filtered) distribution. A request's samples therefore depend on neither batching nor preemption,
there is no generator state, and the step is capturable in a graph. Semantics: `reference.probs`.

Resolution: the uniform has 24 bits and the CDF is an fp32 cumulative sum, so tokens are drawn with their oracle
probability up to about 1e-7 absolute (one fp32 step of the running sum); rarer tokens may be over- or under-drawn.
"""
from __future__ import annotations

import numpy as np
import torch
from holon.api import SampleOut, SamplingParams

_M32 = 0xFFFFFFFF


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (lowbias32) on int64 tensors holding values below 2**32."""
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _M32
    return x ^ (x >> 16)


def uniform(seeds: torch.Tensor, positions: torch.Tensor, stream: int = 0) -> torch.Tensor:
    """One uniform in (0, 1) per row, a pure function of (seed, position, stream); stream 0 is the sampler's own
    draw, other streams are independent ones (e.g. a speculative verifier's acceptance coins)."""
    h = _mix32(seeds & _M32) ^ (seeds >> 32 & _M32) ^ _mix32(positions & _M32)
    if stream:
        h = h ^ _mix32(torch.full_like(positions, (0x9E3779B9 * stream) & _M32))
    h = _mix32(h)
    return (((h >> 8).float() + 0.5) * 2.0**-24).clamp_(max=1 - 2.0**-24)    # the top value rounds to 1.0 in fp32


def _distribution(x: torch.Tensor, top_k: torch.Tensor | None, top_p: torch.Tensor | None,
                  min_p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(probabilities, token order): unfiltered rows keep vocabulary order (order None); filtered rows are sorted
    by logit, with dropped tokens at probability 0 (rows of temperature-scaled logits [R, V])."""
    if top_k is None:
        return x.softmax(-1), None
    sorted_x, order = x.sort(-1, descending=True)
    rank = torch.arange(x.shape[-1], device=x.device)[None, :]
    sorted_x = sorted_x.masked_fill((top_k[:, None] > 0) & (rank >= top_k[:, None]), float("-inf"))
    q = sorted_x.softmax(-1)
    drop = ((top_p[:, None] < 1) & (q.cumsum(-1) - q > top_p[:, None])) | (q < min_p[:, None] * q[:, :1])
    return q.masked_fill(drop, 0.0), order


def draw(q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse CDF: the first index whose cumulative mass exceeds u × total (zero-mass entries are never chosen)."""
    c = q.cumsum(-1)
    idx = torch.searchsorted(c, (u * c[:, -1])[:, None], right=True)[:, 0]
    return idx.clamp_(max=q.shape[-1] - 1)


class Sampler:
    def __init__(self, max_seqs: int, device: torch.device) -> None:
        self.temperature = torch.zeros(max_seqs, dtype=torch.float32, device=device)
        self.top_k = torch.zeros(max_seqs, dtype=torch.int64, device=device)
        self.top_p = torch.ones(max_seqs, dtype=torch.float32, device=device)
        self.min_p = torch.zeros(max_seqs, dtype=torch.float32, device=device)
        self.seed = torch.zeros(max_seqs, dtype=torch.int64, device=device)
        # Host mirrors decide which work a step needs (shapes and branches stay host-side).
        self._greedy = np.ones(max_seqs, dtype=bool)
        self._filtered = np.zeros(max_seqs, dtype=bool)
        self._logprobs = np.full(max_seqs, -1, dtype=np.int64)     # -1: no logprobs

    def admit(self, slot: int, params: SamplingParams) -> None:
        if params.seed is None:
            raise ValueError("the engine gives every request a concrete seed")
        self.temperature[slot], self.top_k[slot] = params.temperature, params.top_k
        self.top_p[slot], self.min_p[slot] = params.top_p, params.min_p
        self.seed[slot] = params.seed
        self._greedy[slot] = params.temperature == 0
        self._filtered[slot] = params.top_k > 0 or params.top_p < 1 or params.min_p > 0
        self._logprobs[slot] = -1 if params.logprobs is None else params.logprobs

    def variant(self, slots: np.ndarray) -> tuple:
        if self._greedy[slots].all():
            return ("greedy", int(self._logprobs[slots].max()))
        return ("sample", bool(self._filtered[slots].any()), int(self._logprobs[slots].max()))

    def distribution(self, logits: torch.Tensor, ctx) -> torch.Tensor:
        """The probabilities `sample` draws each logits row from, in vocabulary order [R, V] (greedy rows: one-hot at
        the argmax)."""
        plan, dev = ctx.batch.plan, ctx.batch.dev
        slots = plan.slots[plan.n_logits > 0]
        rows = dev.slots[dev.logits_rows]
        greedy = torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        if self._greedy[slots].all():
            return greedy
        t = self.temperature[rows]
        x = logits / torch.where(t == 0, 1.0, t)[:, None]
        filters = (self.top_k[rows], self.top_p[rows], self.min_p[rows]) if self._filtered[slots].any() else (
            None, None, None)
        q, order = _distribution(x, *filters)
        if order is not None:                           # filters drop mass: renormalize what is left
            q = torch.zeros_like(q).scatter_(-1, order, q / q.sum(-1, keepdim=True))
        return torch.where((t == 0)[:, None], greedy, q)

    def uniform(self, ctx, stream: int = 0) -> torch.Tensor:
        """Per logits row, the uniform of stream `stream` at the position of the token it draws."""
        dev = ctx.batch.dev
        return uniform(self.seed[dev.slots[dev.logits_rows]], dev.positions[dev.logits_index].long() + 1, stream)

    def logprobs(self, logits: torch.Tensor, tokens: torch.Tensor, ctx) -> SampleOut:
        """`tokens` with the logprobs the step's requests asked for (none asked: tokens alone)."""
        plan = ctx.batch.plan
        k = int(self._logprobs[plan.slots[plan.n_logits > 0]].max())
        if k < 0:
            return SampleOut(tokens)
        logprobs = logits.log_softmax(-1)
        top = logprobs.topk(max(k, 1))
        return SampleOut(tokens, logprobs.gather(-1, tokens[:, None])[:, 0], top.indices, top.values)

    def sample(self, logits: torch.Tensor, ctx) -> SampleOut:
        plan, dev = ctx.batch.plan, ctx.batch.dev
        slots = plan.slots[plan.n_logits > 0]           # host mirror: which work this step needs
        rows = dev.slots[dev.logits_rows]                # device: parameter rows (graph-capturable)
        tokens = logits.argmax(-1)
        if not self._greedy[slots].all():
            t = self.temperature[rows]
            x = logits / torch.where(t == 0, 1.0, t)[:, None]
            filters = (self.top_k[rows], self.top_p[rows], self.min_p[rows]) if self._filtered[slots].any() else (
                None, None, None)
            q, order = _distribution(x, *filters)
            positions = dev.positions[dev.logits_index].long() + 1     # the position of the token being drawn
            drawn = draw(q, uniform(self.seed[rows], positions))
            if order is not None:
                drawn = order.gather(-1, drawn[:, None])[:, 0]
            tokens = torch.where(t == 0, tokens, drawn)
        k = int(self._logprobs[slots].max())
        if k < 0:
            return SampleOut(tokens)
        logprobs = logits.log_softmax(-1)
        top = logprobs.topk(max(k, 1))
        return SampleOut(tokens, logprobs.gather(-1, tokens[:, None])[:, 0], top.indices, top.values)


class Plugin:
    """The `sampler` slot's plugin: creates one sampler per engine."""

    def register(self, regs) -> None:
        pass

    def sampler(self, max_seqs: int, device: torch.device) -> Sampler:
        return Sampler(max_seqs, device)
