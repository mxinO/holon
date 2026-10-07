"""Prompt sets and L2 recordings (design §10.3; format validated by spike S5).

A recording holds teacher-forced top-k logprobs for every scored position of a frozen prompt set. All positions of
all sequences are concatenated; position t of a sequence predicts its token t+1, so a sequence of n tokens owns
n-1 rows:

    topk_ids   int32 [N, k]   top-k token ids, logprob-descending
    topk_lp    fp32  [N, k]   their fp32 log-softmax values
    target_lp  fp32  [N]      logprob of the actual next token
    tail_lp    fp32  [N]      log of the probability mass outside the top-k (the tail bucket of the KL estimate)
    offsets    int64 [S+1]    sequence s owns rows offsets[s]:offsets[s+1]

The safetensors header key "meta" holds JSON provenance: model, revision, weight hashes, dtype, device, versions,
thread count, prompt-set name and sha256, and the per-sequence ids and categories.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TENSORS = ("topk_ids", "topk_lp", "target_lp", "tail_lp", "offsets")


@dataclasses.dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    ids: list[int]


@dataclasses.dataclass(frozen=True)
class PromptSet:
    name: str
    sha256: str
    prompts: list[Prompt]

    @classmethod
    def load(cls, path: str | pathlib.Path) -> PromptSet:
        """Load a JSONL prompt set; a `<path>.sha256` sidecar, if present, must match (the set is frozen)."""
        path = pathlib.Path(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        sidecar = path.with_name(path.name + ".sha256")
        if sidecar.is_file() and sidecar.read_text().split()[:1] != [digest]:
            raise ValueError(f"{path.name}: sha256 {digest} does not match {sidecar.name}")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        prompts = [Prompt(r["id"], r["category"], r["ids"]) for r in rows]
        if short := [p.id for p in prompts if len(p.ids) < 2]:
            raise ValueError(f"{path.name}: prompts {short} have nothing to score (fewer than 2 tokens)")
        return cls(path.name, digest, prompts)

    def meta(self) -> dict:
        return {"prompts": self.name, "prompts_sha256": self.sha256,
                "seq_ids": [p.id for p in self.prompts], "categories": [p.category for p in self.prompts]}


@dataclasses.dataclass
class Recording:
    topk_ids: torch.Tensor
    topk_lp: torch.Tensor
    target_lp: torch.Tensor
    tail_lp: torch.Tensor
    offsets: torch.Tensor
    meta: dict

    @property
    def k(self) -> int:
        return self.topk_ids.shape[1]

    @classmethod
    def from_rows(cls, rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                  meta: dict) -> Recording:
        """Concatenate per-sequence (topk_ids, topk_lp, target_lp, tail_lp) rows in prompt-set order."""
        ids, lp, tgt, tail = zip(*rows, strict=True)
        offsets = torch.tensor([0, *(len(t) for t in tgt)]).cumsum(0)
        return cls(torch.cat(ids).int(), torch.cat(lp).float(), torch.cat(tgt).float(), torch.cat(tail).float(),
                   offsets, dict(meta, k=ids[0].shape[1], n_positions=int(offsets[-1])))

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file({name: getattr(self, name).contiguous() for name in TENSORS}, str(path),
                  metadata={"meta": json.dumps(self.meta, sort_keys=True)})

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Recording:
        with safe_open(str(path), "pt") as f:
            return cls(*(f.get_tensor(name) for name in TENSORS), json.loads(f.metadata()["meta"]))

    def seq_index(self) -> torch.Tensor:
        """Sequence index of every row."""
        return torch.repeat_interleave(torch.arange(len(self.offsets) - 1), self.offsets.diff())

    def same_tensors(self, other: Recording) -> bool:
        return all(torch.equal(getattr(self, n), getattr(other, n)) for n in TENSORS)


def tail_from_topk(topk_lp: torch.Tensor) -> torch.Tensor:
    """Tail log-mass when only the top-k is known (e.g. from an engine): log(1 − Σ p_topk), kept positive because
    fp32 rounding can push the sum past 1. Accurate to ~1e-7 in mass, far below any L2 budget; the recorder
    computes it exactly from the full distribution instead."""
    rest = (1 - topk_lp.double().exp().sum(-1)).clamp_min(1e-12)
    return rest.log().float()
