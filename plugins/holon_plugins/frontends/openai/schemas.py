"""OpenAI request bodies and their mapping to SamplingParams.

Unknown fields are ignored, since OpenAI clients send many; known ones we cannot honour yet are rejected rather than
silently changing what a request means. Extensions follow vLLM's names (top_k, min_p, repetition_penalty,
stop_token_ids, ignore_eos, chat_template_kwargs).
"""
from __future__ import annotations

from typing import Any, get_args

from holon.api import SamplingParams
from pydantic import BaseModel, Field, field_validator, model_validator

_UNSUPPORTED = {"logit_bias": None, "suffix": None, "best_of": 1, "response_format": {"type": "text"},
                "tool_choice": "auto"}   # field → the one value that means "feature not used"


class StreamOptions(BaseModel):
    include_usage: bool = False


class GenerationRequest(BaseModel):
    model: str | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = Field(1, ge=1)
    temperature: float = Field(1.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=-1)            # 0 or -1: off
    min_p: float = Field(0.0, ge=0, le=1)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = Field(1.0, gt=0)
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] = []
    ignore_eos: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key, unused in _UNSUPPORTED.items():
            if data.get(key) not in (None, unused, {}):
                raise ValueError(f"{key!r} is not supported")
        # Clients send null for "unset": a field that cannot be null takes its default instead.
        fields = cls.model_fields
        return {k: v for k, v in data.items()
                if v is not None or k not in fields or type(None) in get_args(fields[k].annotation)}

    @property
    def stops(self) -> list[str]:
        stop = [self.stop] if isinstance(self.stop, str) else self.stop or []
        return [s for s in stop if s]       # "" would match at once

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    def _params(self, max_tokens: int | None, logprobs: int | None, prompt_logprobs: int | None = None
                ) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature, top_p=self.top_p, top_k=max(self.top_k, 0), min_p=self.min_p,
            presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty, n=self.n, max_tokens=max_tokens, ignore_eos=self.ignore_eos,
            stop_token_ids=tuple(self.stop_token_ids), seed=self.seed, logprobs=logprobs,
            prompt_logprobs=prompt_logprobs)


class CompletionRequest(GenerationRequest):
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int | None = Field(16, ge=1)    # OpenAI's default for completions
    echo: bool = False
    logprobs: int | None = Field(None, ge=0, le=20)

    @property
    def prompts(self) -> list[str | list[int]]:
        """One entry per prompt of a batch; each is text or token ids."""
        p = self.prompt
        if isinstance(p, str) or (p and isinstance(p[0], int)):
            return [p]   # type: ignore[list-item]
        return list(p)   # type: ignore[arg-type]

    def params(self) -> SamplingParams:
        # Echoed logprobs score the prompt as well (the lm-eval loglikelihood recipe).
        return self._params(self.max_tokens, self.logprobs, self.logprobs if self.echo else None)


class ChatRequest(GenerationRequest):
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=20)
    chat_template_kwargs: dict[str, Any] = {}

    @field_validator("messages")
    @classmethod
    def _flatten(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Content given as parts becomes text: chat templates expect a string."""
        return [{**m, "content": _text(m["content"])} if isinstance(m.get("content"), list) else m
                for m in messages]

    def params(self) -> SamplingParams:
        logprobs = (self.top_logprobs or 0) if self.logprobs else None
        return self._params(self.max_completion_tokens or self.max_tokens, logprobs)


def _text(parts: list[Any]) -> str:
    if any(not isinstance(p, dict) or p.get("type") != "text" for p in parts):
        raise ValueError("only text content parts are supported")
    return "".join(p["text"] for p in parts)
