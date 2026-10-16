"""OpenAI request bodies and their mapping to SamplingParams and, for chat, to the pipeline's `Conversation`.

Unknown fields are ignored, since OpenAI clients send many; known ones we cannot honour yet are rejected rather than
silently changing what a request means. Extensions follow vLLM's names (top_k, min_p, repetition_penalty,
stop_token_ids, ignore_eos, chat_template_kwargs, return_token_ids). `developer` messages are system messages and
`reasoning_effort` sets the template's effort level (`pipeline.system_first`, `pipeline.Efforts`).
"""
from __future__ import annotations

from typing import Any, get_args

from holon.api import SamplingParams
from pydantic import BaseModel, Field, field_validator, model_validator

from .pipeline import Conversation

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
    return_token_ids: bool = False          # each choice (and stream chunk) also carries its generated token ids

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key, unused in _UNSUPPORTED.items():
            if key not in cls.model_fields and data.get(key) not in (None, unused, {}):
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

    def _params(self, max_tokens: int | None, logprobs: int | None, prompt_logprobs: int | None = None,
                ext: dict[str, Any] | None = None) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature, top_p=self.top_p, top_k=max(self.top_k, 0), min_p=self.min_p,
            presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty, n=self.n, max_tokens=max_tokens, ignore_eos=self.ignore_eos,
            stop_token_ids=tuple(self.stop_token_ids), seed=self.seed, logprobs=logprobs,
            prompt_logprobs=prompt_logprobs, ext=ext or {})


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
    tool_choice: str | dict[str, Any] | None = None     # default: "auto" with tools, else "none"
    parallel_tool_calls: bool = True
    response_format: dict[str, Any] | None = None       # read as a constraint
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=20)
    chat_template_kwargs: dict[str, Any] = {}
    reasoning_effort: str | None = None                 # to the template's level (pipeline.Efforts)

    @field_validator("messages")
    @classmethod
    def _flatten(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Content given as parts becomes text: chat templates expect a string."""
        return [{**m, "content": _text(m["content"])} if isinstance(m.get("content"), list) else m
                for m in messages]

    @field_validator("tools")
    @classmethod
    def _functions(cls, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        for t in tools or []:
            fn = t.get("function")
            if t.get("type") != "function" or not isinstance(fn, dict) or not isinstance(fn.get("name"), str) \
                    or not isinstance(fn.get("parameters", {}), dict | None):
                raise ValueError("tools are function tools: {type: function, function: {name, parameters, …}}")
        return tools or None

    @field_validator("tool_choice")
    @classmethod
    def _choice(cls, choice: Any) -> str | dict[str, str] | None:
        return tool_choice(choice)

    @field_validator("response_format")
    @classmethod
    def _format(cls, fmt: Any) -> dict[str, Any] | None:
        return response_format(fmt)

    def conversation(self) -> Conversation:
        choice = self.tool_choice or ("auto" if self.tools else "none")
        return Conversation(self.messages, self.tools or [], choice, self.parallel_tool_calls, self.response_format,
                            self.chat_template_kwargs, self.reasoning_effort)

    def params(self, ext: dict[str, Any]) -> SamplingParams:
        logprobs = (self.top_logprobs or 0) if self.logprobs else None
        return self._params(self.max_completion_tokens or self.max_tokens, logprobs, ext=ext)


def tool_choice(choice: Any) -> str | dict[str, str] | None:
    """OpenAI's `tool_choice` (chat: `{type: function, function: {name}}`; Responses: `{type: function, name}`) as
    the pipeline's."""
    if choice is None or choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or choice).get("name")
        if isinstance(name, str):
            return {"name": name}
    raise ValueError(f"tool_choice {choice!r}: auto, none, required or a function")


def response_format(fmt: Any) -> dict[str, Any] | None:
    """OpenAI's `response_format` (chat: `{type, json_schema: {name, schema, strict}}`; Responses' `text.format`:
    `{type, name, schema, strict}`) as a constraint (D6), or None for text."""
    kind = fmt.get("type") if isinstance(fmt, dict) else None
    if fmt is None or kind == "text":
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    spec = fmt.get("json_schema", fmt) if kind == "json_schema" else None
    if not isinstance(spec, dict) or not isinstance(spec.get("schema", {}), dict | bool):
        raise ValueError(f"response_format {fmt!r}: text, json_object or json_schema {{name, schema, strict}}")
    return {"type": "json_schema", "schema": spec.get("schema", {}), "strict": bool(spec.get("strict"))}


def _text(parts: list[Any]) -> str:
    if any(not isinstance(p, dict) or p.get("type") != "text" for p in parts):
        raise ValueError("only text content parts are supported")
    return "".join(p["text"] for p in parts)
