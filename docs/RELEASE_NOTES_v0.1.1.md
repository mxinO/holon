# Holon v0.1.1

A patch release: three fixes to the OpenAI-compatible and Anthropic endpoints, found when the pi coding agent drove
`holon serve` on an RTX 5090 (`docs/agent-test-pi-rtx5090.md`). Everything else is as in
[v0.1.0](RELEASE_NOTES_v0.1.0.md).

## Fixes

- **`developer` messages are system messages.** `/v1/chat/completions` passed them to the chat template as they were:
  the HF Qwen3.8 and Qwen3.5 templates refused them (400) and the Qwen3 templates dropped them, with an agent's whole
  system prompt. pi sends its system prompt this way for reasoning models. Every route now renders them as `system`;
  a leading run of system and developer messages that holds one becomes one system prompt.
- **`reasoning_effort` takes effect.** Chat's `reasoning_effort`, the Responses API's `reasoning.effort` and
  Anthropic's `output_config.effort` were ignored. They now set the chat template's level where it has levels (the
  Qwen3.8 templates): a level the template has is used as it is; another level of the clients' scale (`minimal` …
  `max`) takes the nearest one it has (`minimal` the lowest, the higher of two as near), so pi's and Claude Code's
  fixed scales keep working; `none` turns thinking off; any other value is a 400 naming the levels. Templates without
  levels (Qwen3, Qwen3.5) ignore the scale's values, as before.
- **Usage reports the prompt tokens the prefix cache served**, per request: `prompt_tokens_details.cached_tokens`
  (chat and text completions, streamed or not), `input_tokens_details.cached_tokens` (Responses), and Anthropic's
  `cache_read_input_tokens`, with `input_tokens` now the rest of the prompt (Anthropic's convention: clients add the
  two, so Claude Code's displayed input counts change). The engine counts a request's prefix hit at its first
  admission; a request resumed after an engine restart keeps its first count.

## Plugin ABI 1.1

`TokenDelta` gains `cached_tokens: int | None = None`, appended with a default: an additive change, so plugins
declaring `core_abi = ">=1.0,<2.0"` load unchanged (`core/abi.txt`).

## Locks

The core change moved the core's `src`, which every lock pins, so all 22 lock histories were carried to v0.1.1's
code on one clock-screened desktop per card: 22 of 22 approved, each with the judge's L2 inside its floor budget, a
snapshot trace byte-identical to the previous lock's and run-identical over three runs
(`evidence/v011_lock_propagation_2026-10-16.json`).

| card | locks | sweep |
|---|---|---|
| RTX 3090 | 4 | 11 min 23 s |
| RTX 4090 | 8 | 19 min 23 s |
| RTX 5080 | 6 | 8 min 19 s |
| RTX 5090 | 4 | 9 min 31 s |

Performance is unchanged: the same kernels, settings and outputs. The pi re-run on the RTX 5090's new lock
(`evidence/agent_test_pi_v011_rtx-5090_2026-10-16.json`) passed the demo task at three thinking levels, with the level
changing the rendered prompt and pi reporting the server's cache reads.

## Install

```bash
git clone --branch v0.1.1 https://github.com/mxinO/holon.git holon && cd holon
uv sync --frozen
uv run holon prebuild
uv run holon serve unsloth/Qwen3.8-27B-GGUF
```
