# K-M1: re-render divergence (milestone K step 0, `docs/dev/v1-kv-tiers.md` §8)

**Question.** When an agent sends its next request, how much of the previous answer (the engine's own generated
tokens) does the re-rendered prompt repeat token for token? That sets what D8's `answer:<k>` checkpoints can save:
`answer`'s default k, whether it is on by default, and row 3g's prediction.

## Predictions (fixed before measuring)

- **3g.** The previous turn's own output was 44% (27,646 / 62,682) of the uncached input under `prompt_end` in the
  earlier pi test (`docs/dev/agent-test-pi-rtx5090.md`). With `answer:2` it falls to ≤ 10% **if** answers are
  re-rendered verbatim up to their tool calls.
- **Decision rule** (§8): `answer` is on by default only if ≥ 50% of answers are re-rendered verbatim to within one
  block (64 tokens) of a kept checkpoint. Prediction: with the Unsloth Qwen3.8 template and pi's
  `qwen-chat-template` thinking format, the last assistant turn's reasoning is kept verbatim in the next prompt and
  tool calls are re-serialized from JSON, so divergence lands inside or at the start of the tool-call markup:
  ≥ 50% within one block at k = 2 (default k = 2). This is the uncertain prediction; the record holds whatever is
  measured.

## Method

1. **Run.** The pi agent test of `docs/dev/agent-test-pi-rtx5090.md` once more on an RTX 5090 desktop: the same model
   file (Qwen3.8-27B UD-Q4_K_M) at its rtx-5090 lock, pi 0.73.1 with the counted runs' provider config
   (`--compat documented`), the `demo` and `workloads` tasks. `bench/demo/pi_run.py` (unchanged; its harness check
   wants a clean checkout) runs its own capture proxy; its `--server` is a second `bench/capture_proxy.py --raw` in front of
   `holon serve`. In `--raw` mode that proxy asks for `return_token_ids` and keeps each request's body, the engine's
   generated token ids and the reported usage, locally (never committed).
2. **Prompt ids.** `analyse.py` renders every request offline on the CPU with Holon's own chat frontend (the
   `ChatRequest` → `Chat` path of `plugins/holon_plugins/frontends/http`, the GGUF's bundle) and checks that each
   rendered length equals the server's reported `prompt_tokens`, and each generated id list's length its
   `completion_tokens`.
3. **Per turn** (a request i followed by request i+1 of the same session): `P` = prompt length, `G` = generated
   tokens (the terminating `<|im_end|>` included), `A = P + G` the answer's end; `LCP` = the longest common prefix of
   `prompt_i + generated_i` and `prompt_{i+1}`. The decode rows reach positions up to `A − 1` (the last token is
   sampled, never fed), so `answer:<k>` keeps the last k multiples of 64 in `(P, A − 1]`. `prompt_end` checkpoints
   at `⌊P / 64⌋ · 64`. A turn is "re-rendered verbatim to within one block of a kept checkpoint" at k when a kept
   answer checkpoint `c ≤ LCP` exists with `min(LCP, A − 1) − c < 64` (the `min`: a fully verbatim answer whose end
   is aligned cannot have a checkpoint there).
4. **3g.** A replay of the cache: every earlier request's checkpoints (the policy's) on its token sequence, KV
   assumed resident; a request's hit is the deepest checkpoint `c ≤ LCP(its sequence, the prompt)` and
   `c < prompt length`. Uncached input = prompt − hit; of it, the previous turn's own output is
   `G − clamp(hit − P, 0, G)` (under `prompt_end` alone that is `G`, the earlier test's definition). The replay of
   `prompt_end` is checked against the server's measured `cached_tokens`.
5. **Divergence site**, counts only: before the answer, in the reasoning, at or right after `</think>`, in text
   content, at a `<tool_call>` start, inside a tool call, at the end token, or nowhere (verbatim to the end).
6. **Stripped reasoning.** The same with history re-rendered by (a) the template's own default
   (`preserve_thinking` unset: it keeps the reasoning of every assistant turn after the last real user message, so
   in a one-task agent session it strips nothing) and (b) a copy of the template whose assistant turns never render
   their reasoning (an explicit stripping template), each prompt and its successor rendered alike, the engine's
   generated ids appended to the former.

## Results (RTX 5090, 2026-10-18; `evidence/k0_m1_rerender_rtx-5090_2026-10-18.json`)

demo ×3 and workloads ×2 (43 requests, 38 turn pairs). Every rendered prompt had the server's `prompt_tokens` and
every generated id list its `completion_tokens` (43 / 43); the `prompt_end` replay gave the server's `cached_tokens`
on 43 / 43.

| | predicted | measured |
|---|---|---|
| divergence | at or inside the tool-call markup | **none**: 38 / 38 answers re-rendered token for token through `<|im_end|>` (answer end − LCP = 0 on every turn; 11 answers with 2–5 calls, 16 with text before the first call) |
| within one block of a kept checkpoint | ≥ 50% at k = 2 | **100% at every k = 1…4** (k = 2…4 add nothing) |
| decision rule | | **met: `answer` on by default**; k = 1 suffices here |
| 3g: previous output / uncached input | 44% → ≤ 10% with `answer:2` | 39.0% (21,778 / 55,879) → **3.1%** (1,063 / 33,991) with `answer:<k>`, any k; uncached input −39% |
| template default (`preserve_thinking` unset) | | the same prompts (0 / 43 differ): one user message per session |
| a template stripping reasoning from history | | diverges 2 tokens before every answer (the generation prompt's `<think>\n`): 0 / 38 usable at any k |

What remains uncached with `answer` is each answer's tail past its last 64-aligned position (≤ 63 tokens) and the
tool results. One run per series, one agent, one model file; the workloads run 1 hit the harness's 30-minute
timeout (an agent script that did not finish) but its 9 requests count.

## Rerun

```bash
# on the desktop, with `holon serve` of the 27B lock on :8000
python bench/capture_proxy.py --upstream http://127.0.0.1:8000 --port 18000 --out <dir>/raw_requests.jsonl \
    --raw <dir>/raw.jsonl &
TMPDIR=/tmp python bench/demo/pi_run.py --server http://127.0.0.1:18000 --gpu-class rtx-5090 --pi <prefix>/node_modules/.bin/pi \
    --task demo --runs 3 --out <dir>/pi_record.json      # then --task workloads --runs 2
# on any CPU host with the GGUF in the HF cache
python spikes/k0_tiers/m1/analyse.py --raw <dir>/raw.jsonl --model <path to Qwen3.8-27B-UD-Q4_K_M.gguf> \
    --out evidence/k0_m1_rerender_rtx-5090_<date>.json
```

pi's project directories must not lie under the repository (pi would load its `AGENTS.md` as a context file),
hence `TMPDIR`. The raw transcript holds message text: keep it out of the repository. The evidence record holds positions, counts
and statistics only.
