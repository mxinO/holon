# Adding a model

A model is a plugin package under `plugins/holon_plugins/models/`: a manifest, a pure recipe that composes shared
units by `Ref`, a weight map per checkpoint format, and a frontend bundle. The core never names a model
([design §1](design.md#1-invariants-lintci-enforced)). This guide covers what the three existing packages contain,
walks the most recent addition, and lists the checks a new package must pass.

## The three packages

Each package has the same five files: `plugin.toml`, `__init__.py` (empty), `plugin.py`, `recipe.py`, `bundle.py`.

| | `qwen3` (dense) | `qwen3_moe` | `qwen3_5` (hybrid) |
|---|---|---|---|
| Manifest `provides` | `model.qwen3@1`, `arch.Qwen3ForCausalLM@1`, `arch.gguf.qwen3@1` | `model.qwen3_moe@1`, `arch.Qwen3MoeForCausalLM@1`, `arch.gguf.qwen3moe@1` | `model.qwen3_5@1`, `arch.Qwen3_5ForConditionalGeneration@1`, `arch.gguf.qwen35@1`, three `transform.qwen3_5.*@1` |
| `derived_from` | none (the original) | `models.qwen3@1f04987` | `models.qwen3@163c264` |
| `ModelSpec.layers` | `num_hidden_layers` × `LayerDecl(gqa, swiglu, "full_attention")` | the same with `std.moe@1` as the ffn | per layer: `LayerDecl(gdn, swiglu, "linear_attention")` or `LayerDecl(gqa, swiglu, "full_attention")` |
| `Ref` knobs | `std.ops.rmsnorm@1` `eps`; `std.mixers.gqa@1` `heads`, `kv_heads`, `head_dim`, `qk_norm`, `rope` (`std.ops.rope@1` `theta`); `std.ffn.swiglu@1` `inter`; `std.ops.lm_head@1` `tied` | qwen3's, with `std.moe@1` `experts`, `top_k`, `inter`, `renormalize` in place of swiglu | qwen3's, plus rmsnorm `offset=1.0`, gqa `output_gate="sigmoid"`, rope `rotary_dim`; `std.mixers.gdn@1` `heads_k`, `heads_v`, `head_k_dim`, `head_v_dim`, `conv_kernel`, `v_head_order`, `norm_eps` |
| HF `WeightMap` | 7 name rules; `skip=("lm_head.",)` when tied | qwen3's, its `mlp.{rest}` rule replaced by a router rule and 3 expert rules with `stack="e"` | `model.language_model.` prefix; `skip` the vision tower and `mtp.`; conv1d through a transform |
| GGUF `WeightMap` | llama.cpp names, no transforms | expert tensors (`ffn_*_exps`) are whole stacks: no `stack` | norms and `ssm_a` through transforms; MTP blocks skipped |
| Config read | HF keys (a GGUF file's keys mapped to them by the weight source) | qwen3's, plus the MoE facts from GGUF raw `qwen3moe.*` keys or HF keys (`moe_facts`) | HF `text_config` or GGUF raw `qwen35.*` keys (`_facts`) |
| `plugin.py` hooks | `recipe`, `bundle`, `trained_context` | identical to qwen3's | adds `register` (transforms), `draft_head` (MTP); `trained_context` reads `text_config` |
| Bundle | `tool_format = "hermes@1"`, `reasoning_format = "think@1"` | byte-identical to qwen3's | `think@1`; `tool_format` read off the chat template (`qwen_xml@1` or `qwen_xml@2`) |

### qwen3

`recipe(c, source)` raises for a source it has no map for and for config options it does not implement
(`attention_bias`, `use_sliding_window`, a non-default rope type), then returns a `ModelSpec` with
`block=Ref("std.blocks.prenorm@1")`, `embed=Ref("std.ops.embedding@1")`, the final `norm`, `lm_head` and the weight map
for `source` (`"hf_safetensors"` or `"gguf"`). A rule maps a checkpoint name onto a unit parameter, `{i}` being the
layer index and `{rest}` any captured span:
`WeightRule("model.layers.{i}.self_attn.{rest}", "layers.{i}.mixer.{rest}")`.
`trained_context` returns `max_position_embeddings`, which `holon serve --max-model-len auto` caps at 32768.

### qwen3_moe

A copy of qwen3 (diff to it: 26 lines added and 8 removed in `recipe.py`, three lines in `plugin.toml`; `plugin.py`
and `bundle.py` unchanged). The HF experts are one tensor per expert, so each expert rule fills one slice of an
`[E, …]` parameter:

```python
WeightRule("model.layers.{i}.mlp.gate.weight", "layers.{i}.ffn.router"),
*(WeightRule(f"model.layers.{{i}}.mlp.experts.{{e}}.{g}_proj.weight", f"layers.{{i}}.ffn.{g}.weight", stack="e")
  for g in ("gate", "up", "down")),
```

`moe_facts` reads `expert_count`, `expert_used_count` and `expert_feed_forward_length` from a GGUF file (with
`renormalize=True`) or `num_experts` / `num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size` and
`norm_topk_prob` from an HF config, where it refuses dense layers among the MoE ones. Stacking is a core rule
(`WeightRule.stack`, tested in `core/tests/test_stacking.py`); `python -m holon_judge structure` checks that every
stacked expert of an HF load equals the checkpoint's tensor.

## Worked example: qwen3_5 from qwen3

Reproduce the diff with

```bash
diff -u plugins/holon_plugins/models/qwen3/recipe.py plugins/holon_plugins/models/qwen3_5/recipe.py
```

and the same for `plugin.py`, `plugin.toml` and `bundle.py`. There is no inheritance: `derived_from` records lineage
only, and qwen3_5 imports nothing from qwen3 ([design §5.1](design.md#51-model-plugin-contract), "Derivation").

**`plugin.toml`.** A new name, `derived_from = "models.qwen3@163c264"` (the qwen3 commit it was copied from), the two
`arch.*` resources the weight sources report (`architectures[0]` of an HF `config.json`; `gguf.<general.architecture>`
of a GGUF file), and one `transform.<name>@1` per value transform: a plugin may register only what it declares.

**`plugin.py`.** `register(regs)` adds the transforms the weight maps name:

```python
regs.transforms.add("qwen3_5.squeeze", lambda t: t.squeeze(1))                  # conv1d [C, 1, K] -> [C, K]
regs.transforms.add("qwen3_5.minus_one", lambda t: t.float() - 1)             # GGUF norms store w + 1
regs.transforms.add("qwen3_5.a_log", lambda t: torch.log(-t.float()))         # GGUF stores -exp(A_log)
```

`trained_context` reads `text_config` for HF configs. `draft_head("mtp")` (added later, for speculative decoding)
returns the checkpoint's multi-token-prediction layer as an `AuxSpec` over `std.spec.mtp_head@1`, its own weight map
reading only that layer (`only=`); it is built only when a drafter asks for it.

**`recipe.py`.**
- `_facts(c, source)` gathers everything from one of two places: the HF `text_config` (`layer_types`,
  `linear_*` head counts and dims, `partial_rotary_factor`) or the GGUF raw keys (`attention.recurrent_layers` or
  `full_attention_interval`, `ssm.*`, `rope.dimension_count`). GGUF value heads are `"tiled"`, HF's `"grouped"`:
  the `v_head_order` knob of `std.mixers.gdn@1`.
- `_refs` builds the Refs the trunk and the draft head share: rmsnorm with `offset=1.0` (zero-centred norms), gqa
  with `output_gate="sigmoid"` and a partial `rotary_dim`, the new `std.mixers.gdn@1`, swiglu.
- `recipe` picks the mixer per layer: `LayerDecl(r["attn"] if full else r["gdn"], r["mlp"], ...)`.
- The HF map adds the `model.language_model.` prefix, the `linear_attn.*` rules and
  `skip=("model.visual.", "mtp.")`. The GGUF rules send the norms through `qwen3_5.minus_one`, `ssm_a` through
  `qwen3_5.a_log`, and skip the blocks after the trunk.

**`bundle.py`.** The tool format is read off a rendered call (`_tool_format`), and tool-call arguments sent as JSON
text are parsed to objects before rendering (`_parsed`), because the XML templates iterate them. Formats are
implemented once, in `plugins/holon_plugins/frontends/http/formats.py` (`TOOLS`, `REASONING`); a bundle names one.

**Outside the package**, in the commit that added it: the entry point in `plugins/pyproject.toml`, two tests
(`plugins/tests/test_qwen3_5_cpu.py`, `plugins/tests/test_qwen3_5_engine.py`) and the mixer unit `std.mixers.gdn@1`
(`std/holon_std/mixers/gdn/units.py`, registered in that package's `plugin.py` and declared in its `plugin.toml`; the
package's op contracts and oracles, `reference.py`, came one commit earlier). Later commits added the GGUF path's test
(`plugins/tests/test_qwen3_5_gguf.py`), plants (`plugins/tests/test_qwen3_5_plants.py`) and the draft head.

## Mechanics

**Manifest.** `[plugin]` takes `name`, `version`, `entry = "plugin:Plugin"`, `slot = "model"`, `license` and
`core_abi` (required: the ABI range, `">=1.0,<2.0"`); unknown sections or keys are errors
(`core/holon/api/manifest.py`). `[dependencies] python` lists the third-party modules the package imports.
See [design §4.1](design.md#41-manifest).

**Entry point and reinstall.** Add `"models.<name>" = "holon_plugins.models.<name>"` under
`[project.entry-points."holon.plugins"]` in `plugins/pyproject.toml`, then run `uv sync --group dev`: plugins are
discovered from installed entry points. A new third-party dependency goes into the distribution's `pyproject.toml`,
followed by `uv lock`.

**`Ref` / `ModelSpec`.** `recipe.py` imports only `holon.api` (`ModelSpec`, `LayerDecl`, `Ref`, `WeightMap`,
`WeightRule`, `AuxSpec`). `Ref("std.mixers.gqa@1", heads=…)` names a unit and its interface major. Plugins never
import std (an import-linter contract in `pyproject.toml`). The units and their contracts are listed in
`std/contracts.txt`; their code is under `std/holon_std/`. The lock pins the recipe by the hash of the `ModelSpec` it
emits. See [design §5.1](design.md#51-model-plugin-contract).

**`WeightMap` / `stack`.** `WeightMap(source, rules, root="", skip=(), only=())`; `WeightRule(pattern, target,
transform=None, stack=None)` (`core/holon/api/model.py`). A transform is registered by the plugin and declared in its
manifest. See [design §5.5](design.md#55-quantization-and-weight-loading).

**The bundle.** `bundle(source)` returns a `FrontendBundle` (`core/holon/api/frontend.py`): `model_name`, a
tokenizer, `tool_format`, `reasoning_format` and `render_chat`. The checkpoint's tokenizer and chat template come
from the weight source, HF directory or GGUF file alike.

## Static checks

Run `scripts/ci/run_all.sh` (in the `uv sync --group dev` environment). For a model package the ones that bite are:
- `scripts/ci/check_manifest_deps.py`: every module the package or its manifest names is declared by
  `holonlm-plugins` or its dependencies;
- `scripts/ci/check_abi.py`: `plugins/contracts.txt` lists every provided resource but `kernel.*`, `arch.*` and
  `model.*`, so new transforms regenerate it with `python scripts/ci/check_abi.py --write` (an addition is free);
- `scripts/ci/check_core_denylist.py`: no model name in `core/holon`;
- import-linter contracts and ruff;
- `scripts/ci/check_locks.py`: every committed lock re-reads with its digest and `plan_id` (locks are written by the
  tools, never by hand).

## How a model is judged

1. **CPU check against transformers.** A tiny random model built from the recipe equals transformers in fp32:
   `pytest plugins/tests/test_qwen3_cpu.py plugins/tests/test_qwen3_5_cpu.py`. qwen3_5's GGUF path runs a tiny
   checkpoint converted by llama.cpp's own converter (`plugins/tests/test_qwen3_5_gguf.py`).
2. **L0.** A numeric std package's `reference.py` (e.g. `std/holon_std/ops/reference.py`) is the oracle its ops'
   kernels are checked against, under budgets held in `judge/holon_judge/l0.py`.
3. **Goldens and floors.** The golden is HF's teacher-forced top-20 logprobs in fp32 on the CPU over the frozen
   prompt set `judge/prompts/v1.jsonl`; the floor is the same recorder in the model's activation dtype on the GPU.
   A model with another tokenizer gets the set re-tokenized (`python -m holon_judge retokenize`; qwen3_5 uses
   `judge/prompts/v1-qwen3_5.jsonl`). Goldens live under `judge/goldens/` and are written by the tool only:

   ```bash
   python -m holon_judge golden --model <hf dir or id> --prompts judge/prompts/v1.jsonl --out <golden> --threads N
   python -m holon_judge golden --model <hf dir or id> --prompts judge/prompts/v1.jsonl --out <floor> \
       --dtype bfloat16 --device cuda
   ```
4. **L2.** The engine scores the prompt set; it passes when KL p99 and p50 are within 2× the floor's, top-1 agreement
   is within 0.25 points of the floor's and |ΔPPL| ≤ |floor| + 0.25%
   ([design §10.3](design.md#103-judge-tiers-and-numerics-classes)):

   ```bash
   python -m holon_judge l2 --model <checkpoint> --prompts judge/prompts/v1.jsonl --golden <golden> --floor <floor>
   ```
   GGUF files have goldens from their dequantized weights (`--config <hf checkpoint>`); see `judge/README.md`.
5. **First lock.** `python -m holon_tools propagate <model> --new-model --approve --golden <golden> --floor <floor>`
   runs L2 and the snapshot trace and writes the model's first lock under `plans/`
   (`tools/README.md`, [design §4.6](design.md#46-reuse-and-propagation)).

For depth: [design §5](design.md#5-models-ops-kernels-quantization) (models, ops, kernels, quantization),
[§4](design.md#4-plugin-system) (plugins, locks), [§10.3](design.md#103-judge-tiers-and-numerics-classes) (judge tiers).
