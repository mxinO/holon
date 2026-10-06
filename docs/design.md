# plug-serve — Design

Status: draft for review · Date: 2026-10-06 · Branch: `design-doc`

---

## 0. Summary

plug-serve is a microkernel LLM inference engine. A small core (~5k LOC Python, hard cap 6k) fixes only the
timing structure of a step, memory accounting, the plugin resolver and a handful of typed contracts. **Models,
layer types, quantization formats, kernels, cache/offload policies, speculative decoding, sampling, grammars,
schedulers, frontends and parallelism are all plugins.** Every supported (model, GPU class) ships with
best-known, human-tuned defaults. Optionally, a trusted judge and a supervisor let a capable agent — including one
*served by the engine itself* — propose, verify, benchmark, promote and roll back improvements.

### Goals

1. **Isolation with reuse** — adding or changing one model never changes another model's output, while code is
   shared in small units. Enforced by per-model lockfiles that change only through that model's judged transition
   (§4.6) and per-model lock histories (§4.4).
2. **Extensibility** — features in §9 land as plugins with an empty `git diff core/`.
3. **Best-known defaults** — every supported (model, GPU class) ships tuned by us; the engine performs at its best
   with no agent involved.
4. **Optional self-improvement** — when a capable agent and the user choose to, the agent can tune and replace
   plugins under a judge it cannot edit, with automatic rollback (§10). Nothing else depends on this layer.
5. **Small and legible** — every plugin is small enough for a coding agent to hold in context (measured: plan
   M0.3 legibility benchmark).
6. **Competitive single-GPU performance** — within 10% of mini-sglang on decode and on mixed agent traffic, on the
   same GPU, for M0.

### Non-goals (v0)

Multi-node serving, non-NVIDIA platforms, training, a model zoo. These must remain *possible* without core
rewrites (§8) but are not built in v0.

The optional self-improvement layer (§10) is not part of v0; v0 builds the test/benchmark harness it will reuse.
Its first version evaluates candidates on a spare GPU only (accepted 2026-10-06: on single-GPU hosts it is limited
to host-side policies and live tunables).

### Decisions taken (2026-10-06)

| # | Decision |
|---|---|
| D1 | Single consumer GPU first. Datacenter features (TP/EP/PP, PD, DP-attention) must be addable as plugins; the core keeps the seams (§8). |
| D2 | Model ladder: Qwen3 dense → Qwen3-MoE → Qwen3.5 GDN hybrid ([plan §2](plan.md)). |
| D3 | Borrow existing kernels first (FlashInfer, flash-linear-attention, Triton MoE, …) **only if they fit the plugin contract** (§5.6); otherwise write our own. |
| D4 | Python control plane; native/Triton kernels; CUDA graphs on the hot path. The plugin API stays Python; hot core structures and the frontend may later be replaced by Rust modules behind unchanged interfaces, only when profiling triggers it (§12). |
| D5 | Self-improvement is an **optional capability** for capable agents and explicit user intent (e.g. a strong agent served by plug-serve). The engine itself ships best-known, human-tuned defaults and never depends on it. When used, auto-promotion is limited to policy/kernel/tunable scope under the judge; core, judge, ABI and `approx` numerics need a human (§10.5). |
| D6 | References (mini-sglang, nano-vllm, vLLM Model Runner V2, SGLang, FreeToken, Strata) are studied, not copied: control-plane code is our own plugin-native implementation. Only kernels may be borrowed, under §5.6. |
| D7 | Hardware: development on 8× RTX A5000 (sm_86). Consumer performance evidence on consumer desktop machines — RTX 5080 16 GB and RTX 5090 32 GB (sm_120), RTX 4090 (sm_89). Datacenter path on datacenter GPUs (H100, GB200) later. |
| D8 | Expert caching/offload arrives in v1, but its core seams (device SeqTable, MemOps, `quiesce`, stream-memop CPU handshake) are built and spiked in M0 so it lands without core changes. |

---

## 1. Invariants (lint/CI-enforced)

1. The core contains no model name, layer-type name, quant-format name, or feature-specific field.
   (CI: denylist grep over `plugserve/core` — `mamba|gdn|mla|dsa|nvfp4|eagle|qwen|deepseek|lora|…`.)
2. Plugins import only `plugserve.api`, their own package, and third-party libraries declared in their manifest
   `[dependencies]` (e.g. `triton`, `flashinfer`). Other units (incl. std) are referenced with `Ref` and injected,
   never imported (§4.6). Model packages never
   import other model packages. (CI: import-linter contracts + manifest dependency check.)
3. Plugins never import `plugserve.core._*` or `plugserve_judge`. (CI: import-linter.)
4. No module-level mutable state in `plugserve.*`; all state hangs off an explicit `Engine`/`ctx`. (lint)
5. No environment-variable configuration. One typed deployment file feeds the resolver. (lint: no `os.environ` in core/std)
6. No `torch.cuda.*` outside `plugserve.core.platform` and kernel plugins. (lint)
7. Plugin load failure, unresolved requirement, or duplicate provider for a `single` slot is a startup error —
   never a logged warning.
8. Core LOC ≤ 6000 (CI). Growing past it requires a design note.
9. After v0, every milestone PR has an empty diff under `core/` unless it is an explicitly reviewed core ABI change.
10. No `model_type`/architecture branches in std or any shared unit; model-specific code lives in the model's own
    package (`fast/`, §5.7). (lint)
11. One version of each unit per Plan, one Plan per process. Lockfiles and snapshots are written only by
    `plugserve propagate` (incl. `--new-model`), `plugserve tune` and the supervisor (§4.4, §4.6); CI's `isolation-check` re-runs propagate on a trusted GPU
    runner and rejects any lock or snapshot it cannot reproduce byte for byte.

---

## 2. Architecture overview

### 2.1 Processes

```
 ┌──────────────┐  token ids / deltas   ┌──────────────────────────────┐
 │  Frontend    │◄────────── IPC ──────►│  Engine (one per GPU group)  │
 │  (HTTP APIs, │                       │  StepLoop + plugins + Ledger │
 │  tokenizer,  │                       └──────────────┬───────────────┘
 │  templates)  │                                      │ Events (telemetry)
 └──────▲───────┘                                      ▼
        │ OpenAI/Anthropic             ┌──────────────────────────────┐
        │                              │  Supervisor (trusted)        │
   clients / agents ─── improve API ──►│  Lock history · Evidence     │
   (incl. an agent served by           │  lifecycle · rollback · MCP  │
    this very engine)                  └──────────────┬───────────────┘
                                                      │ spawns
                                       ┌──────────────▼───────────────┐
                                       │ Sandbox engine ◄── Judge     │
                                       │  (candidate plans, L0–L4)    │
                                       └──────────────────────────────┘
```

- **Frontend** — HTTP APIs, tokenization/detokenization, chat templates, stop strings. Talks to the engine in
  token ids. Plugins: API dialects; the tokenizer/template/parser bundle comes from the model plugin.
- **Engine** — runs one resolved `Plan`. Single process per GPU group in v0 (rank 0 only, through the SPMD path).
- **Supervisor** — small trusted process: engine lifecycle (start, drain, restart on a lock), health checks,
  crash attribution (§7.7) and rollback. With the optional self-improvement layer (§10) it also owns promotions,
  evidence and the improve API. Never imports agent code.
- **Sandbox engine** (optional layer) — a separate engine process with a candidate Plan in a restricted sandbox
  (§10.6). It may run agent-written code.
- **Judge** — the test/benchmark harness (goldens, snapshots, op oracles, perf). It drives engines **only as a
  client** (HTTP/IPC), never in-process with candidate code. Used in v0 for our own development and CI. Verdict-relevant measurements are taken on the judge side (§10.3).

### 2.2 Packages (one uv workspace)

| Package | Trust | Content |
|---|---|---|
| `plugserve` (`core/`) | core | `plugserve.api` (protocols, dataclasses — the plugin ABI) + `plugserve.core` (runtime). ≤ 6k LOC. |
| `plugserve-std-*` (`std/`) | reviewed | Shared units: `std.ops` (linear, norm, rope, act), `std.blocks` (residual wiring), `std.mixers.gqa`, `std.ffn`, `std.state.paged_kv`, `std.tiered` (TieredStore library), `std.sampling`. One live version each; recipes reference them via `Ref("name@major")`; locks pin `src` + `sem`; changes reach models only through propagation (§4.6). |
| `plugins/*` | reviewed or agent | First-party plugins: models, kernels, dialects, policies, frontends. Each is its own dist package with entry points. |
| `plugserve-judge` (`judge/`) | trusted, read-only to agents | Oracles harness, tolerance budgets, goldens, traces, perf timer. |
| `plugserve-supervisor` (`supervisor/`) | trusted | Lifecycle, crash attribution, rollback; with the optional layer (§10): promotions, evidence, improve API (MCP + HTTP). |
| `workspace/` | agent (optional layer) | Git worktree where agent variants are written (gitignored). A promoted variant is committed by the supervisor to its own branch together with the new lock (§4.4). |

---

## 3. Core vocabulary

Sixteen nouns. Each is a type in `plugserve.api` (the plugin ABI) with its runtime in `plugserve.core`.

| # | Noun | One-line definition | § |
|---|---|---|---|
| 1 | Manifest | Declarative plugin description, readable without import | 4.1 |
| 2 | Plan | Resolved, serializable deployment: plugins, bindings, tunables, fingerprint | 4.3 |
| 3 | Registry | Git-backed lock history per (model, fingerprint class) + evidence; holds tuned defaults | 4.4 |
| 4 | ModelSpec | A model's declaration (emitted by a pure recipe): layers, taps, draft heads, encoders | 5.1 |
| 5 | Op | Typed signature + reference implementation | 5.3 |
| 6 | Kernel | One implementation of an Op with constraints and properties | 5.4 |
| 7 | Layout / QTensor | Number semantics + byte arrangement; typed weight object | 5.5 |
| 8 | StateSpec / StateKind | Per-layer state allocation shape + open-registry meaning | 6.2 |
| 9 | Ledger / Pool | Byte budget over named pools with stable addresses | 6.1 |
| 10 | PrefixIndex | Content-addressed reuse of state across requests | 6.3 |
| 11 | Seq / SeqTable | One request's lifecycle (host) + device-authoritative per-slot state | 7.1 |
| 12 | Batch | One step: host plan (SoA) + static device mirror | 7.2 |
| 13 | Rewindable | Device-side commit/rollback contract for per-sequence state | 7.5 |
| 14 | StepLoop | Fixed timing regions, overlap depth, graphs, control plane | 7.3 |
| 15 | Event | Core-measured telemetry record | 7.9 |
| 16 | Platform | Thin device abstraction (streams, events, pinned mem, graphs, IPC) | 8 |

Hot-structure interfaces — PrefixIndex, slot/page allocator, batch builder — operate on numpy struct-of-arrays
behind narrow interfaces, so a native (Rust via PyO3) implementation can replace them without changing callers
or plugins if profiling demands it (§12). These interfaces call plugin code (e.g. `StateKind.hit_length`) per
request, never per token, so a native implementation pays the Python callback cost only per request.

---

## 4. Plugin system

### 4.1 Manifest

`plugin.toml` at the package root; registered via entry point group `plugserve.plugins` (value = path to manifest).

```toml
[plugin]
name      = "expert_cache.lfu_decay"
version   = "0.3.1"
variant_of = "expert_cache.lfu@src:aa10"  # optional; set for forks (lineage)
implements = "placement_policy@1"         # numeric units: "<unit>@sem:<hash>" — the judge loads that oracle
derived_from = ""                         # optional; copied recipes / explicit old-version copies (lineage only)
entry     = "lfu_decay:Plugin"            # imported only if selected
slot      = "placement_policy"
license   = "Apache-2.0"                  # SPDX; provenance hash for borrowed code in [provenance]
core_abi  = ">=0.1,<0.2"
slot_abi  = "^1"

[resources]
provides = ["placement.expert@1"]
requires = ["store.tiered@^1", "telemetry.router_hist@^1"]

[properties]
numerics    = "policy"     # policy | exact | bounded | approx   (§10.3)
graph_policy = "host"      # full | piecewise | eager | host   (§7.4)
overlap_safe = true

[dependencies]             # third-party imports allowed for this plugin (invariant 2)
python = []

# [reference]              # numeric units only (omitted for policy plugins like this one):
# module = "reference.py"  # the oracle module; sem = hash(reference.py + param schema)

[tunables]
decay     = { type = "float", range = [0.5, 0.999], default = 0.7, apply = "live" }     # live | restart
swap_cap  = { type = "int",   range = [0, 512],     default = 96, log = true, apply = "restart" }
# A tunable is judged under the numerics class of the plugin that owns it.

[support]
platforms = ["cuda>=sm_80"]
```

Resource names are **typed**: each `ns.name@major` is registered in `plugserve.api.resources` and bound to a
Protocol. The resolver matches strings without importing plugins; after import it checks each provider object
against the Protocol (`runtime_checkable` + signature check).

### 4.2 Slots and cardinality

| Cardinality | Meaning | Slots |
|---|---|---|
| `single` | exactly one per Plan | model, platform, launcher, schedule_policy, sampler, verifier, draft_policy, eviction_policy (prefix index) |
| `per_site` | one binding per site (layer, op instance, state spec, store) | mixer, op/kernel, state_kind, moe_op, placement_policy, codec, weight_source |
| `ordered_multi` | ordered list, `after=/before=` constraints | plan_transform, logits_processor, checkpoint_policy |
| `multi` | any number, keyed by name | frontend, dialect, scheme, layout, converter, tier, mover, drafter, encoder, connector, exporter |

Every slot has an ABI version and a judge-owned contract suite (§10.3).

### 4.3 Resolution → Plan

1. Discover manifests (entry points + `plugins/` + `workspace/` if the supervisor enables it).
2. Read the deployment file (model path, pins, budgets).
3. The model plugin's `ModelSpec` enumerates **sites** (each mixer, op, state spec, draft head).
4. For each site: filter candidates by resources, platform/fingerprint constraints, `supports(shape)`, layout
   reachability (§5.5); then pick by deployment pin → the model's lock for this fingerprint class → default priority.
5. Check `ShapeKey` axes are finite; compute graph policy per ShapeKey class (weakest wins).
6. Emit the **Plan** (TOML lockfile): recipe hash, per-unit `src`/`sem` (`[units]`, one version each), site
   bindings + kernel configs, tunables, fingerprint class, `[deploy]` and `[snapshot]` (§4.6).
   `plan_id = sha256(plan)`.
7. Import only the selected entries; each plugin registers its kernels/converters/kinds into explicit registries
   passed to `Plugin.register(regs)` (no module-level registries, invariant 4); construct; bind callables. The hot
   path never does a registry lookup.

The Plan is the unit of deployment, A/B, promotion and rollback. It is serializable so every rank of a future
SPMD deployment can load the identical Plan (§8).

**Model isolation via lockfiles.** Each served model has a lock history per fingerprint class
(`plans/<model_id>/<class>/<n>.lock`, §4.4). The engine starts from the newest lock — never from a fresh
resolve. A fresh
resolve that differs from the lockfile (e.g. because a new, higher-priority kernel was installed for another model)
is reported as a diff and only adopted through `plugserve propagate` (§4.6). Adding a plugin
for model B can therefore never change model A's binding silently.

### 4.4 Registry = lock history (git-backed)

There is no separate database. The **Registry** is the git history of lockfiles plus evidence records:

- **One linear history per (model, fingerprint class):** `plans/<model_id>/<class>/<n>.lock`. The newest lock is
  what the engine runs. Rollback = restart on the previous lock in the history. With the optional layer (§10), a host's history is
  main's locks plus host-local promotion commits on top; when the host pulls a new main, the supervisor re-runs
  propagation from its own newest lock (so promotions are carried forward or dropped by the same checked
  transition), or a promotion is upstreamed to main by PR.
- **Tuned defaults live in the lock.** Kernel configs and tunables are tuned by us per fingerprint class
  (benchmarking tools such as `plugserve tune`, in the spirit of FreeToken's `ft bench bw`), so every class ships
  at the best configuration we know — no agent is involved.
- **Fingerprint class** = GPU SKU + SM + VRAM class + toolchain majors (CUDA, torch, Triton). GPU UUID, driver
  build and measured bandwidths are recorded in evidence only. Classes maintained first: `rtx-5080` and
  `rtx-5090` (sm_120), `rtx-4090` (sm_89), `a5000`/`rtx-3090` (sm_86); evidence is produced on those cards
  (consumer desktop machines, plan §1).
- **Missing or outdated lock for a class** (an outdated lock pins unit `src`s that the deployed commit does not
  contain) → the nearest class's current lock (same SM preferred) is used, flagged `unverified`, with an alert. The engine **never refuses to serve** for lack of a lock; `plugserve tune` on the host produces a
  verified lock for its class.
- Evidence records (command, fingerprint, plan_id, raw numbers) point at lock hashes and are written after the
  lock commit.

### 4.5 Versioning

- `core_abi` and each `slot_abi` are independent semver. Changing the attention metadata ABI does not touch the
  placement-policy ABI.
- Units follow the one-version rule with additive-only interface minors, never-changing defaults and
  "breaker pays" majors; changes reach each model only through propagation (§4.6) — models do not individually
  adopt std versions.
- Core ABI minors keep one-minor compatibility shims (v1+). v0 is pre-1.0: ABI may break, but every break runs
  the full `plugserve check` over first-party plugins.

### 4.6 Reuse and propagation

**Principle.** Code is shared only in small leaf **units**; models are **data** (recipes, §5.1) that reference
units; and **a model's lockfile changes only through a checked transition of that model**. Hash pins give
reproducibility and exact rollback; isolation comes from the transition rule. (Rationale and evidence:
internal design reviews; simplified after an adversarial review.)

**Units.** A unit is one small package: a component (mixer, op, router, norm, block, policy …), its kernels, and a
separate `reference.py` holding its oracle.

| Field | Meaning |
|---|---|
| `src` | git tree hash of the unit's directory |
| `sem` | hash of `reference.py` + parameter schema — the unit's *semantic identity* |
| `iface@major` | interface version; minors are additive only |

- Unit code imports only `plugserve.api`, its own package and declared third-party libraries; other units are
  **injected** through constructor ports (recipes reference them with `Ref`), never imported or inherited (no
  `extends`, mixins or monkeypatching).
- **Defaults never change.** New behaviour is a new knob whose default reproduces the old behaviour, or a new unit.
  Bug fixes are the one sanctioned exception, and they go through approval below.
- Interface majors: "breaker pays" — the change ships a codemod and updates all first-party dependants.
- The oracle for `sem` h is the `reference.py` blob at a reviewed commit (git); the judge never takes an oracle
  from the candidate under test.

**One version per Plan.** A Plan contains exactly one version of each unit; one Plan runs per process.

**Lockfile = deployment record.** In addition to §4.3, every lock carries:

```toml
plan_id = "sha256:…"   recipe = "sha256:…"      # canonical hash of the emitted ModelSpec term
[deploy]   commit = "9e1c…"   uv_lock = "sha256:…"
[units]    "std.mixers.gqa" = { version = "1.3.0", src = "<git tree>", sem = "sha256:77b2…" }
[snapshot] trace = "sha256:…"   uncovered = ["kernel:moe:…"]   # §7.10
```

`(commit, lock)` fully defines a deployment; promoted variant code is part of the commit.

**Propagation.** `plugserve impact <unit>` lists the locks whose closures contain it. `plugserve propagate` runs
inside the PR that changes shared code; CI's `isolation-check` re-runs it on a trusted GPU runner and rejects any
lock or snapshot it cannot reproduce byte for byte. Each affected (model, fingerprint class) gets one of three
outcomes:

| Outcome | Condition | Effect | Human |
|---|---|---|---|
| **Auto-advance** | model's closure unaffected, or snapshot trace byte-identical **and** no `uncovered` site touched **and** every changed line inside the model's closure executed during its snapshot trace | lock rewritten in the PR | 0 |
| **Approve** | otherwise, L2 vs the HF reference passes and no L2 metric regresses beyond δ vs the current snapshot | lock rewritten, snapshot re-recorded | 1 approval for the whole change (all affected models listed) |
| **Fail** | L2 fails | the PR fixes it, or the model keeps the previous unit version through an explicit copy in its own package (visible in the diff, `derived_from` set) | — |

Toolchain bumps (torch, Triton, FlashInfer, CUDA majors) produce a new fingerprint class and run the same sweep.
New models get their first lock from `plugserve propagate --new-model` (L2 vs reference + snapshot, approved).
Maintained consumer classes (`rtx-5080`, `rtx-5090`, `rtx-4090`) are refreshed by remote `isolation-check`
runners on consumer desktops — on shared-change PRs where capacity allows, and nightly at least; a release commit requires
every maintained class refreshed. Until refreshed, an outdated lock is treated as missing (§4.4).

**Deferred machinery, each added only when its trigger fires:**

| Trigger | Add |
|---|---|
| Full propagation sweep > 30 min wall time, or > 1 approval/day | async merge queue + nightly batch (scheduling change only) |
| > 2 models stuck on explicit old-version copies of one unit, or any older than 30 days | holdback tracking with expiry, exception-rate metric per unit |
| Multi-model-per-process serving (v2) | content-addressed unit store with side-by-side loading |
| First model needing sandwich norm / embedding scale / similar glue | closed hook set in `std.blocks` (minor version) |

**Agent variants** (optional layer, §10) are leaves: bound in one model's lock only, never a dependency, expire after
7 days unless promoted. A submission is a Plan delta of `bind`/`set` on existing site keys; `bind` may add an entry
to a `multi`/`ordered_multi` slot already present (auto-promotable only for drafter, exporter, connector,
checkpoint_policy; output-affecting slots are proposals). In v0 agents never edit std units.

---

## 5. Models, ops, kernels, quantization

### 5.1 Model plugin contract

```python
class ModelPlugin(Protocol):
    archs: ClassVar[tuple[str, ...]]                 # HF architectures handled
    def parse_config(self, hf_cfg: dict) -> Any      # opaque to core
    def spec(self, cfg, par: ParallelInfo) -> ModelSpec   # = recipe(cfg) + sharding from ParamSpec/ParallelInfo
    def build(self, cfg, ctx: BuildCtx) -> Module    # uses ctx.mixer(...), ctx.op(...)
    def weights(self, src: WeightSource) -> Iterator[ParamLoad]   # name mapping lives here
    def bundle(self) -> FrontendBundle               # tokenizer, chat template, reasoning/tool parsers

@dataclass
class ModelSpec:
    hidden: int; vocab: int
    layers: list[LayerDecl]          # LayerDecl(idx, mixer=Ref("gqa","1"), ffn=Ref("moe","1"), hparams)
    taps: list[str]                  # "embed", "layer.{i}.out", "pre_norm_last", "attn.{i}.q_prerope", ...
    shared: list[str]                # "embed", "lm_head" (for draft heads)
    draft_heads: list[DraftHeadDecl] # MTP layers, their own LayerDecls, taps consumed
    encoders: list[EncoderDecl]      # VLM towers (v1)
    position_fn: str | None          # e.g. "mrope"; default arange
    params: list[ParamSpec]          # shapes + sharding semantics (col/row/expert/replicated, fused groups)
    block: Ref                       # residual wiring unit, e.g. Ref("std.blocks.prenorm@1")
    norm: Ref; lm_head: Ref          # final norm, output head
    weights: WeightMap               # checkpoint layout, e.g. hf_weights("qwen3_5", root="model.language_model")
```

- **Recipes.** A model is primarily a pure function `recipe(cfg) -> ModelSpec` in `recipe.py` that imports only
  `plugserve.api` (no torch). It composes units with `Ref("std.mixers.gqa@1", heads=…, …)`; `ModelSpec` gains
  `block` (e.g. `std.blocks.prenorm@1`) and `weights` (e.g. `hf_weights("qwen3_5", root="model.language_model")`),
  and `build`/`weights` default to them. The recipe's identity is the canonical hash of the **emitted term**, so
  refactoring `recipe.py` without changing the term changes nothing. Checkpoint layout lives only in `weights`.
- **Derivation = copy with lineage.** A new model copies an existing recipe and records `derived_from`; there is
  no model-to-model link or inheritance (e.g. Qwen3.5 differs from Qwen3 in ~10 recipe lines: GDN mixer for
  linear layers, `output_gate=True`, `rotary_dim=64`, `rmsnorm offset=1.0`, MTP draft head). `git diff` against
  the `derived_from` source shows how copied recipes diverge.
- **Escape hatch.** A model may ship its own `build` (full Python or native forward) that still composes units
  through `BuildCtx` (§5.7 level 5).
- `forward(ids, ctx)` receives an explicit `StepCtx` (no global forward context). Mixers get their bound state
  handle and metadata at build time / via `ctx.meta[group]`.
- **Taps** are named export points the core can attach consumers to (EAGLE-3 hidden states, KVMem-style Q/K
  capture, MTP inputs). A feature needing a new tap requires a model plugin change — never a core change.
- **Weight finalization.** Every built module (model glue, mixer, composite) may implement
  `finalize_weights(params, ctx)`, run once after loading and converters, before graph capture. It covers
  transformations spanning several tensors or semantics a converter cannot see: MLA W_UK/W_UV absorption,
  zero-centred RMSNorm (`x·(1+w)` — kept as an offset-gamma kernel parameter, not folded into bf16), GDN
  `-exp(A_log)` precompute, KV-cache scales. Results live in the module's Pool-backed storage.
- **Nested models.** `BuildCtx` and the weight pipeline are available to drafters, so a draft-*model* drafter
  instantiates a nested `ModelPlugin` whose StateSpecs and ledger requests are namespaced (`draft.*`).

### 5.2 Mixer contract

A mixer is an Op that owns per-request state.

```python
class Mixer(Protocol):
    kind: ClassVar[str]; version: ClassVar[str]
    @classmethod
    def state_specs(cls, cfg, layers, par) -> list[StateSpec]   # from config only, before weights load
    @classmethod
    def graph_policy(cls) -> GraphPolicy
    class Planner:                                   # one per (mixer kind, state group), shared across layers
        needs_host_lens: ClassVar[bool] = False      # True if a kernel needs exact host-side lengths (§7.3)
        def plan_bounds(self, batch: Batch) -> Meta  # host: shapes and upper bounds only
        def prepare_device(self, meta: Meta, seq_table: SeqTable, arena: Arena)
                                                     # graph-capturable: exact lengths/positions from SeqTable
    def finalize_weights(self, params, ctx) -> None  # optional, after load (§5.1)
    def forward(self, x, state: StateHandle, meta: Meta) -> Tensor   # prefill / decode / verify (num_draft>0)
    # oracle: slow pure-torch implementation in the unit's reference.py (sem hash, §4.6) — not a method the
    # candidate could edit; the judge loads the `reference.py` blob named by `sem` at a reviewed commit (§4.6)
```

The state semantics (rollback, prefix hits) live in the mixer's `StateKind` (§6.2), usually shipped in the same
package, so the core can reason about state without instantiating layers.

### 5.3 Ops and composites

```python
@op("linear", sig=Sig(x=Act, w=QTensor, bias=Opt[Tensor]) >> Act)
def linear_ref(x, w, bias=None): return x @ w.layout.scheme.dequant_ref(w).T + (bias or 0)

@composite("add_rmsnorm_quant", decomposition=lambda x, r, g, eps, out_layout: quant(rmsnorm(x + r, g, eps), out_layout))
```

- `@op`, `@composite` and `@register_converter` are **metadata-only markers** (they attach attributes to the
  function); nothing is registered at import. `Plugin.register(regs)` collects marked objects from its own modules
  into the engine's registries (invariant 4).
- Every Op has a reference implementation (pure torch, fp32 accumulate) in its unit's `reference.py`. The
  reference is the oracle and defines the unit's `sem` hash.
- **Fusion in v0 = explicit composites** called by models/mixers. A composite's decomposition is its reference.
  No torch.compile in the core. (Inductor may be used as a kernel provider for a single op.)

### 5.4 Kernels

```python
@dataclass(frozen=True)
class KernelSpec:
    op: str; provider: str; version: str            # version = source hash
    impl: Callable
    in_layouts: dict[str, LayoutPattern]; out_layouts: dict[str, LayoutPattern]
    device: str                                     # "cuda" | "cpu" | ...
    requires: Constraint                            # sm range, CPU ISA, library availability
    supports: Callable[[ShapeCtx], bool] | None
    props: frozenset[str]                           # graph_safe, batch_invariant, deterministic, host_node
    tunables: dict | None
    priority: int = 0                               # agent variants register < 0 until promoted
    config: dict | None = None                      # pinned launch config (tiles, warps, splits) — no autotune

class Plugin(Protocol):
    def register(self, regs: Registries) -> None    # regs.kernels.add(spec), regs.converters.add(...), ...
```

Registries are explicit objects owned by the engine instance (open registry, no per-format tables, no module-
level state).

Selection happens once, at plan time (§4.3). No startup autotune sweeps: tuned configs live in the model's lock for
its fingerprint class (§4.4, produced by `plugserve tune`); without a lock for the class, the nearest class's
configs are used (`unverified`).

### 5.5 Quantization and weight loading

Pipeline: `WeightSource → Dialect → Layout(source) → Converter → Layout(kernel) → QTensor → Pool`.

```python
@dataclass(frozen=True)
class Scheme:  elem: str; scales: tuple[Scale, ...]; zero_point: bool = False
    def dequant_ref(self, t: QTensor) -> Tensor      # mandatory, used by the oracle

@dataclass(frozen=True)
class Layout:  scheme: Scheme; name: str; params: frozenset; readable_by: frozenset[str]
    def tensors(self, logical_shape) -> dict[str, TensorDesc]   # bytes derived, never duplicated

@dataclass
class QTensor: layout: Layout; logical_shape: tuple[int, ...]; parts: dict[str, Tensor]

class Dialect(Protocol):         # gptq, awq, modelopt, compressed-tensors, gguf, fp8-block
    def detect(self, ckpt_meta) -> bool
    def group(self, names) -> dict[str, dict[str, str]]       # param -> {role: ckpt tensor name}
    def source_layout(self, param: ParamSpec, roles) -> Layout

@register_converter(src="gptq4.rowmajor", dst="gptq4.marlin", lossless=True, cost=1.0)
def to_marlin(q: QTensor, *, shard: ShardInfo) -> QTensor: ...
```

- Repacking is owned by **converters**, registered by the plugin that defines the destination layout.
- v0: single-hop, `lossless=True` converters only. Multi-hop path search is v1.
- When two consumers need one weight in two layouts (GPU kernel + CPU expert compute), the resolver asks the
  ledger whether to store both; otherwise picks a layout readable by both.
- Post-repack artifacts are cached on disk keyed by `(source hash, layout, converter version, sm)` (v1).
- KV cache formats are Layouts too, so attention kernels and state kinds negotiate page size/dtype via layouts.

### 5.6 Borrowed-kernel adoption rule (D3)

A third-party kernel is wrapped as a kernel plugin only if **all** hold; otherwise we write our own (Triton first):

1. **Functional**: callable as a pure function over tensors we allocate; no hidden global registries, no
   allocator of its own outside the Ledger (workspaces are declared and allocated from a Pool).
2. **Graph-safe** for its declared `graph_policy` (no host syncs on the captured path; host-side "plan" steps run in
   `Planner.plan_bounds`, e.g. FlashInfer `plan()`; device-side metadata in `prepare_device`).
3. **Layout-declarable**: its weight/KV/activation layouts can be expressed as `Layout`s.
4. **Packaging**: installable without pulling in an engine (no `vllm`/`sglang` wheel dependency); version
   conflicts with other selected kernels are detectable from manifests. Prefer vendoring Triton sources.
5. **License** compatible (Apache-2.0/MIT/BSD) with provenance recorded.
6. **Verifiable**: passes L0 against our reference within judge budgets on our target shapes.
7. **Pinned configuration**: no run-time autotuning. `@triton.autotune` is stripped or reduced to the config
   pinned in `KernelSpec.config` (from the model's lock, else a deterministic default). JIT caches (FlashInfer, Triton)
   are pre-built at install time and their directories are declared, so the sandbox can mount them read-only.

Initial candidates (v0), with findings from the 2026-10-06 review on sm_86 / torch 2.11 / Triton 3.6:
- FlashInfer 0.7 paged attention (decode verified for head_dim 128 and 256; JIT ≈ 8 s/module on first use → pre-build).
- GDN: **vendor the flash-linear-attention ops copy from vLLM** (`vllm/model_executor/layers/fla/ops`, Apache-2.0),
  which works on sm_86 and already supports `ssm_state_indices`/`num_accepted_tokens`. Vendoring strips its
  `vllm.*` imports (e.g. `vllm.triton_utils`) to satisfy rule 4. pip `fla-core` 0.5.2's
  recurrent kernel returns NaN on sm_86 — not adopted.
- Triton fused-MoE (vendored from vLLM/SGLang, Apache-2.0) with W4A16 path.
- Triton W8A16/W4A16 GEMM — written by us. On sm_86 Triton rejects `fp8e4nv`, so FP8 weights are loaded as `uint8`
  and decoded to bf16 with exact bit operations in the K-loop.
- Marlin CUDA needs a compiled extension — adopt only if it builds standalone; else Triton.

### 5.7 Model-specific specialization

A model can be optimized as deeply as a single-model engine (e.g. Strata) without affecting any other model,
because everything model-specific lives in the model's own package and is bound only in its lockfile.

```
plugins/models/<model>/
  recipe.py      # generic composition of std units — portable, correct, and the ORACLE for everything in fast/
  fast/          # model-owned specializations, bound only in this model's lock
    kernels/     # shape-specialized kernels (this model's dims as constexpr); supports() matches only its shapes
    composites/  # fused multi-op / multi-layer paths; decomposition = the generic recipe path they replace
    data/        # routing/expert profiles, table layouts, tuned defaults
  bundle/        # tokenizer, chat template, parsers
```

| Level | Example | Mechanism | Judged |
|---|---|---|---|
| 1 Tunables | PCIe/CPU miss split, swap interval, draft k | the model's lock, per fingerprint class | by owning plugin's class |
| 2 Shape-specialized kernels | GEMV/MoE kernels with constexpr dims | kernel under `fast/kernels`, `implements` the generic op's `sem` **at that site** | bounded (L0 vs the `sem`-named `reference.py` at a reviewed commit); agent may auto-promote |
| 3 Fused composites | whole decoder layer or verify window in one kernel | composite under `fast/composites` that declares **which sites it replaces**; the judge builds the reference itself from the recipe term (the `sem`-pinned references of those sites) — the composite's own decomposition is never used for judging | L0/L1.5/L2 vs the judge-built path; **human proposal in v0** |
| 4 Model-specific policies and data | expert placement profile, PLE row layout, MTP-vs-lookup cost model | policy plugins + `fast/data` | policy rules (§10.3); agent may auto-promote |
| 5 Hand-written forward | full model-owned `build`/`weights`, possibly native | escape hatch; still uses core scheduler, ledger, prefix index, frontends, judge | L2 vs HF + L1.5; **human** |

Rules: model-specific code never appears as `model_type`/architecture branches in shared units (invariant 10); a
specialization that proves generic graduates into std as a new knob or unit through propagation (§4.6).
Level-1, -2 and -4 specializations are kernel/policy/tunable-scope changes, so the self-improving agent may tune and
replace them within §10.5. Level 3 and level 5, any `model` slot binding, and any `build`/`weights` override are
model changes (human), regardless of what the closure diff of recipe and `sem` hashes shows (§10.3).

Limits: the step-loop timing (§7.3) is fixed — a persistent kernel that owns scheduling across steps needs a core
change; fast paths are judged bounded unless they keep the reference reduction order.

---

## 6. Memory and state

### 6.1 Ledger and pools

```python
class Ledger:
    def request(self, name: str, unit_bytes: int, min_units: int, pref_units: int, priority: int,
                resize: Literal["fixed", "recapture", "vmm"], device: DeviceId) -> Pool
    def rebalance(self, targets: dict[str, int]) -> Ticket        # applied between steps (v1)

class Pool:
    base: int                     # stable device address for the pool's lifetime
    capacity_dev: Tensor          # device scalar; in-graph resolvers read it
    def alloc(self, n) -> Tensor; def free(self, ids) -> None; def refcount(...)
    def on_shrink(self, cb) -> None
```

- Startup: the model's `state_specs`, weights, kernel workspaces, draft heads and plugin pools declare needs;
  the ledger profiles free memory and grants.
- Budget split (v0, deterministic): (1) weights + workspaces + graph memory; (2) `Slot` pools sized exactly
  `max_seqs × (1 + draft_slots)` from the deployment file; (3) plugin pools at their declared `pref_units`, or
  `min_units` if short (e.g. checkpoint pool: a byte budget from the deployment file, with its own eviction);
  (4) all remaining memory to `Paged` pools, divided in proportion to bytes-per-token of each spec; specs in one
  page group (`co_index`) get one shared unit count. Startup fails
  with a per-pool report if any `min_units` cannot be met.
- v0 implements `fixed`. The **contract** (stable base, device capacity scalar, declared resize mode) exists from
  day 1 so `vmm` (cuMem-backed live resize, Strata-style) lands in v1 without changing users of `Pool`.
- One ledger per device; Pools carry their device.

### 6.2 StateSpec and StateKind

Allocation shape is closed and tiny; meaning is an open registry.

```python
@dataclass(frozen=True)
class StateSpec:
    name: str                       # unique within the model, e.g. "attn.kv", "gdn.state", "dsa.idx"
    kind: str                       # StateKind registry key
    shape: Paged | Slot             # Paged(unit_bytes, tokens_per_unit) | Slot(unit_bytes)
    layers: tuple[int, ...]
    layout: Layout                  # dtype/arrangement, negotiated with kernels
    alignment: int = 1              # prefix-hit granularity only when intrinsic (e.g. compressed-KV ratio);
                                    # GDN uses 1 — its kernels accept an initial state at any offset
    co_index: str | None = None     # page group: specs sharing page ids, one page table and one unit count,
                                    # allocated/evicted jointly (DSA indexer ↔ MLA KV; drafter KV ↔ target KV)
    draft_slots: int = 0            # Slot: extra slots for verify intermediates
    retention: str | None = None    # retention policy id (sliding window freeing)

class StateKind(Protocol):          # e.g. paged_kv, ssm_checkpoint, ring, dsa_index (plugins)
    def units_needed(self, spec, seq: SeqView, new_tokens: int) -> int
    def reclaimable(self, spec, seq: SeqView) -> list[UnitRef]
    def hit_length(self, spec, idx: PrefixView, upto: int) -> HitLen          # (length, tier cost)
    def publish(self, spec, seq: SeqView, upto: int) -> tuple[Payload, list[MemOp]] | None
    def adopt(self, spec, seq: SeqView, payload: Payload) -> list[MemOp]
    def metadata(self, spec, batch: Batch) -> dict[str, Tensor]       # page tables, slot ids, positions
    def fork(self, spec, src: SeqView, dsts: list[SeqView]) -> list[MemOp]   # n>1, parallel tool branches:
                                    # copy-on-write pages, copied Slot state
    # publish/adopt return MemOps (§6.3); rollback: StateKinds are Rewindable (§7.5). Callbacks are per request
    # and are vectorised over SoA arrays where the core calls them per step.
```

Ring buffers = `Slot` sized to the window, or `Paged` + `reclaimable`. A new mixer with a novel state shape
reduces to `Paged(tokens_per_unit=…)` or `Slot` + its own `StateKind`; global shared state uses a plugin-owned
Pool.

v0 decision: **one pool per StateSpec**. Shared padded pools are an opt-in v1 experiment (adopt if padding waste
< 5% on Qwen3.5 and DSV-class configs).

### 6.3 PrefixIndex

- Core-owned; one per engine. Key: `(namespace, block_hash_chain)`; value: `{spec_name: Payload | Tombstone}`,
  each payload records its tier.
- `namespace = hash(model_id, weight_version, adapter_id, mm_salt, statekind_versions)` — prevents silent
  corruption across weight updates, adapters, multimodal inputs and StateKind upgrades.
- Hit length is a fixed point over all specs (no `isinstance`); `hit_length` returns `(length, cost)` so a hit
  that lives on SSD is not scored like a VRAM hit:

```python
L = upto
while True:
    L2 = align_down(min(k.hit_length(s, idx, L).length for s, k in specs), lcm_alignment)
    if L2 == L: break
    L = L2
```

- Drafter state (e.g. MTP draft-layer KV) takes part in the fixed point like any other spec, so a prefix hit
  restores the draft layer too.
- What to publish (checkpoint policies) and what to evict are plugins. v0: hash chain over full KV pages
  (nano-vllm style). **Eviction is joint per chain**: evicting a recurrent checkpoint also releases the KV pages that
  only it made reachable, so KV and checkpoint LRUs cannot strand each other.
- **Publish and adopt are device MemOps** (§7.3), not host reads of live state: `publish` emits "copy slot →
  checkpoint" ordered after the device commit (in the sample graph) of the step that ends at the boundary — so with
  `replay` verify the copied state is the post-accept state — and the checkpoint is
  registered in the index by `host_commit` only if `n_accept` covers that boundary; `adopt` emits "copy checkpoint →
  active slot" before the forward that uses it; source checkpoints stay pinned until the copy completes. The spec
  transform caps verify windows at checkpoint boundaries.
- Because the index is page-granular, recurrent checkpoints are taken at page boundaries. For agent traffic the
  default policy checkpoints at the **end of each prompt** (chat templates may re-render earlier turns, e.g. strip
  reasoning, so anchors on sampled tool-call tokens are unreliable); the chunk is cut at `floor(end / page) · page`.
- Eviction-policy contract: `victims()` may return **demote** actions (move a payload to a lower tier, keep the
  index entry) as well as drops; payloads with an in-flight async store stay pinned until the store completes.

### 6.4 Async loading

- When a prefix hit lives below VRAM (offload tier, PD transfer), the **connector/tier plugin chooses** per request
  between (a) `BLOCKED(future)` — the Seq is skipped until all its state has arrived — and (b) layer-wise loading —
  the Seq runs while ops wait on per-layer ready events `ctx.ready(spec, layer)`.
- Load futures **reserve their destination pages** in the ledger when issued and are **cancellable**: abort and
  preemption cancel the future and release its pages.
- Both mechanisms exist in v0 (unused by v0 plugins) — retrofitting them is what made vLLM's connector API ~45
  methods wide.

### 6.5 TieredStore (std library, not core)

`plugserve-std-tiered`: `TieredStore[K]` over `Tier`s (vram/pinned/ram/ssd/remote), a `ResidencyMap`, `Mover`s and
`PlacementPolicy` plugins, with two resolver styles:

- `HostPlannedResolver` — misses known before the step (KV prefix/offload, KVMem-style window selection); runs in
  `Planner.plan_bounds`; emits transfers (MemOps) + per-layer events.
- `DeviceResolver` — misses known only inside the step (experts, sparse-KV); fixed-shape, in-graph lookup
  `ids → slot | -1`; host-side `adapt(stats)` (decayed-LFU, profile-seeded, LRU) writes a double-buffered residency map that is swapped
  in between steps (or under `quiesce`, §7.3), never under an in-flight graph.

What to do with a miss is the consumer's decision: the MoE op computes `-1` experts on CPU through the stream-memop
handshake with a native CPU worker (§7.4, FreeToken) or fetches them over PCIe (Strata's `pcie_frac` split). The MoE op declares `placement_invariant`;
if false, residency policies are judged `bounded` rather than `exact` (§10.3).

---

## 7. Execution

### 7.1 Seq

```python
class Seq:
    rid: str; slot: int                          # persistent device row
    tokens: list[int]                            # committed only
    num_computed: int
    status: WAITING | RUNNING | BLOCKED(future) | PREEMPTED | FINISHED(reason)
    params: SamplingParams                       # .ext[ns] for plugin params
    ext: dict[str, Any]                          # namespaced plugin state; core never reads
```

**SeqTable (core, device-authoritative).** Per-slot device tensors — `num_computed`, `last_token`, `n_accept`,
`draft_tokens` — are the source of truth and are updated only inside the sample graph (by `Rewindable.commit`).
The host `Seq.num_computed` is an optimistic mirror: an upper bound that lags by at most `inflight_depth` steps and
becomes exact at commit (the pattern of vLLM Model Runner V2's `states.py`, implemented natively here). This is what
lets overlap and speculative decoding coexist: the host never needs `n_accept` of step N−1 to launch step N.

Scheduling mechanism is phase-free (vLLM v1 idea): each Seq wants `len(tokens) + drafts − num_computed` tokens and
a token budget covers prefill chunks, decode and verify uniformly. The v0 FCFS **policy** never mixes prefill and
decode rows in one step, so decode steps always run in a FULL graph (mini-sglang's choice); a mixing policy is a v1
plugin together with PIECEWISE graphs.

### 7.2 Batch

```python
@dataclass
class BatchPlan:                     # host, struct-of-arrays, mutable by scheduler + transforms
    step: int
    slots, n_query, n_logits: np.ndarray
    start_upper: np.ndarray          # upper bound on first query position; exact value lives in SeqTable
    token_src: np.ndarray            # HOST | DEVICE_LAST | DEVICE_DRAFT  (overlap-safe placeholders)
    host_tokens: np.ndarray
    parents: np.ndarray | None       # tree verify (v2)
    seq_len_upper: np.ndarray
    mem_ops: list[MemOp]
    shape: ShapeKey                  # padded bucket + open plugin axes
    ext: dict[str, Any]

@dataclass(frozen=True)
class ShapeKey:
    n_tokens: int; n_seqs: int | None
    axes: tuple[tuple[str, Hashable], ...]       # e.g. (("spec.q", 4),) — finite, declared by plugins

class Batch:                         # plan + static device mirror built by the core
    plan: BatchPlan
    dev: DeviceBatch                 # positions, token_req, query_start, page_table[spec], slot_index[spec]
```

### 7.3 StepLoop

Fixed timing; plugins attach to named hooks inside the regions.

```python
def step(self):
    while len(self.inflight) > self.depth:                              # depth = max uncommitted steps at schedule time
        self.commit(self.inflight.popleft())
    plan = self.scheduler.schedule(self.seqs, self.ledger, self.inflight)   # CPU; skips Seqs over their own depth
    for t in self.transforms: plan = t.transform(plan, self.sctx)       # ordered, pure
    h = self.exec.launch_forward(plan)                                  # GPU (graph): prepare → model → logits
    if self.depth >= 1 and len(self.inflight) >= self.depth:           # overlap: commit oldest step while GPU runs
        self.commit(self.inflight.popleft())
    host_in = {p.name: p.host_input(plan) for p in self.host_input_stages}       # e.g. grammar bitmasks
    self.exec.launch_sample(h, host_in)    # GPU (graph): logits procs → Sampler|Verifier → Rewindable.commit (device) → drafters
    self.inflight.append(h)

def commit(self, h):                                                    # CPU after D2H event of step h
    for r in self.rewindables: r.host_commit(h.view)                    # frees, ledger trims
    finish/stop/stream; apply deferred frees; apply control-plane requests
```

`inflight_depth` = the maximum number of uncommitted steps when the scheduler runs:

| depth | Behaviour | Use |
|---|---|---|
| 0 | commit before schedule (fully synchronous) | debugging, judge invariant tests, CPU drafters/plugins needing the last token on host |
| 1 | schedule step N while step N−1's sample is still on the GPU; N−1 is committed during step N's forward | default; grammar `host_input` works (it runs after N−1's commit) |
| 2 | two uncommitted steps (deeper pipelining; PP uses N) | v1 perf option |

**MemOps and quiesce.** The plan carries an ordered list of device memory operations (copy, swap, load, checkpoint
publish/adopt, fork) with pins on their sources; the core executes them in-stream at defined points (before the
forward, after the sample). `StepLoop.quiesce(callback)` drains all in-flight steps (commits them), runs the
callback, then resumes — used for VMM resize, `update_weights`, `flush_prefix`, residency-map swaps and binding
swaps, so nothing ever mutates state under an in-flight graph. Resolver maps updated between steps (expert
residency) are double-buffered.

**Host lengths.** For non-speculative Seqs the host mirror is exact (it advances by one per step), so host-side
planning such as FlashInfer `plan()` (run in `plan_bounds`) works at depth 1. The host mirror is inexact only for
Seqs with an uncommitted speculative step (`n_accept` unknown). Planners whose kernels need exact host lengths
declare `needs_host_lens`, and **only those inexact Seqs** are gated to depth 0 for them. Spec-decode paths
therefore use an attention kernel that reads lengths on the device (from the SeqTable) — a device-length path of
FlashInfer or our own Triton paged attention; spike S1 names the kernel per fingerprint class.

Plugins declare the maximum depth they tolerate (`overlap_safe` = tolerates ≥ 1). Per-Seq depth: the scheduler
skips any Seq that has more uncommitted steps in flight than the minimum depth tolerated by the plugins attached
to it, so a CPU prompt-lookup Seq runs every other step while others keep overlapping. Device-side
`Rewindable.commit` runs **inside** the `launch_sample` graph, right after the Sampler/Verifier, so the next step's
device inputs (`DEVICE_LAST`/`DEVICE_DRAFT`) are already consistent.

### 7.4 CUDA graphs

- Graph units: `launch_forward` and `launch_sample`.
- Capture set = bucketed `n_tokens`/`n_seqs` × product of declared `ShapeKey` axes, filtered, capped by a memory
  budget. Unbounded axes are rejected at resolve.
- Each plugin declares `graph_policy` ∈ {`full`, `piecewise`, `eager`, `host`}; the resolver takes the weakest
  per ShapeKey class. `host` means the plugin never runs inside a graph — it runs on the CPU between replays
  (placement-policy `adapt()`, schedulers). v0 implements FULL (decode/verify) and EAGER (prefill); PIECEWISE is v1.
- **CPU work inside a FULL graph** (CPU expert compute, Strata/FreeToken style) uses **stream memory operations**:
  the graph writes a flag (`cuStreamWriteValue`) that a persistent native CPU worker polls, and waits on the
  worker's completion flag (`cuStreamWaitValue`). `Platform` exposes these memops. Per-call `cudaLaunchHostFunc`
  is not used: FreeToken measured 30–50 µs per call (≈ 6 ms per step on a 75-layer model) and replaced it. No
  Python or GIL on this path. Spike S6 in M0 validates the seam before v1 expert offload builds on it.
- All dynamic inputs go through static buffers requested at startup with `ctx.static(name, max_shape)`.

### 7.5 Rewindable

```python
class Rewindable(Protocol):
    def commit(self, slots: Tensor, n_accept: Tensor, path: Tensor | None = None) -> None  # device, graph-capturable
    def host_commit(self, view: CommitView) -> None                                       # optional, CPU
    verify_mode: Literal["trivial", "checkpoint_slots", "replay", "snapshot"]
```

Implemented by every per-sequence state: paged KV (truncate), GDN/Mamba (select `draft_slots[n_accept]` or replay
Strata-style), grammar matcher, stop-token detector, penalties, drafter KV. Memory matters for recurrent state:
Qwen3.5-9B GDN state is fp32, ≈ 48 MiB per sequence per copy, so `checkpoint_slots` with k drafts costs
(k+1)× that per running Seq. Consumer-GPU deployments default to `replay`; `checkpoint_slots` is opt-in with a
capped `draft_slots`. `path` is in the signature from day 1;
it stays `None` until tree verification (v2).

### 7.6 Sampling and decoding

- `LogitsProcessor` chain (ordered multi): `argmax_invariant`, per-slot device params, optional `host_input()`.
  The default sampler is a builtin plugin. Sampler, processors and verifiers declare whether they accept
  vocab-sharded logits or need gathered logits (TP seam, §8).
- Speculative decoding = four roles:
  - `Drafter` (EAGLE-3, MTP, draft model, GPU n-gram, CPU prompt lookup); declares required taps/shared modules.
  - `DraftPolicy` (single): per-Seq `(source, k)` from a finite set; `observe()` for adaptive cost models (Strata).
  - `Verifier` (single): replaces the Sampler during verify (greedy-exact, rejection, typical; tree in v2).
  - `PlanTransform`: appends `DEVICE_DRAFT` rows and sets the `spec.q` ShapeKey axis. Seqs with different k in one
    step are padded to the step's `spec.q` (the max k); the DraftPolicy chooses k from a finite declared set.
- Rollback = core commit fan-out to all Rewindables.

### 7.7 Control plane and errors

- `abort(rid)` (frontend → loop; frees at next commit — safe under overlap), `pause/drain`, `resume`,
  `flush_prefix` (v0). `update_weights(name → tensor)` (in-place through converters into existing QTensor storage;
  bumps `weight_version`) is specified now because it fixes the namespace design, but implemented in v1 (RL weight
  sync, LoRA merges).
- Per-request error isolation: a plugin exception tied to one Seq (grammar, parser) → `FINISHED(error)` for that
  Seq.
- Step-level exception or device fault → engine exits non-zero; the supervisor restarts it **on the same lock**.
  **Crash attribution first:** in-flight requests are resubmitted from committed tokens, rate-limited; a request
  implicated in two crashes is quarantined (`FINISHED(error)`). Only repeated crashes without a poison request lead
  to rollback to the previous lock. Restart time (weights, graph capture, JIT caches) is measured and has an SLO
  set in M0.
- Sampling API coverage in v0 includes `n > 1` (via `StateKind.fork`), output logprobs/top-k logprobs and **prompt
  logprobs** (the judge's teacher-forced re-scoring also uses them).

### 7.8 Frontend

Separate process. API plugins (`openai.chat`, `openai.responses`, `anthropic.messages`) implement
`to_engine(http_req) -> EngineRequest(token_ids, params, ext)` and `from_engine(stream) -> events`, using the model
plugin's `FrontendBundle` (tokenizer, template, reasoning/tool parsers with a token-level `channel()` state
machine). Stop strings are matched in the frontend and trigger `abort`; stop tokens are an engine Rewindable.

The frontend↔engine protocol is a **language-neutral schema** (token ids, sampling params, deltas, logprobs;
msgspec structs from which a protobuf/IDL file is generated and checked in CI), so the frontend can later be replaced by a native (e.g. Rust)
implementation without touching the engine — as vLLM (`vllm/rust`) and SGLang (`rust/`) are doing. Frontend
plugins (API dialects, parsers) would then move to compiled or declarative form; until that trigger (§12), the
frontend stays Python.

### 7.9 Telemetry

`Event` records emitted per step to a ring buffer (shared memory to the supervisor) and optionally to exporters
(Prometheus/OTel plugins):

```
{plan_id, step, ts, phase_us{schedule,transform,forward,host_input,sample,commit},
 batch{n_seqs,n_tokens,shape_key}, ledger{pool: used/cap}, prefix{hits,misses,tokens_saved},
 spec{proposed,accepted}, plugin_counters{ns: {...}}}
```

Phase timings are measured by the core with CUDA events for **diagnosis and agent guidance**. Verdicts and canaries
never rely on them, because candidate code may share the engine process: the judge measures performance
client-side and checks outputs by teacher-forced re-scoring out of process (§10.3, §10.4).

### 7.10 Determinism and snapshots

Two acceptance modes are used in v0 (plan and judge):

| Mode | How | Meaning |
|---|---|---|
| **run-identical** | production Plan replaying the model's **snapshot trace**: a fixed multi-Seq, teacher-forced trace with fixed arrival order and batch composition that covers every captured graph bucket, chunked prefill, depth 1, a prefix hit, `n > 1`, prompt logprobs and — where the deployment enables them — spec decoding and grammar; sampling seeded per (Seq, position); adaptive policies frozen; changed-line coverage recorded | byte-identical output run to run. Used for model isolation ("goldens unchanged") and for auto-advance in propagation (§4.6) |
| **bounded** | production Plan | within judge L2 budgets |

- run-identical needs kernels that are deterministic **run to run** (no split-K or atomic accumulation in pinned
  configs; verified by the spikes and judge L1), not batch-invariant ones. Sites whose production kernel is not
  deterministic are recorded as `uncovered` in the lock; a change touching them can never auto-advance.
- Comparisons that change batch shape, chunking or path — prefix cache on/off, depth 0 vs 1, spec vs non-spec,
  graph replay vs eager (padding changes tile selection; FlashInfer splits KV differently in graph mode) — are
  **bounded**. Spec decoding additionally checks greedy self-consistency: every accepted token equals the argmax of
  the target logits computed in that verify step.
- **Deferred:** a batch-invariant "deterministic Plan" (det-identical comparisons across batch shapes, vLLM
  `batch_invariant`-style kernels). It is added only if a concrete validation need appears.

---

## 8. Platform and datacenter extensibility (D1)

v0 runs one GPU, but the core keeps these seams so datacenter features are plugins:

| Feature | Seam in core (v0) | Plugin (later) |
|---|---|---|
| Tensor parallel | `ParamSpec` sharding semantics; collectives are registered Ops (`allreduce`, `allgather`); Plan is serializable and broadcast; engine always runs the SPMD path (world_size=1 in v0); a `launcher` slot (single) starts ranks — v0 builtin launches one local rank; sampler/processors declare sharded-vs-gathered logits | `tp.nccl` collective kernels; TP-aware linear layouts; `launcher.torchrun`/`launcher.ray` |
| Expert parallel | MoE is a coarse Op owning routing/dispatch | `moe.ep_dispatch` (DeepEP-style) |
| Pipeline parallel | `inflight_depth = N` | `pp.stage_split` |
| DP-attention | PlanTransform with a collective hook in the schedule region | `dp_attention.pad` |
| PD disaggregation | `BLOCKED(future)`, async store import/export, per-layer events | connector + schedule-policy role |
| Multi-device memory | Ledger per device; Pool carries device | — |
| Other platforms | `Platform` (~15 functions); lint bans `torch.cuda` elsewhere | `platform.rocm`, `platform.cpu` |

The dev machine has 8× RTX A5000; v1 includes a **TP=2 milestone** to prove the seam with zero core diff.

---

## 9. Feature → plugin map

| Feature | Plugins | Core seams used |
|---|---|---|
| KV offload (RAM/SSD, remote) | tiers, mover, placement policy, connector | PrefixIndex tier payloads, BLOCKED, per-layer events |
| KVMem-style bounded window | `windowed_retrieval` StateKind, HostPlannedResolver, re-RoPE prepare kernel | StateKind emits positions; namespace |
| Expert cache / offload | expert TieredStore, DeviceResolver, placement policies, MoE op with CPU miss compute | stable Pools, stream-memop CPU handshake, `quiesce`, VMM resize (v1) |
| Quant formats | dialect, scheme, layout, converter, kernels | kernel registry, layout negotiation |
| Linear attention (GDN, KDA, Mamba2) | mixer + `Slot` state (`draft_slots`), page-boundary checkpoints, checkpoint policy + codec | fixed-point prefix hits, device Rewindable |
| DeepSeek sparse attention / V4 compressed KV | `dsa_mla` mixer: MLA `Paged` + indexer `Paged(co_index)` + compressed `Paged(tokens_per_unit)` | `co_index`, `tokens_per_unit`, `alignment` |
| Speculative decoding | drafter, draft policy, verifier, plan transform | taps, shared modules, Rewindable, ShapeKey axis, forward/sample split |
| Structured output / tool calls | xgrammar logits processor (Rewindable matcher), parser `channel()` | `host_input` region |
| LoRA (v1) | adapter op context, ShapeKey axis | namespace, ops read `Batch.dev`/`ext` |
| VLM (v1) | encoder plugin, content-hash encoder cache | `position_fn`, `ext` annotations |

Detailed walk-throughs (DeepSeek-V4-class on 16 GB, Qwen3.8-Flash-Next hybrid, new mixer type X) were worked
through during the design reviews.

---

## 10. Optional self-improvement layer (D5)

**Status:** not part of v0, and nothing else depends on it. The engine ships best-known, human-tuned defaults
(§4.4). v0 builds the judge harness (L0–L2, L4) for our own development and CI; this layer reuses it and is added
after the core feature milestones (plan §10), for capable agents and explicit user intent.

### 10.1 Actors

- **Agent** — a capable agent the user chooses to involve: an external coding agent, or a strong model served by
  plug-serve itself. Untrusted. Writes code only inside `workspace/`.
- **Supervisor** — trusted; serves the improve API, records promotions as lock commits, restarts and rolls back.
- **Sandbox engine** — runs candidate Plans (possibly with agent code) in a restricted sandbox (§10.6).
- **Judge** — trusted, out of process, drives engines only as a client.

### 10.2 Improve API (MCP + HTTP, served by the supervisor, localhost-only, token-authenticated)

| Tool | Purpose |
|---|---|
| `telemetry.query(window, fields)` | Event aggregates for the live Plan (diagnostic) |
| `plan.describe()` / `site.explain(site)` | Which plugin/kernel is bound where, and why |
| `plugins.list(slot)` / `plugin.source(name)` | Read manifests and sources |
| `traces.list()` | List **dev** traces (holdout is never listed or readable) |
| `tunables.get(site)` / `tune.submit(site, space?)` | Tunable search inside the declared space |
| `variant.fork(plugin, new_name)` | Create a variant in `workspace/` (copy-on-fork; sets `variant_of`, `implements`) |
| `variant.test(name, tiers)` | Cheap correctness iteration in the sandbox; no promotion |
| `variant.submit(name)` / `job.status` / `job.cancel` / `evidence.get` | Full judging and its results |
| `registry.status()` / `registry.rollback(model)` | Inspect promotions; roll back to the previous lock (never force-promote) |
| `proposal.open(title, diff, rationale)` | Human-review proposal for out-of-scope changes (§10.5) |

First version: `telemetry.query`, `plan.describe`, `variant.fork`, `variant.test`, `variant.submit`, `job.status`,
`registry.rollback`. The token is issued to the agent harness, not to ordinary API clients. Rollback never depends
on the agent: the supervisor reverts on its own (§10.4).

### 10.3 Judge tiers and numerics classes

| Tier | Check | Pass rule |
|---|---|---|
| L0 op oracle | vs the unit's `reference.py` / `dequant_ref` on the Plan's real shapes + fuzzed edges | budget = f(dtype, reduction length), held in the judge |
| L0-dist | samplers, verifiers, logits processors: fixed logits, ~10⁶ draws vs the reference sampler | TV distance / χ² bound |
| L1 contract | slot conformance, ledger leaks/accounting, graph-capture safety, declared props verified (run-to-run determinism, depth tolerance) | pass/fail |
| L1.5 state invariants | chunked vs one-shot prefill; prefix-hit resume vs recompute; verify+commit(k) vs k decode steps; graph replay vs eager | bounded (§7.10); spec greedy self-consistency exact |
| L2 model golden | teacher-forced logits vs HF fp32 recording on frozen prompts | calibrated noise floor = HF in the model's activation dtype vs HF fp32. Pass: top-1 agreement ≥ floor − δ; KL p50/p99 ≤ floor + δ; ΔPPL ≤ floor + δ |
| Snapshot | the model's snapshot trace (§7.10) | run-identical (isolation, auto-advance) |
| L4 perf | judge-side client measurements (TTFT, TPOT, throughput, peak memory) on a trace portfolio incl. mixed agent traffic, production Plan, interleaved ABAB, N ≥ 5, bootstrap CI; outputs checked by teacher-forced re-scoring against the reference Plan | per trace: CI lower bound of perf ratio ≥ 1 − ε; target gain CI excludes 0; re-scored KL/top-1 within L2 budgets; greedy tokens equal the candidate's own argmax |
| L4-confirm | one fresh run on **holdout** traces after a dev-trace pass | same rule |

- Numerics classes: `policy` — run-identical vs its parent on the snapshot trace only if it leaves batch composition
  and execution paths unchanged (e.g. eviction or placement on traces without memory pressure, with a
  placement-invariant consumer); scheduler, draft and cache-admission policies change batch composition or paths
  and are judged bounded + L4 re-scoring; `exact` (claims run-identical vs its reference path), `bounded`,
  `approx` (human gate). **The judge chooses tests and budgets**; plugin-shipped tests are additional only.
- **Oracle provenance:** oracles are the `reference.py` blobs at reviewed commits named by `sem` (§4.6), never the
  candidate's. **Closure-diff classification** of a submission, site by site: recipe hash changed, or a site's
  `implements` `sem` differs from the parent binding's `sem` at that same site, or any `model` slot binding,
  `build`/`weights` override, or multi-site composite → model change (human); toolchain change → human; otherwise
  §10.5. Every other model's lock must be byte-unchanged.
- Holdout traces exclude anything generated by the agent.

### 10.4 Episode lifecycle

1. The agent reads telemetry and sources; tunes declared tunables first.
2. If tuning plateaus: `variant.fork` → edit in `workspace/` → `variant.test` → `variant.submit`.
3. Evaluation runs in the sandbox engine on a **spare GPU of the same fingerprint class** (first version). On
   single-GPU hosts only host-side policies and live tunables are in scope. In-process A/B and drain-window
   evaluation are deferred.
4. Pass → the supervisor commits the variant and a new lock to its branch (§4.4) and restarts the live engine on it
   → **canary**: golden smoke test, crash-loop detection, and **sampled live re-scoring** — the judge re-scores a
   sample of live outputs with teacher forcing, out of process. This is the check that does not depend on the
   variant being unaware it is judged. (A bucketed TTFT/TPOT canary is added later.)
5. Canary failure or crash loop (after poison-request attribution, §7.7) → restart on the previous lock, variant
   quarantined for this fingerprint class. No agent action needed.
6. One promotion at a time per model.

### 10.5 Scope of autonomy

| Change | Agent may auto-promote? |
|---|---|
| Tunable values within declared ranges | yes |
| New variant of a `policy` plugin (scheduler, cache, placement, draft policy) | yes — run-identical vs parent where batch composition and paths are unchanged, else bounded + L4 re-scoring (§10.3) |
| New kernel variant for an existing Op at a site whose `sem` it implements (incl. model-specific `fast/` kernels, §5.7 level 2) | yes — bounded |
| New plugin added to an existing engine-side slot | only drafter / exporter / connector / checkpoint_policy, if `numerics ≠ approx`, slot ABI unchanged, no new dependency |
| New plugin added to an output-affecting slot (logits_processor, plan_transform, …) | proposal (human) |
| Multi-site composites (§5.7 level 3), model-owned `build`/`weights` (level 5), `model` slot bindings | proposal (human) |
| Model plugin changes that alter goldens; new tap points | proposal (human) |
| Core, judge, std units, `approx` numerics, new third-party dependencies | proposal (human) |
| Frontend-side plugins (API dialects, tokenizer/template bundles, parsers) | proposal (human) |

Limits: per-variant diff-size budget, promotions per model per day capped.

### 10.6 Trust enforcement

Static rules (import contracts, no core-internal imports) are hygiene, not security, and hiding the determinism
setting from plugins is not a control either — a variant can infer it is being judged. The controls are process
and OS boundaries plus judge-owned oracles and sampled live re-scoring (§10.4).

First version:

| Boundary | Can write | Can read | Network |
|---|---|---|---|
| Agent harness | `workspace/` only | repo (read-only), dev traces, evidence summaries | the frontend's and improve API's localhost ports |
| Sandbox container (sandbox engine) | per-run scratch (tmpfs, discarded) | repo + `workspace/` variant (read-only), weights, pre-built JIT caches | none (own network namespace; IPC to the judge only) |
| Live engine (runs promoted agent code) | per-run scratch only | repo + promoted commit (read-only), weights, pre-built JIT caches; **not** holdout, locks or evidence | serving IPC to the frontend only |
| Trusted host processes (supervisor, judge, frontend) | locks/branch, evidence | everything | localhost |

- Yama `ptrace_scope ≥ 2`; the holdout store is readable only by the judge.
- Later: a per-instance network namespace for each engine process, and a separate identity per promoted variant
  if several run concurrently.

---

## 11. Repository layout

```
plug-serve/
  core/            plugserve/{api,core}/          # ≤ 6k LOC
  std/             ops/ blocks/ ffn/(swiglu, moe_topk) mixers_gqa/ state_paged_kv/ tiered/ sampling/
  plugins/         models/<m>/{recipe.py, fast/, bundle/}  for qwen3, qwen3_moe, qwen3_5
                   mixers/{gdn}/  state/{ssm_checkpoint}/
                   kernels/{flashinfer,fla_gdn,triton_moe,triton_wxa16}/  dialects/{gptq,fp8_block}/
                   frontends/{openai,anthropic}/  policies/  drafters/  verifiers/  grammar/
  plans/           <model_id>/<class>/<n>.lock     # lock history per (model, fingerprint class) (§4.4)
  judge/           plugserve_judge/  goldens/  snapshots/  traces/dev/   # holdout stored outside the repo (judge only)
  supervisor/      plugserve_supervisor/
  bench/           trace replay, baselines (separate pinned venvs)
  evidence/        evidence records (command, fingerprint, plan_id, raw numbers)
  workspace/       (gitignored) agent variants — optional layer
  docs/
```

---

## 12. Risks

| Risk | Mitigation / deciding experiment |
|---|---|
| Python per-step overhead at large batch | overlap (depth 1) hides CPU work shorter than the GPU step; M0 measures CPU time per step at B=256 vs nano-vllm; if worse, move per-token input prep to GPU kernels (vLLM Model Runner V2 style) |
| Control-plane CPU still a bottleneck after GPU-side input prep (CPU step > ~50% of GPU step at target batch) | replace the profiled hot structure (PrefixIndex, allocator, per-request scheduling) with a Rust/PyO3 module behind its existing interface (§3); plugin API stays Python |
| Frontend bottleneck (tokenization/streaming under many connections) or multi-instance routing needed | native (Rust) frontend/router over the language-neutral IPC schema (§7.8), following vLLM/SGLang |
| Shared-change sweep cost / approval churn | auto-advance on byte-identical snapshot traces (§4.6); async propagation when the sweep exceeds 30 min or approvals exceed 1/day |
| Models stuck on explicit old-version copies | visible in diffs; holdback tracking added at the §4.6 trigger |
| Overlap + speculative decoding need exact lengths on the host | device-authoritative SeqTable + `prepare_device` + `needs_host_lens` gating (§7.1, §7.3) — built in M0 |
| CPU work inside graphs too slow (host functions) | stream-memop handshake with a persistent native worker (§7.4); spike S6 in M0 |
| Abstractions too generic → slow | composites + direct bound callables; FULL graphs on decode; perf gate vs mini-sglang |
| Wrong core abstraction discovered late | each v0 milestone is chosen to stress one seam (plan §2); seam-breaking discoveries are fixed while v0 is pre-1.0 |
| Borrowed kernels violate contract | adoption rule §5.6; write our own Triton fallback |
| Agent reward hacking / self-harm (optional layer) | §10.5 scope, §10.6 sandbox, judge-owned oracles, client-side measurement, holdout confirm, sampled live re-scoring, supervisor-owned rollback |
| Dev GPUs (A5000, sm_86, 16-channel server host) misrepresent consumer GPUs | perf/tunable evidence only from real consumer desktops: RTX 5080/5090 (sm_120), 4090 (sm_89); A5000 for development and functional tests |
| sm_86 lacks FP8/FP4 tensor cores | weight-only (W8A16/W4A16) kernels on sm_86; native FP8 on sm_89/sm_120 and NVFP4 on sm_120, verified on consumer desktops |
| Determinism claims unachievable with borrowed kernels | §7.10: only run-to-run determinism is required (no batch invariance); non-deterministic sites recorded as `uncovered` |
| Poison requests trigger restart loops and wrong rollbacks | crash attribution before rollback, request quarantine, restart SLO (§7.7) |
| Kernel breakage on sm_86 / torch 2.11 / Triton 3.6 | M0 spikes: uint8-decoded FP8 W8A16, vendored vLLM GDN ops, FlashInfer JIT pre-build |
| Golden recording does not fit one GPU | record on CPU fp32 (251 GB RAM) or across GPUs; quantized models via `dequant_ref` into the dense HF class |
