# plug-serve — Design

Status: draft for review · Date: 2026-10-06 · Branch: `design-doc`

---

## 0. Summary

plug-serve is a microkernel LLM inference engine. A small core (~5k LOC Python, hard cap 6k) fixes only the
timing structure of a step, memory accounting, the plugin resolver and a handful of typed contracts. **Models,
layer types, quantization formats, kernels, cache/offload policies, speculative decoding, sampling, grammars,
schedulers, frontends and parallelism are all plugins.** A trusted judge and a supervisor let an agent — including
an agent *served by the engine itself* — propose, verify, benchmark, promote and roll back improvements.

### Goals

1. **Isolation with reuse** — adding or changing one model never changes another model's output, while code is
   shared in small units. Enforced by per-model lockfiles that change only through that model's judged transition
   (§4.6) and model-scoped Registry rows (§4.4).
2. **Extensibility** — features in §9 land as plugins with an empty `git diff core/`.
3. **Self-improvement** — an agent using the engine can tune and replace plugins under a judge it cannot edit,
   with automatic rollback.
4. **Small and legible** — every plugin is small enough for a coding agent to hold in context.
5. **Competitive single-GPU performance** — within 10% of mini-sglang decode throughput on the same GPU for M0.

### Non-goals (v0)

Multi-node serving, non-NVIDIA platforms, training, a model zoo. These must remain *possible* without core
rewrites (§8) but are not built in v0.

Scope note (D1 × D5): on a host with **one** GPU, v0 self-improvement covers host-side policy plugins and
`apply="live"` tunables (in-process A/B, §10.4). Kernel and graph-path variants need a spare GPU (the dev box has
8) or the drain-window mode, which is a v0 stretch goal and hardened in v1.

### Decisions taken (2026-10-06)

| # | Decision |
|---|---|
| D1 | Single consumer GPU first. Datacenter features (TP/EP/PP, PD, DP-attention) must be addable as plugins; the core keeps the seams (§8). |
| D2 | Model ladder: Qwen3 dense → Qwen3-MoE → Qwen3.5 GDN hybrid ([plan §2](plan.md)). |
| D3 | Borrow existing kernels first (FlashInfer, flash-linear-attention, Triton MoE, …) **only if they fit the plugin contract** (§5.6); otherwise write our own. |
| D4 | Python control plane; native/Triton kernels; CUDA graphs on the hot path. The plugin API stays Python; hot core structures and the frontend may later be replaced by Rust modules behind unchanged interfaces, only when profiling triggers it (§12). |
| D5 | Self-improvement is driven by agents that *use* the engine (e.g. a Qwen3 agent served by plug-serve). Auto-promotion is allowed inside policy/kernel/tunable scope under the judge; core, judge, ABI and `approx` numerics need a human (§10.5). |

---

## 1. Invariants (lint/CI-enforced)

1. The core contains no model name, layer-type name, quant-format name, or feature-specific field.
   (CI: denylist grep over `plugserve/core` — `mamba|gdn|mla|dsa|nvfp4|eagle|qwen|deepseek|lora|…`.)
2. Plugins import only `plugserve.api`, std units (one live version, referenced via `Ref`), their own package, and third-party
   libraries declared in their manifest `[dependencies]` (e.g. `triton`, `flashinfer`). Model packages never
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
    `plugserve propagate` and the supervisor (§4.6); CI's `isolation-check` re-runs propagate on a trusted GPU
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
   clients / agents ─── improve API ──►│  Registry · Evidence ledger  │
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
- **Supervisor** — small trusted process. Owns the Registry, the evidence ledger, engine lifecycle (start, drain,
  restart on a new Plan), health checks, automatic rollback, and the improve API (§10). Never imports agent code.
- **Sandbox engine** — a separate engine process started by the supervisor with a candidate Plan, under a
  restricted OS identity (§10.6). It may run agent-written code.
- **Judge** — a separate trusted process that drives engines (the sandbox, and the live engine during in-process
  A/B) **only as a client** (HTTP/IPC), never in-process with candidate code. Verdict-relevant measurements are taken on the judge side (§10.3).

### 2.2 Packages (one uv workspace)

| Package | Trust | Content |
|---|---|---|
| `plugserve` (`core/`) | core | `plugserve.api` (protocols, dataclasses — the plugin ABI) + `plugserve.core` (runtime). ≤ 6k LOC. |
| `plugserve-std-*` (`std/`) | reviewed | Shared units: `std.ops` (linear, norm, rope, act), `std.blocks` (residual wiring), `std.mixers.gqa`, `std.ffn`, `std.state.paged_kv`, `std.tiered` (TieredStore library), `std.sampling`. One live version each; recipes reference them via `Ref("name@major")`; locks pin `src` + `sem`; changes reach models only through propagation (§4.6). |
| `plugins/*` | reviewed or agent | First-party plugins: models, kernels, dialects, policies, frontends. Each is its own dist package with entry points. |
| `plugserve-judge` (`judge/`) | trusted, read-only to agents | Oracles harness, tolerance budgets, goldens, traces, perf timer. |
| `plugserve-supervisor` (`supervisor/`) | trusted | Registry, evidence ledger, lifecycle, improve API (MCP + HTTP). |
| `workspace/` | agent | Git worktree where agent variants are written (gitignored). Promoted variant code is copied by the supervisor into the content-addressed `registry/units/<src>/` (never into the source tree) and referenced from deployment records (§4.6). |

---

## 3. Core vocabulary

Sixteen nouns. Each is a type in `plugserve.api` (the plugin ABI) with its runtime in `plugserve.core`.

| # | Noun | One-line definition | § |
|---|---|---|---|
| 1 | Manifest | Declarative plugin description, readable without import | 4.1 |
| 2 | Plan | Resolved, serializable deployment: plugins, bindings, tunables, fingerprint | 4.3 |
| 3 | Registry | `(scope, site, fingerprint) → (variant, tunables, evidence)`; autotune DB + promotion table | 4.4 |
| 4 | ModelSpec | A model's declaration (emitted by a pure recipe): layers, taps, draft heads, encoders | 5.1 |
| 5 | Op | Typed signature + reference implementation | 5.3 |
| 6 | Kernel | One implementation of an Op with constraints and properties | 5.4 |
| 7 | Layout / QTensor | Number semantics + byte arrangement; typed weight object | 5.5 |
| 8 | StateSpec / StateKind | Per-layer state allocation shape + open-registry meaning | 6.2 |
| 9 | Ledger / Pool | Byte budget over named pools with stable addresses | 6.1 |
| 10 | PrefixIndex | Content-addressed reuse of state across requests | 6.3 |
| 11 | Seq | One request's lifecycle and committed tokens | 7.1 |
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
derived_from = ""                         # optional; copied recipes / legacy holdback forks (lineage only)
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
2. Read the deployment file (model path, pins, budgets, determinism flag).
3. The model plugin's `ModelSpec` enumerates **sites** (each mixer, op, state spec, draft head).
4. For each site: filter candidates by resources, platform/fingerprint constraints, `supports(shape)`, layout
   reachability (§5.5); then pick by deployment pin → Registry row for this fingerprint → default priority.
5. Check `ShapeKey` axes are finite; compute graph policy per ShapeKey class (weakest wins).
6. Emit the **Plan** (TOML lockfile): recipe hash, per-unit `src`/`sem` (`[units]`, one version each), site
   bindings + kernel configs, tunables, fingerprint, determinism flag, `[deploy]` and `[snapshot]` (§4.6).
   `plan_id = sha256(plan)`.
7. Import only the selected entries; each plugin registers its kernels/converters/kinds into explicit registries
   passed to `Plugin.register(regs)` (no module-level registries, invariant 4); construct; bind callables. The hot
   path never does a registry lookup.

The Plan is the unit of deployment, A/B, promotion and rollback. It is serializable so every rank of a future
SPMD deployment can load the identical Plan (§8).

**Model isolation via lockfiles.** Each served model has a reviewed, checked-in **baseline** lockfile per
fingerprint (`plans/<model_id>/<fingerprint>.lock`). Promotions write **promoted** lockfiles under
`registry/plans/<model_id>/<fingerprint>/<n>.lock` (outside the source tree). The engine starts from the newest
active promoted lockfile, else the baseline — never from a fresh resolve. A fresh
resolve that differs from the lockfile (e.g. because a new, higher-priority kernel was installed for another model)
is reported as a diff and only adopted through `plugserve propagate` (§4.6). Adding a plugin
for model B can therefore never change model A's binding silently.

### 4.4 Registry

One SQLite table owned by the supervisor (read-only to the engine):

```
scope | site_key | fingerprint | variant_hash | tunables_json | evidence_id | promoted_at | status(active|retired|quarantined)
```

- `scope` = `model_id` (v0). A promotion applies only to the model whose goldens/traces were judged. Promoting the
  same variant for another model is a separate judged promotion (v1 may add `shape_class` scopes that require the
  goldens of every model in the class).
- `site_key` examples: `kernel:linear:w=gptq4.tri,x=bf16`, `placement_policy:expert`, `tunable:draft_policy.k`.
- `fingerprint`: **static facts only** — GPU SKU/UUID, SM, VRAM, driver, CUDA, torch, Triton, CPU ISA. Measured
  values (H2D bandwidth, clocks) are recorded in evidence, never in the key, so lookups are deterministic. v0
  lookup is exact-or-default; fallback hierarchy is v1.
- History is append-only. A promotion writes the new row **and** a new promoted lockfile; rollback restarts on the
  previous active lockfile (or the baseline) and marks the row `retired` (§10.4).

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
units; and **a model's lockfile changes only through a judged transition of that model**. Hash pins give
reproducibility and exact rollback; isolation comes from the transition rule. (Rationale and evidence:
internal design reviews.)

**Units.** A unit is one small package: a component (mixer, op, router, norm, block, policy …), its kernels, and a
separate `reference.py` holding its oracle.

| Field | Meaning |
|---|---|
| `src` | hash of the normalized source tree + manifest |
| `sem` | hash of `reference.py` + parameter schema — the unit's *semantic identity* |
| `iface@major` | interface version; minors are additive only |

- Units receive dependencies through constructor ports; they never import sibling units and never inherit across
  plugins (no `extends`, mixins or monkeypatching).
- **Defaults never change.** New behaviour is a new knob whose default reproduces the old behaviour, or a new unit.
  This keeps most shared changes in the cheap "output unchanged" class below.
- Interface majors: "breaker pays" — the change ships a codemod and updates all first-party dependants.

**One version per Plan.** A Plan contains exactly one version of each unit; one Plan runs per process. Two versions
coexist only across processes, via deployment records. There is no hash-namespaced side-by-side loading in v0/v1.

**Lockfile and deployment record.** In addition to §4.3, every lock carries:

```toml
plan_id = "sha256:…"   recipe = "sha256:…"              # canonical hash of the emitted ModelSpec term
[deploy]   commit = "9e1c…"   uv_lock = "sha256:…"
[units]    "std.mixers.gqa" = { version = "1.3.0", src = "sha256:3f9a…", sem = "sha256:77b2…" }
[snapshot] regression = "sha256:…"                      # bit hash of teacher-forced logits, regression Plan
```

A **deployment record** is `(commit, uv.lock, lockfile, variant unit srcs)` — the single definition used
throughout. The supervisor materializes a worktree + venv per record
(cached by uv); rollback = restart on the previous record.

**Propagation.** `plugserve impact <unit>` is a reverse index over all locks. `plugserve propagate` runs inside the
PR that changes shared code (synchronous in v0) and classifies every affected model:

| Change class | Proof | Effect on that model's lock | Human |
|---|---|---|---|
| Not in the model's closures, or both effective closures (production and regression Plan) unchanged | lock diff empty | nothing (early cutoff) | 0 |
| Refactor / no-numerics change | regression-Plan snapshot bit-identical **and** the change touches no `uncovered` site (§7.10); changed-line coverage of covered sites reported | rewritten in the PR | 0 (normal PR review) |
| Change touching an `uncovered` site (production-only, non-deterministic kernel) | bounded L0 on every ShapeKey + L2, enforced 100% changed-line coverage (bitwise proof is impossible for a non-deterministic kernel, so there is no zero-human path) | rewritten | 1 approval |
| Exact perf change (re-layout, same reduction order) | bitwise L0 on every ShapeKey of the Plan + snapshot bit-identical | rewritten | 0 |
| New kernel with different numerics | not a shared change — a per-model Registry binding | per §10.5 | per §10.5 |
| Output-changing bug fix (the sanctioned exception to "defaults never change") | L2 vs the HF reference passes **and** no metric regresses beyond δ (the L2 budget) vs the current snapshot; one `expected-delta` note per change | rewritten; snapshot re-recorded | 1 approval per change |
| New behaviour (additive knob/unit with old default) | `sem` changes, recipe term unchanged, snapshot bit-identical, and touches no `uncovered` site (else the uncovered row applies) | rewritten (new `src`/`sem`), auto-advance; the behaviour reaches a model only when its recipe opts in (a model-local change) | 0 |
| Interface major | contract suite; breaker ships the codemod | rewritten | yes |
| Toolchain / third-party bump (torch, Triton, FlashInfer, driver) | full sweep under the new fingerprint | new per-fingerprint locks; failures stay on the previous deployment record | approve failures |
| Urgent / security fix | same path, preempts the evaluation queue; old `src` yanked | supervisor refuses records containing a yanked hash | on failures |

A model that fails is **held back visibly**: a `plugins/legacy/<unit>_<model>` fork with `derived_from`, bound
only in that model's lock, listed in `registry.status()`. After 14 days the holdback **blocks new deployments and promotions** of that
model and raises an alert until a human fixes the model or explicitly accepts (pins) the holdback. It never
prevents restarting the last running deployment record.

New models get their first lock from `plugserve propagate --new-model` (L2 vs reference + snapshot recording,
human-approved). Locks for fingerprints with no CI runner (e.g. a user's own consumer GPU) are marked `stale` when propagation
advances other fingerprints. Before adopting a new commit, the host's supervisor runs a local
`plugserve propagate --fingerprint=self` (regression snapshot on its own GPU + L2 against the immutable reference
goldens) and auto-advances its own stale locks under the same change-class rules; only failures need a human.
Like holdbacks, `stale` blocks new deployments and promotions and alerts — it never prevents restarting the last
running deployment record.

**Promoted locks follow propagation.** After propagate advances a model's baseline lock, the supervisor rebases
each active promoted lock of that model: it re-applies the promoted `bind`/`set` delta onto the new baseline and
runs the same change-class check — snapshot unchanged → auto-advance; otherwise re-judge, or retire the variant
(falling back to the new baseline). Promoted variant code lives content-addressed in `registry/units/<src>/`; a
deployment record includes their srcs, so every record is reproducible.

**Agent variants are leaves.** A variant is bound in one model's lock only, is never a dependency of anything,
and expires after 7 days unless promoted. A submission is a Plan delta of exactly two operations keyed by existing
`site_key`s — `bind` and `set`. `bind` may also add an entry to a `multi`/`ordered_multi` slot already present in
the lock; an unknown *slot* or unknown `set` key is an error. Additions are agent-auto-promotable only for slots
that cannot change outputs by contract (drafter — verified outputs are unchanged; exporter; connector;
checkpoint_policy); additions to output-affecting slots (logits_processor, plan_transform, …) are proposals. In v0 agents do not edit std units; a good variant is
upstreamed by a human proposal, which then goes through propagation.

**Lineage and GC.** `variant_of`, `derived_from` and `implements` plus Registry history form the lineage graph.
GC roots: baseline locks, active promoted locks, the last 2 retired locks per model, open jobs, live holdbacks.
Git keeps everything else.

**Health metric and triggers.**

| Signal | Action |
|---|---|
| Exception rate f (share of shared-change × affected-model pairs that did not auto-advance) > 10% for a unit | its fan-out is too broad: split it, or demote it to copied code |
| > 2 holdbacks per unit, or any older than 30 days | introduce a unit store (`store/<src>/` on a per-Plan `sys.path`; still one version per process) |
| Full propagation sweep > 30 min wall time, or > 1 re-baseline/day | switch propagate to an async merge queue + nightly batch (scheduling change only; formats unchanged) |
| First model needing sandwich norm / embedding scale / similar glue | add a closed hook set to `std.blocks` (minor version) |

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
  linear layers, `output_gate=True`, `rotary_dim=64`, `rmsnorm offset=1.0`, MTP draft head). `plugserve drift`
  reports how copied recipes diverge (informational).
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
        def plan(self, batch: Batch) -> Meta         # host side
        def prepare(self, meta: Meta, arena: Arena)  # writes static device buffers (graph-safe)
    def finalize_weights(self, params, ctx) -> None  # optional, after load (§5.1)
    def forward(self, x, state: StateHandle, meta: Meta) -> Tensor   # prefill / decode / verify (num_draft>0)
    # oracle: slow pure-torch implementation in the unit's reference.py (sem hash, §4.6) — not a method the
    # candidate could edit; the judge loads it from the parent unit in its own checkout
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

Selection happens once, at plan time (§4.3). No startup autotune sweeps: the Registry holds tuned configs per
fingerprint; missing rows fall back to the default config deterministically.

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
2. **Graph-safe** for its declared `graph_policy` (no host syncs on the captured path; host-side "plan" steps must
   run in `Planner.prepare`, e.g. FlashInfer `plan()`).
3. **Layout-declarable**: its weight/KV/activation layouts can be expressed as `Layout`s.
4. **Packaging**: installable without pulling in an engine (no `vllm`/`sglang` wheel dependency); version
   conflicts with other selected kernels are detectable from manifests. Prefer vendoring Triton sources.
5. **License** compatible (Apache-2.0/MIT/BSD) with provenance recorded.
6. **Verifiable**: passes L0 against our reference within judge budgets on our target shapes.
7. **Pinned configuration**: no run-time autotuning. `@triton.autotune` is stripped or reduced to the config
   pinned in `KernelSpec.config` (from the Registry, else a deterministic default). JIT caches (FlashInfer, Triton)
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
| 1 Tunables | PCIe/CPU miss split, swap interval, draft k | per-model Registry, per fingerprint | by owning plugin's class |
| 2 Shape-specialized kernels | GEMV/MoE kernels with constexpr dims | kernel under `fast/kernels`, `implements` the generic op's `sem` **at that site** | bounded (L0 vs the oracle store's reference); agent may auto-promote |
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
  (4) all remaining memory to `Paged` pools, divided in proportion to bytes-per-token of each spec. Startup fails
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
    co_index: str | None = None     # share page ids with another spec (DSA indexer ↔ MLA KV)
    draft_slots: int = 0            # Slot: extra slots for verify intermediates
    retention: str | None = None    # retention policy id (sliding window freeing)

class StateKind(Protocol):          # e.g. paged_kv, ssm_checkpoint, ring, dsa_index (plugins)
    def units_needed(self, spec, seq: SeqView, new_tokens: int) -> int
    def reclaimable(self, spec, seq: SeqView) -> list[UnitRef]
    def hit_length(self, spec, idx: PrefixView, upto: int) -> int
    def publish(self, spec, seq: SeqView, upto: int) -> Payload | None
    def adopt(self, spec, seq: SeqView, payload: Payload) -> None
    def metadata(self, spec, batch: Batch) -> dict[str, Tensor]       # page tables, slot ids, positions
    # rollback: StateKinds are Rewindable (§7.5)
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
- Hit length is a fixed point over all specs (no `isinstance`):

```python
L = upto
while True:
    L2 = align_down(min(k.hit_length(s, idx, L) for s, k in specs), lcm_alignment)
    if L2 == L: break
    L = L2
```

- What to publish (checkpoint policies: every-N, turn/tool-call anchors) and what to evict are plugins. v0: hash
  chain over full KV pages (nano-vllm style), LRU eviction. Radix tree is v1 if lookup cost warrants it.
- Because the index is page-granular, recurrent checkpoints are taken at page boundaries: an anchor at token `a`
  makes the anchor PlanTransform cut the prefill chunk at `floor(a / page) · page` and publish there.
- Eviction-policy contract (v0 signatures, v1 use): `victims()` may return **demote** actions (move a payload to a
  lower tier and keep the index entry) as well as drops; payloads with an in-flight async store stay pinned
  (refcounted) until the store completes.

### 6.4 Async loading

- Request status `BLOCKED(future)`: a Seq waits for state to arrive (offload tier, PD transfer) and is skipped by
  the scheduler until resolved.
- Per-layer ready events: an op may wait on `ctx.ready(spec, layer)` so layer-wise loading overlaps compute.
- Both exist in v0 (unused by v0 plugins) — retrofitting them is what made vLLM's connector API ~45 methods wide.

### 6.5 TieredStore (std library, not core)

`plugserve-std-tiered`: `TieredStore[K]` over `Tier`s (vram/pinned/ram/ssd/remote), a `ResidencyMap`, `Mover`s and
`PlacementPolicy` plugins, with two resolver styles:

- `HostPlannedResolver` — misses known before the step (KV prefix/offload, KVMem-style window selection); runs in
  `Planner.prepare`; emits transfers + per-layer events.
- `DeviceResolver` — misses known only inside the step (experts, sparse-KV); fixed-shape, in-graph lookup
  `ids → slot | -1`; host-side `adapt(stats)` between replays (decayed-LFU, profile-seeded, LRU).

What to do with a miss is the consumer's decision: the MoE op computes `-1` experts on CPU via a graph host node
(FreeToken) or fetches them over PCIe (Strata's `pcie_frac` split). The MoE op declares `placement_invariant`;
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

Scheduling is phase-free (vLLM v1 idea): each Seq wants `len(tokens) + drafts − num_computed` tokens; a token
budget covers prefill chunks, decode and verify uniformly.

### 7.2 Batch

```python
@dataclass
class BatchPlan:                     # host, struct-of-arrays, mutable by scheduler + transforms
    step: int
    slots, start, n_query, n_logits: np.ndarray
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
- **Host-function nodes are allowed inside FULL graphs** (CPU expert compute, Strata/FreeToken style), with the
  contract that the plugin supplies a **native function pointer** (`cudaLaunchHostFunc`-compatible, no Python, no
  GIL) — a Python callback would deadlock against the waiting main thread. Spike before v1 expert offload.
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
  - `PlanTransform`: appends `DEVICE_DRAFT` rows and sets the `spec.q` ShapeKey axis.
- Rollback = core commit fan-out to all Rewindables.

### 7.7 Control plane and errors

- `abort(rid)` (frontend → loop; frees at next commit — safe under overlap), `pause/drain`, `resume`,
  `flush_prefix` (v0). `update_weights(name → tensor)` (in-place through converters into existing QTensor storage;
  bumps `weight_version`) is specified now because it fixes the namespace design, but implemented in v1 (RL weight
  sync, LoRA merges).
- Per-request error isolation: a plugin exception tied to one Seq (grammar, parser) → `FINISHED(error)` for that
  Seq. A step-level plugin exception or device fault → engine exits non-zero; the supervisor restarts it on the
  last-known-good Plan and the frontend resubmits in-flight requests from committed tokens.

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
never rely on them, because candidate code shares the engine process: the judge measures performance client-side
(§10.3) and the canary uses per-request TTFT/TPOT measured in the frontend process, which runs no agent code
(§10.4).

### 7.10 Determinism

Three acceptance modes are used throughout (plan and judge):

| Mode | Plan | Kernel props required | Meaning |
|---|---|---|---|
| **run-identical** | regression Plan | `deterministic` | byte-identical output run to run with the **same batch composition** (one Seq at a time, depth 0). Used for "goldens unchanged" / model isolation. Resolves for every model. |
| **det-identical** | deterministic Plan | `deterministic` + `batch_invariant` | byte-identical between two configurations that change batch shape, chunking or path (prefix on/off, depth, spec vs non-spec). Only claimed if this Plan resolves for the model. |
| **bounded** | production Plan | — | within judge L2 budgets |

- `deterministic` is verified, not assumed: spikes S1/S3/S4 and judge L1 check run-to-run byte identity, and pinned
  configs used by these Plans must not use split-K or atomic accumulation.
- The **regression Plan** is the production Plan's bindings run at batch 1, depth 0, wherever the production
  kernel is `deterministic`; only where it is not is another kernel substituted, and those sites are recorded as
  `uncovered` in the lock. Snapshots therefore exercise production code wherever possible, and changes to
  `uncovered` sites cannot pass as refactors (§4.6).
- The regression and deterministic Plans select only kernels with the required props and pinned configs, seed
  sampling per (Seq, position), freeze adaptive policies (placement `adapt()`, cost-model draft policies), and
  default to depth 0 (a test overrides depth when depth itself is the thing compared).
- Rule: a comparison is judged det-identical when the model's deterministic Plan resolves, otherwise bounded. This
  applies uniformly to state invariants, `policy` plugins and spec decoding.
- det-identical comparisons are only claimed where a deterministic kernel exists for **every** op on the path. The
  v0 deterministic kernel set (plan M0-H): batch-invariant Triton GEMM (persistent, fixed tiles — vendored from
  vLLM `batch_invariant`), attention with a fixed KV split and identical reduction order for decode, verify and
  prefill rows, pinned-config W8A16/W4A16 GEMM and fused MoE with fixed reduction order. v0 has no batch-invariant
  GDN kernel, so Qwen3.5 models have a regression Plan (run-identical) but no deterministic Plan: their
  comparisons are **bounded**.
- The determinism flag is visible only to the resolver; plugin code cannot read it, so a variant cannot behave
  differently under judging than in production.

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
| Expert cache / offload | expert TieredStore, DeviceResolver, placement policies, MoE op with CPU miss compute | stable Pools, host nodes in graphs, VMM resize (v1) |
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

## 10. Self-improvement (D5)

### 10.1 Actors

- **Agent** — any client using the engine: a Qwen3/Qwen3.5 agent served by this very engine, or an external
  coding agent. Untrusted. Its harness runs under the `ps-agent` OS identity (§10.6) and can write only
  `workspace/`.
- **Supervisor** — trusted; exposes the improve API; owns Registry, lockfiles, evidence, lifecycle, rollback.
- **Sandbox engine** — runs candidate Plans (possibly containing agent code) under the `ps-sandbox` identity.
- **Judge** — trusted, hash-pinned, separate process under the `ps-judge` identity; drives the sandbox engine as a
  client only.

### 10.2 Improve API (MCP + HTTP, served by the supervisor, localhost-only, token-authenticated)

| Tool | Purpose |
|---|---|
| `telemetry.query(window, fields)` | Event aggregates for the live Plan (diagnostic) |
| `plan.describe()` / `site.explain(site)` | Which plugin/kernel is bound where, and why (constraints, registry row, lockfile) |
| `plugins.list(slot)` / `plugin.source(name)` | Read manifests and sources |
| `traces.list()` | List **dev** traces (holdout is never listed or readable) |
| `tunables.get(site)` / `tune.submit(site, space?)` | Request a tunable search inside the declared space |
| `variant.fork(plugin, new_name)` | Create a variant package in `workspace/` (copy-on-fork) |
| `variant.test(name, tiers=[L0, L1, L1.5])` | Cheap correctness iteration on a spare GPU or CPU, run as `ps-sandbox`; no promotion |
| `variant.submit(name)` | Queue full judging (L0–L4); returns a job id |
| `job.status(id)` / `job.cancel(id)` / `evidence.get(id)` | Judge progress, raw numbers, verdict |
| `registry.status()` / `registry.rollback(scope, site)` | Inspect promotions; the agent may roll back, never force-promote |
| `proposal.open(title, diff, rationale)` | Human-review proposal for out-of-scope changes (§10.5) |

The token is issued to the agent harness, not to ordinary API clients, so an arbitrary OpenAI/Anthropic client of
the engine is not an "agent". File editing happens through the harness's own tools under `ps-agent` permissions.
Because the agent may be served by the engine under test, the supervisor (not the engine) serves this API, and
**rollback never depends on the agent** (§10.4).

### 10.3 Judge tiers and numerics classes

| Tier | Check | Pass rule |
|---|---|---|
| L0 op oracle | vs Op reference / `dequant_ref` on the Plan's real shapes + fuzzed edges | budget = f(dtype, reduction length), held in the judge |
| L0-dist | samplers, verifiers, logits processors: fixed logits, ~10⁶ draws vs the reference sampler (verifiers vs plain target sampling) | TV distance / χ² bound |
| L1 contract | slot conformance, ledger leaks/accounting, graph-capture safety, declared props verified (batch-invariant, deterministic, depth tolerance) | pass/fail |
| L1.5 state invariants | chunked == one-shot prefill; prefix-hit resume == recompute; verify+commit(k) == k decode steps; graph replay == eager | det-identical where a deterministic kernel path exists (§7.10), else bounded |
| L2 model golden | teacher-forced logits vs reference recording on frozen prompts | calibrated noise floor = HF in the model's activation dtype (bf16; fp16 for fp16-activation checkpoints such as GPTQ) vs HF fp32 on the same prompts. Pass: top-1 agreement ≥ floor − δ; KL p50/p99 ≤ floor + δ; ΔPPL ≤ floor + δ. Model-isolation checks are run-identical (§7.10) |
| L4 perf | judge-side client measurements (TTFT, TPOT, throughput, peak memory) replaying a trace portfolio, production Plan, interleaved ABAB, N ≥ 5, bootstrap CI. Output check by **teacher-forced re-scoring**: a sample of the candidate's outputs (with its returned logits) is re-run through the reference Plan with teacher forcing (production arms legitimately diverge, so stream checksums are not used) | per trace: CI lower bound of perf ratio ≥ 1 − ε; target metric gain CI excludes 0; at re-scored positions KL and top-1 agreement within L2 budgets; every greedy token equals the argmax of the candidate's own logits; response counts/lengths match the trace requests |
| L4-confirm | one fresh run on **holdout** traces after a dev-trace pass | same rule; avoids selecting on noise across repeated submissions |
| L3 quality (v2) | held-out evals, rotated holdouts | sequential test excludes regression threshold |

- Numerics classes: `policy` (must not change math → det-identical against the parent when the model's
  deterministic Plan resolves, else bounded — §7.10), `exact` (numeric plugin
  claiming det-identical results to its reference path, e.g. a pure re-layout), `bounded` (judge budgets), `approx`
  (human gate). The plugin declares its class; **the judge chooses tests and budgets** — plugin-shipped tests are
  additional, never sufficient.
- A `policy` plugin whose consumer is not `placement_invariant` (e.g. MoE with CPU miss compute) is judged
  `bounded`.
- Holdout traces exclude any trace generated by the agent (poisoning). Agent-session traces may enter the dev set.
- **Oracle provenance.** The judge owns a content-addressed **oracle store** (`sem → reference.py` + param
  schema), filled only from reviewed commits. A numeric unit's oracle is resolved by the `sem` hash named in
  `implements`, from that store — not from the candidate and not from HEAD (so in-flight variants keep their
  oracle when the unit gains a knob). A candidate whose `reference.py` differs from its `implements` is rejected.
- **Reference goldens vs engine snapshots.** Reference goldens are recorded from HF, immutable and human-owned.
  Engine snapshots (`[snapshot]` in the lock) are run-identical hashes of the current Plan, updated by
  propagation; every re-recorded snapshot must still pass L2 against the reference (ratchet — no drift).
- **Closure-diff classification** of an **agent submission** (before anything runs), comparing parent and
  candidate Plans **site by site**:
  - recipe hash changed, or at any site the candidate's `implements` `sem` differs from the parent binding's `sem`
    at that same site → model change (human);
  - any `model` slot binding, `build`/`weights` override, or composite spanning more than one site → model change
    (human), even if all hashes match;
  - `uv.lock` or fingerprint changed → toolchain change (human);
  - otherwise only impl, policy or tunable bindings changed → §10.5 table.
  Every other model's lock must be byte-unchanged. (Shared-code changes by humans follow §4.6 instead, where an
  additive knob may change `sem` and still auto-advance.)

### 10.4 Episode lifecycle

1. Agent reads telemetry and sources; tunes declared tunables first (`tune.submit`).
2. If tuning plateaus: `variant.fork` (sets `variant_of` and `implements`) → edit in `workspace/` →
   `variant.test` (cheap, repeatable) → `variant.submit`. A submission is a `bind`/`set` Plan delta on existing
   site keys of one model (§4.6).
3. The supervisor schedules evaluation, choosing the cheapest valid mode:
   - **Spare GPU (default on the 8-GPU dev box)**: sandbox engine on another GPU of the same SKU; the live engine
     keeps serving.
   - **In-process A/B (single GPU, host-side only)**: for `graph_policy="host"` policy plugins and tunables marked
     `apply="live"`. **Precondition:** L0–L1.5 have passed in the sandbox (CPU or tiny-shape GPU runs) before the
     variant is ever loaded into the live engine. The live engine (identity `ps-engine`, §10.6) then loads both
     bindings and alternates them between benchmark runs (same weights, no restart). The judge drives dedicated
     replay traffic and measures client-side; other requests, including the agent's own, are held in the frontend
     queue during measured runs so they neither perturb nor contaminate the arms. After every in-process A/B the
     live engine **restarts from its current lockfile** (one restart per evaluation), so a rejected variant cannot
     leave modified state behind. Holdout traces are never sent to the live engine: L4-confirm runs in the
     sandbox, except for pure tunables (no agent code) on a live lockfile that contains no agent-authored plugins,
     which may confirm in-process.
   - **Drain window (single GPU, everything else)**: drain the live engine (frontend holds new requests with
     keepalive), run sandbox + judge on the full GPU, restart the live engine. Windows are rate-limited and capped
     in duration; ABAB requires ~2N restarts, so this mode is reserved for kernel variants and is a v0 stretch goal.
4. Verdict pass → new Registry row (scope = model) + new lockfile + evidence record → live engine restarts on the
   new lockfile → **canary**: golden smoke test, then per-request TTFT/TPOT measured **in the frontend process**
   (no agent code), bucketed by request shape (prompt length, output length, spec on/off), compared against the
   evidence for the same buckets during a probation period. Engine `forward_us`/`sample_us` are diagnostic only.
5. Canary failure, crash loop or health-check failure → the supervisor restarts on the **last-known-good
   lockfile**, marks the row `retired`, quarantines the variant for this fingerprint. No agent action needed.
6. One promotion at a time per model; the next candidate's baseline includes the previous promotion.

### 10.5 Scope of autonomy

| Change | Agent may auto-promote? |
|---|---|
| Tunable values within declared ranges | yes |
| New variant of a `policy` plugin (scheduler, cache, placement, draft policy) | yes — det-identical vs parent when the model's deterministic Plan resolves and the consumer is placement-invariant, else bounded |
| New kernel variant for an existing Op at a site, incl. model-specific `fast/` kernels whose `implements` matches that site's `sem` (§5.7 level 2) | yes — bounded |
| Multi-site composites (§5.7 level 3), model-owned `build`/`weights` (level 5), `model` slot bindings | proposal (human) |
| New plugin added to an existing engine-side slot | yes, only for drafter / exporter / connector / checkpoint_policy (§4.6), if `numerics ≠ approx`, slot ABI unchanged, no new dependency (frontend-side slots: see below) |
| New plugin added to an output-affecting slot (logits_processor, plan_transform, …) | proposal (human) |
| Model plugin changes that alter goldens; new tap points | proposal (human) |
| Core, judge, std ABI, `approx` numerics, new third-party dependencies | proposal (human) |
| Frontend-side plugins (frontend/API dialects, tokenizer/template bundles, reasoning/tool parsers) | proposal (human) — the frontend must stay free of agent code (canary trust, §10.6) |

Scope is decided mechanically by the closure diff (§10.3), not by what the variant claims. In v0 agents do not
edit std units directly; upstreaming a variant into std is a human proposal followed by propagation (§4.6).

Limits (v0): per-variant diff-size budget, promotions per model per day capped, evaluation windows per day capped.

### 10.6 Trust enforcement (OS level, v0)

Static rules (import-linter, no core-internal imports) are hygiene, not security: agent code in a Python process
can monkeypatch anything in that process. Enforcement therefore relies on process and OS boundaries:

| Identity | Can write | Can read | Network |
|---|---|---|---|
| `ps-agent` (agent harness) | `workspace/` only | repo (read-only), dev traces, evidence summaries | the frontend's and the improve API's localhost ports only |
| `ps-sandbox` (sandbox engine) | a per-run scratch dir (deleted after the run, never readable by `ps-agent`) | repo + `workspace/` variant + `registry/units/` (read-only), model weights, pre-built JIT caches | none (separate network namespace; IPC socket to the judge only) |
| `ps-engine` (live engine) | a scratch dir only | repo + `registry/units/` promoted variants (read-only), model weights, pre-built JIT caches; **not** holdout, judge or registry write paths | IPC to the frontend |
| `ps-frontend` (frontend; runs only reviewed, baseline-lockfile frontend plugins — never agent code) | none | tokenizer/template bundle (read-only) | localhost serving ports; IPC to the engine |
| `ps-judge` | evidence store | judge package, goldens, dev + holdout traces | IPC to the sandbox; the frontend's localhost ports for in-process A/B |
| `ps-supervisor` | registry (incl. `registry/units/`), lockfiles, evidence, deployment-record worktrees/venvs | everything | localhost |

Deployment-record worktrees and venvs are written only by `ps-supervisor` and are read-only to `ps-engine`,
`ps-sandbox` and `ps-judge`.

- Yama `ptrace_scope ≥ 2`, so no identity can attach to or read another process's memory (protects the
  frontend's canary measurements from engine-side variant code).
- `judge/`, `registry/`, `plans/` and the holdout store are not writable by `ps-agent`, `ps-sandbox` or `ps-engine`; the
  holdout store is readable only by `ps-judge`. The sandbox sees holdout prompts only as requests during a run;
  it has no network and no persistent writable storage, so it cannot exfiltrate or memorise them across runs.
- Verdict perf numbers are measured by the judge as a client; in-engine timers are diagnostic only.
- M0.6 acceptance includes OS-level tests that each forbidden write/read/network action fails.

---

## 11. Repository layout

```
plug-serve/
  core/            plugserve/{api,core}/          # ≤ 6k LOC
  std/             ops/ blocks/ ffn/(swiglu, moe_topk) mixers_gqa/ state_paged_kv/ tiered/ sampling/
  plugins/         models/<m>/{recipe.py, fast/, bundle/}  for qwen3, qwen3_moe, qwen3_5
                   mixers/{gdn}/  state/{ssm_checkpoint}/  legacy/   # time-boxed holdback forks (§4.6)
                   kernels/{flashinfer,fla_gdn,triton_moe,triton_wxa16,det}/  dialects/{gptq,fp8_block}/
                   frontends/{openai,anthropic}/  policies/  drafters/  verifiers/  grammar/
  plans/           <model_id>/<fingerprint>.lock  # per-model Plan lockfiles (model isolation, rollback)
  judge/           plugserve_judge/  goldens/  snapshots/  traces/dev/   # holdout stored outside the repo (ps-judge only)
  supervisor/      plugserve_supervisor/
  bench/           trace replay, baselines (separate pinned venvs)
  workspace/       (gitignored) agent variants
  registry/        registry.sqlite  evidence/  plans/(promoted locks)  units/<src>/(promoted variant code)
  docs/
```

---

## 12. Risks

| Risk | Mitigation / deciding experiment |
|---|---|
| Python per-step overhead at large batch | overlap (depth 1) hides CPU work shorter than the GPU step; M0 measures CPU time per step at B=256 vs nano-vllm; if worse, move per-token input prep to GPU kernels (vLLM Model Runner V2 style) |
| Control-plane CPU still a bottleneck after GPU-side input prep (CPU step > ~50% of GPU step at target batch) | replace the profiled hot structure (PrefixIndex, allocator, per-request scheduling) with a Rust/PyO3 module behind its existing interface (§3); plugin API stays Python |
| Frontend bottleneck (tokenization/streaming under many connections) or multi-instance routing needed | native (Rust) frontend/router over the language-neutral IPC schema (§7.8), following vLLM/SGLang |
| Shared-change sweep cost / re-baseline churn | early cutoff + bit-identical auto-advance (§4.6); async propagation when the sweep exceeds 30 min or re-baselines exceed 1/day |
| Holdback accumulation (models stuck on old units) | 14-day expiry; exception-rate metric f per unit; split or demote units with f > 10% (§4.6) |
| Abstractions too generic → slow | composites + direct bound callables; FULL graphs on decode; perf gate vs mini-sglang |
| Wrong core abstraction discovered late | each v0 milestone is chosen to stress one seam (plan §2); seam-breaking discoveries are fixed while v0 is pre-1.0 |
| Borrowed kernels violate contract | adoption rule §5.6; write our own Triton fallback |
| Agent reward hacking / self-harm | §10.5 scope, §10.6 OS-level identities, judge-side measurement, holdout confirm, supervisor-owned rollback |
| Single-GPU evaluation windows disrupt serving | spare-GPU default on dev box; in-process A/B for host-side policies/live tunables; drain windows rate-limited and capped (stretch); KV offload (v1) to cheapen restarts |
| sm_86 dev GPUs lack FP8/FP4 tensor cores | v0 quant on sm_86 uses weight-only (W4A16/W8A16) kernels; native FP8/NVFP4 kernels are added and verified when sm_89+/sm_120 hardware is available |
| Determinism claims unachievable with borrowed kernels | §7.10: det-identical only where a deterministic kernel path exists; deterministic kernel set is an M0 work item; otherwise bounded |
| Kernel breakage on sm_86 / torch 2.11 / Triton 3.6 | M0 spikes: uint8-decoded FP8 W8A16, vendored vLLM GDN ops, FlashInfer JIT pre-build |
| Golden recording does not fit one GPU | record on CPU fp32 (251 GB RAM) or across GPUs; quantized models via `dequant_ref` into the dense HF class |
