# Holon — Design

Status: draft for review · Date: 2026-10-06

---

## 0. Summary

Holon is a microkernel LLM inference engine. A small core (~5k LOC Python, hard cap 6k) fixes only the
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
| D5 | Self-improvement is an **optional capability** for capable agents and explicit user intent (e.g. a strong agent served by Holon). The engine itself ships best-known, human-tuned defaults and never depends on it. When used, auto-promotion is limited to policy/kernel/tunable scope under the judge; core, judge, ABI and `approx` numerics need a human (§10.5). |
| D6 | References (mini-sglang, nano-vllm, vLLM Model Runner V2, SGLang, FreeToken, Strata) are studied, not copied: control-plane code is our own plugin-native implementation. Only kernels may be borrowed, under §5.6. |
| D7 | Hardware: development on RTX A5000s (sm_86). Consumer performance evidence on consumer desktop machines — RTX 5080 16 GB and RTX 5090 32 GB (sm_120), RTX 4090 (sm_89). Optional Blackwell workstation (RTX PRO 6000 96 GB, sm_120) for sm_120 kernel work and large-memory tasks. Datacenter path (H100, GB200) later. Shared test machines are used through one persistent container per machine: builds, tests and benchmarks run inside it repeatedly rather than as many short batch jobs. |
| D8 | Expert caching/offload arrives in v1, but its core seams (device SeqTable, MemOps, `quiesce`, stream-memop CPU handshake) are built and spiked in M0 so it lands without core changes. |
| D9 | Quantization priority follows the deployment target. Consumer/edge GPUs: llama.cpp GGUF sub-4-bit formats (k-quants, i-quants, e.g. `unsloth/Qwen3.8-27B-GGUF`) — v0. HF-ecosystem GPTQ/AWQ INT4 and FP8 — v1. NVFP4/MXFP4, which mainly serve large-memory datacenter GPUs — v2. |

---

## 1. Invariants (lint/CI-enforced)

1. The core contains no model name, layer-type name, quant-format name, or feature-specific field.
   (CI: denylist grep over `holon/core` — `mamba|gdn|mla|dsa|nvfp4|eagle|qwen|deepseek|lora|…`.)
2. Plugins import only `holon.api`, their own package, and third-party libraries declared in their manifest
   `[dependencies]` (e.g. `triton`, `flashinfer`). Other units (incl. std) are referenced with `Ref` and injected,
   never imported (§4.6). Model packages never
   import other model packages. (CI: import-linter contracts + manifest dependency check.)
3. Plugins never import `holon.core._*` or `holon_judge`. (CI: import-linter.)
4. No module-level mutable state in `holon.*`; all state hangs off an explicit `Engine`/`ctx`. (lint)
5. No environment-variable configuration. One typed deployment file feeds the resolver. (lint: no `os.environ` in core/std)
6. No `torch.cuda.*` outside `holon.core.platform` and kernel plugins. (lint)
7. Plugin load failure, unresolved requirement, or duplicate provider for a `single` slot is a startup error —
   never a logged warning.
8. Core LOC ≤ 6000 (CI). Growing past it requires a design note.
9. After v0, every milestone PR has an empty diff under `core/` unless it is an explicitly reviewed core ABI change.
10. No `model_type`/architecture branches in std or any shared unit; model-specific code lives in the model's own
    package (`fast/`, §5.7). (lint)
11. One version of each unit per Plan, one Plan per process. Lockfiles and snapshots are written only by
    `holon propagate` (incl. `--new-model`), `holon tune` and the supervisor (§4.4, §4.6); CI's `isolation-check` re-runs propagate on a trusted GPU
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
| `holon` (`core/`) | core | `holon.api` (protocols, dataclasses — the plugin ABI) + `holon.core` (runtime). ≤ 6k LOC. |
| `holonlm-std-*` (`std/`) | reviewed | Shared units: `std.ops` (linear, norm, rope, act), `std.blocks` (residual wiring), `std.mixers.gqa`, `std.ffn`, `std.state.paged_kv`, `std.tiered` (TieredStore library), `std.sampling`. One live version each; recipes reference them via `Ref("name@major")`; locks pin `src` + `sem`; changes reach models only through propagation (§4.6). |
| `plugins/*` | reviewed or agent | First-party plugins: models, kernels, dialects, policies, frontends. Each is its own dist package with entry points. |
| `holonlm-judge` (`judge/`) | trusted, read-only to agents | Oracles harness, tolerance budgets, goldens, traces, perf timer. |
| `holonlm-tools` (`tools/`) | trusted | Lock-history commands: `holon impact`, `propagate`, `tune` (§4.4, §4.6); drives engines and the judge. |
| `holon-supervisor` (`supervisor/`) | trusted | Lifecycle, crash attribution, rollback; with the optional layer (§10): promotions, evidence, improve API (MCP + HTTP). |
| `workspace/` | agent (optional layer) | Git worktree where agent variants are written (gitignored). A promoted variant is committed by the supervisor to its own branch together with the new lock (§4.4). |

---

## 3. Core vocabulary

Sixteen nouns. Each is a type in `holon.api` (the plugin ABI) with its runtime in `holon.core`.

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

`plugin.toml` at the package root; registered via entry point group `holon.plugins` (value = path to manifest).

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
slot_abi  = "1"

[resources]
provides = ["placement.expert@1"]
requires = ["store.tiered@1", "telemetry.router_hist@1"]   # matching major = compatible

[properties]
numerics    = "policy"     # policy | exact | bounded | approx   (§10.3)
graph_policy = "host"      # full | piecewise | eager | host   (§7.4); also after/before, shape_axes

[dependencies]             # third-party imports allowed for this plugin (invariant 2)
python = []

# [reference]              # numeric units only (omitted for policy plugins like this one):
# module = "reference.py"  # the oracle module; sem = hash(reference.py + param schema)

[tunables]
decay     = { type = "float", range = [0.5, 0.999], default = 0.7, apply = "live" }     # live | restart
swap_cap  = { type = "int",   range = [0, 512],     default = 96, log = true, apply = "restart" }
# A tunable is judged under the numerics class of the plugin that owns it.

[support]
platforms = ["cuda"]                  # device names only; SM limits live in KernelSpec.requires
```

Resource names are **typed**: each `ns.name@major` is registered in `holon.api.resources` and bound to a
Protocol. The resolver matches strings without importing plugins; after import it checks each provider object
against the Protocol (`runtime_checkable` + signature check).

### 4.2 Slots and cardinality

| Cardinality | Meaning | Slots |
|---|---|---|
| `single` | exactly one per Plan | model, platform, launcher, schedule_policy, sampler, verifier, draft_policy, eviction_policy (prefix index) |
| `per_site` | bound per site (layer, op instance, state spec, store) | unit (mixers, ffn/MoE, norms, blocks, ops — anything a recipe names by `Ref`), kernel, state_kind, placement_policy, codec, weight_source |
| `ordered_multi` | ordered list, `after`/`before` in the manifest | plan_transform, logits_processor (v0: at most one, selected via `EngineConfig.slots`), checkpoint_policy |
| `multi` | any number, keyed by name | frontend, dialect, converter, tier, mover, drafter, encoder, connector, exporter |

A `Ref("a.b.c@1")` is served by the installed plugin whose name is the longest prefix of the unit name (e.g.
`std.ops` serves `std.ops.rmsnorm`), and whose version major equals the Ref's major. Everything the resolver needs
(resources, numerics, graph policy, depth, ordering, shape axes) lives in the manifest; runtime classes never repeat
it.

Every slot has an ABI version and a judge-owned contract suite (§10.3).

### 4.3 Resolution → Plan

1. Discover manifests (entry points + `plugins/` + `workspace/` if the supervisor enables it).
2. Read the deployment file (model path, pins, budgets).
3. The model plugin's `ModelSpec` enumerates **sites** (each mixer, op, state spec, draft head).
4. For each site: filter candidates by resources, platform/fingerprint constraints, `supports(shape)`, layout
   reachability (§5.5); then pick by deployment pin → the model's lock for this fingerprint class → default priority.
5. Check `ShapeKey` axes are finite; compute graph policy per ShapeKey class (weakest wins).
6. Emit the **Plan**: model_id, fingerprint class, recipe hash, commit, uv.lock hash, per-unit `src`/`sem` (one
   version each) and per-site bindings with pinned configs and tunables; `plan_id = canonical_hash(plan)`. The
   **Lock** file stores the Plan plus its verification results (`snapshot`, `uncovered` sites), which may be
   re-recorded without changing `plan_id` (§4.6). Whether a lock was borrowed from the nearest class (`unverified`)
   is decided when the engine starts, not stored.
7. Import only the selected entries; each plugin registers its kernels/converters/kinds into explicit registries
   passed to `Plugin.register(regs)` (no module-level registries, invariant 4); construct; bind callables. The hot
   path never does a registry lookup.

The Plan is the unit of deployment, A/B, promotion and rollback. It is serializable so every rank of a future
SPMD deployment can load the identical Plan (§8).

**Model isolation via lockfiles.** Each served model has a lock history per fingerprint class
(`plans/<model_id>/<class>/<n>.lock`, §4.4). The engine starts from the newest lock — never from a fresh
resolve. A fresh
resolve that differs from the lockfile (e.g. because a new, higher-priority kernel was installed for another model)
is reported as a diff and only adopted through `holon propagate` (§4.6). Adding a plugin
for model B can therefore never change model A's binding silently.

### 4.4 Registry = lock history (git-backed)

There is no separate database. The **Registry** is the git history of lockfiles plus evidence records:

- **One linear history per (model, fingerprint class):** `plans/<model_id>/<class>/<n>.lock`. The newest lock is
  what the engine runs. Rollback = restart on the previous lock in the history. With the optional layer (§10), a host's history is
  main's locks plus host-local promotion commits on top; when the host pulls a new main, the supervisor re-runs
  propagation from its own newest lock (so promotions are carried forward or dropped by the same checked
  transition), or a promotion is upstreamed to main by PR.
- **Tuned defaults live in the lock.** Kernel configs and tunables are tuned by us per fingerprint class
  (benchmarking tools such as `holon tune`, in the spirit of FreeToken's `ft bench bw`), so every class ships
  at the best configuration we know — no agent is involved.
- **Fingerprint class** = GPU SKU (a slug of the product name, e.g. `rtx-5090`, which fixes SM and VRAM) +
  toolchain majors (CUDA, torch; the torch pin implies Triton), e.g. `rtx-5090/cu130-torch2.11`. GPU UUID, driver
  build and measured bandwidths are recorded in evidence only. **Lock classes during development:** `rtx-5080`
  (sm_120) and `rtx-4090` (sm_89), re-propagated on every shared change. **Release classes:** `rtx-5090`,
  `rtx-3090`, tuned once per release. `rtx-a5000` (the local development host, uncommon among users) has no locks.
  **Kernel correctness** (L0, memcheck) runs on every kernel change on sm_86 (the A5000), sm_89 and sm_120a; a card of a covered SM with no lock
  borrows the nearest one (below). Evidence is produced on these cards (plan §1).
- **Missing or outdated lock for a class** (an outdated lock pins unit `src`s that the deployed commit does not
  contain) → the nearest class's current lock (same SM preferred) is used, flagged `unverified`, with an alert. The engine **never refuses to serve** for lack of a lock; `holon tune` on the host produces a
  verified lock for its class.
- Evidence records (command, fingerprint, plan_id, raw numbers) point at lock hashes and are written after the
  lock commit.

**Implemented (M0).** The core reads locks (`holon.core.plans`: `read`/`loads`, `history`/`newest`,
`nearest_class(plans, model_id, fingerprint, current_srcs)` → `(lock, unverified)` — own class, then same GPU, same
SM, same toolchain; with `current_srcs`, locks pinning other code are skipped — and `lock_overrides(lock)` →
EngineConfig overrides: the engine settings and a kernel pin per op, as per-site pins are not expressible yet). The
tools compute and write them (`holon_tools.locks`): `plan_of(engine)` reads the Plan off a built engine — `model_id`
from the weight source (`<hub id or directory name>@<16 hex of config + weight file hashes>`), the class from
`platform.fingerprint`, `recipe = canonical_hash(spec)`, `commit`, `uv_lock`, a `[units]` pin for the core (unit
`holon`) and every plugin whose code runs (the units built, the kernels bound, model, weight source, sampler, state
kinds …; not kernel providers loaded only as candidates), one `[sites]` binding per kernel site (`<site>/<op>` →
provider, facts, props, launch config) and `[engine]`, the EngineConfig settings that shape outputs (memory budgets
are per host and not recorded). `src` is sha256 over a unit directory's sorted relative paths and contents (no hidden
files, caches or build products), so a checkout and an installed copy agree; `sem` is sha256 of the manifest's
reference module. Lock files are canonical TOML (`[deploy]`, `[engine]`, `[snapshot]`, then `[units]`/`[sites]` one
sorted line per entry, so a changed binding is one changed line) carrying a `digest` of the whole content and the
`plan_id`, both re-checked on read, so a hand edit fails to load; `sm` is recorded for the same-SM preference. A
lock whose plan has uncovered sites and no run-identical trace records no snapshot. Snapshots are stored by content
next to the history (`<class>/snapshots/<hash>.json`). The snapshot trace is the judge's
(`holon_judge.snapshot.trace`: long context, a prefix hit, the decode batch shrinking through every graph bucket up
to 64, seeded n > 1, logprobs, prompt logprobs; each run records the step widths and prefix tokens saved). The
commands are `python -m holon_tools impact | propagate | tune` (`tools/README.md`). Until changed-line coverage is
recorded, every changed plan needs one approval with the byte-identical trace as evidence; auto-advance is an explicit
opt-in (`--allow-uncovered-advance`). Not yet: changed-line coverage, the engine starting from the lock (the
supervisor's job, with `unverified` pins as soft preferences), L2-regression-vs-current-lock for approvals, and the
`isolation-check` CI job.

### 4.5 Versioning

- `core_abi` and each `slot_abi` are independent semver. Changing the attention metadata ABI does not touch the
  placement-policy ABI.
- Units follow the one-version rule with additive-only interface minors, never-changing defaults and
  "breaker pays" majors; changes reach each model only through propagation (§4.6) — models do not individually
  adopt std versions.
- Core ABI minors keep one-minor compatibility shims (v1+). v0 is pre-1.0: ABI may break, but every break runs
  the full `holon check` over first-party plugins.

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

- Unit code imports only `holon.api`, its own package and declared third-party libraries; other units are
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

**Propagation.** `holon impact <unit>` lists the locks whose closures contain it. `holon propagate` runs
inside the PR that changes shared code; CI's `isolation-check` re-runs it on a trusted GPU runner and rejects any
lock or snapshot it cannot reproduce byte for byte. Each affected (model, fingerprint class) gets one of three
outcomes:

| Outcome | Condition | Effect | Human |
|---|---|---|---|
| **Auto-advance** | model's closure unaffected, or snapshot trace byte-identical **and** no `uncovered` site touched **and** every changed line inside the model's closure executed during its snapshot trace | lock rewritten in the PR | 0 |
| **Approve** | otherwise, L2 vs the HF reference passes and no L2 metric regresses beyond δ vs the current snapshot | lock rewritten, snapshot re-recorded | 1 approval for the whole change (all affected models listed) |
| **Fail** | L2 fails | the PR fixes it, or the model keeps the previous unit version through an explicit copy in its own package (visible in the diff, `derived_from` set) | — |

Toolchain bumps (torch, Triton, FlashInfer, CUDA majors) produce a new fingerprint class and run the same sweep.
New models get their first lock from `holon propagate --new-model` (L2 vs reference + snapshot, approved).
The development lock classes (`rtx-5080`, `rtx-4090`, §4.4) are refreshed by `isolation-check` runners on
consumer desktop machines — a scheduled (renewable) allocation per class that starts the pinned container image
and runs the queued checks inside it (D7 working method); the image digest is recorded in evidence as part of the toolchain —
on shared-change PRs where capacity allows, and nightly at least; a release commit requires
every development and release class refreshed. Until refreshed, an outdated lock is treated as missing (§4.4).

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
    def bundle(self, source) -> FrontendBundle       # tokenizer, chat template, parsers of this checkpoint

@dataclass
class ModelSpec:
    hidden: int; vocab: int
    layers: list[LayerDecl]          # LayerDecl(idx, mixer=Ref("gqa","1"), ffn=Ref("moe","1"), hparams)
    taps: list[str]                  # "embed", "layer.{i}.out", "pre_norm_last", "attn.{i}.q_prerope", ...
    shared: list[str]                # "embed", "lm_head" (for draft heads)
    draft_heads: list[DraftHeadDecl] # MTP layers, their own LayerDecls, taps consumed
    encoders: list[EncoderDecl]      # VLM towers (v1)
    position_fn: str | None          # e.g. "mrope"; default arange
    params: list[ParamSpec]          # shapes + sharding semantics (col/row/expert/replicated, fused groups);
                                     # a fused group (qkv, gate/up) is legal only if all members share one
                                     # source Layout — mixed-type GGUF files ("UD" builds) fall back to separate
                                     # GEMMs per site, and bindings may differ per layer
    block: Ref                       # residual wiring unit, e.g. Ref("std.blocks.prenorm@1")
    norm: Ref; lm_head: Ref          # final norm, output head
    weights: WeightMap               # checkpoint layout, e.g. hf_weights("qwen3_5", root="model.language_model")
```

- **Recipes.** A model is primarily a pure function `recipe(cfg) -> ModelSpec` in `recipe.py` that imports only
  `holon.api` (no torch). It composes units with `Ref("std.mixers.gqa@1", heads=…, …)`; `ModelSpec` gains
  `block` (e.g. `std.blocks.prenorm@1`) and `weights` (e.g. `hf_weights("qwen3_5", root="model.language_model")`),
  and `build`/`weights` default to them. The recipe's identity is the canonical hash of the **emitted term**, so
  refactoring `recipe.py` without changing the term changes nothing. Checkpoint layout lives only in `weights`.
- **Weight sources and model identity.** The signature is `recipe(cfg, source) -> ModelSpec`, where `source` is
  the detected weight source (`hf_safetensors`, `gguf`); the recipe selects the matching weight map
  (`hf_weights(...)` / `gguf_weights(...)`), so the bf16 term is unchanged when a GGUF map is added. `model_id`
  includes the checkpoint build (repo + file hash): every GGUF quant has its own lock history. A weight map carries
  **value transforms** as well as names — e.g. Qwen3.5/3.8 GGUF stores zero-centred norms as `w+1` (except
  `ssm_norm`), `ssm_a` as `-exp(A_log)`, conv1d squeezed to 2D, and GDN value heads in *tiled* rather than grouped
  order. The GDN mixer therefore has a `v_head_order = grouped | tiled` knob (q/k use `repeat` instead of
  `repeat_interleave` when tiled, and `out_proj`'s input activations are permuted rather than its packed weights),
  and norm offset / `A_log` handling follow the source. A second weight source can feed `draft_heads` (e.g. a
  separate MTP GGUF file).
- **Stacked parameters (M0.2 D7).** `WeightRule(pattern, target, stack="e")`: checkpoint tensors that differ only in
  the digits captured by `{e}` (HF's per-expert tensors) fill slice `e` of one `[E, …]` parameter, in place on the
  first slice. Out-of-range or twice-filled slices, a quantized tensor under a stack, a shape mismatch, a parameter
  filled both whole and by slices and missing slices are load errors. GGUF stores expert stacks whole and needs no
  rule; the unit sees the same `[E, …]` stack from either source.
- **GGUF is self-contained (M0.1 D3), as in llama.cpp.** The file alone gives the configuration (`<arch>.*`
  keys mapped to the HF keys recipes read), the tokenizer (BPE over `tokenizer.ggml.tokens/merges`; the
  pre-tokenizer chosen by `tokenizer.ggml.pre` from llama.cpp's table, e.g. `qwen2` with NFC; CONTROL tokens
  special, USER_DEFINED atomic), the chat template and the stop set (eos/eot plus llama.cpp's end-of-turn texts).
  No HF base model is needed; tests check tokenization, templates (agent transcripts with tools and reasoning) and
  the stop set against the HF checkpoint the file was made from. Fields a future architecture's GGUF lacks (e.g.
  `mrope_section`) come from a per-architecture table in its weight map, not from an HF download.
  `mmproj-*.gguf` is ignored when `text_only`; v1 VLM support loads it as the encoder's weight source.
- **Attention masks (M0.4 D4).** `std.mixers.gqa` takes `causal=False` (a row's new tokens see each other both ways:
  a block drafted at once) and `window=W` (sliding-window attention: a query at p sees no key at or before p − W,
  llama.cpp's convention); both reach every attention kernel it selects as facts, so a provider that cannot mask so
  is never chosen. `GQA.write_kv(x, ctx)` writes a step's keys and values without attending (a drafter's context
  from another model's features).
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
  `-exp(A_log)` precompute, KV-cache scales. Results live in the module's Pool-backed storage. After every
  module has finalized (merges included), the core calls `bind_kernels(ctx)` on each module that has it: units
  whose kernel depends on the loaded weights (a quantized layout, M0.1) select it there, so a unit merged away
  binds none.
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

M0.3's `std.mixers.gdn` is the first mixer with `Slot` state: a row's state is its pool's unit at the row's SeqTable
slot, padding rows use the pool's null unit (the padding slot's) and start from zero state; the planner computes
has-initial-state on the device from positions (a row at position 0 starts from zero, so a reused slot's stale state
is never read), builds the chunk kernel's indices from the host plan (else it syncs), and keys decode graphs by rows.

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
- **MoE (M0.2, `std.moe`):** `unit.std.moe@1` composes three ops, each with a reference: `moe_topk@1` (f32
  softmax, top-k, renormalised), `linear_experts@1` (a stack of E experts applied per (token, slot), format-general
  through a `layout` fact; a merged gate/up stack applies SwiGLU) and `moe_combine@1` (the weighted sum in f32, in
  slot order). The router is f32 and the expert path f32 between the stacks (M0.2 D3).

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
its fingerprint class (§4.4, produced by `holon tune`); without a lock for the class, the nearest class's
configs are used (`unverified`).

**Selection contract (implemented in E2).** An op is owned by the unit that defines it: it is declared as a versioned
resource (`op.attention.paged_decode@1`), its `reference.py` implementation is registered in `regs.ops` as the L0
oracle, and its instance API is documented next to it (kernel plugins cannot import std, so the contract is
duck-typed). Kernel plugins provide `kernel.<op>@<major>`. `BuildCtx.select_kernel(op, site=…, **facts)` takes
canonical facts (shapes, `q_dtype`/`kv_dtype`, layout, …): it imports only the providers whose manifest
dependencies are installed and whose platforms match the device (no import attempts), filters by
`requires.allows(device, sm)` and `supports(facts)`, takes the highest priority (a tie is an error), honours a
deployment pin (`kernels = {op: provider}`, later the lock), and records provider, facts and props per site for the
lock. It returns the `KernelSpec`; the caller — a mixer's planner — creates and owns its instances with
`spec.impl(ctx, **facts)`, because instances hold per-step plans (FlashInfer wrappers) and depth 1 needs two of
them. Layouts are facts the unit passes in (GQA: `kv.hnd`), so every kernel of one state group agrees by
construction; rows of length 0 (graph padding) must be harmless. The same mechanism serves the small per-token ops: std.ops owns
`rmsnorm@1`, `add_rmsnorm@1` (residual add fused into the norm) and `rope@1`, std.ffn owns `silu_mul@1`; units pick
their kernel once at build.

**References never serve (M0.1 D1).** A `reference.py` is the op's oracle only: it is never registered as a kernel and
production code never calls it. Selection sees production kernel plugins only; no match is a startup error naming
the op, facts and device (invariant 7) — there is no silent slow path. Tests get the references as CPU kernels from
the test-only `holonlm-testing` distribution (`testing.ref_kernels`, installed in the development environment, never
with Holon; an import contract keeps production code from importing it), which also serves `attention.causal@1`, the
whole-sequence attention of a model built without state. Projections that
read the same input (q/k/v, gate/up) are merged into one GEMM in `finalize_weights` when the linear units allow it
(`Linear.merge` rebinds each unit's weight as a view of the merged buffer, so recipes and weight maps stay as they
are).

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

@register_converter(src="gguf.q4_k.native", dst="gguf.q4_k.mmq_tiled", lossless=True, cost=1.0)
def to_mmq_tiles(q: QTensor, *, shard: ShardInfo) -> QTensor: ...   # GGUF kernels usually read gguf.native directly
```

- Repacking is owned by **converters**, registered by the plugin that defines the destination layout.
- v0: single-hop, `lossless=True` converters only. Multi-hop path search is v1.
- When two consumers need one weight in two layouts (GPU kernel + CPU expert compute), the resolver asks the
  ledger whether to store both; otherwise picks a layout readable by both.
- Post-repack artifacts are cached on disk keyed by `(source hash, layout, converter version, sm)` (v1).
- KV cache formats are Layouts too, so attention kernels and state kinds negotiate page size/dtype via layouts.

**As built (M0.1).** Weight sources are detected from manifest `[support] detect` patterns (`config.json`,
`*.gguf`), so no plugin is imported to decide. The model is built on the meta device; `load_weights` swaps each dense
parameter in place and hands each `QTensor` to the unit owning the parameter (`adopt(ctx, name, qtensor)`), which
stores it in its own form or refuses (a load error naming unit and layout); the core never looks inside a layout.
GGUF tensors keep their stored blocks (`Layout(Scheme("ggml.<type>"), "gguf.native")`): llama.cpp's kernels read
them directly, so v0 has no Dialect and no converter for GGUF. Units bind the kernel of the layout they adopted in
`bind_kernels`, after merges: same-type q/k/v and gate/up concatenate by rows (ggml rows are independent block
runs) and run as one quantized matmul; a v of another type stays apart. A tied quantized head reads the
embedding's blocks.

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

Adopted kernels, decided by the M0 spikes (evidence and raw numbers in `spikes/s*/README.md`):

- **Attention** (spike S1, FlashInfer 0.6.12; sm_86 and sm_120):
  - prefill: FlashInfer FA2 paged prefill;
  - non-speculative decode: FlashInfer tensor-core decode, host `plan()` in `plan_bounds` (exact at depth 1 —
    non-spec host lengths are exact). At overlap depth ≥ 1, two wrappers per head geometry are alternated
    because a wrapper reuses one pinned staging buffer;
  - speculative decode at depth 1 and quantized KV: **our own Triton paged attention** reading lengths from the
    SeqTable (median 0.95–0.98× FlashInfer's time over 216 decode/verify configurations; bf16 and q8_0 KV; fp8 and
    prefill rows still to add); XQA is a borrowed alternative on sm_120;
  - not adopted: FlashInfer CUDA-core decode (rejects GQA group 6, up to 3.4× slower at batch 1); FlashInfer's
    graph wrapper with `disable_split_kv=True` (reads unwritten indices in 0.6.12);
  - JIT modules are pre-built per class (≈ 30 s with `prebuild.py`; `ninja` on PATH).
- **GGUF** (spike S2): **port llama.cpp's current ggml-cuda quantized kernels** (`mmvq`, `mmq` incl. i-quants,
  `mul_mat_id` for MoE; MIT) behind a ≈ 200-line shim that supplies the ggml helpers and allocates from our
  workspaces (≈ 5 developer-days; same code as the baseline, so parity is direct). The SGLang-carried set was
  measured and not adopted: it decodes IQ3_S with an obsolete grid (4.3% of weights wrong) and is far behind
  llama.cpp for multi-row decode (0.2–0.8×) and batched/MoE prefill (0.05–0.45×). Types needed for the unsloth UD
  builds: Q8_0, Q3_K–Q6_K, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS.
  **Built in M0.1** (`kernels.ggml_cuda`): llama.cpp `51ce9c11` vendored with a mechanical `vendor.patch` (bf16
  activations in, bf16 or f32 out, no environment reads), 18 types, `mmvq` up to llama.cpp's row threshold and `mmq`
  in 512-row chunks above it, the q8_1 workspace declared to the ledger, no weight padding (`in` % 256 = 0; checked
  under compute-sanitizer). On the RTX 5080 Qwen3-8B Q4_K_M prefills at 1.04× and decodes at 0.98× llama.cpp
  (UD-IQ3_XXS: 1.06×, 0.97×).
  **Extended in M0.2** with llama.cpp's MoE kernels: `mul_mat_id` (mmvq with ids, its fused SwiGLU on the merged
  gate/up stack; mmq with ids, then our f32 SwiGLU), `topk-moe.cu` and `moe-weighted-reduction.cu`, patched to the
  contracts' layouts (ids at row stride k; the combine stores bf16). Qwen3-30B-A3B prefills at 1.02–1.11× and
  decodes at 0.97–1.02× llama.cpp (RTX 5080 UD-IQ3_XXS; Q4_K_M on 4090 and A5000). Dense bf16 expert stacks (an HF
  checkpoint) run on our own `kernels.triton_experts` grouped GEMM.
  Reference dequantization: llama.cpp's gguf-py.
- **GDN** (spike S3): vendored subset of vLLM's flash-linear-attention copy (commit `6dd813e0`, 14 files,
  ≈ 2.75k code lines, Apache-2.0 over MIT), rebuilt by `vendor.sh` + `vendor.patch`: `vllm.*` imports and env
  switches removed, identity-keyed `tensor_cache` removed (stale with reused static buffers), autotune replaced by
  one pinned config per kernel, and a `v_head_tiled` switch added (GGUF order without repacking, bit-identical).
  Contract details: slot 0 is reserved as null; verify uses `[N, k+1]` slot indices; prefill takes no slot
  indices (the mixer gathers initial and scatters final states); outputs and prefill scratch move to our buffers;
  conv1d and gated RMSNorm kernels still to add. Eager prefill pays ≈ 0.5 ms launch overhead per GDN layer, so
  GDN prefill steps are graph-captured per length bucket. pip `fla-core` is not used (NaN on sm_86).
- (v1, HF-ecosystem formats) Triton fused-MoE (vendored from vLLM/SGLang, Apache-2.0) with a W4A16 path; Triton
  W8A16/W4A16 GEMM written by us — on sm_86 Triton rejects `fp8e4nv`, so FP8 weights are loaded as `uint8` and
  decoded to bf16 with exact bit operations in the K-loop; Marlin only if it builds standalone.

Consumer cards (RTX 4090 / 5080 / 5090 desktops; "Consumer cards" in the spike READMEs) confirm these choices;
the RTX PRO 6000 (128 MB L2, 188 SMs) is not a consumer proxy:
- attention: our Triton paged attention stays at parity with FlashInfer on every class (median time ratio
  0.99–1.01×) with configs pinned per class; fp8 verify on sm_120 uses XQA; q8_0 KV still needs a tuning pass on
  sm_120 (1.1–1.7× FlashInfer fp8 time; at the memory roofline on the 4090);
- GGUF: dequantize + cuBLAS cannot stand in for quantized prefill on GeForce cards, whose int8 tensor throughput
  is ≈ 4× their bf16 (0.44–0.73× llama.cpp prefill), so llama.cpp's int8 `mmq` is needed from M0.1 on, not later;
  the ported kernels are correct and graph-safe for all 10 types on sm_89 and sm_120.

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
  `(max_seqs + 1) × (1 + draft_slots)` from the deployment file (the extra slot is the padding slot, whose unit is
  the pool's null unit); (3) plugin pools (`BuildCtx.pool`, M0.3) at their declared `pref_units`, or
  `min_units` if short (e.g. checkpoint pool: a byte budget from the deployment file, with its own eviction);
  (4) all remaining memory to `Paged` pools, divided in proportion to bytes-per-token of each spec; specs in one
  page group (`page_group`) get one shared unit count. Startup fails
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
    layout: str                     # dtype/arrangement, negotiated with kernels (format settled by spike S1)
    alignment: int = 1              # prefix-hit granularity only when intrinsic (e.g. compressed-KV ratio);
                                    # GDN uses 64, the chunk size: its kernels accept an initial state at any
                                    # offset, but resume is bitwise equal to recomputing only at chunk boundaries
    page_group: str | None = None   # specs sharing page ids, one page table and one unit count, allocated and
                                    # evicted jointly (DSA indexer ↔ MLA KV; drafter KV ↔ target KV)
# Slot(unit_bytes, draft_slots=0): draft_slots = extra slots for verify intermediates

class StateKind(Rewindable, Protocol):   # e.g. paged_kv, ssm_checkpoint (plugins); bound by the core to ONE spec
                                         # and its pool: factory(spec, pool, ctx) -> StateKind
    def units_needed(self, seq: SeqView, new_tokens: int) -> int
    def reclaimable(self, seq: SeqView) -> list[int]
    def hit_length(self, idx: PrefixView, upto: int) -> HitLen               # (length, tier cost)
    def publish(self, seq: SeqView, upto: int) -> tuple[Payload, list[MemOp]] | None
    def adopt(self, seq: SeqView, payload: Payload) -> list[MemOp]
    def fork(self, src: SeqView, dsts: list[SeqView]) -> list[MemOp]         # n>1, parallel tool branches
    # optional (M0.3), for state reused through payloads: cut(seq, start, n) -> n' (a prefill row may end where
    # the kind publishes), next_publish(seq) -> position | None, payload_pool (the pool the index may evict from)
    # Page tables and slot indices are filled into DeviceBatch by the core (it owns allocation); mixer planners
    # read them. publish/adopt return MemOps (§6.3). Callbacks are per request; the core vectorises them.
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
  checkpoint" ordered after the sample of the step that ends at the boundary, and the checkpoint is registered in
  the index at that step's commit only if committed tokens cover it; `adopt` emits "copy checkpoint → active slot"
  before the forward that uses it; source checkpoints stay pinned until the copy completes. Verify rows, and requests
  with an uncommitted one, publish nothing: their end is known only on the device (M0.4 §3.7).
- Because the index is page-granular, recurrent checkpoints are taken at page boundaries. For agent traffic the
  default policy checkpoints at the **end of each prompt** (chat templates may re-render earlier turns, e.g. strip
  reasoning, so anchors on sampled tool-call tokens are unreliable); the chunk is cut at `floor(end / page) · page`.
- Eviction-policy contract: `victims()` may return **demote** actions (move a payload to a lower tier, keep the
  index entry) as well as drops; payloads with an in-flight async store stay pinned until the store completes.
- **v0 (implemented in E4)**: paged state only (the index is off when any spec is not paged; checkpoints arrive in
  M0.3). Blocks are `lcm(tokens_per_unit, alignment)` tokens; keys chain from a namespace derived from the model's
  canonical hash; entries verify their tokens, so a hash collision is a miss. Pages are shared by refcount — the
  index holds one reference per published unit, an adopting request one more — so publish and adopt need no MemOps.
  Admission adopts the longest cached prefix, always leaving one token to compute; every completed block,
  generated tokens included, is published after the step that computed it (so the next turn of a conversation
  hits). Eviction drops least-recently-used leaves whose units only the index holds, and runs before any
  preemption; `could_fit` counts such units as available, which keeps the scheduler's liveness argument. `n > 1`
  siblings wait until the first sequence's prompt is published and then adopt it (each recomputes at most one
  partial block); `fork` with a partial-page copy replaces this when MemOps land.
- **M0.3**: `Slot` state joins through payloads. A kind that publishes asks the scheduler to `cut` prefill rows at
  its policy's positions (`std.state.recurrent`: the prompt's last aligned block, or every n tokens); when a row's
  step ends at `next_publish`, `publish` returns a payload unit of the kind's plugin pool and a copy MemOp run after
  the sample; the core registers the payload on the block's entry at the step's commit only if committed tokens
  cover it. Admission takes the fixed point above; the adopting request pins the payload until its copy MemOp is enqueued
  (before the forward of its first step). MemOps run eagerly around the step, never inside a graph, unit by unit
  with no temporary. A full payload pool evicts its own least-recently-used payloads, interior entries included,
  never paged units; an evicted payload only shortens the hits that would have resumed there (the kind's
  `hit_length` falls back to an earlier payload). `fork` is not built.

### 6.4 Async loading

- When a prefix hit lives below VRAM (offload tier, PD transfer), the **connector/tier plugin chooses** per request
  between (a) `BLOCKED(future)` — the Seq is skipped until all its state has arrived — and (b) layer-wise loading —
  the Seq runs while ops wait on per-layer ready events `ctx.ready(spec, layer)`.
- Load futures **reserve their destination pages** in the ledger when issued and are **cancellable**: abort and
  preemption cancel the future and release its pages.
- Both mechanisms exist in v0 (unused by v0 plugins) — retrofitting them is what made vLLM's connector API ~45
  methods wide.

### 6.5 TieredStore (std library, not core)

`holonlm-std-tiered`: `TieredStore[K]` over `Tier`s (vram/pinned/ram/ssd/remote), a `ResidencyMap`, `Mover`s and
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
def step(self):                                                         # Engine.step, depth 1 (as implemented)
    prev = self.inflight                                                # launched, not committed
    unblock()                                       # requests whose logits processor became ready may be scheduled
    if prev has a constrained row: commit(prev)     # its next mask needs that row's tokens (A′, M0.5 D2)
    plan = self.scheduler.schedule()                # CPU; skips constrained Seqs with a row in flight, and BLOCKED
    launch(plan)    # GPU (one graph for decode/verify): prepare → model → logits → mask (logits processor, rows
                    # filled on the host at launch) → Sampler|Verifier → store → rewind (device) → drafter
    commit(prev)                                    # CPU, while the step runs
    self.inflight = plan

def commit(step):                                                       # CPU after D2H event of the step
    processor.commit (may fail the request); finish/stop/stream; observe drafts; deferred frees; control requests
```

The design's forward/sample split (`launch_forward` → commit N−1 → host input → `launch_sample`, two graphs per
step) is not built: a constrained Seq (one whose logits a processor masks from its committed tokens) runs at depth 0
inside depth-1 steps instead, and the split is built only if M0.5 D2's measurement asks for it (its seam is
sketched there).

A step whose rows all stop mid-prompt (chunked prefill) samples nothing: it runs the trunk, which writes their state
(and prompt logprobs, when asked), and no head.

`inflight_depth` = the maximum number of uncommitted steps when the scheduler runs:

| depth | Behaviour | Use |
|---|---|---|
| 0 | commit before schedule (fully synchronous) | debugging, judge invariant tests, CPU drafters/plugins needing the last token on host |
| 1 | schedule step N while step N−1's sample is still on the GPU; N−1 is committed during step N's forward | default; constrained Seqs are gated to depth 0 (below) |
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

**Overlap invariant (implemented in E5).** All engine device work runs on one stream and the launch path never
waits for the device: per-step inputs (batch fields, page lengths and write indices, block-table rows) are staged
in two alternating pinned buffers and copied asynchronously into static device buffers; sampled tokens come back
the same way, with an event per step. The host mirror `Seq.pending` counts tokens sampled but not committed: decode
rows read their input from the device SeqTable (`last_token`), a request preempted while its decode is in flight
waits for that commit, and requests reaching a length limit are not scheduled again. Everything a commit releases
— finished requests' slots, pages and sampler rows, `_swap`ped prefix pages, preempted requests' pages — is reused
only by launches enqueued later on the same stream, so an in-flight step never sees it change. An extra row for a
request that stopped at the previous commit is discarded. Sampled-token logprobs come back like the tokens (pinned
host copies read at commit), so those steps run at depth 1, eagerly; steps that report prompt logprobs, which are
extracted on the host, commit at once.

**Host-token gate (M0.5 D2).** A logits processor builds a row's mask on the host from the Seq's committed tokens
(and, for a verify row, the drafts copied back with the step's results), so a constrained Seq is never scheduled
while it has a row in flight or while it is `BLOCKED` (its processor not ready). The engine commits the in-flight
step first whenever it holds a constrained row, so constrained Seqs decode every step and that step's other rows lose
their overlap (D2's A′, chosen by measurement; the variant that left the Seq out of the step was removed). A
manifest-declared per-plugin depth (`max_depth`) was removed unread: the gate comes from the processor's
`open`. Device-side `rewind` runs inside the step's graph, right after the Sampler/Verifier, so the next step's
device inputs (`DEVICE_LAST`/`DEVICE_DRAFT`) are already consistent.

### 7.4 CUDA graphs

- Graph units: `launch_forward` and `launch_sample`.
- Capture set = bucketed `n_tokens`/`n_seqs` × product of declared `ShapeKey` axes, filtered, capped by a memory
  budget. Unbounded axes are rejected at resolve.
- Each plugin declares `graph_policy` ∈ {`full`, `piecewise`, `eager`, `host`}; the resolver takes the weakest
  per ShapeKey class. `host` means the plugin never runs inside a graph — it runs on the CPU between replays
  (placement-policy `adapt()`, schedulers). v0 implements FULL (decode/verify) and EAGER (prefill); PIECEWISE is v1.
- **v0 (implemented in E6)**: decode-only steps without logprobs are padded to a batch-size bucket (1, 2, 4, 8, 16,
  24, 32, 48, 64, … up to `max_seqs`) and replayed from a graph keyed by (bucket, row width, the sampler's `variant` — its host-side branches —, each planner's `graph_key`). A planner whose
  kernel offers no `graph_key` keeps the step eager. The graph covers forward, sampling and the SeqTable write; inputs
  are staged into the static buffers and results copied out around the replay. Padding rows use a dedicated slot whose
  block-table row points at each paged pool's reserved, zeroed null unit, so they read and write nothing real. Every
  bucket × width × sampler variant (greedy, sampled, filtered) is captured at startup (with a logits processor
  selected, every step applies its masks — over padding rows only when it has no constrained rows — so one graph
  serves both), largest bucket first so smaller graphs reuse its blocks of the one
  shared pool; kernels keep their graph keys independent of per-step lengths (Triton's split count follows the batch
  and the table width), so serving never stops to capture. The memory profile subtracts the pool the largest bucket's
  graphs leave reserved, and the CUDA allocator uses expandable segments so a long mix of large prefill activations
  and the graph pool does not fragment the budget.
- **CPU work inside a FULL graph** (CPU expert compute, Strata/FreeToken style) uses a one-word handshake per
  layer with a persistent native CPU worker (spike S6): the graph signals REQ, the worker computes on pinned
  memory and answers DONE, the graph waits. `Platform` exposes this as a signal/wait pair with two
  implementations chosen per fingerprint class by measurement — a one-thread spin kernel (1.9–2.8 µs per
  handshake on the A5000 server and RTX 4090/5080/5090 desktops; the default) and stream memory operations
  (`cuStreamWriteValue`/`WaitValue`, 6–10 µs under load; for display GPUs with a kernel watchdog, laptops and
  power-sensitive hosts — a resident spinner draws 12–32 W). `cudaLaunchHostFunc` is not used (21–39 µs). The
  CPU worker pool takes one physical core fewer than the host has and the Python loop is pinned to the free core:
  sharing a core with a spinning worker slows the loop 1.4–2.3× on 8-core desktops. Expert inputs
  and outputs are read and written in pinned memory directly (no memcpy nodes, which cost ≈ 14 µs per layer). GPU
  work between signal and wait overlaps the CPU. Contract: stop the worker only after draining in-flight replays
  (or abort, drain, stop); residency-map swaps are written to the inactive buffer and flipped **on the replay
  stream between replays** — no `quiesce` needed; `quiesce` is for changes not ordered on the stream (host or
  worker state, side-stream VRAM refills, worker restarts). No Python or GIL on this path.
- All dynamic inputs go through static buffers requested at startup with `ctx.static(name, max_shape)`.

### 7.5 Rewindable

```python
class Rewindable(Protocol):
    def rewind(self, ctx: StepCtx, path: Tensor | None = None) -> None   # device, graph-capturable
```

After the verifier has written `SeqTable.n_accept` for a step's verify rows, the core calls `rewind(ctx)` on every
module of the target and every loaded slot object (sampler, verifier, drafter, policy) that defines it, before the
drafter runs. Paged state needs none — it rewinds by length. Recurrent state rolls back by **masked replay** (M0.4
D6): a verify row keeps the state it started from (its final state goes to a sink unit, its inputs to a replay
buffer), and `rewind` reruns the recurrence from it with the update masked to identity past the accepted tokens. Per-
token state copies (`checkpoint_slots`) were dropped: (k + 1) × 147 MiB per sequence for Qwen3.8-27B. An auxiliary
model a drafter runs (§7.6) is rolled back by that drafter, after its own pass. `path` stays `None` until tree
verification (v2).

### 7.6 Sampling and decoding

- **Logits processor** (M0.5, `docs/m05-grammar.md` §3): at most one, opt-in (`--slots logits_processor=…`, recorded
  in the lock's `[engine]`), built with `build(ctx, vocab: VocabView)` (padded vocabulary size, stop set, the weight
  source's tokenizer). It follows the requests it constrains on the host — `open(rid, params, prompt) -> bool` at
  `Engine.add` (a request whose `SamplingParams.ext` names a resource no active plugin provides is rejected),
  `ready(rid)` polled while the request is `BLOCKED` (compiling off the engine thread), `commit(seq, out)` with the
  tokens each step appends (before finishing or streaming; once after `ready` for a resumed request), `close(rid)` —
  writes the masks of a step's constrained rows at launch (`fill(rows: list[MaskRow], buf)`, never raising; a verify
  row's `MaskRow.drafts` are the drafts it checks, copied to the host with the step's results), and applies them on
  the device between the head and the Sampler/Verifier on every path (`apply(logits, mask, rows)`, in the graph).
  The core stages only the constrained rows (a compact `[rows, mask_width]` int32 block and their logits rows, padded
  with an out-of-range index to the graph's row count) through the Uploader and never reads mask bits; the format and
  kernel are the plugin's. Logprobs are of the masked distribution (frontends clamp −∞ to −9999). Its `open`,
  `ready` or `commit` raising `RequestError(kind, message)` ends the request (§7.7).
- The default sampler is a builtin plugin (`std.sampling`, implemented in E3): per-slot parameter tensors set on
  admission; temperature → top-k → top-p over the renormalized remainder → min-p; one counter-based uniform per row,
  a hash of (seed, position), picks the token by inverse CDF, so samples depend on neither batching nor preemption
  and the step needs no generator state (resolution ≈ 1e-7 absolute probability: 24-bit uniform, fp32 CDF). The
  engine gives every sequence a concrete seed (`n > 1` siblings derive
  theirs from the request's), and computes prompt logprobs from the trunk's hidden states in row chunks. Sampler, processors and verifiers declare whether they accept
  vocab-sharded logits or need gathered logits (TP seam, §8).
- **Speculative decoding** (M0.4, `docs/m04-spec.md`) = three roles, all plugins:
  - `DraftPolicy` (`draft_policy` slot): k **per step**, `k(seqs)` from a finite declared set (`ks()`: the widths
    captured as verify graphs); `observe(seq, k, n_draft, n_accept)` after each commit (Strata-style cost models
    later). The slot's plugin object is the policy, or its optional `build(ctx)` returns one configured from the
    deployment's options (`std.spec.fixed_k`: `spec.k`, `spec.fallback`).
  - `Drafter` (`drafter` slot; MTP, DFlash, GPU n-gram; EAGLE-3 and draft models later): built against a
    `TargetView` (the target's shared `embed`/`norm`/`lm_head`, its checkpoint, its model plugin's `draft_head`);
    `admit(slot, history)` at admission; `run(ctx, h)` after **every** step's sample stage with the trunk's output, in
    the step's graph: it writes its own state for the step's tokens, then `SeqTable.draft_tokens` and `n_draft` for
    the rows that sampled. It may build **auxiliary models** (`BuildCtx.aux_model`: units in a child context whose
    state names, pools, tables and planners live under a namespace, with options of its own — `planner.device_lens`
    for a model that only runs as sub-steps; `StepCtx.view(namespace)` at step time), open its own checkpoint
    (`BuildCtx.weight_source(path)`: the source plugin that recognises it; the plan records its identity under the
    option naming it, not the host path) and run
    **sub-steps** (`StepCtx.substep(q, offset)`: its own rows at positions past the step's, planned at launch with
    per-sub-step buffers and device-length planners). `lookahead(k)` positions past each row are reserved for it. It
    may **tap** the target (`BuildCtx.tap(layer, consume)`: the trunk hands `consume` the residual stream after that
    layer, every step, inside its graph), and a drafter that samples its drafts publishes the laws it drew them from
    (`q_support` = Kq: `SeqTable.draft_q_ids` / `draft_q` [S, k_max, Kq]).
  - `Verifier` (`verifier` slot; exact-match, rejection): replaces the Sampler on verify steps, drawing through it
    (`ctx.sampler`: `distribution`, counter-based `uniform` streams, `logprobs`), and never accepts past a row's
    `n_draft`; `std.spec.rejection` accepts d with min(1, p(d)/q(d)) against the drafter's law q (one-hot at d for
    deterministic drafts) and redraws from max(p − q, 0).
  Several plugins may be installed for each slot: `EngineConfig.slots` (`--slots drafter=… verifier=…`) picks one,
  and locks record the choice in `[engine]`. `Engine.stats()` counts real drafts and verify rows by drafts accepted.
  Every running row of a speculating step is a verify row of the step's width 1 + k (rows with fewer real drafts pad
  with token 0); block tables carry overflow columns at the null unit for positions past `max_model_len`.
- **The SeqTable seam (implemented, M0 acceptance §3.4-5).** Every step's positions, lengths and page write indices
  are derived on the device from the SeqTable's `num_computed` (`Executor._prepare`, inside the graph for decode); the
  host writes the starts it knows exactly (prefill chunks, padding rows, requests without an uncommitted verify
  row) through a per-row mask, so a verify row's outcome never has to reach the host before the next step is
  planned. Verify rows read their drafts from `SeqTable.draft_tokens`; the verifier returns the tokens and
  `n_accept`; the sample stage writes `last_token`, `num_computed = start + 1 + n_accept` and `n_accept`; rewind and
  the drafter follow. Commit appends the accepted tokens (stopping at a stop token or length) and makes the host
  length exact again. Requests with an uncommitted verify row skip steps whose decode kernel plans from host lengths
  (`Planner.needs_host_lens`).

### 7.7 Control plane and errors

- `abort(rid)` (frontend → loop; frees at next commit — safe under overlap), `pause/drain`, `resume`,
  `flush_prefix` (v0). `update_weights(name → tensor)` (in-place through converters into existing QTensor storage;
  bumps `weight_version`) is specified now because it fixes the namespace design, but implemented in v1 (RL weight
  sync, LoRA merges).
- Per-request error isolation (M0.5): `RequestError(kind, message)` from the logits processor (`open`, `ready`,
  `commit`) ends every sequence of its request (`finish_reason` "error", `TokenDelta.kind`: "client" — the request
  is wrong, "busy" — the server cannot take it now, "engine" — it failed while generating); a plain `ValueError`
  from `Engine.add` is kind "client". Frontends map "client" to 400, "busy" to 503, anything else to a 5xx (or
  `finish_reason` "error" once streaming). Other exceptions in a step still end the engine.
- Step-level exception or device fault → engine exits non-zero; the supervisor restarts it **on the same lock**.
  **Crash attribution first:** in-flight requests are resubmitted from committed tokens, rate-limited; a request
  implicated in two crashes is quarantined (`FINISHED(error)`). Only repeated crashes without a poison request lead
  to rollback to the previous lock. Restart time (weights, graph capture, JIT caches) is measured and has an SLO
  set in M0.
- Sampling API coverage in v0 includes `n > 1` (via `StateKind.fork`), output logprobs/top-k logprobs and **prompt
  logprobs** (the judge's teacher-forced re-scoring also uses them).

**Implemented (M0).** The v0 supervisor is `holon.core.supervisor` (the core owns the engine lifecycle; the
`supervisor/` package keeps the optional layer of §10). It is the frontends' `EngineClient` (`holon serve
--supervised`, the default). Health: the IPC client notices a dead process (the engine also sends `failed` when a step
raises), and a periodic `stats` ping with a timeout catches a hung one. After a crash the engine restarts on the same
config and every in-flight request is resubmitted from its committed tokens — each unfinished sequence as prompt +
tokens streamed, marked as output (`EngineRequest.resume_output`: they count towards `max_tokens`, and a logits
processor replays them), its own seed (`sequence_seed`); since sampling is a pure function of (seed, position), the
stream continues without duplicate or missing tokens. An "error" delta on one of a request's engine requests aborts
the others. Attribution: each crash strikes the requests in flight; a request with `poison_crashes` (2) strikes that
is the only such request in a crash, with other requests completed since its first strike, is quarantined, and the
crashes it was in no longer count against the config. Several repeat suspects go on probation (rerun one at a time
beside other requests) so a later crash isolates the culprit. Resubmission is bounded per request rather than by
restarts per minute (a restart takes seconds, so such a rate would rarely trigger): a request that has lived through
`max_crashes` (5) crashes, in flight or waiting, fails with an error. Every crash no quarantine explains is charged to
the config: `rollback_crashes` (4) charges, or as many failed starts in a row, roll back to the previous entry of the
config history. On the oldest entry the engine keeps restarting; the supervisor gives up (and `serve` exits) only when
the engine cannot start. Strikes and charges are forgotten after `stable_s` (60 s) of engine uptime without a crash —
uptime, not wall time, so slow restarts or hang timeouts do not make repeated crashes look unrelated. The history is a
plain list of `EngineConfig`s, oldest first — the seam where the lock registry (§4.4) plugs in. Limits of v0
attribution: a poison request alone on an idle server cannot be told from a broken config (it fails by `max_crashes`,
and its crashes count toward rollback); two poison requests submitted together can be probed before any survivor
exists and charge enough crashes to roll back (rollback is permanent until restart); a crash caused by a fresh request
while a suspect is on probation is blamed on the suspect; one long probe holds the other suspects back until it
finishes, and several poison requests probed first can push innocent suspects past `max_crashes`; a config that
crashes only after `stable_s` of uptime is treated as flaky and never rolls back.
Measured (`bench/restart.py`, Qwen3-0.6B, rtx-a5000, overlap + CUDA graphs, 5 kills): restart (process launch →
ready) 6.6–7.1 s (median 6.75 s), of which ~1.3 s imports and ~5.5 s engine init (weights, pool sizing, 27 graph captures); the
stall a client sees is 7.6–8.1 s (≤ 1 s detection, teardown, restart); the continued greedy stream equals the
uninterrupted one. **SLO (M0):** restart ≤ 10 s and client-visible stall ≤ 12 s for Qwen3-0.6B on every maintained
class, re-measured with each class's tuned lock; larger models add their weight-load time, measured and recorded
the same way before an SLO is set for them.

### 7.8 Frontend

Separate process. One frontend package serves the API dialects (M0.5 D7: OpenAI chat/completions, OpenAI Responses,
Anthropic Messages as routers of one app over one pipeline: normalize → render → generate → reasoning/tool parse →
typed events), using the model plugin's `FrontendBundle`: tokenizer, chat template, and the formats its template
writes reasoning and tool calls in (`reasoning_format`, `tool_format`, versioned names such as "hermes@1"); the
frontend implements each named format once — its streaming parser (typed events: `Text`, `Reasoning`,
`ToolCallStart`, `ToolArgsDelta`, `ToolCallEnd`) and its constraint builder (a plain-dict resource in
`SamplingParams.ext` that the logits processor's plugin declares and translates). The frontend plugin gets the
deployment's plugin options and reads its own namespace (e.g. `http.drop_billing_header`). Stop strings are matched in
the frontend and trigger `abort`; stop tokens are checked at the engine's commit.

The frontend↔engine protocol is a **language-neutral schema** (token ids, sampling params, deltas, logprobs;
msgspec structs from which a protobuf/IDL file is generated and checked in CI), so the frontend can later be replaced by a native (e.g. Rust)
implementation without touching the engine — as vLLM (`vllm/rust`) and SGLang (`rust/`) are doing. Frontend
plugins (API dialects, parsers) would then move to compiled or declarative form; until that trigger (§12), the
frontend stays Python.

### 7.9 Telemetry

`Event` records emitted per step to a ring buffer (shared memory to the supervisor) and optionally to exporters
(Prometheus/OTel plugins):

```
{plan_id, step, ts, phase_us{schedule,transform,forward,fill,sample,commit},
 batch{n_seqs,n_tokens,shape_key}, ledger{pool: used/cap}, prefix{hits,misses,tokens_saved},
 spec{proposed,accepted}, plugin_counters{ns: {...}}}
```

Slot objects (sampler, verifier, drafter, draft policy, logits processor) may count what they do: an optional
`stats()` whose counters `Engine.stats()` reports as `<plugin name>.<key>` beside the core's (M0.5).

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
- The bounded check for prefix cache on/off and depth 1 vs 0 (`python -m holon_judge bounded`, v0): every
  configuration runs a multi-turn trace (sessions sharing a system prompt, each turn resending the history) twice —
  greedy without logprobs on its production path (graphs, depth 1), and with logprobs — and a reference engine
  (prefix cache off, depth 0, eager) re-scores the generated tokens teacher-forced. The first run is judged by
  argmax agreement (overall and at confident positions), the second by the paired KL(reference ‖ configuration) at
  the same positions, as in L2. The floor is the baseline configuration (prefix off, depth 0) measured the same way
  — two bf16 paths disagree at near-ties whatever the change, so the L2 floor does not transfer — and the margins
  are L2's (`TOP1_DELTA`, `KL_FACTOR`) or 2.5 standard errors of a difference between two runs like the floor
  (sessions as clusters), whichever is larger. Planted bugs (`holon_judge.planted`: a prefix hit claiming one token
  too many, a wrong adopted block, a stale device token at depth 1, also at ~1% of steps) must fail exactly the
  configurations using their path; that is tested on CPU and calibrated on Qwen3-8B. Graph replay with logprobs is
  not covered (logprob steps run eagerly): graph replay vs eager is its own comparison.
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
| DeepSeek sparse attention / V4 compressed KV | `dsa_mla` mixer: MLA `Paged` + indexer `Paged(page_group)` + compressed `Paged(tokens_per_unit)` | `page_group`, `tokens_per_unit`, `alignment` |
| Speculative decoding | drafter, draft policy, verifier, plan transform | taps, shared modules, Rewindable, verify graphs, sub-steps, aux models |
| Structured output / tool calls | xgrammar logits processor (host matcher rolled back after each draft probe), frontend parsers and constraint builders by the bundle's formats | logits-processor lifecycle, host-token gate, drafts on the host, `RequestError`, `resume_output`, masks staged through the Uploader on every sampling path |
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
  Holon itself. Untrusted. Writes code only inside `workspace/`.
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
| L2 model golden | teacher-forced top-20 logprobs vs an HF fp32 CPU recording (pinned thread count) on the frozen prompt set v1 (chat, code, long-context recall, agent/tool transcripts; spike S5) | floor = HF in the model's activation dtype vs fp32. Pass: KL p99 ≤ 2× floor (the discriminating metric), KL p50 ≤ 2× floor, top-1 agreement ≥ floor − 0.25 points, \|ΔPPL\| ≤ \|floor\| + 0.25%. δ is re-checked on the first real-engine run. Bugs below bf16 noise are L0's job. Quantized models: golden from dequantized weights; floor calibrated per quant type |
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
holon/           (repository root)
  core/            holon/{api,core}/          # ≤ 6k LOC
  std/             ops/ blocks/ ffn/(swiglu, moe_topk) mixers_gqa/ state_paged_kv/ tiered/ sampling/
  plugins/         models/<m>/{recipe.py, fast/, bundle/}  for qwen3, qwen3_moe, qwen3_5, qwen3_8
                   mixers/{gdn}/  state/{ssm_checkpoint}/
                   kernels/{flashinfer,fla_gdn,ggml_cuda}/  weight_sources/{safetensors,gguf}/  dialects/{ggml}/
                   frontends/{http}/  policies/  drafters/  verifiers/  grammar/
  plans/           <model_id>/<class>/<n>.lock     # lock history per (model, fingerprint class) (§4.4)
  judge/           holon_judge/  goldens/  snapshots/  traces/dev/   # holdout stored outside the repo (judge only)
  tools/           holon_tools/                    # impact, propagate, tune (lock history tooling)
  supervisor/      holon_supervisor/
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
| GGUF kernel performance lags llama.cpp | borrowed ggml kernels measured against llama.cpp on the same card and quant from M0.1; types that lag get our own Triton kernels (kill/pivot criterion, plan §9) |
| sm_86 lacks FP8/FP4 tensor cores | not a v0 issue (GGUF kernels run on all classes); v1 FP8 uses W8A16 on sm_86 and native FP8 on sm_89/sm_120 |
| Determinism claims unachievable with borrowed kernels | §7.10: only run-to-run determinism is required (no batch invariance); non-deterministic sites recorded as `uncovered` |
| Poison requests trigger restart loops and wrong rollbacks | crash attribution before rollback, request quarantine, restart SLO (§7.7) |
| Kernel breakage on sm_86 / torch 2.11 / Triton 3.6 | M0 spikes: standalone GGUF kernel build (S2), vendored vLLM GDN ops, FlashInfer JIT pre-build |
| Golden recording does not fit one GPU | record on CPU fp32 or across GPUs; quantized models via `dequant_ref` into the dense HF class |
