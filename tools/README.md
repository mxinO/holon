# holon_tools — lock history commands (M0)

The `holon impact | propagate | tune` commands of design §4.4 and §4.6, run as `python -m holon_tools <command>`.
They build engines in-process and call the judge (snapshot trace, L2), so they live outside the core and the judge:
the core only reads lock files (`holon.core.plans`), the judge chooses and measures the tests, and these tools compute
plans (`locks.plan_of`) and decide what goes into a model's lock history (`locks.dumps`/`append`). Locks are written
by these tools only, never by hand.

## Lock history

`plans/<model_id>/<class>/<n>.lock`, the newest `n` is what runs, rollback is the one before. `model_id` is
`<hub id or checkpoint dir name>@<16 hex of config + weight hashes>`; the class is `<gpu>/<toolchain>`, e.g.
`rtx-a5000/cu130-torch2.11`. A lock is TOML:

```toml
digest = "sha256:…"                  # canonical hash of everything below; re-checked on read (hand edits fail)
plan_id = "sha256:…"                 # canonical hash of the plan; re-checked on read
model_id = "Qwen/Qwen3-0.6B@b2a7fdcbde68f255"
fingerprint_class = "rtx-a5000/cu130-torch2.11"
sm = 86
recipe = "sha256:…"                  # canonical hash of the model's ModelSpec

[deploy]
commit = "…"                         # git HEAD the lock was verified at
uv_lock = "none"                     # sha256 of uv.lock, if any

[engine]                             # EngineConfig settings that shape outputs (memory budgets are per host)
cuda_graphs = true
dtype = "bfloat16"
max_step_tokens = 4096
…

[snapshot]
trace = "sha256:…"                   # hash of the snapshot trace's outputs; stored in <class>/snapshots/<hash>.json
uncovered = []                       # sites whose kernel is not run-to-run deterministic

[units]                              # the code that runs, plus the core as "holon"
"std.ops" = { sem = "sha256:…", src = "sha256:…", version = "1.0.0" }

[sites]                              # one kernel binding per site: "<site>/<op>"
"kv/attention.paged_decode@1" = { config = { facts = { … }, props = ["deterministic", "graph_safe"] }, impl = "kernels.triton_attn" }
```

`[units]` pins the core, the plugins of the units the model built, the kernel providers bound at its sites and the
engine's other plugins (model, weight source, sampler, state kinds …); kernel providers loaded only as candidates
are not pinned. `src` is sha256 over a unit directory's sorted relative paths and file contents, without hidden
files, `__pycache__`, `*.pyc`/`*.pyo`/`*.so`, `build/`, `dist/` and `*.egg-info`: content only, so a git checkout
and an installed copy agree. `model_id` hashes the weight files: free for hub-cache snapshots (their blobs are named
by sha256); other checkpoint directories are read once and remembered in the user's cache (`~/.cache/holon/weights`,
never the checkpoint), keyed by each file's device, inode, size, mtime and ctime.

## Commands

```
# which locks pin a changed unit: unit names, paths (every unit whose directory holds it), or a git range (units
# whose directory tree differs); anything else is an error
python -m holon_tools impact std.ops
python -m holon_tools impact std/holon_std/ops/kernels.py
python -m holon_tools impact origin/main..HEAD

# carry a model's lock to the checked-out code (one model, this machine's class)
CUDA_VISIBLE_DEVICES=<gpu> python -m holon_tools propagate Qwen/Qwen3-0.6B \
    --golden judge/goldens/qwen3-0.6b/v1-fp32.safetensors --floor judge/goldens/qwen3-0.6b/v1-bfloat16-floor.safetensors
# a new model's first lock (always one approval)
... propagate <model> --new-model --approve --golden … --floor …
# pin the fastest provider per decode op on this GPU, then the same checks as propagate
... tune <model> [--ops attention.paged_decode rmsnorm …] [--batch 8] [--tokens 64] --golden … --floor …
```

Common flags: `--plans <dir>` (default `plans`), `--class C` (start from that class's newest lock; default this
machine's class, else the nearest class's), `--prompts` (prompt set of L2 and of the snapshot trace; default
`judge/prompts/v1.jsonl`), `--approve`, `--allow-uncovered-advance`, `--allow-approx`, and every EngineConfig field as
a flag (`--device`, `--dtype`, `--max-seqs`, …). `propagate`/`tune` exit 0 when nothing needs doing or the lock was
written, 2 when the new lock waits for `--approve`, 1 on failure. A plan that binds a plugin declaring
`numerics = "approx"` (e.g. the route tail skip, `docs/v1-offload.md` §6) is never written without `--allow-approx`:
such a lock is a human gate, never a default.

`propagate` builds the plan the checked-out code resolves with the lock's kernel pins (they isolate the model from
kernels installed since) and the engine settings given as flags (a changed setting is a plan change like any other),
then:

| outcome | condition | effect |
|---|---|---|
| unchanged | the plan equals the lock's except for `commit` | nothing written |
| approve | judge L2 vs the HF golden passes the floor budgets; the snapshot trace (run twice) and whether it is byte-identical to the lock's are recorded as evidence | written with `--approve` |
| fail | L2 fails, or the new snapshot trace is not run-identical while every kernel claims determinism | nothing written |
| advance | only with `--allow-uncovered-advance`: the trace is byte-identical to the lock's and no `uncovered` site (old or new) changed | next lock appended |

Design §4.6 auto-advances only when, in addition, every changed line in the model's closure ran during the trace.
Changed-line coverage is not recorded yet, so every changed plan gets one approval by default. A plan with uncovered
sites may have no run-identical trace; its lock then records no snapshot and every change goes through L2. The commit
does not count as a change: a lock committed in commit C can never name C, so counting it would make every re-run
write another lock. Core changes are seen through the core's own `holon` unit pin. Starting from another class's lock
(no lock for this class yet), only the pins whose provider runs here are kept.

`tune` builds the base plan, lists for each decode op the providers that run here and support every site, and
benchmarks one op at a time; a pinned provider is replaced only when beaten by more than 3%, so re-tuning does not
flip pins on noise.

The snapshot trace is the judge's (`holon_judge.snapshot.trace`): a 2048-token prompt, the same prompt extended and
submitted once the first has finished (a prefix-cache hit), and 62 short requests with staggered lengths, so the
decode batch shrinks through every graph bucket up to 64; seeded sampling (n = 2 included), logprobs and prompt
logprobs. Each run records the step widths it covered and the prefix tokens saved. On Qwen3-0.6B (RTX A5000, default
engine settings) it is run-identical across 3 runs, covers widths 1–65 except a few in 50–60 (every bucket up to 64)
and saves 2064 prefix tokens. Timings for Qwen3-0.6B on an RTX A5000: a first lock (two trace runs + L2 over v1)
≈ 18 s, a no-op re-run ≈ 10 s, `tune` over the five decode ops ≈ 63 s.

## Not yet

- Changed-line coverage (§4.6), the condition for auto-advance without the opt-in flag.
- Approval also requires "no L2 metric regresses beyond δ vs the current lock": needs the lock's L2 metrics stored.
- The trace's long context stops at the prompt set's longest prompt (2048 tokens); buckets above 64 (`max_seqs` >
  64), spec decoding and grammar are not exercised.
- CI `isolation-check` (re-run propagate on trusted GPU runners and reject unreproducible locks), sweeping all models.
- A lock with an offloaded MoE's CPU split (F5b, `docs/v1-offload.md` §6): L2 scores the lock's own settings, and
  F5b's path-aware floor needs a frozen placement at one token per step, so `propagate` and `tune --cpu-split` (without
  `--no-lock`) stop with the judge's `ValueError`; F5b's bounded L2 runs by hand (`spikes/x6_cpu/l2.py`, then
  `holon_judge golden --cpu-split` and `compare`). Plan §10's "F5b locks".
- The engine/supervisor starting from the newest lock (`nearest_class` + `lock_overrides`), with an `unverified`
  lock's pins as soft preferences: a borrowed lock may pin kernels this GPU cannot run.
