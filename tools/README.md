# holon_tools — lock history commands (M0)

The `holon impact | propagate | tune` commands of design §4.4 and §4.6, run as `python -m holon_tools <command>`.
They build engines in-process and call the judge (snapshot, L2), so they live outside the core and the judge: the
core only knows plans and lock files (`holon.core.plans`), the judge only measures, and these tools decide what goes
into a model's lock history. Locks are written by these tools only, never by hand.

## Lock history

`plans/<model_id>/<class>/<n>.lock`, the newest `n` is what runs, rollback is the one before. `model_id` is
`<hub id or checkpoint dir name>@<16 hex of config + weight hashes>`; the class is `<gpu>/<toolchain>`, e.g.
`rtx-a5000/cu130-torch2.11`. A lock is TOML:

```toml
plan_id = "sha256:…"                 # canonical hash of the plan below; re-checked on read
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

[units]                              # every plugin the engine loaded, plus the core as "holon"
"std.ops" = { sem = "sha256:…", src = "<git tree hash>", version = "1.0.0" }

[sites]                              # one kernel binding per site: "<site>/<op>"
"kv/attention.paged_decode@1" = { config = { facts = { … }, props = ["deterministic", "graph_safe"] }, impl = "kernels.triton_attn" }
```

`src` is the git tree hash of the unit's directory when the checkout matches HEAD there, else a `sha256:` content
hash (uncommitted edits), so a pin never names code other than what ran. `model_id` hashes the weight files: free
for hub-cache snapshots (their blobs are named by sha256), a full read for plain directories.

## Commands

```
# which locks pin a changed unit: unit names, paths (every unit whose directory holds it), or a git range (units
# whose src tree differs); anything else is an error
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
`judge/prompts/v1.jsonl`), and every EngineConfig field as a flag (`--device`, `--dtype`, `--max-seqs`, …).
`propagate`/`tune` exit 0 when nothing needs doing or the lock was written, 2 when the new lock waits for
`--approve`, 1 on failure.

`propagate` builds the plan the checked-out code resolves with the lock's kernel pins (they isolate the model from
kernels installed since) and the engine settings given as flags (a changed setting is a plan change like any other),
then:

| outcome | condition | effect |
|---|---|---|
| unchanged | the plan equals the lock's except for `commit` | nothing written |
| advance | the snapshot trace (run twice, must be run-identical) is byte-identical to the lock's and no `uncovered` site (old or new) changed | next lock appended |
| approve | otherwise, and judge L2 vs the HF golden passes the floor budgets | written with `--approve` |
| fail | L2 fails, or the new snapshot trace is not run-identical while every kernel claims determinism | nothing written |

A plan with uncovered sites may have no run-identical trace; its lock then records no snapshot and every change goes
through L2. The commit does not count as a change: a lock committed in commit C can never name C, so counting it
would make every re-run write another lock. Core changes are seen through the core's own `holon` unit pin. Starting
from another class's lock (no lock for this class yet), only the pins whose provider runs here are kept.

`tune` builds the base plan, lists for each decode op the providers that run here and support every site, and
benchmarks one op at a time; a pinned provider is replaced only when beaten by more than 3%, so re-tuning does not
flip pins on noise.

The snapshot trace (`propagate.snapshot_trace`) takes the first 8 prompts of the prompt set (≤ 128 tokens): greedy
requests, seeded sampling with n = 2 and logprobs, prompt logprobs, staggered lengths. A first lock for Qwen3-0.6B on
an RTX A5000 (two snapshot runs + L2 over v1) takes ≈ 20 s; a no-op re-run ≈ 10 s; `tune` over the five decode ops
≈ 63 s.

## Not yet

- Changed-line coverage (§4.6): auto-advance should also require every changed line in the model's closure to have
  executed during the snapshot trace. Needs line coverage recorded while the trace runs.
- Approval also requires "no L2 metric regresses beyond δ vs the current lock": needs the lock's L2 metrics stored.
- The snapshot trace lacks timed arrivals (prefix hits across requests) and spec decoding/grammar (later milestones).
- CI `isolation-check` (re-run propagate on trusted GPU runners and reject unreproducible locks), sweeping all models.
- The engine/supervisor starting from the newest lock (`nearest_class` + `lock_overrides`), with an `unverified`
  lock's pins as soft preferences: a borrowed lock may pin kernels this GPU cannot run.
