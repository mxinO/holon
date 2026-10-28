# Spikes

Experiments kept because a document cites them as how its numbers were produced: the M0 spikes behind the adopted
kernels and the judge's prompt sets (`s1_attention`, `s2_gguf`, `s3_gdn`, `s5_golden`, `s6_cpu_handshake`; design
§5.6, plan "Spike outcomes", `judge/README.md`) and `x4_gate`, which made the placement profile the README's offload
lock names. One folder per question, named after the milestone step that asked it. They are records, not part of the
engine or its tests: they ran from `spikes/<name>/` at the commits they name and may not run against today's code.

Other spikes the development records cite (`spikes/<name>/…`) are not kept in the repository; the tools that outlived
them are in `judge/` and `bench/`. New experiments live in the untracked `local/`.

Paths under `local/` (e.g. `local/fn/`, `local/models/`, `local/tmp/`) name the authors' untracked working files:
downloads of public models, clones of the reference engines at their public commits, files built from those with the
engines' own public tools, and scratch output. They are not part of the repository and hold no other source of
information.
