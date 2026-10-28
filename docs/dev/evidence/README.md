# Evidence

Measurement records behind the claims in the README, the design and the development records (`docs/dev/`): one JSON
(or a folder) per run, with the command, the commit or lock it ran, the GPU class and the raw numbers. A record is
written once and never edited; a re-measurement is a new record. Names follow `<topic>_<gpu-class>_<date>`.

Only records a tracked document cites are kept here. A run starts in the untracked `local/` (the bench tools write
their records to `local/evidence/` by default); a record moves here when a document cites it, under the same name.
Records written before this directory existed name the paths of their time: `evidence/` for this directory and
`spikes/<name>/` for a spike (kept ones are under `docs/dev/spikes/`).

Paths under `local/` (e.g. `local/fn/`, `local/models/`, `local/tmp/`) name the authors' untracked working files:
downloads of public models, clones of the reference engines at their public commits, files built from those with the
engines' own public tools, and scratch output. They are not part of the repository and hold no other source of
information.
