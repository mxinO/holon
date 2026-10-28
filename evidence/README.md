# Evidence

Measurement records behind the claims in the README, the design and the development records (`docs/dev/`): one JSON
(or a folder) per run, with the command, the commit or lock it ran, the GPU class and the raw numbers. A record is
written once and never edited; a re-measurement is a new record. Names follow `<topic>_<gpu-class>_<date>`.

Paths under `local/` (e.g. `local/fn/`, `local/models/`, `local/tmp/`) name the authors' untracked working files:
downloads of public models, clones of the reference engines at their public commits, files built from those with the
engines' own public tools, and scratch output. They are not part of the repository and hold no other source of
information.
