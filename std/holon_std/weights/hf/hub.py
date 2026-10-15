"""`holon serve <org>/<name>`: a Hugging Face Hub repository holding a safetensors checkpoint, fetched into the HF cache
(release design §14.3). The hub library is imported only here, when a reference is resolved, so serving a local path
never needs it; its own configuration applies (`HF_HOME`, `HF_TOKEN`, `HF_HUB_OFFLINE`)."""
from __future__ import annotations

import fnmatch
import pathlib
import re

from holon.api import Resolved

REPO = re.compile(r"\w[\w.-]*/\w[\w.-]*")          # `<org>/<name>`: a hub repository id
FILES = ("*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken")   # what a checkpoint directory reads


def resolve(ref: str) -> Resolved | None:
    """A repository with `config.json` and safetensors weights (files or an index): its top-level files the source
    reads. Anything else is not this plugin's (None)."""
    if not REPO.fullmatch(ref):
        return None
    files = {p: n for p, n in listing(ref).items() if "/" not in p and any(fnmatch.fnmatch(p, f) for f in FILES)}
    if "config.json" not in files or not any(p.endswith((".safetensors", ".safetensors.index.json")) for p in files):
        return None

    def fetch() -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(ref, allow_patterns=sorted(files))

    return Resolved(f"{ref} ({len(files)} files)", sum(files.values()), fetch)


def listing(repo: str) -> dict[str, int]:
    """Path in the repository -> size: the hub's listing; offline or unreachable, the cached snapshot's. A ValueError
    says why neither is there."""
    from huggingface_hub import HfApi, constants, errors

    error = "HF_HUB_OFFLINE is set"
    if not constants.HF_HUB_OFFLINE:
        try:
            return {f.path: f.size for f in HfApi().list_repo_tree(repo, recursive=True) if hasattr(f, "size")}
        except errors.GatedRepoError:
            raise ValueError(f"{repo} is gated: accept its terms on the hub, then `hf auth login`") from None
        except errors.RepositoryNotFoundError:
            raise ValueError(f"{repo}: no such repository on the hub (private ones need `hf auth login`)") from None
        except Exception as e:  # noqa: BLE001 — a connection failure: the cache may still have it
            error = f"the hub is unreachable ({type(e).__name__})"
    root = pathlib.Path(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
    ref = root / "refs" / "main"
    snapshot = root / "snapshots" / ref.read_text().strip() if ref.is_file() else None
    if snapshot is None or not snapshot.is_dir():
        raise ValueError(f"{error} and {repo} is not in the HF cache (offline, set HF_HUB_OFFLINE=1 to serve cached "
                         "models without the network)")
    return {p.relative_to(snapshot).as_posix(): p.stat().st_size for p in snapshot.rglob("*") if p.is_file()}
