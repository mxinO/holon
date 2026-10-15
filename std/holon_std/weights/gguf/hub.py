"""`holon serve <org>/<repo>:<TAG>` or `<org>/<repo>/<stem>`: one GGUF file of a Hugging Face Hub repository, fetched
into the HF cache (release design §14.3). The hub library is imported only here, when a reference is resolved, so
serving a local path never needs it; its own configuration applies (`HF_HOME`, `HF_TOKEN`, `HF_HUB_OFFLINE`)."""
from __future__ import annotations

import pathlib
import re

from holon.api import Resolved

REPO = re.compile(r"\w[\w.-]*/\w[\w.-]*")          # `<org>/<repo>`: a hub repository id
SPLIT = re.compile(r"-\d{5}-of-\d{5}$")          # a split file's suffix: `<stem>-00001-of-00003.gguf`


def resolve(ref: str) -> Resolved | None:
    """`<repo>:<TAG>` (the file the tag names), `<repo>/<stem>[.gguf]` (that file, in any directory: the source's name
    drops it), or a bare `<repo>` with `.gguf` files (its only one). Anything else is not this plugin's (None)."""
    if ":" in ref:
        (repo, tag), stem = ref.split(":", 1), ""
    else:
        parts = ref.split("/")
        repo, tag, stem = "/".join(parts[:2]), None, "/".join(parts[2:]).removesuffix(".gguf")
    if not REPO.fullmatch(repo):
        return None
    found, cached = listing(repo)
    files = {p: n for p, n in found.items() if p.endswith(".gguf") and "mmproj" not in p.lower()}
    if not files and tag is None and not stem:
        if cached:                                    # the hub unreachable: say what the cache lacks
            raise ValueError(f"{repo}: its cached snapshot holds no GGUF file")
        return None                                   # not a GGUF repository: another source's
    path = pick(sorted(files), tag, stem, repo)

    def fetch() -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, path)

    return Resolved(f"{repo}/{path}", files[path], fetch)


def pick(files: list[str], tag: str | None, stem: str, repo: str) -> str:
    """The file a reference names among the repository's model files (paths in it): with `tag`, of the files whose
    stem ends in `-TAG` or `.TAG` (any case), the shortest stem (`Q4_K_M` takes `X-Q4_K_M` over `X-UD-Q4_K_M`); with
    `stem`, the file of that stem; else the only file. A split file is refused: the source reads one file."""
    whole = {p: SPLIT.sub("", pathlib.PurePosixPath(p).stem) for p in files}      # the stem without a split suffix
    if tag is not None:
        found = [p for p, s in whole.items() if s.lower().endswith((f"-{tag.lower()}", f".{tag.lower()}"))]
        found = [p for p in found if len(whole[p]) == min(len(whole[q]) for q in found)]
    else:
        found = [p for p, s in whole.items() if not stem or s == stem]
    split = [p for p in found if whole[p] != pathlib.PurePosixPath(p).stem]
    if split and len(split) < len(found):             # a single file beside a split set of the same name: the file
        found, split = [p for p in found if p not in split], []
    if len(found) == 1 and not split:
        return found[0]
    if found and len(split) == len(found) and len({whole[p] for p in found}) == 1:
        raise ValueError(f"{repo}: {whole[found[0]]} is split into {len(found)} files; the GGUF source reads one file")
    what = "several GGUF files" if found else f"no GGUF file {stem}" if stem else f"no GGUF file tagged {tag}"
    raise ValueError(f"{what} in {repo}; name one as {repo}:<TAG> or {repo}/<name>: "
                     + (", ".join(sorted(set(whole.values()))) or "it has none"))


def listing(repo: str) -> tuple[dict[str, int], bool]:
    """Path in the repository -> size, and whether it came from the cache: the hub's listing; offline or unreachable,
    the cached snapshot's. A ValueError says why neither is there."""
    from huggingface_hub import HfApi, constants, errors

    error = "HF_HUB_OFFLINE is set"
    if not constants.HF_HUB_OFFLINE:
        try:
            return {f.path: f.size for f in HfApi().list_repo_tree(repo, recursive=True) if hasattr(f, "size")}, False
        except errors.HFValidationError as e:
            raise ValueError(f"{repo}: not a hub repository id ({e})") from None
        except errors.GatedRepoError:
            raise ValueError(f"{repo} is gated: accept its terms on the hub, then `hf auth login`") from None
        except errors.RepositoryNotFoundError:
            raise ValueError(f"{repo}: no such repository on the hub (private ones need `hf auth login`)") from None
        except Exception as e:  # noqa: BLE001 — a connection failure: the cache may still have it
            error = f"the hub is unreachable ({type(e).__name__}; offline, HF_HUB_OFFLINE=1 serves cached models)"
    root = pathlib.Path(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
    ref = root / "refs" / "main"
    snapshot = root / "snapshots" / ref.read_text().strip() if ref.is_file() else None
    if snapshot is None or not snapshot.is_dir():
        raise ValueError(f"{error}, and {repo} is not in the HF cache")
    return {p.relative_to(snapshot).as_posix(): p.stat().st_size for p in snapshot.rglob("*") if p.is_file()}, True
