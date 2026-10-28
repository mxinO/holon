"""What fits Qwen3.8-Flash-Next's locks to a machine (docs/dev/v1-auto-config.md D1, D5), asked by `holon serve`:
`spill`, the PLE n-gram table read from the SSD when RAM is short (byte-identical, `v1-flash-next.md` §22.1), and
`make_source`, the MTP pack the locks name, downloaded once the project publishes it."""
from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import time
from collections.abc import Mapping

MTP = "mtp-q2_0-q8@a55280a0417d4b19"     # the pack the locks name (`[sources] mtp.path`; v1-flash-next.md §21.11)
# identity -> (Hub repository, file, revision) of the packs the project publishes: none yet (the owner's go). The
# file's sha256 must start with the identity's 16 hex digits.
PUBLISHED: dict[str, tuple[str, str, str]] = {}
NETWORK = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "ceph", "glusterfs", "fuse.sshfs", "9p", "gpfs",
           "beegfs", "afs"}
RETRY_S = 3600                           # a failed fetch of a published pack is tried again an hour later


def storage(path: str, mountinfo: str = "/proc/self/mountinfo", sysfs: str = "/sys") -> str:
    """"network", "hdd", "ssd" or "unknown" for the filesystem holding `path`: its type, else whether the block
    devices under it rotate (through LUKS, LVM and md to the real drives: `slaves/`)."""
    real, best = os.path.realpath(path), None
    with open(mountinfo) as f:
        for fields in (line.split() for line in f):
            mount = fields[4]
            if (real == mount or real.startswith(mount.rstrip("/") + "/")) and (
                    best is None or len(mount) >= len(best[4])):
                best = fields
    if best is None:
        return "unknown"
    if best[best.index("-") + 1] in NETWORK:
        return "network"
    rotational = _rotational(pathlib.Path(sysfs, "dev", "block", best[2]).resolve())
    return "unknown" if not rotational else "hdd" if any(rotational) else "ssd"


def _rotational(dev: pathlib.Path) -> list[bool]:
    """Whether each drive under a sysfs block device rotates: its stacked devices' (`slaves/`), else its own queue's,
    else (a partition) its parent's."""
    if (slaves := sorted((dev / "slaves").glob("*")) if (dev / "slaves").is_dir() else []):
        return [r for s in slaves for r in _rotational(s.resolve())]
    for queue in (dev / "queue", dev.parent / "queue"):
        if (flag := queue / "rotational").is_file():
            return [flag.read_text().strip() == "1"]
    return []


def spill(options: Mapping[str, object], checkpoint: str) -> tuple[dict[str, str], str] | None:
    """The PLE table from the checkpoint's file on the SSD instead of pinned RAM, where a lock keeps it in RAM and the
    file is not on a rotating or network disk (`unknown` is offered, and said)."""
    place = [p.strip() for p in str(options.get("rows.place", "")).split(",") if p.strip()]
    if "ple:host" not in place or (disk := storage(checkpoint)) in ("hdd", "network"):
        return None
    return ({"rows.place": ",".join("ple:ssd" if p == "ple:host" else p for p in place),
             "cpu_task.io_threads": "4"},
            f"the PLE table read from the checkpoint ({disk} disk) instead of ≈ 27 GiB of pinned RAM, byte-identical; "
            "prefill −16%/−26% at 1K/4K and MTP decode −6%/−7% on an RTX 5090, unmeasured on an RTX 5080")


def make_source(identity: str, directory: str) -> str | None:
    """The MTP pack `identity` names, in `directory` (a link to the hub cache's file), downloaded by identity from the
    project's published packs and checked against it; None for another identity; ValueError why not (a failed fetch
    of a published pack is remembered for an hour beside `directory`, keyed by what was published)."""
    if identity != MTP:
        return None
    out = pathlib.Path(directory) / "mtp-q2_0-q8.gguf"
    if out.is_file():
        return str(out)
    if identity not in PUBLISHED:
        raise ValueError("not published yet; the README's Qwen3.8-Flash-Next section builds it by hand. Without it "
                         "decode is ≈ 30% slower (est.)")
    memo, key = pathlib.Path(f"{directory}.failed"), repr(PUBLISHED[identity])
    if memo.is_file() and time.time() - memo.stat().st_mtime < RETRY_S and memo.read_text().startswith(key + "\n"):
        raise ValueError(memo.read_text().partition("\n")[2] + " (remembered; tried again an hour later)")
    try:
        from huggingface_hub import hf_hub_download

        repo, name, revision = PUBLISHED[identity]
        path = hf_hub_download(repo, name, revision=revision)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(1 << 24):
                digest.update(block)
        if not digest.hexdigest().startswith(identity.rpartition("@")[2]):
            raise ValueError(f"{repo}/{name}@{revision} is not {identity} (sha256 {digest.hexdigest()[:16]})")
        link = out.with_name(f".{out.name}.{os.getpid()}")
        link.unlink(missing_ok=True)
        link.symlink_to(path)
        os.replace(link, out)                # atomic: a concurrent first start finds one whole link
        return str(out)
    except Exception as e:  # noqa: BLE001 — offline, not found, a wrong file: serve without it, and say why
        with contextlib.suppress(OSError):  # a cache that cannot be written only loses the memo
            memo.write_text(f"{key}\n{e}")
        raise ValueError(str(e)) from None
