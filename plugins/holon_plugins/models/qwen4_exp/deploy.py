"""What fits Qwen3.8-Flash-Next's locks to a machine (docs/dev/v1-auto-config.md D1, D5), asked by `holon serve`:
`spill`, the PLE n-gram table read from the SSD when RAM is short (byte-identical, `v1-flash-next.md` §22.1), and
`source_hint`, how to make the MTP pack the locks name with Strata's public tools (Holon loads it as it is)."""
from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping

MTP = "mtp-q2_0@a22206a4d74eafbf"        # the pack the locks name (`[sources] mtp.path`; v1-flash-next.md §13)
NETWORK = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "ceph", "glusterfs", "fuse.sshfs", "9p", "gpfs",
           "beegfs", "afs"}


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
    return ({"rows.place": ",".join("ple:ssd" if p == "ple:host" else p for p in place)},
            f"the PLE table read from the checkpoint ({disk} disk) instead of ≈ 27 GiB of pinned RAM, byte-identical; "
            "prefill −66%/−80%/−81% and decode −2% to −3% at 1K/4K/32K on an RTX 5090, prefill −60%/−71%/−71% and "
            "decode −2% on an RTX 5080")


def source_hint(identity: str) -> str | None:
    """How to make the MTP pack the locks name (the README's Qwen3.8-Flash-Next section), and the cost of serving
    without it; None for another identity."""
    if identity != MTP:
        return None
    out = f"~/.cache/holon/sources/{MTP}"
    return ("serving without the MTP head costs 24–37% of decode (measured on an RTX 5080 and 5090). Strata's public "
            "tools make it (README, Qwen3.8-Flash-Next):\n"
            "    git clone https://github.com/Niko1221/Strata.git strata && git -C strata checkout 296b59d\n"
            "    uv run python strata/tools/mtp_fetch.py fetch --out mtp\n"
            f"    mkdir -p {out}\n"
            '    STRATA_GGUF_PY=$(uv run python -c "import gguf, pathlib; '
            'print(pathlib.Path(gguf.__file__).parents[1])") '
            f"uv run python strata/tools/mtp_pack.py --src mtp --experts q2_0 --out {out}/mtp-q2_0.gguf")
