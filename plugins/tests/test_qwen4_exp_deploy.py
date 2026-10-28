"""models.qwen4_exp's deployment hooks (docs/dev/v1-auto-config.md D1, D5): the PLE spill only onto a disk that is
not positively rotating or networked, the drives found through stacked devices; the MTP pack, unpublished, refused
with the reason, remembered."""
import hashlib
import sys
import types

import pytest
from holon_plugins.models.qwen4_exp import deploy


def _host(tmp_path, fstype="ext4", rotational=("0",), stacked=True):
    """A mountinfo with `/data` on 253:0 (a LUKS / LVM `dm-0` over `slaves`, or a plain partition 259:2 of nvme0n1)."""
    sysfs, dev = tmp_path / "sys", "253:0" if stacked else "259:2"
    block = sysfs / "dev" / "block"
    block.mkdir(parents=True)
    if stacked:
        dm = sysfs / "devices" / "dm-0"
        for i, flag in enumerate(rotational):
            drive = sysfs / "devices" / f"sd{i}"
            (drive / "queue").mkdir(parents=True)
            (drive / "queue" / "rotational").write_text(flag + "\n")
            (dm / "slaves").mkdir(parents=True, exist_ok=True)
            (dm / "slaves" / f"sd{i}").symlink_to(drive)
        (block / dev).symlink_to(dm)
    else:
        part = sysfs / "devices" / "nvme0n1" / "nvme0n1p2"
        part.mkdir(parents=True)
        (part.parent / "queue").mkdir()
        (part.parent / "queue" / "rotational").write_text(rotational[0] + "\n")
        (block / dev).symlink_to(part)
    info = tmp_path / "mountinfo"
    info.write_text(f"1 0 8:1 / / rw - ext4 /dev/sda1 rw\n2 1 {dev} / /data rw - {fstype} /dev/x rw\n")
    return str(info), str(sysfs)


@pytest.mark.parametrize("case, want", [("nvme partition", "ssd"), ("luks on nvme", "ssd"), ("raid with a hdd", "hdd"),
                                        ("nfs", "network"), ("no sysfs", "unknown")])
def test_the_spill_targets_any_disk_not_known_to_rotate_or_be_remote(tmp_path, monkeypatch, case, want):
    info, sysfs = _host(tmp_path, "nfs4" if case == "nfs" else "ext4",
                        ("0", "1") if case == "raid with a hdd" else ("0", "0"), stacked=case != "nvme partition")
    if case == "no sysfs":
        sysfs = str(tmp_path / "nowhere")
    assert deploy.storage("/data/model.gguf", info, sysfs) == want
    monkeypatch.setattr(deploy, "storage", lambda path: want)
    got = deploy.spill({"rows.place": "ple:host,token_embd:host"}, "/data/model.gguf")
    if want in ("hdd", "network"):
        assert got is None
    else:
        assert got[0] == {"rows.place": "ple:ssd,token_embd:host", "cpu_task.io_threads": "4"} and want in got[1]
    assert deploy.spill({"rows.place": "ple:ssd"}, "/data/model.gguf") is None        # nothing in RAM to spill


def test_the_pack_is_fetched_by_identity_and_a_failure_remembered_only_for_what_was_published(tmp_path, monkeypatch):
    """Unpublished: refused with the cost, never remembered (publishing it serves it at the next start); a published
    pack's failed fetch is remembered an hour, keyed by what was published; a fetched pack is linked atomically and
    checked against the identity."""
    directory = tmp_path / deploy.MTP
    directory.mkdir()
    assert deploy.make_source("other@0123456789abcdef", str(directory)) is None
    with pytest.raises(ValueError, match="not published yet.*30% slower"):
        deploy.make_source(deploy.MTP, str(directory))
    assert not (tmp_path / f"{deploy.MTP}.failed").exists()
    blob, fetched = tmp_path / "blob", []
    hub = types.SimpleNamespace(hf_hub_download=lambda repo, name, revision: fetched.append(revision) or str(blob))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(deploy.PUBLISHED, deploy.MTP, ("org/packs", "mtp-q2_0-q8.gguf", "r1"))
    blob.write_bytes(b"not the pack")
    for _ in range(2):                                     # the second from the memo, without fetching again
        with pytest.raises(ValueError, match="is not mtp-q2_0-q8@a55280a0417d4b19"):
            deploy.make_source(deploy.MTP, str(directory))
    # republished at r2 (here: the identity of what r2 holds): the memo of r1 does not stop the fetch
    monkeypatch.setattr(deploy, "MTP", "mtp-q2_0-q8@" + hashlib.sha256(b"not the pack").hexdigest()[:16])
    monkeypatch.setitem(deploy.PUBLISHED, deploy.MTP, ("org/packs", "mtp-q2_0-q8.gguf", "r2"))
    assert deploy.make_source(deploy.MTP, str(directory)) == str(directory / "mtp-q2_0-q8.gguf")
    assert fetched == ["r1", "r2"] and (directory / "mtp-q2_0-q8.gguf").resolve() == blob
