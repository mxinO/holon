"""std.tiered's host-side pieces (docs/v1-offload.md D4, D10): extents read back into host memory exactly, through
aligned windows (O_DIRECT where the filesystem has it), and residency-map edits applied at launch c + 2, entry by
entry."""
import os

import numpy as np
import torch
from holon.api import Extent
from holon_std.tiered import files
from holon_std.tiered.residency import ResidencyMap


def test_extents_read_back_bitwise_through_aligned_windows(tmp_path, monkeypatch):
    """Unaligned extents in any order, adjacent, overlapping a window's end, larger than a window, at the file's
    end, from two files."""
    monkeypatch.setattr(files, "WINDOW", 3 * files.ALIGN)
    data = {p: np.random.default_rng(i).integers(0, 256, 9 * files.ALIGN + 77, dtype=np.uint8)
            for i, p in enumerate((tmp_path / "a", tmp_path / "b"))}
    for p, d in data.items():
        p.write_bytes(d.tobytes())
    spans = [(5, 100), (105, 4000), (8200, 5 * files.ALIGN), (3 * files.ALIGN - 7, 20), (len(data[tmp_path / "a"]) - 50,
                                                                                      50)]
    jobs = [(Extent(str(p), off, n), np.zeros(n, dtype=np.uint8)) for p in data for off, n in reversed(spans)]
    files.read(jobs)
    for extent, dst in jobs:
        assert np.array_equal(dst, data[type(tmp_path)(extent.path)][extent.offset:extent.offset + extent.nbytes])
    assert os.O_DIRECT                                      # the reader asks for it (Linux)


class _Ctx:
    def static(self, name, shape, dtype):
        return torch.zeros(shape, dtype=dtype)


def test_map_edits_apply_at_launch_c_plus_2_and_only_where_they_name():
    """An edit decided at commit 7 reaches the device at launch 9 as its own entries: what the device changed in
    between elsewhere in the map (a device resolver's admission) stays; the edit list is empty after its launch."""
    m = ResidencyMap(_Ctx(), "t", (2, 4, 2))
    m.host[0, 1] = [5, 6]
    m.seed()
    assert m.device[0, 1].tolist() == [5, 6]
    m.edit(7, [(0, 1, 0, -1), (1, 3, 1, 9), (1, 3, 1, 8)])        # the last edit of an entry wins
    assert m.writes(8) == [] and m.host[0, 1, 0] == 5
    m.device[1, 0, 0] = 42                                        # the device admits an expert meanwhile
    for dst, values in m.writes(9):                               # as the core runs `before_launch` writes
        dst.copy_(torch.from_numpy(values))
    m.apply()                                                     # the step's first resolve
    assert m.device[0, 1].tolist() == [-1, 6] and m.device[1, 3, 1] == 8 and m.device[1, 0, 0] == 42
    assert m.host[1, 3, 1] == 8 and m.writes(9) == []
    m.device[0, 1, 1] = 7
    m.apply()                                                     # a step without edits changes nothing
    assert m.device[0, 1].tolist() == [-1, 7]
