"""A minimal GGUF (v2/v3) reader: the metadata and tensor table read once, and tensor views over a copy-on-write
memory map opened only while weights load, nothing copied.

gguf-py's `GGUFReader` builds Python objects for every element of every array (≈ 5 s for a 152K-token vocabulary);
this reads numeric arrays with numpy and strings in one pass (≈ 1 s, the token and merge strings). gguf-py still
supplies the type tables, and stays the dequantization oracle.
"""
from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator
from typing import Any

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

_SCALARS = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32,
            7: np.bool_, 10: np.uint64, 11: np.int64, 12: np.float64}
_STRING, _ARRAY = 8, 9


@dataclasses.dataclass(frozen=True)
class TensorInfo:
    name: str
    type: GGMLQuantizationType
    shape: tuple[int, ...]           # torch order (slowest dimension first)
    offset: int                      # from the file's start
    rows: int
    row_bytes: int                   # a row is whole blocks (or elements, for dense types)


class _Cursor:
    def __init__(self, buf: np.ndarray, path: pathlib.Path) -> None:
        self.buf, self.path, self.at = buf, path, 0

    def take(self, n: int) -> np.ndarray:
        if self.at + n > len(self.buf):
            raise ValueError(f"{self.path}: truncated GGUF header")
        out = self.buf[self.at:self.at + n]
        self.at += n
        return out

    def scalar(self, dtype: Any) -> Any:
        return self.take(np.dtype(dtype).itemsize).view(dtype)[0].item()

    def string(self) -> str:
        return self.take(self.scalar(np.uint64)).tobytes().decode("utf-8", errors="replace")

    def value(self, kind: int) -> Any:
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            elem, n = self.scalar(np.uint32), self.scalar(np.uint64)
            if elem == _STRING:
                return [self.string() for _ in range(n)]
            if elem not in _SCALARS:
                raise ValueError(f"{self.path}: unsupported GGUF array of type {elem}")
            dtype = np.dtype(_SCALARS[elem])
            return self.take(n * dtype.itemsize).view(dtype).tolist()
        if kind not in _SCALARS:
            raise ValueError(f"{self.path}: unsupported GGUF value type {kind}")
        return self.scalar(_SCALARS[kind])


def read(path: str | pathlib.Path) -> tuple[dict[str, Any], list[TensorInfo]]:
    """The file's metadata (key -> value) and its tensor table, checked against the file's size."""
    path = pathlib.Path(path)
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    cur = _Cursor(buf, path)
    if cur.take(4).tobytes() != b"GGUF" or cur.scalar(np.uint32) not in (2, 3):
        raise ValueError(f"{path}: not a GGUF v2 or v3 file")
    n_tensors, n_fields = cur.scalar(np.uint64), cur.scalar(np.uint64)
    fields = {}
    for _ in range(n_fields):
        key = cur.string()
        fields[key] = cur.value(cur.scalar(np.uint32))
    table = []
    for _ in range(n_tensors):
        name = cur.string()
        dims = [cur.scalar(np.uint64) for _ in range(cur.scalar(np.uint32))]
        table.append((name, GGMLQuantizationType(cur.scalar(np.uint32)), dims, cur.scalar(np.uint64)))
    align = fields.get("general.alignment", 32)
    start = -(-cur.at // align) * align
    infos = []
    for name, kind, dims, offset in table:
        block, size = GGML_QUANT_SIZES[kind]
        if not dims or dims[0] % block or offset % align:
            raise ValueError(f"{path}: {name}: rows of {dims[:1]} not whole {kind.name} blocks of {block}, or offset "
                             f"{offset} not aligned to {align}")
        rows = 1
        for n in dims[1:]:
            rows *= n                                   # Python ints: corrupt sizes cannot wrap around
        info = TensorInfo(name, kind, tuple(reversed(dims)), start + offset, rows, dims[0] // block * size)
        if info.offset + info.rows * info.row_bytes > len(buf):
            raise ValueError(f"{path}: {name} runs past the end of the file")
        infos.append(info)
    return fields, infos


def views(path: str | pathlib.Path, infos: list[TensorInfo]) -> Iterator[tuple[TensorInfo, np.ndarray]]:
    """Each tensor's bytes, uint8 [rows, row_bytes], over one copy-on-write map (torch wraps them without a copy, and
    the file is never written); the map closes when the last view is dropped — after loading on a GPU, and with the
    engine on the CPU, whose parameters stay views into it."""
    buf = np.memmap(path, dtype=np.uint8, mode="c")
    for t in infos:
        yield t, buf[t.offset:t.offset + t.rows * t.row_bytes].reshape(t.rows, t.row_bytes)
