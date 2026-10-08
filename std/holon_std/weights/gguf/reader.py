"""A minimal GGUF (v2/v3) reader: metadata and tensor views over one copy-on-write memory map, nothing copied.

gguf-py's `GGUFReader` builds Python objects for every element of every array (≈ 5 s for a 152K-token vocabulary);
this reads numeric arrays with numpy and strings in one pass (≈ 1 s, the token and merge strings). gguf-py still supplies the type tables, and
stays the dequantization oracle.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

_SCALARS = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32,
            7: np.bool_, 10: np.uint64, 11: np.int64, 12: np.float64}
_STRING, _ARRAY = 8, 9


@dataclasses.dataclass(frozen=True)
class Tensor:
    name: str
    type: GGMLQuantizationType
    shape: tuple[int, ...]           # torch order (slowest dimension first)
    data: np.ndarray                 # uint8 [rows, row_bytes] for block types, else the element type


class _Cursor:
    def __init__(self, buf: np.ndarray) -> None:
        self.buf, self.at = buf, 0

    def scalar(self, dtype: Any) -> Any:
        v = self.buf[self.at:self.at + np.dtype(dtype).itemsize].view(dtype)[0]
        self.at += np.dtype(dtype).itemsize
        return v.item()

    def string(self) -> str:
        n = self.scalar(np.uint64)
        s = self.buf[self.at:self.at + n].tobytes().decode("utf-8", errors="replace")
        self.at += n
        return s

    def value(self, kind: int) -> Any:
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            elem, n = self.scalar(np.uint32), self.scalar(np.uint64)
            if elem == _STRING:
                return [self.string() for _ in range(n)]
            dtype = np.dtype(_SCALARS[elem])
            out = self.buf[self.at:self.at + n * dtype.itemsize].view(dtype).tolist()
            self.at += n * dtype.itemsize
            return out
        return self.scalar(_SCALARS[kind])


def read(path: str | pathlib.Path) -> tuple[dict[str, Any], list[Tensor]]:
    """The file's metadata (key -> value) and its tensors (views into the memory map; copy-on-write, so torch can
    wrap them without a copy and the file is never written)."""
    buf = np.memmap(path, dtype=np.uint8, mode="c")
    cur = _Cursor(buf)
    cur.at = 4
    if buf[:4].tobytes() != b"GGUF" or cur.scalar(np.uint32) < 2:
        raise ValueError(f"{path}: not a GGUF v2+ file")
    n_tensors, n_fields = cur.scalar(np.uint64), cur.scalar(np.uint64)
    fields = {}
    for _ in range(n_fields):
        key = cur.string()
        fields[key] = cur.value(cur.scalar(np.uint32))
    infos = []
    for _ in range(n_tensors):
        name = cur.string()
        dims = [cur.scalar(np.uint64) for _ in range(cur.scalar(np.uint32))]
        infos.append((name, GGMLQuantizationType(cur.scalar(np.uint32)), dims, cur.scalar(np.uint64)))
    align = fields.get("general.alignment", 32)
    start = -(-cur.at // align) * align
    tensors = []
    for name, kind, dims, offset in infos:
        block, size = GGML_QUANT_SIZES[kind]
        row_bytes = dims[0] // block * size
        rows = int(np.prod(dims[1:], dtype=np.int64))
        data = buf[start + offset:start + offset + rows * row_bytes].reshape(rows, row_bytes)
        tensors.append(Tensor(name, kind, tuple(reversed(dims)), data))
    return fields, tensors
