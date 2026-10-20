"""The oracle of `state_copy@1` (kernels.py): one gather and one scatter of whole rows, on the views' device."""
from __future__ import annotations

import torch


def state_copy(src: torch.Tensor, src_rows: torch.Tensor, dst: torch.Tensor, dst_rows: torch.Tensor) -> None:
    if len(src_rows):
        dst[dst_rows.to(dst.device)] = src[src_rows.to(src.device)].to(dst.device)
