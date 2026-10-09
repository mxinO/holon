# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
#
# Holon (spike S3): vllm.platforms / env-var switches removed (NVIDIA only, TMA off, no FLA_* env), tensor_cache
# removed, and triton.autotune replaced by `autotune` below: one pinned config per kernel (design §5.6 rule 7).
import contextlib
import functools
from collections.abc import Callable

import torch
import triton

# Default chunk size used across FLA triton kernels (chunk, chunk_o, etc.)
FLA_CHUNK_SIZE = 64

is_amd = False
is_nvidia_hopper = False  # only consulted for chunk_o's warp list; Hopper is not a Holon v0 target
is_tma_supported = False
is_gather_supported = hasattr(triton.language, "gather")

# Pinned configs: one table for sm_86 and sm_120, picked from the per-arch winners of S3's tune.py
# (spikes/s3_gdn/results/tune_*.json) at Qwen3.5-9B / Qwen3.8-27B shapes, T=8192.
PINNED: dict[str, dict] = {
    "chunk_local_cumsum_scalar_kernel": dict(kwargs={}, num_warps=2, num_stages=3),
    "chunk_scaled_dot_kkt_fwd_kernel": dict(kwargs={"BK": 64}, num_warps=4, num_stages=2),
    "merge_16x16_to_64x64_inverse_kernel": dict(kwargs={}, num_warps=2, num_stages=3),
    "recompute_w_u_fwd_kernel": dict(kwargs={}, num_warps=2, num_stages=3),
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": dict(kwargs={"BV": 32}, num_warps=4, num_stages=3),
    "chunk_fwd_kernel_o": dict(kwargs={"BK": 64, "BV": 64}, num_warps=4, num_stages=2),
}


def autotune(configs: list, key: list[str], **_: object):
    """Drop-in for `triton.autotune`: one config (PINNED, else the first listed), so nothing is benchmarked at the
    first call and every run launches the same configuration."""

    def wrap(fn):
        pinned = PINNED.get(fn.__name__)
        return triton.autotune(configs=[triton.Config(**pinned) if pinned else configs[0]], key=key)(fn)

    return wrap


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.accelerator.device_index(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    # vLLM's per-arch table reduced to the arch="none" threshold it is always called with (102400 B). False on
    # both sm_86 and sm_120 (99 KiB opt-in shared memory), which selects the smaller BKV_LIST / BS_LIST.
    try:
        max_shared = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)["max_shared_mem"]
        return max_shared >= 102400
    except Exception:
        return False
