"""Tier B's floor emulates llama.cpp's int8 matmuls with offsets read from the file's blocks, which keep llama.cpp's
tiled value-head order, while the floor's HF weights are untiled (`gguf.hf_state`): the offsets must be permuted
the same way. On the real UD-IQ3_S file (skipped without it): emulating a tensor in the file's order and in HF's
order gives the same result, for a column permutation (`ssm_out`, Q4_K) and a row one (`attn_qkv`, Q2_K)."""
import glob
import os

import gguf
import numpy as np
import pytest
import torch
from holon_judge import gguf as judge_gguf
from holon_judge import q8_1

FILE = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-UD-IQ3_S.gguf"))
pytestmark = pytest.mark.skipif(not FILE, reason="needs the Qwen3.8-27B UD-IQ3_S file")


@pytest.mark.parametrize("name", ["blk.0.ssm_out.weight", "blk.14.attn_qkv.weight"])
def test_the_floors_offsets_follow_the_untiled_weights(name):
    reader = gguf.GGUFReader(FILE[0])
    f = judge_gguf._qwen35(reader.fields)
    t = next(t for t in reader.tensors if t.name == name)
    kind, blocks = t.tensor_type.name.lower(), np.asarray(t.data)
    g = q8_1.blocking(kind, q8_1.MMQ).sum_group
    assert g, f"{kind} has no stored-sum offsets on mmq: nothing to check"
    w = torch.from_numpy(gguf.quants.dequantize(blocks, t.tensor_type).astype(np.float64))
    offs = q8_1.offsets(blocks, kind, g)
    x = torch.randn(3, w.shape[1], generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    tiled = q8_1.matmul(x, w, None, kind, q8_1.MMQ, offs)
    order = lambda a: judge_gguf._qwen35_order(name, a, f)                        # noqa: E731
    x_hf = judge_gguf._untile(x, f, 1) if name.endswith("ssm_out.weight") else x
    if name.endswith("ssm_out.weight"):            # columns: the input is in the order the weights take it
        got = q8_1.matmul(x_hf, order(w), None, kind, q8_1.MMQ, order(offs))
        torch.testing.assert_close(got, tiled, rtol=1e-12, atol=1e-9)
    else:                                          # rows: the output comes out untiled
        got = q8_1.matmul(x, order(w), None, kind, q8_1.MMQ, order(offs))
        torch.testing.assert_close(got, judge_gguf._untile(tiled, f, 1, start=2 * f["hk"] * f["dk"]),
                                   rtol=1e-12, atol=1e-9)
    assert not torch.allclose(q8_1.matmul(x_hf, order(w), None, kind, q8_1.MMQ, offs), got)   # plant: unpermuted
