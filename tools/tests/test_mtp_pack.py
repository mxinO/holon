"""`holon_tools mtp-pack` on a tiny Strata-format pack: the ten dense projections become Q8_0 within half a step of
their BF16 values, every other tensor and the metadata are copied byte for byte, and a converted pack is refused. (On
Strata's real pack the output's sha256 is the one FN's locks name, `mtp-q2_0-q8@a55280a0417d4b19`.)"""
import gguf
import numpy as np
import pytest
from holon_tools import mtp_pack
from holon_tools.__main__ import main

BF16, Q8_0 = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.Q8_0
KEPT = {"mtp.layers.0.mlp.gate.weight": BF16, "mtp.norm.weight": gguf.GGMLQuantizationType.F32,
        "mtp.layers.0.mlp.experts.down_proj.weight": Q8_0}


def _bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_mtp_pack(tmp_path, capsys):
    rng = np.random.default_rng(0)
    src = tmp_path / "mtp-q2_0.gguf"
    w = gguf.GGUFWriter(src, "qwen4exp")
    w.add_uint32("qwen4exp.block_count", 1)
    w.add_array("tokenizer.ggml.tokens", ["a", "b"])
    kinds = {(f"mtp.{n}" if n.startswith("fc_") else f"mtp.layers.0.{n}"): BF16 for n in mtp_pack.Q8} | KEPT
    tensors = {}
    for name, kind in sorted(kinds.items()):
        x = rng.standard_normal((3, 64)).astype(np.float32)
        tensors[name] = data = {BF16: _bf16(x), Q8_0: gguf.quants.quantize(x, Q8_0)}.get(kind, x)
        w.add_tensor(name, data, raw_dtype=kind)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    out = tmp_path / "mtp-q2_0-q8.gguf"
    assert main(["mtp-pack", str(src), str(out)]) == 0
    assert "10 tensors in Q8_0" in capsys.readouterr().out
    a, b = gguf.GGUFReader(src), gguf.GGUFReader(out)
    assert {f.name: f.contents() for f in a.fields.values() if not f.name.startswith("GGUF.")} == \
        {f.name: f.contents() for f in b.fields.values() if not f.name.startswith("GGUF.")}
    got = {t.name: t for t in b.tensors}
    assert got.keys() == tensors.keys()
    for name, data in tensors.items():
        t = got[name]
        if mtp_pack.short(name) in mtp_pack.Q8:
            assert t.tensor_type == Q8_0
            x = (data.astype(np.uint32) << 16).view(np.float32)
            y = gguf.quants.dequantize(np.asarray(t.data), Q8_0).reshape(x.shape)
            step = np.abs(x).reshape(-1, 32).max(1, keepdims=True) / 127        # a 32-block's step, its scale fp16
            assert (np.abs(y - x).reshape(-1, 32) <= step * (0.5 + 127 * 2.0**-11)).all()
        else:
            assert t.tensor_type == KEPT[name] and np.asarray(t.data).tobytes() == data.tobytes()

    with pytest.raises(SystemExit, match="not BF16"):
        main(["mtp-pack", str(out), str(tmp_path / "again.gguf")])
