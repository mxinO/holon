"""GGUF models end to end on the CPU (the test-only reference kernels): a tiny random Qwen3 written as a GGUF with
quantized weights serves exactly what the same model serves from its dequantized weights (M0.1 step 1)."""
import re

import gguf
import numpy as np
import pytest
import torch
from holon.api import QTensor, SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import PluginError
from holon_testing.gguf_files import write
from safetensors.torch import save_file

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures

Q = gguf.GGMLQuantizationType


def _engine(model, **kw) -> Engine:
    cfg = dict(model=str(model), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20)
    return Engine(EngineConfig(**cfg | kw))


def _prompt_logprobs(engine: Engine, prompts) -> list:
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=2)) for p in prompts]
    out = {}
    while engine.has_work():
        out |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    return [out[r] for r in rids]


@pytest.mark.parametrize(("qtype", "types"), [(Q.Q8_0, None), (Q.Q4_0, None),
                                              (Q.Q8_0, {"attn_v": Q.Q4_0, "ffn_up": Q.Q4_0})])
def test_a_quantized_gguf_serves_what_its_dequantized_weights_serve(tiny_checkpoint, tiny_case, tmp_path, qtype,  # noqa: F811
                                                                    types):
    from holon_judge.gguf import engine_shapes

    prompts, _ = tiny_case
    file = tmp_path / f"tiny-{qtype.name}.gguf"
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "config.json").write_text((tiny_checkpoint / "config.json").read_text())
    save_file({k: torch.from_numpy(np.ascontiguousarray(v))
               for k, v in write(tiny_checkpoint, file, qtype, types=types).items()}, dense / "model.safetensors")
    ours, theirs = _engine(file), _engine(dense)
    assert ours.source.kind == "gguf" and ours.source.arch == "gguf.qwen3"
    assert ours.generate(prompts, GREEDY) == theirs.generate(prompts, GREEDY)
    assert _prompt_logprobs(ours, prompts) == _prompt_logprobs(theirs, prompts)      # bit for bit
    # One quantized matmul per merged group (q/k/v, gate/up) plus o, down and the head; the merged-away
    # projections bound no kernel, so they leave no site in a lock.
    linears = [k for k in ours.ctx.kernels if k.endswith("/qlinear@2")]
    assert any(k.endswith("/qembedding@1") for k in ours.ctx.kernels)
    outs = sorted(int(re.search(r",out=(\d+)", k)[1]) for k in linears)
    if types is None:
        assert outs == [64, 64, 64 + 2 * 32, 2 * 128, 512], outs     # o, down, qkv, gate+up, lm_head
    # Projections of other types stay apart (q/k merge, v alone; gate, up): the judge's per-shape floors read the
    # same matmul shapes (docs/m07-perf.md D2).
    sites = {(f["layout"], f["in"], f["out"]) for f in (s["facts"] for k, s in ours.ctx.kernels.items()
                                                        if k.endswith("/qlinear@2"))}
    assert sites == {(f"ggml.{t}/gguf.native", k, o) for t, k, o in engine_shapes(file).values()}


def test_quantized_parts_say_where_their_bytes_lie_in_the_file(tiny_checkpoint, tmp_path):  # noqa: F811
    """Weight extents (docs/v1-offload.md S4): each quantized tensor's blocks read back bitwise from its extent, so an
    offload tier can read them again from the file."""
    from holon_std.weights.gguf.source import GGUFSource

    file = tmp_path / "tiny.gguf"
    write(tiny_checkpoint, file, Q.Q8_0, types={"ffn_down": Q.Q4_0})
    quantized = [q for _, q in GGUFSource(file).tensors() if isinstance(q, QTensor)]
    assert len(quantized) > 4 and all(q.extents.keys() == q.parts.keys() for q in quantized)
    with open(file, "rb") as f:
        for q in quantized:
            extent = q.extents["blocks"]
            f.seek(extent.offset)
            assert extent.path == str(file) and f.read(extent.nbytes) == q.parts["blocks"].numpy().tobytes()


def test_a_split_file_is_one_checkpoint(tiny_checkpoint, tiny_case, tmp_path):  # noqa: F811
    """A split GGUF (FN D14), opened by its first file: the whole file's name, configuration and tensors, each extent
    in its own shard; it serves what the whole file serves. Refused: a later shard named, shards out of their order
    (`split.no`), a missing shard."""
    from holon_judge.gguf import shards
    from holon_std.weights.gguf.source import GGUFSource

    whole, split = tmp_path / "tiny.gguf", tmp_path / "split" / "tiny.gguf"
    split.parent.mkdir()
    write(tiny_checkpoint, whole, Q.Q8_0)
    write(tiny_checkpoint, split, Q.Q8_0, split=5)
    files = sorted(split.parent.iterdir())
    a, b = GGUFSource(whole), GGUFSource(files[0])
    assert len(files) > 2 and b.files == files == shards(files[0]) and (b.name, b.config()) == (a.name, a.config())
    got, want = dict(b.tensors()), dict(a.tensors())
    assert got.keys() == want.keys()
    for name, t in got.items():
        if not isinstance(t, QTensor):
            assert torch.equal(t, want[name])
            continue
        extent = t.extents["blocks"]
        with open(extent.path, "rb") as f:
            f.seek(extent.offset)
            assert f.read(extent.nbytes) == t.parts["blocks"].numpy().tobytes() == want[name].parts["blocks"].numpy(
                ).tobytes()
    assert {t.extents["blocks"].path for t in got.values() if isinstance(t, QTensor)} != {extent.path}
    prompts, _ = tiny_case
    assert _engine(files[0]).generate(prompts, GREEDY) == _engine(whole).generate(prompts, GREEDY)
    with pytest.raises(ValueError, match="by its first"):
        GGUFSource(files[1])
    files[1].rename(tmp_path / "x")
    files[2].rename(files[1])
    (tmp_path / "x").rename(files[2])
    with pytest.raises(ValueError, match="split.no 2 of .*, not shard 1"):
        GGUFSource(files[0])
    files[-1].unlink()
    with pytest.raises(ValueError, match="are missing"):
        GGUFSource(files[0])


def test_a_unit_that_cannot_read_a_layout_fails_at_load(tiny_checkpoint, tmp_path):  # noqa: F811
    file = tmp_path / "norms-quantized.gguf"
    write(tiny_checkpoint, file, Q.Q8_0, quantize_norms=True)          # RMSNorm has no quantized form
    with pytest.raises(PluginError, match="cannot read layout"):
        _engine(file)


def test_a_damaged_file_fails_when_opened(tiny_checkpoint, tmp_path):  # noqa: F811
    from holon_std.weights.gguf import reader

    file = tmp_path / "tiny.gguf"
    write(tiny_checkpoint, file, Q.Q8_0)
    data = file.read_bytes()
    reader.read(file)
    (tmp_path / "cut.gguf").write_bytes(data[:-100])                  # the last tensor runs past the end
    (tmp_path / "header.gguf").write_bytes(data[:200])                # the metadata itself is cut
    (tmp_path / "v9.gguf").write_bytes(data[:4] + (9).to_bytes(4, "little") + data[8:])
    for name, match in [("cut", "past the end"), ("header", "truncated"), ("v9", "v2 or v3")]:
        with pytest.raises(ValueError, match=match):
            reader.read(tmp_path / f"{name}.gguf")


def test_nvfp4_tensors_and_nan_mxfp4_scales_are_refused(tiny_checkpoint, tmp_path):  # noqa: F811
    """llama.cpp's NVFP4 GGUFs keep each tensor's scale apart, as a `.scale` tensor (docs/m06-lowp.md D1); an MXFP4
    e8m0 scale of 0xFF is NaN, refused as std.weights.hf refuses it."""
    from holon_std.weights.gguf import reader
    from holon_std.weights.gguf.source import GGUFSource

    file = tmp_path / "nvfp4.gguf"
    w = gguf.GGUFWriter(file, "qwen3")
    w.add_tensor("blk.0.ffn_down.weight", np.zeros((4, 36), np.uint8), raw_dtype=Q.NVFP4)     # 4 rows of 64
    w.add_tensor("blk.0.ffn_down.scale", np.ones(1, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match=r"NVFP4 tensors \(blk\.0\.ffn_down\.weight, …\) are not served from GGUF"):
        GGUFSource(file)
    file = tmp_path / "no-tokenizer.gguf"                              # (and a file without a tokenizer is no model)
    w = gguf.GGUFWriter(file, "qwen3")
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(4, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="no tokenizer in this file"):
        GGUFSource(file).tokenizer()
    file = tmp_path / "mxfp4.gguf"
    write(tiny_checkpoint, file, Q.Q8_0, types={"ffn_down": Q.MXFP4})
    down = next(t for t in reader.read(file)[1] if t.name == "blk.1.ffn_down.weight")
    with open(file, "r+b") as f:                                       # the last row's last block's scale
        f.seek(down.offset + down.rows * down.row_bytes - 17)
        f.write(b"\xff")
    with pytest.raises(ValueError, match=r"blk\.1\.ffn_down\.weight: an MXFP4 scale is NaN"):
        list(GGUFSource(file).tensors())


def test_the_reference_kernels_decode_mxfp4_as_gguf_py():
    """The reference kernels' vectorized MXFP4 decode (ref_kernels.ggml_mxfp4) is gguf-py's by value (its code 8 is
    +0, ours −0), over every scale byte (but NaN's 0xFF) and code."""
    from holon_testing.ref_kernels.kernels import ggml_mxfp4

    blocks = torch.randint(0, 256, (255, 3, 17), dtype=torch.uint8)
    blocks[:, :, 0] = torch.arange(255, dtype=torch.uint8)[:, None]
    want = gguf.quants.dequantize(blocks.reshape(255, -1).numpy(), Q.MXFP4)
    assert torch.equal(ggml_mxfp4(blocks.reshape(255, -1)), torch.from_numpy(want))
