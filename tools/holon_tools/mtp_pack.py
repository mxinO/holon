"""`holon_tools mtp-pack`: Qwen3.8-Flash-Next's MTP pack as its locks name it (`mtp-q2_0-q8`,
docs/dev/v1-flash-next.md §21.11). The input is Strata's pack (its `tools/mtp_fetch.py fetch`, then
`tools/mtp_pack.py --experts q2_0`); the head's large dense projections — fc_embedding, fc_hidden, q / k / v / o, the
indexer's fused q / k and the shared expert — are rewritten from BF16 to Q8_0 by ggml's reference rounding, as
Strata's own draft head runs them (its `tools/mtp_rt.py`). The hyper-connection and router weights stay BF16, the
experts Q2_0, and every other tensor and the metadata are copied as they are. Holon reads the Q8_0 tensors as any
GGUF's (`mtp.path`), so only the drafts change: the verify decides the output."""
from __future__ import annotations

import hashlib
import pathlib

import gguf
import numpy as np

Q8 = frozenset({"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
                "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight"})


def short(name: str) -> str:
    """Strata's name for a pack tensor (`mtp.` and `layers.0.` dropped)."""
    return name.removeprefix("mtp.").removeprefix("layers.0.")


def convert(src: str | pathlib.Path, out: str | pathlib.Path) -> str:
    """Write `src`'s pack with the `Q8` tensors in Q8_0 to `out`; returns `out`'s sha256. ValueError if `src` is not
    a Strata pack (a `Q8` tensor missing or not BF16)."""
    src, out = pathlib.Path(src), pathlib.Path(out)
    if out.exists() and out.resolve() == src.resolve():
        raise ValueError(f"{out}: the output would overwrite the input")
    r = gguf.GGUFReader(src)
    names = {short(t.name): t for t in r.tensors}
    if missing := sorted(Q8 - names.keys()):
        raise ValueError(f"{src}: not Strata's MTP pack: no {missing}")
    if wrong := sorted(n for n in Q8 if names[n].tensor_type != gguf.GGMLQuantizationType.BF16):
        raise ValueError(f"{src}: {wrong} are not BF16 (converted already?)")
    w = gguf.GGUFWriter(out, r.fields["general.architecture"].contents())
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        sub = f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=sub)
    for t in r.tensors:
        data = np.asarray(t.data)
        if short(t.name) in Q8:
            f32 = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            w.add_tensor(t.name, gguf.quants.quantize(f32, gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    digest = hashlib.sha256()
    with out.open("rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()
