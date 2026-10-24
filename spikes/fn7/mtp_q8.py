"""FN step 7: the MTP pack with its head's large dense projections in Q8_0, as Strata drafts with them (its
`tools/mtp_rt.py`: fc_embedding, fc_hidden, q / k / v / o, the indexer's fused q / k and the shared expert, quantized
by ggml's reference rounding; the hyper-connection and router weights stay BF16, the experts Q2_0). Every other tensor
and the metadata copied as they are; Holon reads the Q8_0 tensors as any GGUF's (`mtp.path`), so only the drafts
change — the verify decides the output. `--check <Strata rt dir>`: each Q8_0 tensor's bytes against Strata's
`dense.bin`. Usage: mtp_q8.py <mtp-q2_0.gguf> <out.gguf> [--check <rt dir>]"""
from __future__ import annotations

import argparse
import pathlib

import gguf
import numpy as np

Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def short(name: str) -> str:
    """Strata's name for a pack tensor (`mtp.` and `layers.0.` dropped)."""
    s = name.removeprefix("mtp.")
    return s.removeprefix("layers.0.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--check")
    a = ap.parse_args()
    r = gguf.GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(a.out, arch)
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        sub = f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=sub)
    quantized = {}
    for t in r.tensors:
        data = np.asarray(t.data)
        if short(t.name) in Q8:
            assert t.tensor_type == gguf.GGMLQuantizationType.BF16, (t.name, t.tensor_type)
            f32 = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            q = gguf.quants.quantize(f32, gguf.GGMLQuantizationType.Q8_0)
            quantized[short(t.name)] = q
            w.add_tensor(t.name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.out}: {len(quantized)} tensors in Q8_0")
    if a.check:
        rt = pathlib.Path(a.check)
        blob = (rt / "dense.bin").read_bytes()
        for line in (rt / "dense.txt").read_text().splitlines():
            name, kind, _, _, off, n = line.split()
            if kind == "q8_0":
                same = quantized[name].tobytes() == blob[int(off):int(off) + int(n)]
                print(f"  {name}: {'identical to' if same else 'DIFFERS from'} Strata's")
                if not same:
                    raise SystemExit(1)


if __name__ == "__main__":
    main()
