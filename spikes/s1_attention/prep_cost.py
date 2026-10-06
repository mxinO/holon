"""Per-step device cost of writing FlashInfer CSR metadata from the SeqTable (one Triton launch, in-graph).

  CUDA_VISIBLE_DEVICES=0 python prep_cost.py
"""

import torch

from common import GEOMS, graph_time_us, make_paged_kv, save
from fi_kernels import FIDecode


def main():
    rows = []
    g = GEOMS["qwen3_8b"]
    for page in (16, 64):
        for batch in (1, 8, 64):
            kv = make_paged_kv(g, [32768] * batch, page)[0]
            fi = FIDecode(g, kv, True, "device")
            us = graph_time_us([fi.prepare] * 32)        # amortize graph launch over 32 launches
            rows.append({"page": page, "batch": batch, "ctx": 32768, "us": us})
            print(rows[-1], flush=True)
            del kv, fi
            torch.cuda.empty_cache()
    print("saved", save("prep_cost", {"rows": rows}))


if __name__ == "__main__":
    main()
