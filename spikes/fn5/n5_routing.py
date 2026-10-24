"""FN step 5, N5 (docs/v1-flash-next.md §8): Holon's own decode routing on the RTX PRO 6000, all-resident, written in
Strata's `--dump-routing` record format (int32 layer, int32 k, k x int32 expert ids, k x float32 weights; one record
per layer per decode token), so `spikes/fn0/routing.py` and N3's roofline (`n3_roofline.py --trace`) read it as they
read Strata's. Requests: D11's corpus spans (`bench/offload_ctx.py --corpus fn`) at 1K and 4K, each generating
`--tokens` greedily, one at a time, eager (the hooks see every decode step).

    python spikes/fn5/n5_routing.py --model <...-00001-of-00002.gguf> --out <routing.bin> [--tokens 256]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    corpus = offload_ctx.fn_corpus()
    engine = Engine(EngineConfig(a.model, max_seqs=1, max_model_len=8192, max_step_tokens=8192, cuda_graphs=False,
                                 prefix_cache=False))
    records: list[np.ndarray] = []
    for i, layer in enumerate(engine.executor.model.layers):
        moe = layer.ffn.parts[0]
        route = moe._route

        def traced(x, route=route, i=i):
            weights, ids, xin = route(x)
            if x.shape[0] == 1:                              # a decode token
                k = ids.shape[-1]
                records.append(np.concatenate([np.array([i, k], np.int32), ids[0].int().cpu().numpy(),
                                               weights[0].float().cpu().numpy().view(np.int32)]))
            return weights, ids, xin
        moe._route = traced
    params = SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True)
    n = len(corpus)
    for c, context in enumerate((1024, 4096)):
        for k in range(3):
            start = (k * (n // 6) + c * (n // 18)) % n
            engine.generate([offload_ctx.span(corpus, start, context)], params)
    layers = len(engine.executor.model.layers)
    engine.close()
    np.concatenate(records).tofile(a.out)
    print(f"{len(records) // layers} decode tokens -> {a.out}")


if __name__ == "__main__":
    main()
