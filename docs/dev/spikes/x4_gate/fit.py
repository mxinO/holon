"""The offload store's natural fit (docs/dev/v1-offload.md D9): what its arenas hold when the expert cache fills the VRAM
left above a paged floor of `--floor` tokens (default: the context), batch 1. Prints each class's cache slots and the
`moe.cache` value that pins them (D4: a lock pins the slot counts through the store's own option), as JSON.

    python spikes/x4_gate/fit.py <model.gguf> [--ctx 33792] [--floor N] [--step 4096] [--cache '<moe.cache>'] \
        [--opts k=v ...]
"""
import argparse
import json

import torch
from holon.core.engine import Engine, EngineConfig
from holon_std.moe.offload import KEY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=33792)
    ap.add_argument("--floor", type=int)
    ap.add_argument("--step", type=int, default=4096, help="max_step_tokens")
    ap.add_argument("--cache", default="")
    ap.add_argument("--host-memory", type=int, default=20 << 30)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": a.cache} | dict(o.split("=", 1) for o in a.opts)
    engine = Engine(EngineConfig(model=a.model, max_seqs=1, max_model_len=a.ctx, paged_min_tokens=a.floor,
                                 max_step_tokens=a.step, host_memory=a.host_memory, prefix_cache=False,
                                 plugin_options=opts))
    store = engine.ctx.shared_objects[KEY]
    classes = {c.name: {"unit_bytes": c.unit_bytes, "pieces": len(c.pieces), "null": c.null, "staging": c.staging,
                        "cache_slots": c.size - c.null - c.staging, "arena_slots": c.size}
               for c in store.classes.values()}
    resident = (store.residency.host >= 0).all(-1)
    free, total = torch.cuda.mem_get_info()
    out = {"ctx": a.ctx, "floor": a.floor or a.ctx, "max_step_tokens": a.step, "cache_option": a.cache,
           "pin": ",".join(f"{n}={c['cache_slots']}" for n, c in classes.items()), "classes": classes,
           "cached_experts": int(resident.sum()), "experts": int(resident.size),
           "cached_share": round(float(resident.mean()), 4),
           "cached_bytes": sum(c["cache_slots"] * c["unit_bytes"] for c in classes.values()),
           "expert_bytes": sum(c["pieces"] * c["unit_bytes"] for c in classes.values()),
           "pools": {n: {"capacity": p.capacity, "unit_bytes": p.unit_bytes} for n, p in engine.ledger.pools.items()},
           "free_mib": free >> 20, "total_mib": total >> 20}
    out["cached_byte_share"] = round(out["cached_bytes"] / out["expert_bytes"], 4)
    print("FIT " + json.dumps(out), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
