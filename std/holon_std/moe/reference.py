"""Reference for std.moe (oracles; also the floors' MoE block in the judge): softmax top-k routing, expert stacks
applied per (token, slot), the slot-ordered combine. fp32 throughout, as the op contracts specify."""
from __future__ import annotations

import torch

CLAMP = 6.1e-5            # llama.cpp's guard on the top-k sum (never reached by softmax: the sum is ≥ k/E)


def moe_topk(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The `moe_topk@1` oracle: `(weights [T, k] f32, ids [T, k] int32)`."""
    x = logits.float()
    x = torch.where(x.isnan(), torch.finfo(torch.float32).min, x)
    probs = x.softmax(-1)
    order = probs.sort(dim=-1, descending=True, stable=True)            # ties to the lower index
    weights, ids = order.values[:, :top_k], order.indices[:, :top_k]
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(CLAMP)
    return weights.contiguous(), ids.int().contiguous()


def expert_parts(parts: dict[str, torch.Tensor], experts: int, merged: bool) -> list[list[dict[str, torch.Tensor]]]:
    """A `linear_experts@2` stack's parts per expert, then per unit it was merged from (gate's, then up's when
    `merged`; else one): a part of rank ≥ 3 is per row, [E, rows, …], each unit taking its share of the expert's rows;
    one of rank ≤ 2 is per tensor, [E] or [E, 1], or merged [2, E] (gate's row, then up's). For a dequantizer that
    reads one unit's parts at a time."""
    units = 2 if merged else 1

    def part(p: torch.Tensor, e: int, u: int) -> torch.Tensor:
        if p.dim() >= 3:
            return p[e].chunk(units)[u]
        return p[u, e] if merged else p.reshape(experts)[e]
    return [[{r: part(p, e, u) for r, p in parts.items()} for u in range(units)] for e in range(experts)]


def linear_experts(x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, glu: str = "none",
                   operand_dtype: torch.dtype | None = None) -> torch.Tensor:
    """The `linear_experts@2` oracle over the stack's weights `weight` [E, out, in] (dequantized, or dense):
    `y [T, k, out_y]` f32. `operand_dtype`: a dense layout rounds the activation to the weight dtype first."""
    t, k = ids.shape
    xs = x.expand(t, k, x.shape[-1]) if x.shape[1] == 1 else x
    xs = xs.to(operand_dtype).float() if operand_dtype is not None else xs.float()
    y = xs.new_zeros(t, k, weight.shape[1])
    flat = ids.long()
    for e in flat.unique().tolist():                                    # one product per routed expert
        rows = flat == e
        y[rows] = xs[rows] @ weight[e].float().T
    if glu == "swiglu":
        gate, up = y.chunk(2, dim=-1)
        y = torch.nn.functional.silu(gate) * up
    elif glu != "none":
        raise ValueError(f"unknown glu {glu!r}")
    return y


def moe_combine(y: torch.Tensor, weights: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The `moe_combine@1` oracle: `Σ_j w[t, j] · y[t, j]` in f32, slot by slot, cast to `dtype`."""
    acc = torch.zeros(y.shape[0], y.shape[2], dtype=torch.float32, device=y.device)
    for j in range(y.shape[1]):
        acc += weights[:, j, None].float() * y[:, j].float()
    return acc.to(dtype)


def moe_resolve(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                limits: torch.Tensor, staging: int, ring: torch.Tensor | None = None,
                stage: torch.Tensor | None = None, plan: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """The `moe_resolve@1` oracle: `(slots [R, T, k], fetch [R, 2, F] (−1 past n_fetch), n_fetch [R], counts [E],
    routed [E], errors, cpu [T, k], jobs)`, `counts` the increments of this call, `routed` the experts it saw routed,
    `errors` the bits it sets, `jobs` the CPU's job rows (`cpu_share`; empty without a `plan`). `ring` defaults to
    none, `stage` to the staging slots k + i, `plan` to zero (no CPU share)."""
    from .kernels import DECODE, PADDING, RANGE

    t, k = ids.shape
    experts, roles = residency.shape
    ring = torch.full((experts, roles), -1) if ring is None else ring.long().cpu()
    stage = (k + torch.arange(staging)).expand(roles, staging) if stage is None else stage.long().cpu()
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    residency, host = residency.long().cpu(), host.long().cpu()
    live = (kinds[:, None] != PADDING) & (ids >= 0)                     # [T, k]: routed entries
    routed = ids.clamp_min(0)
    taken, cpu, jobs = cpu_share(ids, kinds, residency, ring, host, plan)
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    width = stage.shape[1]
    fetch = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    for r in range(roles):
        missing = sorted({e for e in routed[live].tolist() if residency[e, r] < 0 and ring[e, r] < 0} - taken)
        rank = {e: i for i, e in enumerate(missing)}
        n_fetch[r] = len(missing)
        for i, e in enumerate(missing[:width]):
            fetch[r, :, i] = torch.tensor([host[e, r] if host[e, r] >= 0 else -2 - e, stage[r, i]])
        for a, j in live.nonzero().tolist():
            e = int(routed[a, j])
            at = int(residency[e, r]) if residency[e, r] >= 0 else int(ring[e, r])
            if e in taken:
                continue                                                # the CPU's: null slot j
            if e in rank:
                slots[r, a, j] = stage[r, min(rank[e], width - 1)]
            elif k + staging <= at < int(limits[r]):
                slots[r, a, j] = at
            else:                                                       # corrupt: null slot j, flagged
                errors |= RANGE
    counted = routed[live & (kinds[:, None] == DECODE)]
    counts = torch.bincount(counted, minlength=experts)
    routed_ = torch.zeros(experts, dtype=torch.int32)
    routed_[routed[live]] = 1
    routed_[sorted(taken)] |= 4                                         # the CPU's
    return slots.int(), fetch.int(), n_fetch.int(), counts.int(), routed_, errors, cpu, jobs


def moe_tiers(ids: torch.Tensor, kinds: torch.Tensor, group: int, fetch: torch.Tensor, n_fetch: torch.Tensor,
              stage: torch.Tensor, records: list[tuple[int, int, int]], residency: torch.Tensor, host: torch.Tensor,
              piece: torch.Tensor, classes: torch.Tensor, owner: torch.Tensor, capacity: torch.Tensor,
              heat: torch.Tensor | None) -> dict:
    """The `moe_tiers@1` oracle, one layer call after its resolve (`records`: the call's delta records): `fetch`,
    `residency`, `host`, `owner` after it, `io` per role (n, then (piece, unit) pairs, as rows), `moves` [R, 4, F]
    (demotions: slot, unit; moves: slot, slot) and `n_moves` [R, 2]."""
    from .kernels import DECODE, PADDING

    g, (_, experts, roles) = group, residency.shape
    fetch, residency, host, owner = fetch.clone(), residency.clone(), host.clone(), owner.clone()
    width = stage.shape[1]
    live = (kinds.long()[:, None] != PADDING) & (ids >= 0)
    routed = {int(e) for e in ids[live].tolist()}
    decode = {int(e) for e in ids[live & (kinds.long()[:, None] == DECODE)].tolist()}
    cls = classes[g].tolist()
    moves = torch.full((roles, 4, width), -1, dtype=torch.int32)
    n_moves = torch.zeros(roles, 2, dtype=torch.int32)
    placed: set[tuple[int, int]] = set()

    def free_units() -> list[int] | None:
        units = [next((u for u in range(int(capacity[c])) if owner[c, u] < 0), None) for c in cls]
        return None if None in units else units

    def coldest(than: float) -> tuple[int, int] | None:
        """RAM's coldest whole expert of the layer's classes, strictly colder than `than` (not routed, not placed)."""
        if heat is None:
            return None
        best = None
        for gg in range(residency.shape[0]):
            if classes[gg].tolist() != cls:
                continue
            for e in range(experts):
                if (residency[gg, e] == -1).all() and (gg, e) not in placed and not (gg == g and e in routed):
                    key = (float(heat[gg, e]), gg, e)
                    best = key if best is None or key < best else best
        return None if best is None or not best[0] < than else best[1:]

    def place(e: int, units: list[int] | None) -> list[int] | None:
        """Expert e of this layer into RAM: `units` if given, else free units, else the coldest's; None: the SSD."""
        if units is None:
            units = free_units()
        if units is None and (x := coldest(float(heat[g, e]) if heat is not None else 0.0)) is not None:
            units = host[x[0], x[1]].tolist()
            residency[x[0], x[1]] = -2
            host[x[0], x[1]] = -1
        if units is None:
            residency[g, e], host[g, e] = -2, -1
            return None
        for c, u in zip(cls, units, strict=True):
            owner[c, u] = g * experts + e
        residency[g, e], host[g, e] = -1, torch.tensor(units, dtype=host.dtype)
        placed.add((g, e))
        return units

    admitted = {a for _, a, _ in records if a >= 0}
    staged = (n_fetch.long() - len(admitted)).tolist()
    entry = {(r, int(fetch[r, 1, i])): i for r in range(roles) for i in range(min(int(n_fetch[r]), width))}
    j = 0
    for _, a, b in records:                                              # 1. admissions, in record order
        if a < 0:
            continue
        slots = [int(residency[g, a, r]) for r in range(roles)]          # a's new slots (b's, or empty ones)
        source = [int(fetch[r, 0, entry[r, slots[r]]]) for r in range(roles)]
        for r, c in enumerate(cls):
            if b >= 0:                                                   # into a victim's slot: staged, then moved
                fetch[r, 1, entry[r, slots[r]]] = stage[r, staged[r] + j]
                moves[r, 2:4, n_moves[r, 1]] = torch.tensor([int(stage[r, staged[r] + j]), slots[r]])
                n_moves[r, 1] += 1
            if source[r] >= 0:
                owner[c, source[r]] = -1                                 # (b takes it, below, if it came from RAM)
        if b >= 0:
            units = place(b, source if all(u >= 0 for u in source) else None)
            if units is not None:
                for r in range(roles):
                    moves[r, 0:2, n_moves[r, 0]] = torch.tensor([slots[r], units[r]])
                    n_moves[r, 0] += 1
            j += 1
        host[g, a] = -1
    io = [[] for _ in range(roles)]
    ssd = sorted({-2 - int(fetch[r, 0, i]) for r in range(roles) for i in range(min(int(n_fetch[r]), width))
                  if int(fetch[r, 0, i]) <= -2})
    for e in ssd:                                                        # 2. keeps, 3. the reads
        kept = None
        if e not in admitted and e in decode:
            kept = place(e, None)
        for r, c in enumerate(cls):
            i = next(i for i in range(min(int(n_fetch[r]), width)) if int(fetch[r, 0, i]) == -2 - e)
            unit = kept[r] if kept is not None else int(capacity[c]) + i
            io[r].append((int(piece[g, e, r]), unit))
            fetch[r, 0, i] = unit
    return {"fetch": fetch, "residency": residency, "host": host, "owner": owner, "io": io, "moves": moves,
            "n_moves": n_moves}


def cpu_share(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, ring: torch.Tensor,
              host: torch.Tensor, plan: torch.Tensor | None) -> tuple[set[int], torch.Tensor, list[list[int]]]:
    """`moe_resolve@1`'s CPU share (F5b): the experts the CPU takes, `cpu [T, k]` (t·k + j for their entries, else
    −1) and the job rows `(expert, gate/up unit, down unit, n_tok, entries…)` by expert id."""
    from .kernels import PADDING

    t, k = ids.shape
    live = (kinds.long()[:, None] != PADDING) & (ids >= 0)
    best: dict[int, int] = {}
    for a, j in live.nonzero().tolist():
        e = int(ids[a, j])
        best[e] = min(best.get(e, k), j)
    held = ((residency == -1) & (ring < 0)).all(-1)                     # missing everywhere, in the read tier
    cands = sorted((best[e], e) for e in best if held[e])
    n = len(cands)
    c = 0 if plan is None or not n else min(int(plan[n]), n)
    taken = {e for _, e in cands[n - c:]}
    cpu = torch.full((t, k), -1, dtype=torch.int32)
    jobs = []
    for e in sorted(taken):
        entries = [a * k + j for a, j in live.nonzero().tolist() if int(ids[a, j]) == e]
        for x in entries:
            cpu.view(-1)[x] = x
        jobs.append([e, int(host[e, 0]), int(host[e, 1]), len(entries), *entries])
    return taken, cpu, jobs


def cpu_experts(jobs: torch.Tensor, x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, rows: torch.Tensor,
                gate_up_type: str, down_type: str, inter: int, top_k: int) -> torch.Tensor:
    """The `cpu_experts@1` oracle: `rows` with each job's entries written (float64 sums, written as f32): weights
    dequantized by gguf-py, activations quantized as ggml-cpu's `from_float` of the type's `vec_dot_type` does
    (`quantize_cpu`)."""
    import gguf

    out = rows.clone()
    for job in jobs[1:1 + int(jobs[0, 0])].tolist():
        _, u, v, n, *entries = job
        gu = torch.from_numpy(gguf.quants.dequantize(gate_up[u].numpy(), _ggml(gate_up_type))).double()
        dn = torch.from_numpy(gguf.quants.dequantize(down[v].numpy(), _ggml(down_type))).double()
        gu, dn = gu.reshape(2 * inter, -1), dn.reshape(x.shape[1], inter)
        for entry in entries[:n]:
            xq = quantize_cpu(x[entry // top_k].float(), gate_up_type)
            h = (gu @ xq).float()                                       # f32, as ggml-cpu's dot writes
            g, up = h[:inter], h[inter:]
            act = g / (1.0 + torch.exp(-g.double()).float()) * up        # expf, correctly rounded
            out[entry] = (dn @ quantize_cpu(act, down_type)).float()
    return out


# ggml-cpu's activation type per weight type (`type_traits_cpu[...].vec_dot_type`) for the types the CPU path serves
Q8_0_DOT = {"q4_0", "q5_0", "q8_0", "iq4_nl"}
Q8_K_DOT = {"q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s",
            "iq4_xs"}


def _ggml(layout: str):
    import gguf

    return gguf.GGMLQuantizationType[layout.split("/")[0].removeprefix("ggml.").upper()]


def quantize_cpu(x: torch.Tensor, layout: str) -> torch.Tensor:
    """x (f32 [n]) as ggml-cpu quantizes it for a dot with weights of `layout` (`ggml.<type>/…`), dequantized to
    float64: Q8_K (`quantize_row_q8_K_ref`: per 256 values iscale = −127 / the signed value of largest magnitude, the
    first one, q = nearest(iscale · x) up to 127, d = 1 / iscale) or Q8_0 (the x86 path: per 32 values d = amax / 127
    stored fp16, q = nearest(x · 127 / amax), ties to even)."""
    kind = layout.split("/")[0].removeprefix("ggml.")
    x = x.float()
    if kind in Q8_K_DOT:
        b = x.reshape(-1, 256)
        first = b.abs().argmax(-1, keepdim=True)                       # the first of the largest magnitudes
        top = b.gather(-1, first)
        iscale = torch.where(top == 0, 0.0, -127.0 / torch.where(top == 0, 1.0, top))
        q = torch.round(iscale * b).clamp(max=127)
        d = torch.where(top == 0, 0.0, 1.0 / torch.where(iscale == 0, 1.0, iscale))
        return (q.double() * d.double()).reshape(x.shape)
    if kind in Q8_0_DOT:
        b = x.reshape(-1, 32)
        amax = b.abs().amax(-1, keepdim=True)
        inv = torch.where(amax == 0, 0.0, 127.0 / torch.where(amax == 0, 1.0, amax))
        q = torch.round(b * inv)
        return (q.double() * (amax / 127.0).half().double()).reshape(x.shape)
    raise ValueError(f"cpu_experts@1: no CPU activation type for {layout}")


def moe_patch_rows(y: torch.Tensor, rows: torch.Tensor, cpu: torch.Tensor) -> torch.Tensor:
    """The `moe_patch_rows@1` oracle: `y [T, k, H]` with each CPU entry's row."""
    out = y.clone()
    at = cpu.reshape(-1).long()
    flat = out.view(-1, out.shape[-1])
    flat[at >= 0] = rows[at[at >= 0]].to(out.device, out.dtype)
    return out


def moe_publish_rows(x: torch.Tensor, cpu: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The `moe_patch_rows@1` `publish` oracle: `out` with x's rows (f32) of the tokens that have a CPU entry."""
    res = out.clone()
    used = (cpu >= 0).any(-1).cpu()
    res[:len(x)][used] = x.float().cpu()[used].to(res.device)
    return res


def cpu_task(task) -> None:
    """The `cpu_task@1` oracle: the task, run now."""
    task()


def copy_task(roles) -> None:
    """`cpu_task@1`'s copy task, run now: per role (descriptors [P, X, 4], units [U, unit], requests [1 + F, 2]) each
    requested piece's extents (fd, direct, offset, nbytes) read in order into its unit (an O_DIRECT descriptor through
    its aligned superset, into page-aligned memory)."""
    import mmap
    import os

    for desc, units, req in roles:
        for piece, unit in req[1:1 + int(req[0, 0])].tolist():
            at = 0
            for fd, direct, offset, nbytes in desc[piece].tolist():
                if not nbytes:
                    continue
                lo, hi = (offset // 4096 * 4096, -(-(offset + nbytes) // 4096) * 4096) if direct else \
                    (offset, offset + nbytes)
                with mmap.mmap(-1, hi - lo) as buf:
                    got = os.preadv(fd, [buf], lo)
                    if got < offset + nbytes - lo:
                        raise OSError(f"copy task: {nbytes} bytes at {offset}, {got - (offset - lo)} read")
                    data = bytes(buf[offset - lo:offset - lo + nbytes])
                units[unit, at:at + nbytes] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                at += nbytes


def expert_fetch(fetch: torch.Tensor, n: int, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """The `expert_fetch@1` oracle: `dst` with units `src[fetch[0, i]]` in slots `fetch[1, i]` for i < min(n,
    staging)."""
    out = dst.clone()
    for unit, slot in fetch[:, :n].T.tolist():
        out[slot] = src[unit].to(out.device)
    return out
