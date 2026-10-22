"""kernels.ggml_cpu's `cpu_experts@1` (L0, docs/v1-offload.md D7) on this host's CPU tier: every job's rows within the
oracle's budget (std.moe.reference: gguf-py's weights, ggml-cpu's activation quantization emulated, float64 sums: a
row may differ by a requantized SwiGLU value that sits at a rounding tie, which the oracle's sums put a hair to the
other side, so the budget is tight for nine rows in ten and one such flip's size for the rest),
and a token's row bitwise the same alone (one job, one thread) as grouped with other tokens and jobs at 1 and 7
threads — the batch-invariance check, which catches a kernel that splits K for grouped tokens (L0's own mutation
test); a malformed job table fails the task instead of reading past a buffer. Synthetic Q8_0 / Q4_0 experts always,
and Q8_0 / Q2_0 (FN's down type); the development vehicle's own classes (Q4_K with Q6_K or Q4_K down, UD-IQ3_XXS's)
and FN IQ2_XS's three (inter 640) when their files are cached. Prebuild links each tier in its own process."""
import pathlib

import gguf
import numpy as np
import pytest
import torch
from holon.core import platform
from holon_std.moe import reference
from huggingface_hub.constants import HF_HUB_CACHE

VEHICLE = pathlib.Path(HF_HUB_CACHE) / "models--unsloth--Qwen3-30B-A3B-GGUF" / "snapshots"
FN = pathlib.Path(HF_HUB_CACHE) / "models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF" / "snapshots"
CASES = [("synthetic", None, None), ("synthetic", None, "q2_0"),
         ("Q4_K_M", "Qwen3-30B-A3B-Q4_K_M.gguf", 0),   # Q6_K down
         ("Q4_K_M", "Qwen3-30B-A3B-Q4_K_M.gguf", 6), ("UD-IQ3_XXS", "Qwen3-30B-A3B-UD-IQ3_XXS.gguf", 0),
         # FN's IQ2_XS (inter 640, not a multiple of 256): IQ2_S, IQ2_XXS and IQ1_M gate/up with Q2_0 down
         *(("IQ2_XS", "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf", layer) for layer in (0, 1, 8))]
K, T, J = 8, 8, 16                       # top-k, tokens and jobs per run
BUDGET, FLIP = 1e-4, 2e-2               # |kernel − oracle| per row, relative to the oracle row's RMS


@pytest.fixture(scope="module")
def plugin():
    from holon_plugins.kernels.ggml_cpu import extension, plugin

    tier = platform.cpu_tier()
    try:
        extension.load(tier)
    except extension.NotBuilt as e:
        pytest.skip(str(e))
    return plugin, tier


def _experts(name: str, file: str | None, layer: int):
    """(gate/up layout, down layout, hidden, inter, gate/up units [E, bytes], down units [E, bytes])."""
    if file is None:                    # 4 experts, hidden 512, inter 256; down Q4_0, or `layer`'s type (Q2_0: FN's)
        down = layer or "q4_0"
        rng = np.random.default_rng(0)
        gu = np.stack([gguf.quants.quantize(rng.standard_normal((512, 512), dtype=np.float32) * 0.05,
                                            gguf.GGMLQuantizationType.Q8_0).reshape(-1) for _ in range(4)])
        if down == "q2_0":            # random codes: ggml's quantizer (d = amax) never writes code 3 (+2), the files do
            blocks = np.zeros((4, 512 * 256 // 64, 18), np.uint8)
            blocks[..., :2] = (rng.uniform(0.01, 0.05, blocks.shape[:2]).astype(np.float16)).view(np.uint8).reshape(
                *blocks.shape[:2], 2)
            blocks[..., 2:] = rng.integers(0, 256, (*blocks.shape[:2], 16), dtype=np.uint8)
            assert (((blocks[..., 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3) == 3).any()
            dn = blocks.reshape(4, -1)
        else:
            dn = np.stack([gguf.quants.quantize(rng.standard_normal((512, 256), dtype=np.float32) * 0.05,
                                                gguf.GGMLQuantizationType[down.upper()]).reshape(-1) for _ in range(4)])
        return "ggml.q8_0/gguf.native", f"ggml.{down}/gguf.native", 512, 256, torch.from_numpy(gu), torch.from_numpy(dn)
    path = next((VEHICLE if "/" not in file else FN).glob(f"*/{file}"), None)
    if path is None:
        pytest.skip(f"{file} not cached")
    t = {x.name: x for x in gguf.GGUFReader(path).tensors}
    g, u, d = (t[f"blk.{layer}.ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
    e = g.data.shape[0]
    gu = np.concatenate([np.asarray(g.data).reshape(e, -1), np.asarray(u.data).reshape(e, -1)], 1)
    return (f"ggml.{g.tensor_type.name.lower()}/gguf.native", f"ggml.{d.tensor_type.name.lower()}/gguf.native",
            int(g.shape[0]), int(g.shape[1]), torch.from_numpy(gu.copy()),
            torch.from_numpy(np.asarray(d.data).reshape(e, -1).copy()))


def _table(spec: list[tuple[int, list[int]]]) -> torch.Tensor:
    """A job table: (expert, its entries t·k + j) per job; units = expert ids."""
    jobs = torch.zeros(1 + J, 4 + T, dtype=torch.int32)
    jobs[0, 0] = len(spec)
    for i, (e, entries) in enumerate(spec):
        jobs[1 + i, :4 + len(entries)] = torch.tensor([e, e, e, len(entries), *entries])
    return jobs


@pytest.mark.parametrize("name, file, layer", CASES)
def test_cpu_experts_match_the_oracle_and_a_token_is_bitwise_alone_or_grouped(plugin, name, file, layer):
    plugin, tier = plugin
    gu_layout, dn_layout, hidden, inter, gu, dn = _experts(name, file, layer)
    kernel = plugin.CpuExperts(plugin.Modules(), tier, None, gate_up=gu_layout, down=dn_layout, hidden=hidden,
                               inter=inter, top_k=K, tokens=T, jobs=J)
    x = torch.randn(T, hidden, generator=torch.Generator().manual_seed(1))
    e = len(gu)
    spec = [(1 % e, [0, 9, 18]), (2 % e, [1]), (e - 1, [2, 11, 20, 29, 38, 47, 56, 63]), (0, [3, 12])]
    jobs, used = _table(spec), [x for _, entries in spec for x in entries]
    rows = torch.full((T * K, hidden), 7.0)
    want = reference.cpu_experts(jobs, x, gu, dn, rows, gu_layout, dn_layout, inter, K)
    alone = rows.clone()
    for expert, entries in spec:
        for entry in entries:
            out = rows.clone()
            kernel(_table([(expert, [entry])]), x, gu, dn, out)
            alone[entry] = out[entry]
    err = (alone[used] - want[used]).abs().amax(-1) / want[used].pow(2).mean(-1).sqrt()
    assert (err < BUDGET).float().mean() >= 0.9 and err.max() < FLIP, err
    for threads in (1, 7, 40):
        got = rows.clone()
        kernel(jobs, x, gu, dn, got, threads=threads)
        assert torch.equal(got, alone), threads                        # entries not named are left as they were
    kernel.plant = 2                                                    # the mutation: grouped tokens split K
    got = rows.clone()
    kernel(jobs, x, gu, dn, got, threads=3)
    assert not torch.equal(got[used], alone[used])
    kernel.plant = 0
    bad = jobs.clone()
    bad[2, 1] = e                                                       # a unit past the pool
    with pytest.raises(RuntimeError, match="malformed job table"):
        kernel(bad, x, gu, dn, rows.clone())


def test_prebuild_links_each_tier_in_its_own_process(monkeypatch, tmp_path):
    """torch names a module linked a second time in one process `<name>_v1`, which load() never finds: a second tier
    built beside the first is a fresh process's (found on an AVX-512 desktop, X's close)."""
    from holon_plugins.kernels.ggml_cpu import extension, prebuild

    calls = []
    monkeypatch.setattr(extension, "module_path", lambda tier: tmp_path / tier)          # nothing built yet
    monkeypatch.setattr(extension, "build", lambda tier, verbose: calls.append(("in-process", tier)))
    monkeypatch.setattr(prebuild.subprocess, "run", lambda cmd, check: calls.append((cmd[1:3], cmd[-1])))
    prebuild.main([])
    assert calls == [(["-m", "holon_plugins.kernels.ggml_cpu.prebuild"], t) for t in sorted(extension.TIERS)]
    calls.clear()
    prebuild.main(["--tier", "x86-avx512"])                                             # one tier: in this process
    assert calls == [("in-process", "x86-avx512")]
