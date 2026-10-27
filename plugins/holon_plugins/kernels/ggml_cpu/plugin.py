"""`cpu_experts@1` (std.moe.kernels; docs/dev/v1-offload.md D7, F5b): the CPU's share of a layer's missed experts, over
their GGUF blocks in the offload store's host pools, with ggml-cpu's own quantizers and dot products (vendored at
kernels.ggml_cuda's llama.cpp pin, MIT; csrc/cpu_experts.cpp). A token's output is bitwise the same alone or grouped
and at any thread count (rows split across threads, never K): D7, checked by L0.

One binary per CPU tier (`Constraint.cpu_isa`; `holon.core.platform.CPU_TIERS`): ggml-cpu's reductions may differ by
ISA, so a lock that runs these kernels names its tier in its class. An AVX-512 host takes the AVX-512 binary (M3 found
it bitwise the AVX2 one's for every class it measured, and equally fast). The modules are built ahead of time
(`python -m holon_plugins.kernels.ggml_cpu.prebuild`); without one an instance cannot be created and the error names
that command. `CpuExperts.plant` is judge-only: the calibration plants set its bits; production never does."""
from __future__ import annotations

import functools

from holon.api import Constraint, KernelSpec

from . import extension

OP = "cpu_experts"
# The weight types the CPU path serves and their ggml type ids (activations: Q8_0 for the first five, Q8_K for the
# rest; std.moe.reference `Q8_0_DOT`, `Q8_K_DOT`). Q2_0 (type 42): Qwen3.8-Flash-Next's down experts (FN D1).
TYPES = {"q2_0": 42, "q4_0": 2, "q5_0": 6, "q8_0": 8, "iq4_nl": 20, "q2_k": 10, "q3_k": 11, "q4_k": 12, "q5_k": 13,
         "q6_k": 14, "iq2_xxs": 16, "iq2_xs": 17, "iq3_xxs": 18, "iq1_s": 19, "iq3_s": 21, "iq2_s": 22, "iq4_xs": 23,
         "iq1_m": 29}


def ggml_type(layout: str) -> int | None:
    """`ggml.<type>/gguf.native` -> its ggml type id, if the CPU path serves it."""
    elem, _, name = layout.partition("/")
    return TYPES.get(elem.removeprefix("ggml.")) if elem.startswith("ggml.") and name == "gguf.native" else None


class Modules:
    """The prebuilt module of a tier, loaded once on first use (never compiled here)."""

    def __init__(self) -> None:
        self.loaded: dict[str, object] = {}

    def __call__(self, tier: str):
        if tier not in self.loaded:
            self.loaded[tier] = extension.load(tier)
        return self.loaded[tier]


class Task:
    """A `cpu_task@1` task over fixed buffers: callable (runs now, one thread), `native` its ABI handle."""

    def __init__(self, native) -> None:
        self._task, self.native = native, native.native()

    def __call__(self) -> None:
        if code := self._task.run(1):
            raise RuntimeError(f"cpu_experts@1: the task failed (code {code}: a malformed job table)")


class CpuExperts:
    plant = 0                  # judge-only: 1 = SwiGLU reads the merged gate/up halves swapped (cpu_gate_up_order)

    def __init__(self, modules: Modules, tier: str, ctx, *, gate_up: str, down: str, hidden: int, inter: int,
                 top_k: int, tokens: int, jobs: int) -> None:
        self.lib = modules(tier)
        self.experts = self.lib.Experts(ggml_type(gate_up), ggml_type(down), hidden, inter, top_k, tokens, jobs)

    def __call__(self, jobs, x, gate_up, down, rows, threads: int = 1) -> None:
        if code := self.lib.Task(self.experts, jobs, x, gate_up, down, rows, self.plant).run(threads):
            raise RuntimeError(f"cpu_experts@1: code {code} (a malformed job table)")

    def task(self, jobs, x, gate_up, down, rows) -> Task:
        return Task(self.lib.Task(self.experts, jobs, x, gate_up, down, rows, self.plant))


SMALL_BLOCKS = {"q2_0", "q4_0", "q5_0", "q8_0", "iq4_nl"}       # 32 values a block (and their dot type's, Q8_0); else
                                                                  # QK_K = 256 (and Q8_K)


def _block(layout: str) -> int:
    return 32 if layout.split("/")[0].removeprefix("ggml.") in SMALL_BLOCKS else 256


def _supports(f) -> bool:
    """Each matrix's reduction a whole number of its type's blocks: gate/up's over hidden, down's over inter (FN's 640
    with Q2_0 down)."""
    return ggml_type(f["gate_up"]) is not None and ggml_type(f["down"]) is not None \
        and f["hidden"] % _block(f["gate_up"]) == 0 and f["inter"] % _block(f["down"]) == 0


class Plugin:
    def register(self, regs) -> None:
        modules = Modules()
        # an AVX-512 host prefers its own build (priority), an AVX2 host can run only the AVX2 one
        for tier, hosts, priority in (("x86-avx2", {"x86-avx2", "x86-avx512"}, 10), ("x86-avx512", {"x86-avx512"}, 11)):
            regs.kernels.add(OP, KernelSpec(
                op=OP, provider="kernels.ggml_cpu", impl=functools.partial(CpuExperts, modules, tier),
                requires=Constraint(devices=frozenset({"cuda"}), cpu_isa=frozenset(hosts)), supports=_supports,
                props=frozenset({"deterministic"}), priority=priority, config={"tier": tier}))
