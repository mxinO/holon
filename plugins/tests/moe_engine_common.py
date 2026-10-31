"""The MoE engine tests' shared models, engines and checks (`test_moe_engine*.py`, and the offload plugins' engine
tests): the tiny CPU Qwen3-MoE and its engine, the GPU models (`gpu_moe`, `gpu_moe_wide`) with the resident store's
outputs, GLM mode (ii)'s terms (`_mode2`, RAM's share by `fit` at a set f: `_fit_at`), and the exclusive-tier store's
invariants (`_bytes_exact`, `_held_once`,
...). The fixtures, registered in `conftest.py`, are module-scoped: each test module that uses one builds its own
(xdist spreads the modules over the workers)."""
import contextlib
import math
from unittest import mock

import gguf
import numpy as np
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_std.tiered.ring import Lent
from holon_testing.gguf_files import write
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

Q = gguf.GGMLQuantizationType
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny_moe(tmp_path_factory):
    """A random two-layer Qwen3-MoE (8 experts, top-2), and its prompts."""
    return _tiny_moe(tmp_path_factory, 2)


@pytest.fixture(scope="module")
def tiny_moe_deep(tmp_path_factory):
    """Four layers of it (a ring's two halves fit a half-cached layout once lending spreads over the layers)."""
    return _tiny_moe(tmp_path_factory, 4)


def _tiny_moe(tmp_path_factory, layers: int):
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp(f"tiny_qwen3_moe{layers}")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():                                  # a router sharp enough that routing matters
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
            layer.mlp.experts.gate_up_proj.mul_(20)          # SiLU out of its linear range
            layer.mlp.experts.down_proj.mul_(5)               # the MoE output matters beside the residual
    model.save_pretrained(path)
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (3, 17, 8, 25)]
    return path, prompts


def _gguf_and_dense(checkpoint, tmp_path, qtype, **kw):
    """The checkpoint as a GGUF file, and an HF checkpoint of the file's dequantized weights (the oracle's)."""
    file, dense = tmp_path / f"moe-{qtype.name}.gguf", tmp_path / "dense"
    dequantized = write(checkpoint, file, qtype, **kw)
    dense.mkdir()
    (dense / "config.json").write_text((checkpoint / "config.json").read_text())
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in dequantized.items()},
              dense / "model.safetensors")
    return file, dense


def _engine(model, **kw) -> Engine:
    cfg = dict(model=str(model), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20)
    return Engine(EngineConfig(**cfg | kw))


def _served(engine, prompts, mirrored=None) -> tuple[list, dict]:
    """Greedy streams and prompt logprobs of `prompts` (`mirrored`: a store whose host mirror must be its device map,
    tier codes too, once the prompt steps are committed)."""
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=3)) for p in prompts]
    got = {}
    while engine.has_work():
        got |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    if mirrored is not None:
        assert np.array_equal(mirrored.residency.device.cpu().numpy(), mirrored.residency.host)
    return engine.generate(prompts, GREEDY), [(got[r].token, got[r].top_ids, got[r].top) for r in rids]


@contextlib.contextmanager
def _tail_checked(store):
    """D11's terms, asserted around `Lent` on every step that lends a tail (its tables copied on the device as the step
    goes, checked at its end: no host sync inside the step): call 0 lends n slots of the tail's class (`lends` row 0)
    and evicts, in every role, every expert holding a slot of the range, which the rounds then never pick; the range
    (`tail_view`) is the class's last n slots, contiguous, and holds the bytes its clients' passes in this step take
    (no more slots than those, nor than the tail's); no slot of the
    range is in the pool, the stage lists or the ring, nor mapped — at the step's start and after every layer —; what
    the step holds per class is what its pool holds (the range counted in neither), so a held ring has a slot for every
    missing piece and no step is short; at its end the empty slots it found lie below the lent ones, as they were, and
    the lent ones lie above them whole, by call then expert id. Yields each tail step's (empty slots found, `lends`
    rows, ring held)."""
    begin, done, steps = Lent.begin, Lent.done, []

    def checked_begin(self):
        before, found = store.residency.device.clone(), (store.empty.clone(), store.n_empty.clone())
        begin(self)
        if self.tail is None:
            return
        cls, n = self.tail
        c, classes = list(store.classes.values())[cls], store._class_index
        view = store.tail_view()
        need = max(f(store.step_tokens) for f in store.tail_bytes)       # the step's passes', at most the tail's
        assert n <= store.tail[1] and need <= max(f() for f in store.tail_bytes)
        assert view.is_contiguous() and len(view) == n * c.unit_bytes >= need > 0 and len(view) < need + c.unit_bytes
        assert view.data_ptr() == c.slots()[c.size - n].data_ptr()
        self._snap = {"before": before, "found": found, "maps": [store.residency.device.clone()],
                      "held": self._held(classes), "want": self._want(store.residency.device, classes)} | {
            name: getattr(store, name).clone() for name in ("pool", "stage_tab", "ring", "lent_call", "lend_needs",
                                                            "ring_ok", "errors")}

    def checked_done(self, g):
        done(self, g)
        if self.tail is None:
            return
        snap = self._snap
        snap["maps"].append(store.residency.device.clone())
        if g < len(store.layers) - 1:
            return
        t = {k: v.cpu() if isinstance(v, torch.Tensor) else [x.cpu() for x in v] for k, v in snap.items()}
        (cls, n), roles = self.tail, torch.from_numpy(store.class_table == self.tail[0])        # [G, R]
        lo, size = list(store.classes.values())[cls].size - n, list(store.classes.values())[cls].size

        def inside(slots):
            return (slots >= lo) & (slots < size)
        held = (inside(t["before"]) & roles[:, None, :]).any(-1)                             # [G, E]: the range's
        call, ring_ok = t["lent_call"], bool(t["ring_ok"])
        assert (t["maps"][0][held] == -1).all() and (call[held] == 0).all()
        assert (call != 0).eq(~held).all()                                  # the rounds never pick a holder
        rounds = store.lent_slots.cpu().permute(0, 2, 1)[roles][(call > 0)[roles.any(1)]]
        assert (rounds < lo).all(), "a round lent a slot past the step's lend end (held would count it, the pool not)"
        assert t["lend_needs"][0].tolist() == [n * (k == cls) for k in range(len(store.classes))]
        assert not any(inside(m.permute(0, 2, 1)[roles]).any() for m in t["maps"]), "a piece mapped in the tail"
        assert not inside(t["pool"][cls, :-1]).any(), "the pool holds the tail"              # (past it: the sink)
        assert torch.equal(t["held"], (t["pool"][:, :-1] >= 0).sum(1)), "held counts other slots than the pool's"
        stage, ring = t["stage_tab"][roles], t["ring"].permute(0, 2, 1)[roles]
        assert not inside(stage).any() and not inside(ring).any()
        missing = t["maps"][0].permute(0, 2, 1)[roles] < 0
        assert not ring_ok or ((t["held"] >= t["want"]).all() and (ring[missing] >= 0).all())
        assert not int(t["errors"]) & 4                                                     # never short
        (stack, n_found), empty, lent_slots = t["found"], store.empty.cpu(), store.lent_slots.cpu()
        for layer in range(len(store.layers)):                  # given back: found below, then the lent, whole
            k = int(n_found[layer])
            order = sorted(call[layer].ge(0).nonzero()[:, 0].tolist(), key=lambda e: (int(call[layer, e]), e))
            assert int(store.n_empty[layer]) == k + len(order)
            assert torch.equal(empty[layer, :, :k], stack[layer, :, :k])
            assert torch.equal(empty[layer, :, k:k + len(order)], lent_slots[layer, order].t())
        steps.append((int(n_found.sum()), t["lend_needs"].numpy(), ring_ok))
    with mock.patch.object(Lent, "begin", checked_begin), mock.patch.object(Lent, "done", checked_done):
        yield steps


@pytest.fixture(scope="module")
def gpu_moe(tmp_path_factory):
    """A random Qwen3-MoE as a GGUF of two expert classes, hidden and expert widths 1024 (an mmq tile then runs
    several K iterations: where stream-k engages it splits them), its prompts (decode-length to chunked prefill),
    and the resident store's outputs on the GPU."""
    torch.manual_seed(1)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=1024, intermediate_size=1024, moe_intermediate_size=1024,
                         num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=128,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp("gpu_moe")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
    model.save_pretrained(path / "hf")
    write(path / "hf", path / "moe.gguf", Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (1, 7, 40, 300)]
    engine = _gpu_engine(path / "moe.gguf")
    want = _served(engine, prompts)
    engine.close()
    return path / "moe.gguf", prompts, want


@pytest.fixture(scope="module")
def gpu_moe_wide(tmp_path_factory):
    """`gpu_moe` wide enough for a staging step's rounds (docs/dev/v1-glm.md D8s: a class lends a fixed set of a round's
    RAM slots and the landing rows' past its tail), its down experts two classes on disjoint layers as GLM's (D8w's
    tails, one a class): 128 experts, 8 a token, widths 512, four layers, layers 0-1's down Q5_0, 2-3's Q4_0."""
    return _wide(tmp_path_factory, Q.Q8_0, {"0.ffn_down_exps": Q.Q5_0, "1.ffn_down_exps": Q.Q5_0,
                                            "ffn_down_exps": Q.Q4_0})


def _wide(tmp_path_factory, qtype, types) -> tuple:
    """`gpu_moe_wide`'s model as a GGUF of `qtype` stacks (`types` by name part), its prompts and the resident store's
    outputs on the GPU."""
    torch.manual_seed(2)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=512, intermediate_size=512, moe_intermediate_size=512,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                         num_experts=128, num_experts_per_tok=8, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp("gpu_moe_wide")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
    model.save_pretrained(path / "hf")
    write(path / "hf", path / "moe.gguf", qtype, types=types)
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (1, 40, 300, 450)]
    engine = _gpu_engine(path / "moe.gguf")
    want = _served(engine, prompts)
    engine.close()
    return path / "moe.gguf", prompts, want


def _mode2(**over) -> dict:
    """GLM mode (ii)'s deployment terms on the tiny models (docs/dev/v1-glm.md, the process section: an exclusive-tier
    change's engine tests run under them too): exclusive tiers with RAM's share sized from the host budget (`fit`,
    docs/dev/v1-glm-ram.md: `_fit_at` sets the f it solves), F10 (`placement.promote@1`, the spares a tail step
    releases), the tail lent with the step-peak loan required, RAM's own heat, the loan's margin and largest range,
    split calls and gathers beside the compute — the cache fraction the tiny models'."""
    return {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "fit",
            "moe.placement": "placement.promote@1", "moe.ram_heat": "0.999", "moe.loan_margin": "1.12",
            "moe.loan_largest": "0.55", "moe.split": "on",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"} | over


@pytest.fixture
def mode2():
    """`_mode2`, the deployment's terms (overrides by keyword), `fit` solving f = `share` (default half the experts)
    for the engines the test builds after."""
    with contextlib.ExitStack() as stack:
        def terms(share: float = 0.5, **over) -> dict:
            stack.enter_context(_fit_at(share))
            return _mode2(**over)
        yield terms


@contextlib.contextmanager
def _fit_at(share: float | None):
    """`moe.ram=fit` solving f = `share` (None: the working floor's, `fit.floor`): the host memory the store is told is
    left (`BuildCtx.host_left`) is fit's reserve and that share of every piece's bytes (`fit`'s arithmetic on real
    readings: its own tests)."""
    from holon.core.build import BuildContext
    from holon_std.moe.offload import KEY
    from holon_std.tiered import fit

    def left(ctx) -> int:
        store = ctx.shared_objects[KEY]
        classes = {c.name: (len(c.pieces), c.unit_bytes) for c in store.classes.values()}
        f = fit.floor(classes, store._keeps) if share is None else share
        return fit.RESERVE + math.ceil(f * sum(n * unit for n, unit in classes.values()))
    with mock.patch.object(BuildContext, "host_left", left):
        yield


@contextlib.contextmanager
def _no_host_sync(cls, *methods, mode="error"):
    """`cls`'s methods raise on any operation that waits for the device (a borrowing step's lending and ring), or
    (`mode` "warn") warn of each."""
    def strict(fn):
        def run(*args, **kw):
            torch.cuda.set_sync_debug_mode(mode)
            try:
                return fn(*args, **kw)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        return run
    with contextlib.ExitStack() as stack:
        for name in methods:
            stack.enter_context(mock.patch.object(cls, name, strict(getattr(cls, name))))
        yield


def _gpu_engine(model, **kw) -> Engine:
    return Engine(EngineConfig(**dict(model=str(model), device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                                      max_step_tokens=256, max_model_len=512, cuda_graphs=True, overlap=True) | kw))


def _bytes_exact(store, cache: dict | None = None) -> int:
    """Exclusive tiers: how many pieces' RAM units or VRAM slots do not hold the bytes the file has for the piece the
    map names there (an idle engine); `cache`: the file's bytes kept by piece across calls."""
    from holon_std.tiered import files
    from holon_std.tiered.store import _offsets

    torch.cuda.synchronize()
    res, host, wrong = store.residency.device.cpu().numpy(), store.host_units.cpu().numpy(), 0
    cache = {} if cache is None else cache
    for g, classes in enumerate(store.layer_classes):
        for i in range(store.items):
            for r, c in enumerate(classes):
                if res[g, i, r] < -1:
                    continue
                n = store._piece_index(c, g, i, r)
                if (c.name, n) not in cache:
                    cache[c.name, n] = want = np.empty(c.unit_bytes, dtype=np.uint8)
                    files.read([(e, want[o:o + e.nbytes]) for e, o in _offsets(c.pieces[n][3])])
                want = cache[c.name, n]
                got = c.host_view()[int(host[g, i, r])].numpy() if res[g, i, r] == -1 else \
                    c.slots()[int(res[g, i, r])].cpu().numpy()
                wrong += not np.array_equal(got[:c.unit_bytes], want)
    return wrong


def _held_once(store) -> None:
    """Exclusive tiers: every expert in one tier — a RAM unit's piece exactly where the map says (tier code −1), never
    beside a VRAM copy; each RAM unit owned once, by the piece whose host unit it is (no copy a promotion that raced a
    demotion could leave twice: Maya's duplicate-copy GC, step 7) — or reserved for a parked expert whose D8x move
    has not mapped it yet (`reservations`), which no piece's host unit names (A's D8x2-M1; the expert may have been
    routed into another tier meanwhile: phase 3 frees the unit then)."""
    res, host = store.residency.device.cpu().numpy(), store.host_units.cpu().numpy()
    assert ((res == -1) == (host >= 0)).all() and ((res >= 0).all(-1) == (res >= 0).any(-1)).all()
    reserved = {(k, u): item for k, u, item in store.reservations()}
    for k, c in enumerate(store.classes.values()):        # each RAM unit holds one piece, the one the map says
        owner = store.owner[k].cpu().numpy()
        units = [u for u in np.flatnonzero(owner >= 0) if (k, u) not in reserved]
        g, e = np.divmod(owner[units], store.items)
        role = [next(r for r, x in store.groups[store.layers[gg]].items() if x is c) for gg in g]
        assert len(set(owner[owner >= 0])) == np.count_nonzero(owner >= 0) and (host[g, e, role] == units).all()
        named = {int(host[gg, ee, rr]) for gg in range(len(store.layers)) for ee in range(store.items)
                 for rr, x in store.groups[store.layers[gg]].items() if x is c}
        assert all(owner[u] == item and u not in named for (kk, u), item in reserved.items() if kk == k)


def _slots_held(store) -> torch.Tensor:
    """Each layer's cache slots per role between steps: mapped, empty, a policy's spares' — conserved (none leaks)."""
    spare = store.state.get("spare")
    spares = (spare[..., 0] > 0).sum(1) if spare is not None else 0
    return ((store.residency.device >= 0).sum(1) + (store.n_empty + spares)[:, None]).cpu()


def _loan_free(store) -> bool:
    """D8w: no piece the map names sits in this step's activation loan — its other classes' tails and its tail's
    class's last bytes (`_in_loan`'s ranges)."""
    res, classes = store.residency.device, list(store.classes.values())
    cls = store._class_index.long()[:, None, :].expand_as(res)
    bad = torch.zeros_like(res, dtype=torch.bool)
    for c, m in store._step_down.items():
        bad |= (cls == c) & (res >= classes[c].size - m)
    if (tail := store._loan_slots()) is not None and store._step_gu_loan:
        bad |= (cls == tail[0]) & (res >= tail[1])
    return not bool(bad.any())


def _slots_once(store) -> bool:
    """Each class's slot is mapped by one piece or on one layer's empty stack, never both nor twice."""
    res, empty, n = (t.cpu().numpy() for t in (store.residency.device, store.empty, store.n_empty))
    classes = store._class_index.cpu().numpy()
    for c in range(len(store.classes)):
        mine = [(g, r) for g in range(len(n)) for r in range(res.shape[2]) if classes[g, r] == c]
        held = [res[g, :, r][res[g, :, r] >= 0] for g, r in mine] + [empty[g, r, :n[g]] for g, r in mine]
        held = np.concatenate(held)
        if len(set(held.tolist())) != len(held):
            return False
    return True
