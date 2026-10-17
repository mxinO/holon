import dataclasses
import pathlib
import subprocess
import sys

import pytest
from holon.api import (
    Binding,
    LayerDecl,
    Lock,
    Manifest,
    ModelSpec,
    Plan,
    Ref,
    Resource,
    UnitPin,
    WeightMap,
    canonical_hash,
)


def _spec(heads: int = 32, **extra) -> ModelSpec:
    attn = Ref("std.mixers.gqa@1", heads=heads, kv_heads=8)
    mlp = Ref("std.ffn.swiglu@1", inter=12288)
    return ModelSpec(
        hidden=4096, vocab=151936, layers=tuple(LayerDecl(attn, mlp, "full_attention") for _ in range(2)),
        block=Ref("std.blocks.prenorm@1"), embed=Ref("std.ops.embedding@1"),
        norm=Ref("std.ops.rmsnorm@1", eps=1e-6), lm_head=Ref("std.ops.lm_head@1", tied=False),
        weights=WeightMap("hf_safetensors", rules=()), **extra,
    )


def test_ref_identity_ignores_keyword_order():
    a = Ref("std.mixers.gqa@1", heads=32, kv_heads=8)
    b = Ref("std.mixers.gqa@1", kv_heads=8, heads=32)
    assert a == b and hash(a) == hash(b) and canonical_hash(a) == canonical_hash(b)
    assert (a.name, a.major, a.target, a.param("heads")) == ("std.mixers.gqa", 1, "std.mixers.gqa@1", 32)
    assert a.with_(heads=16).param("heads") == 16 and a.param("heads") == 32


@pytest.mark.parametrize("bad", ["std.mixers.gqa", "a@x", "@1", "a@"])
def test_ref_rejects_bad_targets(bad):
    with pytest.raises(ValueError):
        Ref(bad)


def test_ref_rejects_non_canonical_params():
    with pytest.raises(ValueError, match="canonical"):
        Ref("a@1", fn=len)
    assert Ref("a@1", x=1) != Ref("a@1", x=True)


def test_hash_is_pinned():
    # Changing the term encoding changes every identity; this pin makes such a change deliberate.
    assert canonical_hash(_spec()) == "sha256:8c69dae8b5c5625aa249b7bef98748eb8a3440a1ceaad71ee303c82194316c58"


def test_defaulted_fields_do_not_change_identity():
    rooted = lambda root: dataclasses.replace(_spec(), weights=WeightMap("hf_safetensors", rules=(), root=root))  # noqa: E731
    assert canonical_hash(_spec()) == canonical_hash(rooted(""))
    assert canonical_hash(_spec()) != canonical_hash(_spec(heads=16))
    assert canonical_hash(_spec()) != canonical_hash(rooted("model."))


def test_hash_is_order_independent_for_dicts_and_sets():
    assert canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
    assert canonical_hash(frozenset({1, "a", (2, "x")})) == canonical_hash(frozenset({(2, "x"), "a", 1}))
    with pytest.raises(TypeError):
        canonical_hash({1: "x"})
    assert canonical_hash(1e-6) == canonical_hash(0.000001)


def _plan(**kw) -> Plan:
    base = dict(model_id="qwen3-8b", fingerprint_class="a5000/cu130-torch2.11-triton3.6", recipe="sha256:r",
                commit="abc", uv_lock="sha256:u",
                units={"std.ops": UnitPin("1.0.0", "tree1"), "std.ffn": UnitPin("1.0.0", "tree2")},
                sites={"layers.0.mixer": Binding("std.mixers.gqa", config={"page": 16})})
    return Plan(**{**base, **kw})


def test_plan_id_is_order_independent_and_excludes_verification():
    a = _plan()
    b = _plan(units=dict(reversed(list(a.units.items()))))
    assert a.plan_id == b.plan_id and a.plan_id != _plan(commit="def").plan_id
    assert Lock(a, snapshot="sha256:s1").plan.plan_id == Lock(a, snapshot="sha256:s2").plan.plan_id


DESIGN = pathlib.Path(__file__).parents[2] / "docs" / "design.md"


def test_manifest_loads_design_example(tmp_path: pathlib.Path):
    """Design §4.1's manifest example, read from the design, parses."""
    text = DESIGN.read_text()
    start = text.index("```toml\n", text.index("### 4.1 Manifest")) + len("```toml\n")
    (tmp_path / "plugin.toml").write_text(text[start:text.index("```", start)])
    m = Manifest.load(tmp_path / "plugin.toml")
    assert m.provides == (Resource("draft_policy.adaptive_k", 1),) and m.cardinality.value == "single"
    assert m.options["spec.k_max"].choices == ("2", "3", "4") and m.options["spec.k_max"].default == "3"


@pytest.mark.parametrize("abi, extra, match", [
    (">=1.0,<2.0", "[surprise]\nk=1\n", "unknown section"),
    (">=1.0,<2.0", "[resources]\nprovide=['a@1']\n", "unknown keys"),
    (">=1.0,<2.0", "[properties]\nnumerics='fast'\n", "invalid numerics"),
    (">=1.0,<2.0", "[properties]\nmax_depth=1\n", "unknown keys"),  # removed: the depth gate comes from the processor
    (">=1.0,<2.0", "[plugin_extra]\n", "unknown section"),
    (">=1.0,<2.0", "[options]\n'a.b' = { default = '1' }\n", "help"),            # an option needs its help line
    (">=1.0,<2.0", "[options]\n'a.b' = { help = 'x', choices = 'a8' }\n", "strings"),
    (">=1.0,<2.0", "[options]\n'a.b' = { help = 'x', choices = [4, 8] }\n", "strings"),
    (None, "", "core_abi missing"),                                  # required (release design §7)
    (">=0.2,<0.3", "[surprise]\n", "does not admit this core's plugin ABI 1.2"),   # checked before the keys
    ("~=0.1", "", "expected clauses"),
])
def test_manifest_rejects_mistakes_with_path(tmp_path: pathlib.Path, abi, extra, match):
    base = '[plugin]\nname="x"\nversion="0"\nentry="x:P"\nlicense="MIT"\nslot="unit"\n'
    (tmp_path / "plugin.toml").write_text(base + (f"core_abi={abi!r}\n" if abi else "") + extra)
    with pytest.raises(ValueError, match=match) as e:
        Manifest.load(tmp_path / "plugin.toml")
    assert "plugin.toml" in str(e.value)


def test_set_like_abi_items_change_by_element(monkeypatch):
    """`SLOTS`, the manifest's keys, `KERNEL_PROPS` and the string vocabularies are listed one element per line: an
    element added at 1.1 is a minor addition, one removed or changed is major. An ordered vocabulary consumers switch
    over (finish reasons, error kinds) stays one line, so any new value is major."""
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ci"))
    from check_abi import bump_error, elements, render, sections
    from holon.api import Cardinality as C

    def listing(version, slots, keys, props=("deterministic",), reasons=("stop",)):
        return "\n".join([f"abi {version}", *render("SLOTS", slots), *sections({"plugin": keys, "options": None}),
                          *render("KERNEL_PROPS", frozenset(props)), *elements("Numerics", ["exact"]),
                          *render("FINISH_REASONS", tuple(reasons))]) + "\n"

    base = listing("1.0", {"model": C.SINGLE}, {"name"})
    for added in (listing("1.1", {"model": C.SINGLE, "tier": C.MULTI}, {"name"}),
                  listing("1.1", {"model": C.SINGLE}, {"name", "variant_of"}),
                  listing("1.1", {"model": C.SINGLE}, {"name"}, props=("deterministic", "batch_invariant"))):
        assert bump_error(base, added) is None
    for broken in (listing("1.1", {}, {"name"}), listing("1.1", {"model": C.MULTI}, {"name"}),
                   listing("1.1", {"model": C.SINGLE}, set()), listing("1.1", {"model": C.SINGLE}, {"name"}, props=()),
                   listing("1.1", {"model": C.SINGLE}, {"name"}, reasons=("stop", "length"))):
        assert "needs a major" in bump_error(base, broken)


def test_api_imports_without_torch_or_numpy():
    code = "import sys, holon.api; print(any(m in sys.modules for m in ('torch', 'numpy')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_dataclass_replace_on_ref_is_not_supported():
    with pytest.raises(TypeError):
        dataclasses.replace(Ref("a@1", x=1), name="b")


LISTING = ("abi {}\nprotocol Sampler() required, implemented by plugins\n  sample(self)\nconst X = 1\n"
           "protocol Ctx() required, implemented by core\n  pool(self, name: 'str', *, n: 'int' = 0) -> 'Pool'\n"
           "dataclass E\n  #0 c: C = C(x=1)\ndataclass D\n  #0 a: int\n  #1 b: int = 0\n")
FIELD = "  #1 b: int = 0\n"
POOL = "  pool(self, name: 'str', *, n: 'int' = 0) -> 'Pool'\n"
CONTRACTS = "op.rmsnorm@1  (std.ops)\n"


@pytest.mark.parametrize("old, new, error", [
    ("0.1", LISTING.format("0.1.1") + "const Y = 2\n", None),                      # an addition: minor
    ("0.1", LISTING.format("0.1") + "const Y = 2\n", "needs a minor"),
    ("1.0", LISTING.format("1.1") + "const Y = 2\n", None),
    ("1.0", LISTING.format("1.1").replace("const X = 1\n", ""), "needs a major"),      # a removal: major
    ("0.1", LISTING.format("0.2").replace("const X = 1\n", ""), None),                 # (below 1.0: the minor number)
    ("1.0", LISTING.format("1.1").replace("  sample(self)\n", "  sample(self)\n  seed(self)\n"),   # plugins lack it
     "required plugin member"),
    ("1.0", LISTING.format("2.0").replace("  sample(self)\n", "  sample(self)\n  seed(self)\n"), None),
    ("1.0", LISTING.format("1.1") + "  #2 c: int = 1\n", None),                       # a field appended with a default
    ("1.0", LISTING.format("1.1") + "  #2 c: int\n", "needs a major"),               # without: callers lack it
    ("1.0", LISTING.format("1.1").replace(FIELD, "  #1 c: int = 1\n  #2 b: int = 0\n"), "needs a major"),  # inserted
    # A member plugins only call gains keyword-only parameters with defaults; a default shows its class's new field.
    ("1.0", LISTING.format("1.1").replace(POOL, POOL.replace("0)", "0, host: 'bool' = False)")), None),
    ("1.0", LISTING.format("1.1").replace(POOL, POOL.replace("0)", "0, host: 'bool')")), "needs a major"),
    ("1.0", LISTING.format("1.1").replace(POOL, POOL.replace("*, n", "n")), "needs a major"),
    ("1.0", LISTING.format("1.1").replace("  sample(self)", "  sample(self, *, x: 'int' = 0)"), "needs a major"),
    ("1.0", LISTING.format("1.1").replace("C(x=1)", "C(x=1, y=2)"), None),
    ("1.0", LISTING.format("1.1").replace("C(x=1)", "C(x=2, y=2)"), "needs a major"),
])
def test_abi_listing_needs_the_right_bump(monkeypatch, old, new, error):
    """`scripts/ci/check_abi.py`: what an ABI change requires of the version; contracts are the plugins' own and
    change by their majors (release design §7)."""
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ci"))
    from check_abi import bump_error, contract_error, contracts

    why = bump_error(LISTING.format(old), new)
    assert why is None if error is None else error in why
    assert contract_error(CONTRACTS, CONTRACTS + "op.new@1  (std.ops)\n") is None             # new contracts are free
    assert "without a new major" in contract_error(CONTRACTS, "op.rmsnorm@1  (other)\n")
    assert contract_error(CONTRACTS, "op.rmsnorm@2  (std.ops)\n") is None
    listed = contracts("std") + contracts("plugins")                   # every provided resource but implementations
    for resource in ("state_kind.paged@1", "grammar.constraint@1", "transform.qwen3_5.squeeze@1"):
        assert f"{resource}  (" in listed
    assert "kernel." not in listed and "arch." not in listed and "model." not in listed
