"""The GGUF weight source on the cached unsloth Qwen3-8B files, checked against the HF checkpoint they were made
from (M0.1 design §4.2): the reader, the configuration, the self-contained tokenizer, the stop set, the template. And
the hub references both weight sources resolve (release design §14.3), on listings faked here: no network."""
import json
import pathlib

import gguf
import pytest
from holon.core.assembly import detect_source
from holon.core.plugins import PluginSet, discover
from holon_std.weights.gguf import reader
from holon_std.weights.gguf.source import GGUFSource

HUB = pathlib.Path.home() / ".cache/huggingface/hub"
FILES = sorted(HUB.glob("models--unsloth--Qwen3-8B-GGUF/snapshots/*/Qwen3-8B-*.gguf"))
HF = max(HUB.glob("models--Qwen--Qwen3-8B/snapshots/*"), default=None)
PROMPTS = pathlib.Path(__file__).resolve().parents[2] / "judge" / "prompts" / "v1.jsonl"
ADVERSARIAL = ["Hello world!", "<think>\n\n</think>\n\n", "  leading  spaces\tand\ttabs\r\nCRLF\n\n\n", "1234567890123",
               "中文测试，日本語のテスト", "emoji 🤖🚀👍🏽", "é vs é", "가 Hangul jamo", "don't I'll they're",
               "<tool_call>{\"a\": 1}</tool_call><|im_end|>", "[PAD151669] text", "non breaking", " 　 ",
               "<|im_start|>user\nhi<|im_end|>"]


@pytest.fixture(scope="module")
def hf_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF)


@pytest.fixture(scope="module", params=[f.name for f in FILES] if HF else [None])
def source(request):            # every test of the source needs it: they skip without the cached files
    if request.param is None:
        pytest.skip("needs the cached Qwen3-8B GGUF files and HF model")
    return GGUFSource(next(f for f in FILES if f.name == request.param))


def test_the_reader_sees_what_gguf_py_sees(source):
    ref = gguf.GGUFReader(source.path)
    ours = {t.name: (t, data) for t, data in reader.views(source.path, source._tensors)}
    assert set(ours) == {t.name for t in ref.tensors}
    for t in ref.tensors[:24]:                                       # every kind of tensor, bytes and shapes
        info, data = ours[t.name]
        assert info.type == t.tensor_type and data.tobytes() == bytes(t.data)
        assert info.shape == tuple(int(n) for n in reversed(t.shape))
    for key in ("general.architecture", "tokenizer.ggml.pre", "qwen3.rope.freq_base", "tokenizer.ggml.merges"):
        assert source._fields[key] == ref.fields[key].contents()


def test_config_equals_the_hf_checkpoints(source):
    hf = json.loads((HF / "config.json").read_text())
    ours = {k: v for k, v in source.config().items() if k != "gguf"}
    assert ours == {k: hf[k] for k in ours}, {k: (v, hf.get(k)) for k, v in ours.items() if v != hf.get(k)}


def test_tokenizer_encodes_as_hf(source, hf_tok):
    tok = source.tokenizer()
    for text in ADVERSARIAL:
        assert tok.encode(text, add_special_tokens=False) == hf_tok.encode(text, add_special_tokens=False), text
    for i, added in hf_tok.added_tokens_decoder.items():             # every special/added token is atomic
        assert tok.encode(added.content, add_special_tokens=False) == [i]
    for line in PROMPTS.read_text().splitlines():                    # the judge's prompts round-trip
        ids = json.loads(line)["ids"]
        text = hf_tok.decode(ids)
        assert tok.decode(ids) == text and tok.encode(text, add_special_tokens=False) == ids


def test_stop_set_equals_hf_generation_config(source):
    gen = json.loads((HF / "generation_config.json").read_text())["eos_token_id"]
    assert source.eos_token_ids == frozenset(gen)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
AGENT = [{"role": "system", "content": "You are a helpful agent."},
         {"role": "user", "content": "Weather in Paris and Rome?"},
         {"role": "assistant", "reasoning_content": "Two cities: two calls.", "content": "",
          "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                         {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
         {"role": "tool", "content": "18C, sunny"}, {"role": "tool", "content": "24C, clear"},
         {"role": "assistant", "content": "<think>\nBoth known.\n</think>\n\nParis 18C, Rome 24C."},
         {"role": "user", "content": "Thanks!"}]
CHATS = {"plain": ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3+3?"}], {}),
         "agent with tools and reasoning": (AGENT, {"tools": TOOLS}),
         "thinking off": (AGENT[:2], {"enable_thinking": False})}


@pytest.mark.parametrize("chat", CHATS)
def test_chat_template_renders_like_hf(source, hf_tok, chat):
    messages, kw = CHATS[chat]
    ours = source.tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    assert ours == hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def test_identity_and_detection(source):
    model, digest = source.identity().split("@")
    assert model == f"unsloth/Qwen3-8B-GGUF/{source.path.stem}" and len(digest) == 16
    plugins = PluginSet(discover())
    assert detect_source(plugins, str(source.path)) == "gguf" and detect_source(plugins, str(HF)) == "hf_safetensors"


REPOS = {"o/gguf": {"Qwen3-8B-Q4_K_M.gguf": 5, "Qwen3-8B-UD-Q4_K_M.gguf": 6, "Qwen3-8B-UD-IQ3_XXS.gguf": 4,
                    "BF16/Qwen3-8B-BF16-00001-of-00002.gguf": 9, "BF16/Qwen3-8B-BF16-00002-of-00002.gguf": 9,
                    "UD-Q8_K_XL/Qwen3-8B-UD-Q8_K_XL.gguf": 8, "mmproj-F16.gguf": 1, "config.json": 1, "README.md": 1,
                    "Qwen3-8B-F16.gguf": 3, "F16/Qwen3-8B-F16-00001-of-00002.gguf": 2,
                    "F16/Qwen3-8B-F16-00002-of-00002.gguf": 2},
         "o/one": {"Qwen3-8B-Q4_K_M.gguf": 5},
         "o/two": {"Qwen3-8B-Q2_K.gguf": 1, "BF16/Qwen3-8B-BF16-00001-of-00002.gguf": 2,
                   "BF16/Qwen3-8B-BF16-00002-of-00002.gguf": 2},
         "o/hf": {"config.json": 1, "model-00001-of-00002.safetensors": 7, "model-00002-of-00002.safetensors": 7,
                  "model.safetensors.index.json": 1, "tokenizer.json": 2, "original/consolidated.safetensors": 99,
                  "consolidated.safetensors": 99, "README.md": 1}}
REPOS["o/hf-unindexed"] = REPOS["o/hf"]                # its index cannot be fetched: every top-level weight file


G, H = "gguf", "hf"
UNREACHABLE = "the hub is unreachable (ConnectError; offline, HF_HUB_OFFLINE=1 serves cached models), and"


@pytest.mark.parametrize("ref, want", [
    ("o/gguf:Q4_K_M", {G: ("o/gguf/Qwen3-8B-Q4_K_M.gguf", 5)}),            # the plain file, not UD-Q4_K_M
    ("o/gguf:ud-q4_k_m", {G: ("o/gguf/Qwen3-8B-UD-Q4_K_M.gguf", 6)}),
    ("o/gguf/Qwen3-8B-UD-Q8_K_XL", {G: ("o/gguf/UD-Q8_K_XL/Qwen3-8B-UD-Q8_K_XL.gguf", 8)}),   # in a directory
    ("o/gguf:BF16", {G: "o/gguf: Qwen3-8B-BF16 is split into 2 files"}),
    ("o/gguf:F16", {G: ("o/gguf/Qwen3-8B-F16.gguf", 3)}),                    # the single file beside a split set
    ("o/gguf:Q2_K", {G: "no GGUF file tagged Q2_K in o/gguf; name one as o/gguf:<TAG> or o/gguf/<name>: Qwen3-8B-B"}),
    ("o/gguf", {G: "several GGUF files"}),               # its config.json does not make it the HF source's
    ("o/one", {G: ("o/one/Qwen3-8B-Q4_K_M.gguf", 5)}),
    ("o/two", {G: "several GGUF files in o/two"}),       # a split set is a file too: not "the only one"
    ("o/hf", {H: ("o/hf (5 files)", 18)}),               # top-level files the index names: not original/, README.md,
                                                         # or a consolidated copy beside the shards
    ("o/hf-unindexed", {H: ("o/hf-unindexed (6 files)", 117)}),
    ("./o/hf", {}),                                      # a path, never a reference
    ("cached:o/gguf:Q4_K_M", {G: ("o/gguf/Qwen3-8B-Q4_K_M.gguf", 5),        # the hub unreachable: the cache's listing
                              H: None}),
    ("cached:o/gguf", {G: ("o/gguf/Qwen3-8B-Q4_K_M.gguf", 5),               # the only cached file
                       H: "o/gguf: its cached snapshot holds no safetensors checkpoint"}),
    ("cached:o/hf", {G: f"{UNREACHABLE} o/hf is not in the HF cache", H: f"{UNREACHABLE} o/hf is not in the HF cache"}),
])
def test_hub_references_are_claimed_by_one_source(tmp_path, monkeypatch, ref, want):
    """Each source's `resolve` on a faked hub: a reference is claimed by the source whose files the repository holds
    (HF: config.json and safetensors; GGUF: .gguf files), the file a tag names chosen without the listing's order
    mattering; claiming downloads nothing (the fetch runs here on faked download functions); unreachable, the cached
    snapshot is listed, and what it lacks is said. `want`: per source, its claim (what, bytes), its error's start, or
    None (not its reference; a source left out is None)."""
    import huggingface_hub
    from holon_std.weights.gguf import hub as gguf_hub
    from holon_std.weights.hf import hub as hf_hub
    from huggingface_hub import constants

    class ConnectError(Exception):
        pass

    def tree(api, repo, recursive):
        if cached:
            raise ConnectError(repo)
        return [type("RepoFile", (), {"path": p, "size": n})() for p, n in REPOS[repo].items()]

    cached, ref = ref.startswith("cached:"), ref.removeprefix("cached:")
    snapshot = tmp_path / "models--o--gguf" / "snapshots" / "abc"            # the cache: one GGUF file of o/gguf
    snapshot.mkdir(parents=True)
    (snapshot / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"12345")
    (tmp_path / "models--o--gguf" / "refs").mkdir()
    (tmp_path / "models--o--gguf" / "refs" / "main").write_text("abc")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", tree)
    index = tmp_path / "index.json"                                           # the HF index: the shards only
    index.write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                                "b": "model-00002-of-00002.safetensors"}}))
    def download(repo, path):
        if repo == "o/hf-unindexed":
            raise OSError(repo)
        return str(index) if path.endswith(".index.json") else f"fetched {repo} {path}"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, allow_patterns: f"fetched {allow_patterns}")
    for name, hub in ((G, gguf_hub), (H, hf_hub)):
        expect = want.get(name)
        try:
            claim = hub.resolve(ref)
        except ValueError as e:
            assert isinstance(expect, str) and str(e).startswith(expect), (name, str(e))
            continue
        if expect is None:
            assert claim is None, name
            continue
        assert (claim.what, claim.nbytes) == expect, name
        if name == G:
            org, repo, path = expect[0].split("/", 2)
            assert claim.fetch() == f"fetched {org}/{repo} {path}"
        else:
            files = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "model.safetensors.index.json", "tokenizer.json"]
            files = sorted(files + ["consolidated.safetensors"]) if ref.endswith("unindexed") else files
            assert claim.fetch() == f"fetched {files}"
