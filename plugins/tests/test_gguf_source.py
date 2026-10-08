"""The GGUF weight source on the cached unsloth Qwen3-8B files, checked against the HF checkpoint they were made
from (M0.1 design §4.2): the reader, the configuration, the self-contained tokenizer, the stop set, the template."""
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
pytestmark = pytest.mark.skipif(not FILES or HF is None, reason="needs the cached Qwen3-8B GGUF files and HF model")
PROMPTS = pathlib.Path(__file__).resolve().parents[2] / "judge" / "prompts" / "v1.jsonl"
ADVERSARIAL = ["Hello world!", "<think>\n\n</think>\n\n", "  leading  spaces\tand\ttabs\r\nCRLF\n\n\n", "1234567890123",
               "中文测试，日本語のテスト", "emoji 🤖🚀👍🏽", "é vs é", "가 Hangul jamo", "don't I'll they're",
               "<tool_call>{\"a\": 1}</tool_call><|im_end|>", "[PAD151669] text", "non breaking", " 　 ",
               "<|im_start|>user\nhi<|im_end|>"]


@pytest.fixture(scope="module")
def hf_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF)


@pytest.fixture(scope="module", params=[f.name for f in FILES])
def source(request):
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
