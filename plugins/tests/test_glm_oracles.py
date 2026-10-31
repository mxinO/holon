"""GLM-5.3-Flash's oracles outside the golden venv (docs/dev/v1-glm.md §6.1): the tokenizer a GGUF file makes
(`std.weights.gguf`, `tokenizer.ggml.pre` `glm4`) against llama.cpp's own (`llama-tokenize` at 27c7a76) and the
model's `tokenizer.json`, on a multilingual round-trip corpus. The file is the tiny CPU fixture
(`holon_testing.glm5next_files`, built in the golden venv): `GLM_TINY_GGUF` names it, `GLM_LLAMA_TOKENIZE` the tool."""
import json
import os
import pathlib
import subprocess
import warnings

import pytest

SKIP = ("GLM's tokenizer check (§6.1) did not run: it needs GLM_TINY_GGUF (the tiny CPU fixture, "
        "`python -m holon_testing.glm5next_files` in the golden venv) and GLM_LLAMA_TOKENIZE (llama.cpp 27c7a76's "
        "llama-tokenize): docs/dev/release.md §22")
READY = bool(os.environ.get("GLM_TINY_GGUF")) and bool(os.environ.get("GLM_LLAMA_TOKENIZE"))
if not READY:
    warnings.warn(SKIP, stacklevel=1)       # loud: a skipped proof is not a passed one

CORPUS = [
    "Hello, world! It's a test; we'll see what's what.",
    "The year 2026 has 365 days, 8,760 hours and 31536000 seconds; π ≈ 3.14159265358979.",
    "  leading spaces, trailing spaces   \n\nand blank lines\r\n\tand tabs\t\t",
    "def f(x):\n    return x ** 2 + 1  # comment\n\nprint(f(12345678))",
    "中文测试：今天天气很好，我们去公园散步吧。数字１２３和123。",
    "日本語のテキスト：東京は日本の首都です。カタカナとひらがな。",
    "한국어 문장입니다. 서울은 대한민국의 수도입니다.",
    "Русский текст: Москва — столица России. Ёлка, ёж.",
    "النص العربي: القاهرة هي عاصمة مصر.",
    "हिन्दी पाठ: नई दिल्ली भारत की राजधानी है।",
    "Ελληνικά: η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    "Tiếng Việt: Hà Nội là thủ đô của Việt Nam.",
    "ไทย: กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
    "emoji 😀🎉👩‍💻🇫🇷 and symbols ©®™ ∑∫√∞ ≠ ≤ ≥",
    "Combining: é vs é; Å vs Å; ﬁ ligature; full-width ＡＢＣ",
    "URLs https://example.com/path?q=1&r=two#frag and emails a.b@c.org",
    "<|user|>\nWhat is 2+2?<|assistant|><think>Four.</think>4<|endoftext|>",
    "<|system|>You are helpful.<|observation|>{\"k\": [1, 2, 3]}",
    "I'M SHOUTING, YOU'RE NOT, THEY'VE GONE, SHE'D GO, HE'LL STAY.",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "1234567890123456789012345678901234567890 0.000001 -42 +7 1e-9",
    "Mixed 混合 text with English and 中文 and 日本語 in one line.",
    "见 参考 和 参考资料 而 者 王 根据 据 演员",                 # words BPE's merges would split (ignore_merges)
    "Markdown: **bold**, _italic_, `code`, # Heading\n- item 1\n- item 2\n\n| a | b |\n|---|---|",
    "",
    " ",
    "\n",
]


@pytest.mark.skipif(not READY, reason=SKIP)
def test_the_files_tokenizer_is_llama_cpps_and_the_models(tmp_path):
    from holon_std.weights.gguf.source import GGUFSource

    file = pathlib.Path(os.environ["GLM_TINY_GGUF"])
    tok = GGUFSource(file).tokenizer()
    reference = None
    hub = file.parent / "tokenizer.json"                     # the tool copies the model's tokenizer.json beside it
    if hub.exists():
        from tokenizers import Tokenizer

        reference = Tokenizer.from_str(hub.read_text())
    for i, text in enumerate(CORPUS):
        ids = tok.encode(text, add_special_tokens=False)
        prompt = tmp_path / f"{i}.txt"
        prompt.write_text(text)
        out = subprocess.run([os.environ["GLM_LLAMA_TOKENIZE"], "-m", str(file), "-f", str(prompt), "--ids",
                              "--no-bos", "--log-disable"], capture_output=True, text=True, check=True).stdout
        assert ids == json.loads(out.strip().splitlines()[-1] if out.strip() else "[]"), text
        if reference is not None:
            assert ids == reference.encode(text, add_special_tokens=False).ids, text
        assert tok.decode(ids) == text, text


def test_the_ram_short_spill_is_mode_ii_as_measured():
    """`holon serve`'s spill for the RTX 5090 lock (docs/dev/v1-auto-config.md D1): mode (ii)'s measured settings, each
    an option an installed plugin declares, with a value in its choices; none for the RTX 5080's lock (mode (ii) was
    measured on the 5090 only) or for options already setting RAM's share."""
    from holon.core import cli
    from holon.core.plugins import discover
    from holon_plugins.models.glm5_next import deploy

    lock = {"moe.store": "std.moe.offload@1", "moe.lend_peak": "require", "moe.ring_lend": "tail"}
    options, note = deploy.spill(lock, "/data/glm.gguf")
    declared = cli.declared_options(discover())
    assert options == deploy.MODE2 and set(options) <= set(declared) and not cli.outside_choices(options, declared)
    assert "mode (ii)" in note and "1.7–1.9×" in note
    assert deploy.spill(lock | {"moe.lend_peak": "off"}, "/data/glm.gguf") is None             # the RTX 5080's
    assert deploy.spill(lock | {"moe.ram": "*=0.5"}, "/data/glm.gguf") is None
