"""The judge on Qwen3.8-Flash-Next (FN D2), in the golden venv (transformers 5.19.0, `local/envs/golden-qwen4`): the
tiny model's GGUF, rebuilt by llama.cpp's converter to its pinned sha256, reads back as the checkpoint
(`gguf.hf_state`, PLE's rows lazily); the streamed golden of it equals HF's whole forward bit for bit under the
indexer's tie rule, which the prompt's selections exercise (boundary ties occur); and the committed fixtures are what
the tool writes."""
import hashlib
import os
import pathlib

import pytest
import torch
import transformers

pytestmark = pytest.mark.skipif(not hasattr(transformers, "Qwen4ExpTextConfig") or not os.environ.get("FN_LLAMA_CPP"),
                                reason="needs transformers with qwen4_exp and FN_LLAMA_CPP (llama.cpp at 27c7a76)")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from holon_testing import qwen4exp_files as q

    model = q.tiny_hf()
    path = q.save(model, tmp_path_factory.mktemp("tiny_qwen4exp"))
    return model, path, q.convert(path, pathlib.Path(os.environ["FN_LLAMA_CPP"]))


def test_the_judge_reads_the_converted_file_as_the_checkpoint(tiny):
    from holon_judge import gguf
    from holon_testing import qwen4exp_files as q

    model, _, file = tiny
    assert hashlib.sha256(file.read_bytes()).hexdigest() == q.GGUF_SHA256
    ref = model.state_dict()
    table = next(k for k in ref if k.endswith("ngram_embedding.weight"))
    got = dict(gguf.hf_state(file))
    assert set(got) == set(ref) - {table}                       # the MTP block skipped, the table read lazily
    # A_log through the converter's exp and our log, a norm's w + 1 - 1 (f32 at 1: an ulp of 1.2e-7); the rest exact
    rounded = {k for k in got if k.endswith("A_log") or "norm" in k}
    assert all(torch.equal(got[k], ref[k]) for k in got.keys() - rounded)
    assert all((got[k] - ref[k]).abs().max() <= 1.2e-7 for k in rounded)
    ids = torch.tensor([[0, 5, 4095], [17, 17, 1000]])
    assert torch.equal(gguf.PleRows(file).rows(ids), ref[table][ids])


def test_the_streamed_golden_is_hfs_forward(tiny):
    from holon_judge import golden
    from holon_judge.recording import Prompt, PromptSet

    model, path, file = tiny
    torch.manual_seed(1)
    ids = torch.randint(0, 248000, (100,)).tolist()
    ids[40] = model.config.eos_token_id
    ties = []
    stable = golden._stable_topk

    def counting(x, k, dim=-1, largest=True, sorted=True):
        values = torch.sort(x, descending=True, stable=True).values
        ties.append(k < x.numel() and bool(values[k - 1] == values[k]))
        return stable(x, k, dim, largest, sorted)
    rec = golden.record(str(file), PromptSet("tiny", "", [Prompt("p", "chat", ids)]), config=str(path), k=20)
    golden._stable_topk = counting
    try:
        for m in model.modules():
            if type(m).__name__.endswith("QSAIndexer"):
                m.forward = golden._indexer_forward(m.forward)
        with torch.inference_mode():
            logits = model(torch.tensor([ids])).logits[0, :-1]
    finally:
        golden._stable_topk = stable
    ref = golden.topk_rows(logits, torch.tensor(ids[1:]), 20)
    assert torch.equal(rec.topk_ids, ref[0]) and torch.equal(rec.target_lp, ref[2])
    assert rec.meta["deviations"] == [golden.TIE_RULE] and any(ties)


def test_the_committed_fixtures_are_the_tools():
    from holon_testing import qwen4exp_files as q

    tensors, meta = q.fixtures(q.tiny_hf())                  # a fresh model: the golden test wraps the indexers
    committed, committed_meta = q.load_fixtures()
    assert tensors.keys() == committed.keys() and all(torch.equal(tensors[k], committed[k]) for k in tensors)
    assert all(int(meta[k]) > 0 and meta[k] == committed_meta[k] for k in meta)        # boundary ties occur
