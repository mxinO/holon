"""The judge on GLM-5.3-Flash (docs/dev/v1-glm.md D2, §6.1), in the golden venv (transformers 5.19.0, `local/envs/
golden-glm`): the tiny models' GGUFs, rebuilt by llama.cpp's converter (and `llama-quantize` for the GPU copy) to their
pinned sha256; the f32 file reads back as the checkpoint (`gguf.hf_state`: the KDA convs and MLA's `kv_b_proj` fused
back, `A_log` from `ssm_a`); the streamed golden of it equals HF's whole forward over the same weights bit for bit
under the indexer's and the router's tie rule, which the prompt's selections exercise; both fixtures meet D2's init
(the clamp binds on >= 5% of every path's rows, most comb rows peaked, the GPU head's logit std >= 2); and the
committed CPU fixtures are what the tool writes."""
import hashlib
import json
import os
import pathlib
import warnings

import pytest
import torch
import transformers

SKIP = ("GLM's golden against HF on the tiny models (D2) did not run: it needs transformers with glm5_next (the golden "
        "venv) and GLM_LLAMA_CPP (llama.cpp at 27c7a76; GLM_LLAMA_QUANTIZE its llama-quantize for the GPU copy): "
        "docs/dev/release.md §22")
READY = hasattr(transformers, "Glm5NextTextConfig") and bool(os.environ.get("GLM_LLAMA_CPP"))
if not READY:
    warnings.warn(SKIP, stacklevel=1)       # loud: a skipped proof is not a passed one
pytestmark = pytest.mark.skipif(not READY, reason=SKIP)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from holon_testing import glm5next_files as g

    return g.build(tmp_path_factory.mktemp("tiny_glm5next"), pathlib.Path(os.environ["GLM_LLAMA_CPP"]), "cpu")


def test_the_judge_reads_the_converted_file_as_the_checkpoint(tiny):
    from holon_judge import gguf
    from holon_testing import glm5next_files as g

    assert hashlib.sha256(tiny.read_bytes()).hexdigest() == g.GGUF_SHA256["cpu"]
    ref, got = g.tiny_hf("cpu").state_dict(), dict(gguf.hf_state(tiny))
    assert got.keys() == ref.keys()
    a_log = {k for k in got if k.endswith("A_log")}              # through the converter's −exp and our log: an ulp
    assert all(torch.equal(got[k], ref[k]) for k in got.keys() - a_log)
    assert all((got[k] - ref[k]).abs().max() <= 6e-8 for k in a_log)


def test_the_streamed_golden_is_hfs_forward(tiny):
    from holon_judge import gguf, glm5next, golden
    from holon_judge.recording import Prompt, PromptSet
    from holon_testing import glm5next_files as g

    ids = g.prompts()[0]
    prompts = PromptSet("tiny", "", [Prompt("p", "chat", ids)])
    rec = golden.record(str(tiny), prompts, config=str(tiny.parent), k=20)
    model, ties = g.tiny_hf("cpu"), []                           # its own: the tie rule wraps its modules
    model.load_state_dict(dict(gguf.hf_state(tiny)))
    glm5next.wrap_topk(model, ties.append)
    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits[0, :-1]
    ref = golden.topk_rows(logits, torch.tensor(ids[1:]), 20)
    assert torch.equal(rec.topk_ids, ref[0]) and torch.equal(rec.target_lp, ref[2])
    assert rec.meta["deviations"] == [glm5next.TIE_RULE] and sum(ties) > 0


@pytest.mark.skipif(not os.environ.get("GLM_LLAMA_QUANTIZE"), reason="needs GLM_LLAMA_QUANTIZE (llama-quantize)")
def test_the_gpu_fixture_is_pinned_and_meets_d2(tmp_path):
    from holon_testing import glm5next_files as g

    q8 = g.build(tmp_path, pathlib.Path(os.environ["GLM_LLAMA_CPP"]), "gpu",
                 pathlib.Path(os.environ["GLM_LLAMA_QUANTIZE"]))
    assert hashlib.sha256(q8.read_bytes()).hexdigest() == g.GGUF_SHA256["gpu"]
    model, ids = g.tiny_hf("gpu"), g.prompts("gpu")
    with torch.inference_mode():
        assert model(torch.tensor([ids[0]])).logits[0].std(-1).mean() >= 2        # A-N2
    _meets_d2(g.stats(model, ids))


def _meets_d2(stats: dict) -> None:
    """D2: the clamp binds on >= 5% of every path's rows, on each bound; most comb rows peaked."""
    rows = {k: v for k, v in stats.items() if k != "comb_peaked" and not k.endswith("elements")}
    assert len(rows) == 9 and min(rows.values()) >= 0.05 and stats["comb_peaked"] >= 0.5, stats


def test_the_release_f32_tensors_stay_full_precision(tiny):
    """Step 2's review A-m1: the tensors the release keeps F32 are not bf16 values in the fixture, so a recipe that
    loads one in the model dtype shows."""
    from holon_testing import glm5next_files as g

    state = g.tiny_hf("cpu").state_dict()
    kept = [k for k in state if any(n in k for n in g.F32)]
    assert {n for n in g.F32 if any(n in k for k in kept)} == set(g.F32)
    assert all(not torch.equal(state[k], state[k].bfloat16().float()) for k in kept), kept


def test_the_committed_fixtures_are_the_tools(tiny):
    from holon_testing import glm5next_files as g

    tensors, meta = g.fixtures(g.tiny_hf("cpu"), tiny)
    committed, committed_meta = g.load_fixtures()
    assert tensors.keys() == committed.keys() and all(torch.equal(tensors[k], committed[k]) for k in tensors)
    assert all(meta[k] == committed_meta[k] for k in meta)
    assert all(int(meta[k]) > 0 for k in meta if k.endswith("boundary_ties"))        # ties occur
    _meets_d2(json.loads(meta["stats"]))
