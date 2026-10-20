"""The judge on Qwen3.8-Flash-Next (FN D2), in the golden venv (transformers 5.19.0, `local/envs/golden-qwen4`): the
tiny model's GGUF, rebuilt by llama.cpp's converter to its pinned sha256, reads back as the checkpoint
(`gguf.hf_state`, PLE's rows lazily); the streamed golden of it equals HF's whole forward bit for bit under the
indexer's tie rule, which the prompt's selections exercise (boundary ties occur), and its bf16 floor runs on the CPU;
the floors' MoE mirror is HF's sparse-MoE block with the gated shared expert; the committed fixtures are what the tool
writes; and the MTP head's pooled reference reads the tiny Strata-format pack as the checkpoint's tensors."""
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
    assert gguf.emulate_q8_1(model, file, "mmq") == {}      # a floor passes unquantized tensors the map lacks (indexer)


def _prompt(model) -> list[int]:
    torch.manual_seed(1)
    ids = torch.randint(0, 248000, (100,)).tolist()
    ids[40] = model.config.eos_token_id
    return ids


def test_the_streamed_golden_is_hfs_forward_and_its_bf16_floor_runs(tiny):
    from holon_judge import golden, qwen4exp
    from holon_judge.recording import Prompt, PromptSet
    from holon_testing import qwen4exp_files as q

    _, path, file = tiny
    model = q.tiny_hf()                                          # its own: the tie rule wraps its indexers
    ids, ties = _prompt(model), []
    prompts = PromptSet("tiny", "", [Prompt("p", "chat", ids)])
    rec = golden.record(str(file), prompts, config=str(path), k=20)
    qwen4exp.wrap_indexers(model, ties.append)
    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits[0, :-1]
    ref = golden.topk_rows(logits, torch.tensor(ids[1:]), 20)
    assert torch.equal(rec.topk_ids, ref[0]) and torch.equal(rec.target_lp, ref[2])
    assert rec.meta["deviations"] == [qwen4exp.TIE_RULE] and any(ties)
    floor = golden.record(str(file), prompts, dtype=torch.bfloat16, config=str(path), k=20)    # lazy rows in bf16
    assert floor.meta["dtype"] == "bfloat16" and floor.target_lp.isfinite().all()
    assert (floor.topk_ids[:, 0] == rec.topk_ids[:, 0]).float().mean() > 0.9


def test_the_floors_moe_mirror_is_hfs_block_with_its_shared_expert():
    from holon_judge import moe
    from holon_testing import qwen4exp_files as q

    block = q.tiny_hf().model.layers[0].mlp
    x = torch.randn(1, 50, block.gate.hidden_dim, generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        want = block(x)
        assert moe.mirror(block) == 1
        got = block(x)
        shared = torch.sigmoid(block.shared_expert_gate(x)) * block.shared_expert(x)
    assert (got - want).abs().max() < 1e-6 and shared.abs().max() > 1e-4         # the gated shared term is in


@pytest.mark.skipif(not os.environ.get("FN_STRATA"), reason="needs FN_STRATA (Strata at 296b59d: tools/mtp_pack.py)")
def test_the_mtp_pooled_reference_reads_the_strata_pack_as_the_checkpoint(tiny, tmp_path):
    from holon_judge import qwen4exp
    from holon_testing import qwen4exp_files as q

    model = q.tiny_hf()
    pack = q.mtp_pack(model, tmp_path, pathlib.Path(os.environ["FN_STRATA"]), pathlib.Path(os.environ["FN_LLAMA_CPP"]))
    assert hashlib.sha256(pack.read_bytes()).hexdigest() == q.MTP_PACK_SHA256
    packed = qwen4exp.pack_state(pack)
    source = {k: v.float() for k, v in q.mtp_tensors(model).items()}
    experts = [k for k in source if k.endswith(("gate_up_proj", "down_proj")) and ".experts." in k]
    for k in experts:                       # Q2_0 in the pack: its layout is the source's (gate rows before up rows)
        err = lambda x, k=k: ((x - source[k]).pow(2).mean() / source[k].pow(2).mean()).item()   # noqa: E731
        swapped = packed[k].roll(packed[k].shape[1] // 2, dims=1) if "gate_up" in k else packed[k].flip(0)
        assert err(packed[k]) < 0.25 < err(swapped)
        source[k] = packed[k]
    ids = torch.tensor(_prompt(model))
    with torch.inference_mode():
        trunk = {}
        hook = model.model.hyper_connection_mixer.register_forward_pre_hook(
            lambda m, args: trunk.__setitem__("h", args[0][0]))
        model.model(ids[None])
        hook.remove()
        embeds = model.model.embed_tokens(ids)
    want = qwen4exp.mtp_pooled(source, model.config, embeds, trunk["h"])
    got = qwen4exp.mtp_pooled(packed, model.config, embeds, trunk["h"])
    assert (got - want).abs().max() < 1e-4


def test_the_committed_fixtures_are_the_tools(tmp_path):
    """With FN_STRATA, the MTP fixtures too (from the pack the tool makes, pinned); else the trunk's."""
    from holon_testing import qwen4exp_files as q

    model = q.tiny_hf()                                      # a fresh model: the golden test wraps the indexers
    pack = (q.mtp_pack(model, tmp_path, pathlib.Path(os.environ["FN_STRATA"]), pathlib.Path(os.environ["FN_LLAMA_CPP"]))
            if os.environ.get("FN_STRATA") else None)
    tensors, meta = q.fixtures(model, pack)
    committed, committed_meta = q.load_fixtures()
    if pack is None:
        committed = {k: v for k, v in committed.items() if not k.endswith(".mtp_x")}
    assert tensors.keys() == committed.keys() and all(torch.equal(tensors[k], committed[k]) for k in tensors)
    assert all(int(meta[k]) > 0 and meta[k] == committed_meta[k] for k in meta)        # boundary ties occur
