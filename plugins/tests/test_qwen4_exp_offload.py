"""Qwen3.8-Flash-Next served at full width on a 24 GB card: the offload store's MTP deployment with 64 sequences at
`--memory-fraction 0.97` runs the snapshot trace (holon_judge.snapshot: every graph bucket, sampled and filtered rows
with logprobs, a 2K prompt beside 62 short ones in one prefill step) without running out of memory. It did not
before FN's close (§22.2): the startup profile left out the device memory the graph execs of every bucket take outside
the allocator (1.3 GB at 64 sequences), and the sampler sorted all of a 256-row verify step's vocabulary at once.

Needs the real model and the MTP pack: `FN_GGUF=<…-IQ2_XS-00001-of-00002.gguf>`, `FN_MTP=<the Strata-format pack>`;
a GPU of at most 24 GB (on a larger one the case is not the edge it tests)."""
import os
import pathlib

import pytest
import torch

GGUF, MTP = os.environ.get("FN_GGUF"), os.environ.get("FN_MTP")
pytestmark = pytest.mark.skipif(GGUF is None or MTP is None or not torch.cuda.is_available(),
                                reason="needs FN_GGUF, FN_MTP and a GPU")
PREBUILT = ("ggml_cuda", "expert_io")  # CUDA extensions built if missing (conftest)


def test_sixty_four_sequences_at_097_fit():
    import argparse
    import dataclasses
    import shlex

    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from holon_judge import snapshot
    from holon_judge.recording import PromptSet

    flags = ("--memory-fraction 0.97 --max-step-tokens 4096 --host-memory 70000000000 --max-draft-tokens 3 "
             "--slots drafter=drafters.mtp draft_policy=std.spec.fixed_k verifier=std.spec.greedy --plugin-options "
             "moe.store=std.moe.offload@1 moe.ring_lend=tail ggml_cuda.pass_rows=4096 "
             "rows.place=ple:host,token_embd:host moe.ring_copy=dma moe.router_dtype=bfloat16 hc.prefill=bf16x3 "
             f"spec.k=3 mtp.path={MTP} moe.cache=aux.*=0.3")
    prompts = PromptSet.load(pathlib.Path(__file__).resolve().parents[2] / "judge/prompts/v1-long.jsonl")
    parser = argparse.ArgumentParser()
    engine_flags(parser)
    e = Engine(dataclasses.replace(engine_config(GGUF, parser.parse_args(shlex.split(flags))), max_seqs=64,
                                   max_model_len=33792))
    try:
        snap = snapshot.record(e, snapshot.trace(prompts, long_len=2048))
    finally:
        e.close()
    assert len(snap.streams) == 64 and all(s for s in snap.streams)
