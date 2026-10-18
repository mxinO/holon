"""A prompt set held out from the gate's prompts (the judge corpus, judge/prompts/v1.jsonl) for `holon_tools profile`:
chat prompts asking about excerpts of this repository's design documents, tokenized with the model's tokenizer.

    python spikes/x4_gate/heldout.py <out.jsonl> [--tokenizer Qwen/Qwen3-30B-A3B] [--n 32] [--words 300]
"""
import argparse
import json
import pathlib

from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ("docs/design.md", "docs/plan.md", "docs/v1-offload.md", "docs/release.md")
ASKS = ("Explain the following passage to a new engineer.", "Summarize the key decisions in this text.",
        "What could go wrong with the design described here?", "Write Python pseudocode for what this describes.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--words", type=int, default=300)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    words = [w for d in DOCS for w in (ROOT / d).read_text().split()]
    step = len(words) // a.n
    rows = []
    for i in range(a.n):
        text = " ".join(words[i * step:i * step + a.words])
        msgs = [{"role": "user", "content": f"{ASKS[i % len(ASKS)]}\n\n{text}"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        rows.append({"id": f"heldout-{i}", "category": "docs", "ids": list(ids)})
    pathlib.Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} prompts, {sum(len(r['ids']) for r in rows)} tokens")


if __name__ == "__main__":
    main()
