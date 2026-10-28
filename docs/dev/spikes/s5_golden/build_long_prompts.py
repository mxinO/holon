"""Build the frozen long-context prompt set `judge/prompts/v1-long.jsonl` (docs/dev/v1-flash-next.md acceptance 1b and
D11's comparison corpus; run once, the JSONL is the artifact and this script its provenance).

Eight conversations of 4K-32K tokens in Qwen3.8-Flash-Next's own chat template and tokenizer; four are recall
prompts (a needle sentence at a fixed depth of a long document, then a question about it). Unlike `v1.jsonl`, the
final assistant turn is fixed text - the needle's answer, the dataset's reference highlights or solution, the files'
titles - not a model's greedy output: no model runs, so the set is reproducible on the CPU, and the scored answer
positions of a recall prompt are predictable only by attending back to the needle (the QSA plants' target).

    python spikes/s5_golden/build_long_prompts.py --out judge/prompts/v1-long.jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import re

import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser(
    "~/.cache/huggingface")), "hub")
MODEL = "models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
SYSTEM = "You are a helpful, precise assistant."


def parquet(pattern: str, column: str | None = None):
    path = sorted(glob.glob(f"{HUB}/{pattern}"))[0]
    table = pq.read_table(path)
    return table.column(column).to_pylist() if column else table.to_pylist()


def wiki_articles() -> list[str]:
    """wikitext-103 train articles (v1.jsonl's recall documents come from validation), one string per article."""
    lines = parquet("datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/train-00000-of-00002.parquet",
                    "text")
    starts = [i for i, x in enumerate(lines) if x.startswith(" = ") and not x.startswith(" = = ")] + [len(lines)]
    def detok(s: str) -> str:
        return re.sub(r" ([,.;:)'])", r"\1", s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
                      .replace("( ", "("))
    return [detok("".join(lines[a:b])).strip() for a, b in zip(starts, starts[1:], strict=False)]


class Builder:
    def __init__(self, tok: PreTrainedTokenizerFast):
        self.tok = tok

    def ids(self, messages: list[dict], tools=None, think=False, prompt=False) -> list[int]:
        out = self.tok.apply_chat_template(messages, tools=tools, tokenize=True, enable_thinking=think,
                                           add_generation_prompt=prompt, return_dict=False)
        return list(out["input_ids"]) if isinstance(out, dict) else list(out)

    def n(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def fill(self, pieces: list[str], budget: int) -> list[str]:
        """The leading pieces whose tokens fit `budget` (the last one cut at a paragraph)."""
        out, used = [], 0
        for p in pieces:
            k = self.n(p)
            if used + k <= budget:
                out.append(p)
                used += k
                continue
            paras = p.split("\n")
            while paras and used + self.n("\n".join(paras)) > budget:
                paras.pop()
            if paras:
                out.append("\n".join(paras))
            break
        return out


NEEDLES = [  # (needle, question, answer), one per recall prompt
    ("The maintenance password for the observatory's north dome is Halcyon-482.",
     "What is the maintenance password for the observatory's north dome?",
     "The maintenance password for the observatory's north dome is Halcyon-482."),
    ("Professor Ingrid Solberg keeps her rare-books collection in a cedar cabinet in room 1147.",
     "Where does Professor Ingrid Solberg keep her rare-books collection?",
     "Professor Ingrid Solberg keeps her rare-books collection in a cedar cabinet in room 1147."),
    ("The ferry to Kestrel Island leaves at 06:40 on Tuesdays and returns at 19:15.",
     "When does the Tuesday ferry to Kestrel Island leave, and when does it return?",
     "On Tuesdays the ferry to Kestrel Island leaves at 06:40 and returns at 19:15."),
    ("The prototype battery passed its stress test after 3,912 charge cycles at 45 degrees Celsius.",
     "After how many charge cycles, and at what temperature, did the prototype battery pass its stress test?",
     "The prototype battery passed its stress test after 3,912 charge cycles at 45 degrees Celsius."),
]
# (id, total tokens, needle index, needle depth, first wikitext article)
RECALL = [("recall-4k", 4096, 0, 0.25, 0), ("recall-8k", 8192, 1, 0.75, 40), ("recall-16k", 16384, 2, 0.10, 120),
          ("recall-32k", 32000, 3, 0.50, 260)]


def recall(b: Builder, arts: list[str], name: str, total: int, k: int, depth: float, first: int) -> dict:
    needle, question, answer = NEEDLES[k]
    head = "Read the following document carefully.\n\n<document>\n"
    tail = f"\n</document>\n\n{question}"
    frame = len(b.ids([{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + tail},
                       {"role": "assistant", "content": answer}]))
    doc = "\n\n".join(b.fill(arts[first:], total - frame - b.n(needle) - 8))
    paras = doc.split("\n")
    paras.insert(int(len(paras) * depth), needle)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + "\n".join(paras) + tail}]
    return dict(id=name, category="recall", think=False, messages=msgs, answer=answer)


def summary(b: Builder, total: int) -> dict:
    rows = parquet("datasets--abisee--cnn_dailymail/snapshots/*/3.0.0/validation-*.parquet")
    head = "Here are several news articles.\n\n"
    tail = "\n\nGive the key points of each article, in order, as a short list per article."
    arts, highlights, used = [], [], 0
    for r in rows:
        text = f"Article {len(arts) + 1}:\n{r['article']}"
        point = f"Article {len(arts) + 1}:\n" + "\n".join(f"- {h.strip()}" for h in r["highlights"].split("\n"))
        k = b.n(text) + b.n(point) + 4
        if used + k > total - 96:
            break
        arts.append(text)
        highlights.append(point)
        used += k
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + "\n\n".join(arts) + tail}]
    return dict(id="summary-6k", category="chat", think=False, messages=msgs, answer="\n\n".join(highlights))


def math_think(b: Builder, total: int) -> dict:
    """One OpenR1-Math problem whose own reasoning trace brings the conversation closest to `total` tokens."""
    rows = parquet("datasets--open-r1--OpenR1-Math-220k/snapshots/*/data/train-00000-of-00010.parquet")
    best = None
    for r in rows[:4000]:
        for gen, ok in zip(r["generations"], r["correctness_math_verify"], strict=True):
            if not ok or "</think>" not in gen:
                continue
            k = b.n(r["problem"]) + b.n(gen)
            if k <= total - 64 and (best is None or k > best[0]):
                best = (k, r["problem"], gen)
    _, problem, gen = best
    reasoning, answer = gen.split("</think>", 1)
    msgs = [{"role": "user", "content": problem},
            {"role": "assistant", "content": answer.strip(),
             "reasoning_content": reasoning.replace("<think>", "").strip()}]
    return dict(id="math-think-12k", category="chat", think=True, messages=msgs, answer=None)


def chat(b: Builder, total: int) -> dict:
    """Nemotron chat exchanges in sequence: one long multi-turn conversation."""
    rows = parquet("datasets--nvidia--Nemotron-Post-Training-Dataset-v2/snapshots/*/data/chat-00000-of-*.parquet")
    msgs, used = [{"role": "system", "content": SYSTEM}], 0
    for r in rows:
        turns = [m for m in r["messages"] if m["role"] in ("user", "assistant") and m["content"]]
        if len(turns) < 2 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant" or \
                any(re.search(r"[一-鿿]", m["content"]) for m in turns):
            continue
        k = sum(b.n(m["content"]) + 8 for m in turns)
        if used + k > total - 64:
            if used > total * 0.9:
                break
            continue
        msgs += [{"role": m["role"], "content": m["content"].removeprefix("<think></think>").strip()} for m in turns]
        used += k
    return dict(id="chat-multiturn-20k", category="chat", think=False, messages=msgs, answer=None)


def agent(b: Builder, arts: list[str], total: int) -> dict:
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a text file.",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    msgs = [{"role": "system", "content": "You are an agent. Use tools when helpful."},
            {"role": "user", "content": "The workspace has files notes_1.txt, notes_2.txt and so on. Read them one "
             "by one and then tell me the title of each file's article."}]
    titles, used, i = [], 0, 400
    while True:
        art = arts[i]
        title = art.split("\n")[0].strip(" =")
        path = f"notes_{len(titles) + 1}.txt"
        k = b.n(art) + 48
        if used + k > total - 64 - 16 * (len(titles) + 1):
            break
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                 "name": "read_file", "arguments": {"path": path}}}]},
                 {"role": "tool", "content": json.dumps({"path": path, "content": art}, ensure_ascii=False)}]
        titles.append(f"{path}: {title}")
        used += k
        i += 1
    return dict(id="agent-files-24k", category="agent", think=False, messages=msgs, tools=tools,
                answer="The files' articles:\n" + "\n".join(f"- {t}" for t in titles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="judge/prompts/v1-long.jsonl")
    a = ap.parse_args()
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{HUB}/{MODEL}/tokenizer.json")
    tok.chat_template = open(f"{HUB}/{MODEL}/chat_template.jinja").read()
    b, arts = Builder(tok), [x for x in wiki_articles() if len(x) > 2000]
    specs = [recall(b, arts, *RECALL[0]), summary(b, 6144), recall(b, arts, *RECALL[1]), math_think(b, 12288),
             recall(b, arts, *RECALL[2]), chat(b, 20480), agent(b, arts, 24576), recall(b, arts, *RECALL[3])]
    rows = []
    for s in specs:
        msgs = s["messages"]
        if s["answer"] is not None:
            msgs = msgs + [{"role": "assistant", "content": s["answer"]}]
        ids = b.ids(msgs, s.get("tools"), s["think"])
        prompt = b.ids(msgs[:-1], s.get("tools"), s["think"], prompt=True)   # up to the last assistant turn
        gen_start = len(prompt)
        if ids[:gen_start] != prompt:
            print(f"WARNING {s['id']}: the rendered prompt is not a prefix of the conversation")
        rows.append(dict(id=s["id"], category=s["category"], think=s["think"], n_tokens=len(ids),
                         gen_start=gen_start, ids=ids, text=tok.decode(ids)))
        print(f"{s['id']:20s} total={len(ids):6d} gen_start={gen_start:6d}", flush=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    with open(a.out, "wb") as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    with open(a.out + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(a.out)}\n")
    print(f"{len(rows)} sequences, {sum(r['n_tokens'] for r in rows)} tokens, sha256={digest}")


if __name__ == "__main__":
    main()
