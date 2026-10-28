"""Build the frozen L2 prompt set (run once; the JSONL is the artifact, this script is provenance).

Each sequence = a Qwen3-chat-template-rendered conversation whose final assistant turn is generated greedily by
Qwen3-8B (HF bf16, GPU) so that the teacher-forced positions mix off-policy context (user turns, documents, tool
output) with on-policy model text. Token ids are frozen in the JSONL; the generator never runs again.

    CUDA_VISIBLE_DEVICES=0 python build_prompts.py --out prompts/v1.jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import re

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_MODEL = "Qwen/Qwen3-8B"
MIN_LEN, MAX_LEN = 512, 2048
HUB = os.path.expanduser("~/.cache/huggingface/hub")

SYSTEM = "You are a helpful, precise assistant."

# --- chat ---------------------------------------------------------------------------------------------------------
CHAT = [
    ("chat-explain", False, "Explain how a modern CPU's branch predictor works, why mispredictions are expensive, and "
     "give two concrete code-level techniques to reduce them. Be detailed."),
    ("chat-math-think", True, "A bag has 5 red, 4 blue and 3 green marbles. Three marbles are drawn without "
     "replacement. What is the probability that all three are different colours, and what is the probability that "
     "at least two are red? Show your work."),
    ("chat-zh", False, "请详细介绍一下中国古代四大发明，以及它们对世界文明发展的影响。"),
    ("chat-story", False, "Write a short story (about 500 words) about a lighthouse keeper who discovers that the "
     "light has been signalling to something under the sea."),
    ("chat-advice-think", True, "I'm choosing between PostgreSQL and MongoDB for a new SaaS product that stores "
     "invoices, customers and audit logs. Compare them for this use case and recommend one."),
    ("chat-multiturn", False, None),  # filled below
]
CHAT_MULTITURN = [
    {"role": "user", "content": "What's the difference between a process and a thread?"},
    {"role": "assistant", "content": "A process is an independent program in execution with its own address space, "
     "file descriptors and resources. A thread is a unit of execution inside a process; threads of one process "
     "share its address space and resources but each has its own stack, registers and program counter. Creating "
     "and switching threads is cheaper, but shared memory means you need synchronisation."},
    {"role": "user", "content": "OK. Then why does Python have a GIL, and when should I use multiprocessing vs "
     "threading vs asyncio? Give examples for each."},
]

# --- code ---------------------------------------------------------------------------------------------------------
BUGGY = '''def merge_intervals(intervals):
    intervals.sort()
    out = []
    for s, e in intervals:
        if out and s < out[-1][1]:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out
'''
CODE = [
    ("code-lru", False, "Implement an LRU cache in Python with O(1) get and put, without using OrderedDict or "
     "functools. Include type hints, docstrings and a few unit tests with pytest."),
    ("code-debug", False, "This function is supposed to merge overlapping intervals but gives wrong results for "
     "some inputs. Find all bugs, explain them, and give a corrected version with tests.\n\n```python\n"
     + BUGGY + "```"),
    ("code-cuda", False, "Write a CUDA C++ kernel for row-wise softmax over a [rows, cols] float matrix where "
     "cols <= 4096, using one block per row and warp shuffles for the reductions. Explain the numerical-stability "
     "trick and give the launch code."),
    ("code-rust-think", True, "In Rust, write a function that parses a simple arithmetic expression with +, -, *, / "
     "and parentheses into an AST and evaluates it. Handle errors with Result."),
    ("code-lcb-0", False, "lcb:0"),
    ("code-lcb-1", True, "lcb:7"),
]

# --- long-context recall --------------------------------------------------------------------------------------------
NEEDLES = [
    ("The access code for the east archive room is 7341-K.", "What is the access code for the east archive room?"),
    ("Dr. Imelda Varga's favourite fruit is the persimmon.", "What is Dr. Imelda Varga's favourite fruit?"),
    ("The backup generator was serviced on 14 March by a technician named Okafor.",
     "Who serviced the backup generator, and on what date?"),
    ("The secret ingredient in the lab's coffee is cardamom.", "What is the secret ingredient in the lab's coffee?"),
    (None, "Summarise the document above in five bullet points."),
    (None, "List every year mentioned in the document above, with what happened in that year."),
]
RECALL_ARTICLES = [0, 2, 5, 8, 10, 13]

# --- agent / tool-call transcripts ------------------------------------------------------------------------------
def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


TOOLS = [
    fn("search_web", "Search the web and return the top results.", {"query": {"type": "string"}}, ["query"]),
    fn("read_file", "Read a text file from the workspace.", {"path": {"type": "string"}}, ["path"]),
    fn("run_python", "Execute Python code and return stdout/stderr.", {"code": {"type": "string"}}, ["code"]),
    fn("get_weather", "Current weather for a city.", {"city": {"type": "string"},
       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("send_email", "Send an email.", {"to": {"type": "string"}, "subject": {"type": "string"},
       "body": {"type": "string"}}, ["to", "subject", "body"]),
]


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": args}}]}


def tool(obj):
    return {"role": "tool", "content": json.dumps(obj)}


CSV_HEAD = "date,region,units,revenue\n" + "\n".join(
    f"2024-{m:02d}-01,{r},{(m * 37 + i * 11) % 90 + 10},{((m * 37 + i * 11) % 90 + 10) * 129}"
    for m in range(1, 13) for i, r in enumerate(["north", "south", "east", "west"]))
AGENT = [
    ("agent-weather", False, [
        {"role": "user", "content": "I'm flying from Berlin to Lisbon tomorrow. Check the weather in both and tell "
         "me what to pack."},
        call("get_weather", city="Berlin", unit="celsius"),
        tool({"city": "Berlin", "temp_c": 6, "condition": "light rain", "wind_kmh": 22, "humidity": 88}),
        call("get_weather", city="Lisbon", unit="celsius"),
        tool({"city": "Lisbon", "temp_c": 19, "condition": "sunny", "wind_kmh": 9, "humidity": 55}),
    ]),
    ("agent-csv", False, [
        {"role": "user", "content": "Load sales.csv, compute total revenue per region and tell me which region "
         "grew most from Q1 to Q4."},
        call("read_file", path="sales.csv"),
        tool({"path": "sales.csv", "content": CSV_HEAD}),
    ]),
    ("agent-pycode", True, [
        {"role": "user", "content": "How many prime numbers are there below 1,000,000, and what is their sum? Use "
         "the Python tool."},
        call("run_python", code="import sympy\nps = list(sympy.primerange(2, 10**6))\nprint(len(ps), sum(ps))"),
        tool({"stdout": "", "stderr": "ModuleNotFoundError: No module named 'sympy'", "exit_code": 1}),
    ]),
    ("agent-search", False, [
        {"role": "user", "content": "Find out what the James Webb Space Telescope's primary mirror is made of and "
         "how big it is, then email a two-paragraph summary to alex@example.com."},
        call("search_web", query="James Webb Space Telescope primary mirror material size"),
        tool({"results": [
            {"title": "JWST Mirrors - NASA", "snippet": "The primary mirror is 6.5 metres across and consists of 18 "
             "hexagonal segments made of beryllium, each coated with a thin layer of gold to improve reflectivity "
             "of infrared light."},
            {"title": "Why beryllium?", "snippet": "Beryllium is strong and light and holds its shape across a "
             "wide range of temperatures, which matters because the telescope operates at about 40 K."},
            {"title": "Mirror segment facts", "snippet": "Each segment weighs about 20 kg and measures 1.32 m "
             "flat to flat. Actuators adjust each segment to nanometre precision."}]}),
    ]),
    ("agent-multistep", False, [
        {"role": "user", "content": "Read config.yaml and tell me whether the database settings are safe for "
         "production. Fix anything that is not and show me the corrected file."},
        call("read_file", path="config.yaml"),
        tool({"path": "config.yaml", "content": "server:\n  host: 0.0.0.0\n  port: 8080\n  debug: true\n"
              "database:\n  url: postgres://admin:admin123@db:5432/app\n  pool_size: 200\n  ssl: false\n"
              "  timeout_s: 0\nlogging:\n  level: DEBUG\n  file: /tmp/app.log\n"}),
    ]),
    ("agent-retry", True, [
        {"role": "user", "content": "What's the weather in Tokyo in fahrenheit? If the tool fails, retry once and "
         "then explain the error to me."},
        call("get_weather", city="Tokyo", unit="fahrenheit"),
        tool({"error": "upstream timeout after 10s", "code": 504}),
        call("get_weather", city="Tokyo", unit="fahrenheit"),
        tool({"error": "upstream timeout after 10s", "code": 504}),
    ]),
]


def wiki_articles() -> list[str]:
    path = glob.glob(f"{HUB}/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*.parquet")[0]
    lines = pq.read_table(path).column("text").to_pylist()
    starts = [i for i, x in enumerate(lines) if x.startswith(" = ") and not x.startswith(" = = ")] + [len(lines)]
    detok = lambda s: re.sub(r" ([,.;:)'])", r"\1", s.replace(" @-@ ", "-").replace(" @,@ ", ",")
                             .replace(" @.@ ", ".").replace("( ", "("))
    return [detok("".join(lines[a:b])).strip() for a, b in zip(starts, starts[1:])]


def lcb_problem(i: int) -> str:
    path = glob.glob(f"{HUB}/datasets--livecodebench--code_generation_lite/snapshots/*/test.jsonl")[0]
    with open(path) as f:
        for j, line in enumerate(f):
            if j == i:
                q = json.loads(line)
                return (f"Solve this competitive-programming problem in Python 3. Read from stdin and write to "
                        f"stdout.\n\n{q['question_title']}\n\n{q['question_content']}")
    raise IndexError(i)


def specs(tok) -> list[dict]:
    out = []
    for name, think, q in CHAT:
        msgs = CHAT_MULTITURN if q is None else [{"role": "user", "content": q}]
        out.append(dict(id=name, category="chat", think=think, messages=[{"role": "system", "content": SYSTEM}] + msgs))
    for name, think, q in CODE:
        q = lcb_problem(int(q[4:])) if q.startswith("lcb:") else q
        out.append(dict(id=name, category="code", think=think, messages=[{"role": "user", "content": q}]))
    arts = wiki_articles()
    for k, (a, (needle, question)) in enumerate(zip(RECALL_ARTICLES, NEEDLES)):
        paras = [p for p in arts[a].split("\n") if p.strip()]
        if needle:
            paras.insert(len(paras) * (k + 1) // (len(NEEDLES) + 1), needle)
        doc = "\n".join(paras)
        ids = tok(doc).input_ids[:1400]  # keep context ≤ ~1.4k tokens so prompt + answer stays ≤ MAX_LEN
        doc = tok.decode(ids)
        out.append(dict(id=f"recall-{k}", category="recall", think=False, messages=[
            {"role": "user", "content": f"Read the following document carefully.\n\n<document>\n{doc}\n</document>"
             f"\n\n{question}"}]))
    for name, think, msgs in AGENT:
        out.append(dict(id=name, category="agent", think=think, tools=TOOLS,
                        messages=[{"role": "system", "content": "You are an agent. Use tools when helpful."}] + msgs))
    return out


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts/v1.jsonl")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    rows = []
    for s in specs(tok):
        prompt = tok.apply_chat_template(s["messages"], tools=s.get("tools"), add_generation_prompt=True,
                                         enable_thinking=s["think"], tokenize=True, return_dict=False)
        prompt = list(prompt["input_ids"]) if isinstance(prompt, dict) else list(prompt)
        x = torch.tensor([prompt], device="cuda:0")
        gen = model.generate(x, max_new_tokens=MAX_LEN - len(prompt), do_sample=False, temperature=None,
                             top_p=None, top_k=None)[0, len(prompt):].tolist()
        ids = prompt + gen
        if not MIN_LEN <= len(ids) <= MAX_LEN:
            print(f"WARNING {s['id']}: {len(ids)} tokens outside [{MIN_LEN}, {MAX_LEN}]")
        rows.append(dict(id=s["id"], category=s["category"], think=s["think"], n_tokens=len(ids),
                         gen_start=len(prompt), ids=ids, text=tok.decode(ids)))
        print(f"{s['id']:22s} prompt={len(prompt):5d} total={len(ids):5d}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    with open(args.out, "wb") as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    with open(args.out + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(args.out)}\n")
    print(f"{len(rows)} sequences, {sum(r['n_tokens'] for r in rows)} tokens, sha256={digest}")


if __name__ == "__main__":
    main()
