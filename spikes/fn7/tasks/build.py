"""FN's task-accuracy set (docs/v1-flash-next.md §18 (d)): 72 items with checkable answers, all synthetic — written
for this repository or generated here from a fixed seed, so they carry the repository's license (Apache-2.0) and no
third-party text. Every item is a chat (`messages`) with an `answer` and a `check`:

- `math` (24): multi-step problems from eight templates (CRT, digit sums of powers, inclusion-exclusion counts,
  modular recurrences, 3×3 integer linear systems, lattice paths around a blocked point, change from bulk pricing,
  train arrival times); exact answers computed here.
- `code` (14): Python functions specified in prose, checked by running the item's asserts against the answer's last
  ```python block.
- `reasoning` (16): multiple choice (A–E) — seating orders from generated constraints with one solution, knights and
  knaves with one consistent assignment, weekday arithmetic.
- `recall` (12): a needle, a two-hop needle pair or two numbers to add, hidden in generated filler (fictional
  ledgers) of about 8K and 32K tokens.
- `multiturn` (6): facts given and changed over three user turns with fixed assistant replies, then a question.

Answers end with a line `Final answer: <value>` (the prompts ask for it); `run.py` scores them.

    python spikes/fn7/tasks/build.py --tokenizer <a Qwen3.8-Flash-Next .gguf> --out spikes/fn7/tasks/fn_tasks.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import pathlib
import random

SEED = 7
FINAL = "End your reply with a line of the form `Final answer: <answer>`."
SOURCE = "synthetic: spikes/fn7/tasks/build.py, seed 7 (Apache-2.0, this repository)"


def item(cat: str, n: int, messages: list[dict], answer, check: str, **extra) -> dict:
    return {"id": f"{cat}-{n:02d}", "category": cat, "messages": messages, "answer": answer, "check": check,
            "source": SOURCE, **extra}


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ---- math -------------------------------------------------------------------------------------------------------
def math_items(r: random.Random) -> list[dict]:
    out = []

    def add(text: str, answer, check: str = "number") -> None:
        out.append(item("math", len(out), user(f"{text}\n\n{FINAL}"), answer, check))

    for _ in range(3):
        ms = r.sample([5, 7, 8, 9, 11, 13], 3)
        while math.gcd(ms[0], ms[1]) > 1 or math.gcd(ms[0], ms[2]) > 1 or math.gcd(ms[1], ms[2]) > 1:
            ms = r.sample([5, 7, 8, 9, 11, 13], 3)
        rs = [r.randrange(1, m) for m in ms]
        n = next(n for n in itertools.count(1) if all(n % m == x for m, x in zip(ms, rs, strict=True)))
        add(f"Find the smallest positive integer n such that n leaves remainder {rs[0]} when divided by {ms[0]}, "
            f"remainder {rs[1]} when divided by {ms[1]}, and remainder {rs[2]} when divided by {ms[2]}.", n)
    for _ in range(3):
        b, e = r.choice([2, 3, 7]), r.randrange(30, 61)
        add(f"What is the sum of the decimal digits of {b}^{e}?", sum(map(int, str(b ** e))))
    for _ in range(3):
        n, a, b, c = r.randrange(1000, 5001), *r.sample([3, 4, 5, 6, 7, 11], 2), r.choice([2, 10, 13])
        ans = sum(1 for k in range(1, n + 1) if (k % a == 0 or k % b == 0) and k % c)
        add(f"How many integers k with 1 <= k <= {n} are divisible by {a} or by {b}, but not divisible by {c}?", ans)
    for _ in range(3):
        x, k, c, m = r.randrange(2, 50), r.randrange(3, 20), r.randrange(1, 40), r.choice([97, 101, 103, 127])
        a = x
        for _ in range(11):
            a = (k * a + c) % m
        add(f"A sequence starts with a_1 = {x} and continues with a_(n+1) = ({k} * a_n + {c}) mod {m}. "
            "What is a_12?", a)
    for _ in range(3):
        while True:
            sol = [r.randrange(-9, 10) for _ in range(3)]
            mat = [[r.randrange(-5, 6) for _ in range(3)] for _ in range(3)]
            det = (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                   - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
                   + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]))
            if det:
                break
        rhs = [sum(a * s for a, s in zip(row, sol, strict=True)) for row in mat]
        eqs = "\n".join(f"{row[0]}x + {row[1]}y + {row[2]}z = {v}".replace("+ -", "- ") for row, v in
                        zip(mat, rhs, strict=True))
        add(f"Solve the system of equations\n{eqs}\nand give the value of x*y + z.", sol[0] * sol[1] + sol[2])
    for _ in range(3):
        w, h = r.randrange(5, 9), r.randrange(4, 8)
        bx, by = r.randrange(1, w), r.randrange(1, h)
        paths = [[0] * (h + 1) for _ in range(w + 1)]
        for i in range(w + 1):
            for j in range(h + 1):
                paths[i][j] = 0 if (i, j) == (bx, by) else 1 if i == j == 0 else \
                    (paths[i - 1][j] if i else 0) + (paths[i][j - 1] if j else 0)
        add(f"A robot walks on grid points from (0, 0) to ({w}, {h}), each step moving one unit in the +x or +y "
            f"direction. The point ({bx}, {by}) is blocked and may not be visited. How many different paths are "
            "there?", paths[w][h])
    for _ in range(3):
        single, pack, n = r.randrange(40, 90), 0, r.randrange(50, 140)
        pack = single * 12 - r.randrange(60, 200)
        bill = r.choice([5000, 10000])
        cost = n // 12 * pack + n % 12 * single
        while cost > bill:
            n -= 12
            cost = n // 12 * pack + n % 12 * single
        add(f"A shop sells pens for {single} cents each, or in packs of 12 for {pack} cents per pack. Ana needs "
            f"exactly {n} pens: she buys as many full packs as she can use and the rest as single pens. She pays "
            f"with a {bill // 100}-dollar bill. How much change, in cents, does she get?", bill - cost)
    for _ in range(3):
        hh, mm = r.randrange(5, 12), r.choice([0, 10, 15, 25, 40, 50])
        speed, dist, stop = r.choice([60, 80, 90, 120]), r.choice([300, 360, 420, 540]), r.choice([7, 12, 15])
        minutes = dist * 60 // speed + (math.ceil(dist / 100) - 1) * stop
        t = hh * 60 + mm + minutes
        add(f"A train leaves at {hh:02d}:{mm:02d} and travels {dist} km at a constant {speed} km/h. It stops for "
            f"{stop} minutes after every 100 km travelled, except at its destination. At what time does it "
            "arrive? Give the time as HH:MM on a 24-hour clock.", f"{t // 60 % 24:02d}:{t % 60:02d}", "text")
    return out


# ---- code -------------------------------------------------------------------------------------------------------
CODE = [
    ("rle", "rle(s: str) -> str", "returns the run-length encoding of s: each maximal run of one character becomes the "
     "character followed by the run length (e.g. 'aaabcc' -> 'a3b1c2'); the empty string maps to ''",
     ["rle('aaabcc') == 'a3b1c2'", "rle('') == ''", "rle('z') == 'z1'", "rle('aabbbbbbbbbbb') == 'a2b11'"]),
    ("balanced", "balanced(s: str) -> bool", "tells whether the brackets ()[]{} in s are balanced and properly nested "
     "(other characters are ignored)", ["balanced('a(b[c]{d})e')", "not balanced('(]')", "balanced('')",
                                        "not balanced('(()')", "not balanced('{[}]')"]),
    ("roman", "roman(s: str) -> int", "converts a Roman numeral (I, V, X, L, C, D, M, subtractive forms allowed) to an "
     "integer", ["roman('MCMXCIV') == 1994", "roman('IV') == 4", "roman('LVIII') == 58", "roman('MMMCMXCIX') == 3999"]),
    ("merge", "merge(intervals: list[list[int]]) -> list[list[int]]", "merges overlapping closed intervals "
     "(touching ones too, e.g. [1, 3] and [3, 5]) and returns them sorted by start",
     ["merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]", "merge([[1,4],[4,5]]) == [[1,5]]",
      "merge([]) == []", "merge([[5,7],[1,2],[2,3]]) == [[1,3],[5,7]]"]),
    ("spiral", "spiral(m: list[list[int]]) -> list[int]", "returns the elements of a rectangular matrix in clockwise "
     "spiral order starting at the top-left", ["spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
                                               "spiral([[1,2,3,4]]) == [1,2,3,4]", "spiral([[1],[2],[3]]) == [1,2,3]",
                                               "spiral([[1,2],[3,4],[5,6]]) == [1,2,4,6,5,3]"]),
    ("islands", "islands(grid: list[str]) -> int", "counts the islands in a grid of '1' (land) and '0' (water) "
     "strings, where land cells connect horizontally and vertically",
     ["islands(['11000','11000','00100','00011']) == 3", "islands(['0']) == 0", "islands(['101','010','101']) == 5",
      "islands(['111','101','111']) == 1"]),
    ("lis", "lis(a: list[int]) -> int", "returns the length of the longest strictly increasing subsequence of a",
     ["lis([10,9,2,5,3,7,101,18]) == 4", "lis([]) == 0", "lis([7,7,7]) == 1", "lis([0,1,0,3,2,3]) == 4"]),
    ("edit", "edit(a: str, b: str) -> int", "returns the Levenshtein distance (insertions, deletions, substitutions) "
     "between a and b", ["edit('kitten','sitting') == 3", "edit('','abc') == 3", "edit('flaw','lawn') == 2",
                         "edit('same','same') == 0"]),
    ("evaluate", "evaluate(expr: str) -> int", "evaluates an integer arithmetic expression with +, -, *, parentheses "
     "and spaces, with the usual precedence; do not use eval or exec",
     ["evaluate('2+3*4') == 14", "evaluate('(2+3)*4') == 20", "evaluate(' 10 - 2 - 3 ') == 5",
      "evaluate('2*(3+(4-1))*2') == 24", "evaluate('7') == 7"]),
    ("topk", "topk(words: list[str], k: int) -> list[str]", "returns the k most frequent words, ordered by frequency "
     "descending and then alphabetically", ["topk(['a','b','a','c','b','a'], 2) == ['a','b']",
                                            "topk(['x','y','z'], 2) == ['x','y']",
                                            "topk(['b','a','b','a','c'], 3) == ['a','b','c']"]),
    ("insert_at", "insert_at(a: list[int], x: int) -> int", "returns the leftmost index at which x could be inserted "
     "into the sorted list a keeping it sorted, using binary search",
     ["insert_at([1,3,5,7], 4) == 2", "insert_at([], 1) == 0", "insert_at([1,2,2,2,3], 2) == 1",
      "insert_at([1,2,3], 9) == 3"]),
    ("toposort_ok", "toposort_ok(n: int, edges: list[tuple[int, int]]) -> bool", "tells whether the directed graph on "
     "nodes 0..n-1 with edges (u, v) meaning u -> v has no cycle",
     ["toposort_ok(3, [(0,1),(1,2)])", "not toposort_ok(3, [(0,1),(1,2),(2,0)])", "toposort_ok(1, [])",
      "not toposort_ok(2, [(1,1)])"]),
    ("kth_two", "kth_two(a: list[int], b: list[int], k: int) -> int", "returns the k-th smallest element (1-based) of "
     "the union (with repeats) of two sorted lists", ["kth_two([1,3,5],[2,4,6],4) == 4", "kth_two([],[1,2],2) == 2",
                                                      "kth_two([1,1],[1,2],4) == 2", "kth_two([5],[1,2,3],1) == 1"]),
    ("wrap", "wrap(text: str, width: int) -> list[str]", "greedily wraps the words of text (split on whitespace) into "
     "lines of at most width characters, words joined by single spaces; a word longer than width gets its own line",
     ["wrap('the quick brown fox', 10) == ['the quick', 'brown fox']", "wrap('', 5) == []",
      "wrap('a bb ccc', 3) == ['a', 'bb', 'ccc']", "wrap('extraordinary day', 5) == ['extraordinary', 'day']"]),
]


def code_items() -> list[dict]:
    return [item("code", i, user(f"Write a Python function `{sig}` that {spec}. Reply with the complete function in "
                                 "one ```python block (no tests, no input reading)."), None, "code", tests=tests,
                 function=name)
            for i, (name, sig, spec, tests) in enumerate(CODE)]


# ---- reasoning ----------------------------------------------------------------------------------------------------
NAMES = ["Ada", "Bram", "Cleo", "Dov", "Esme", "Faro", "Gita", "Hugo"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [("January", 31), ("February", 28), ("March", 31), ("April", 30), ("May", 31), ("June", 30), ("July", 31),
          ("August", 31), ("September", 30), ("October", 31), ("November", 30), ("December", 31)]


def mc(r: random.Random, n: int, question: str, correct: str, wrong: list[str]) -> dict:
    options = [correct, *r.sample(wrong, 4)]
    r.shuffle(options)
    letter = "ABCDE"[options.index(correct)]
    text = question + "\n" + "\n".join(f"{'ABCDE'[i]}. {o}" for i, o in enumerate(options))
    return item("reasoning", n, user(f"{text}\n\nAnswer with the letter of the correct option. {FINAL}"), letter,
                "letter")


def reasoning_items(r: random.Random) -> list[dict]:
    out = []
    for _ in range(6):                                  # seating: constraints until one order fits
        people = r.sample(NAMES, 5)
        truth = people[:]
        r.shuffle(truth)
        pos = {p: i for i, p in enumerate(truth)}
        clues, orders = [], list(itertools.permutations(people))
        while len(orders) > 1:
            a, b = r.sample(people, 2)
            kind = r.choice(["left", "adjacent", "not_end", "gap"])
            if kind == "left" and pos[a] < pos[b]:
                clue, ok = f"{a} sits somewhere to the left of {b}.", lambda o, a=a, b=b: o.index(a) < o.index(b)
            elif kind == "adjacent" and abs(pos[a] - pos[b]) == 1:
                clue, ok = f"{a} sits next to {b}.", lambda o, a=a, b=b: abs(o.index(a) - o.index(b)) == 1
            elif kind == "not_end" and 0 < pos[a] < 4:
                clue, ok = f"{a} does not sit at either end.", lambda o, a=a: 0 < o.index(a) < 4
            elif kind == "gap" and abs(pos[a] - pos[b]) == 2:
                clue, ok = f"Exactly one person sits between {a} and {b}.", \
                    lambda o, a=a, b=b: abs(o.index(a) - o.index(b)) == 2
            else:
                continue
            left = [o for o in orders if ok(o)]
            if len(left) < len(orders):
                clues.append(clue)
                orders = left
        k = r.randrange(5)
        q = ("Five people sit in a row of five chairs, numbered 1 to 5 from left to right.\n" + "\n".join(clues)
             + f"\nWho sits in chair {k + 1}?")
        out.append(mc(r, len(out), q, truth[k], [p for p in people if p != truth[k]]))
    for _ in range(5):                                  # knights (always truthful) and knaves (always lie)
        people = r.sample(NAMES, 3)
        while True:
            stmts = []
            for p in people:
                o = r.choice([x for x in people if x != p])
                kind = r.choice(["is_knave", "same", "is_knight"])
                stmts.append((p, o, kind))

            def holds(assign, o, kind):
                return {"is_knave": not assign[o], "is_knight": assign[o]}.get(kind, None)
            sols = []
            for bits in itertools.product([True, False], repeat=3):
                assign = dict(zip(people, bits, strict=True))
                good = True
                for p, o, kind in stmts:
                    claim = assign[p] == assign[o] if kind == "same" else holds(assign, o, kind)
                    good &= claim == assign[p]
                if good:
                    sols.append(assign)
            if len(sols) == 1:
                break
        say = {"is_knave": "{o} is a knave.", "is_knight": "{o} is a knight.",
               "same": "{o} and I are of the same kind."}
        q = ("On an island every inhabitant is a knight, who always tells the truth, or a knave, who always lies. "
             + " ".join(f"{p} says: \"{say[k].format(o=o)}\"" for p, o, k in stmts) + "\nWhich is true?")

        def describe(assign, people=people):
            return ", ".join(f"{p} is a {'knight' if assign[p] else 'knave'}" for p in people)
        wrong = [describe(dict(zip(people, bits, strict=True))) for bits in itertools.product([True, False], repeat=3)]
        right = describe(sols[0])
        out.append(mc(r, len(out), q, right, [w for w in wrong if w != right]))
    for _ in range(5):                                  # weekday arithmetic in a non-leap year
        m1, m2 = sorted(r.sample(range(12), 2))
        d1, d2 = r.randrange(1, MONTHS[m1][1] + 1), r.randrange(1, MONTHS[m2][1] + 1)
        w1 = r.randrange(7)
        day = lambda m, d: sum(n for _, n in MONTHS[:m]) + d       # noqa: E731
        w2 = (w1 + day(m2, d2) - day(m1, d1)) % 7
        q = (f"In a year that is not a leap year, {MONTHS[m1][0]} {d1} is a {WEEKDAYS[w1]}. "
             f"What day of the week is {MONTHS[m2][0]} {d2} of the same year?")
        out.append(mc(r, len(out), q, WEEKDAYS[w2], [w for w in WEEKDAYS if w != WEEKDAYS[w2]]))
    return out


# ---- recall -------------------------------------------------------------------------------------------------------
GUILDS = ["coopers", "chandlers", "tanners", "glaziers", "masons", "weavers", "vintners", "smiths", "potters"]
TOWNS = ["Ostrava Vale", "Kellmarch", "Durnhollow", "Brisk Harbor", "Quellstone", "Aldmere", "Fenwick Cross"]
GOODS = ["barrels of pitch", "bolts of linen", "crates of slate", "casks of cider", "sacks of barley", "coils of rope"]


def filler(r: random.Random) -> str:
    year, g, t, n, goods = r.randrange(1700, 1900), r.choice(GUILDS), r.choice(TOWNS), r.randrange(3, 400), \
        r.choice(GOODS)
    return r.choice([
        f"In {year} the {g} of {t} recorded {n} {goods} received before the spring fair.",
        f"The ledger of the {g} notes that {t} sent {n} {goods} by river in {year}.",
        f"By the autumn of {year}, {n} {goods} had been stored in the {g}' hall at {t}.",
        f"A clerk in {t} wrote that the {g} paid for {n} {goods} with silver in {year}.",
    ])


def recall_items(r: random.Random, tokenizer) -> list[dict]:
    out = []
    per_sentence = None
    for target in [8000] * 6 + [31000] * 6:
        kind = ["code", "hop", "sum"][len(out) % 3]
        if kind == "code":
            vault, code = r.choice(["amber", "cobalt", "violet", "saffron"]), r.randrange(100000, 1000000)
            needles = [f"The combination of the {vault} vault is {code}."]
            q, answer, check = f"What is the combination of the {vault} vault?", str(code), "number"
        elif kind == "hop":
            holder, room = r.choice(["Orsolya", "Tamsin", "Ignatz", "Perpetua"]), r.randrange(100, 1000)
            needles = [f"The key to the archive is kept by {holder}.", f"{holder} lives in room {room}."]
            q, answer, check = "In which room does the person who keeps the key to the archive live?", str(room), \
                "number"
        else:
            a, b = r.randrange(20, 500), r.randrange(20, 500)
            needles = [f"The red crate weighs {a} kilograms.", f"The blue crate weighs {b} kilograms."]
            q, answer, check = "What is the combined weight in kilograms of the red crate and the blue crate?", \
                str(a + b), "number"
        if per_sentence is None:
            sample = " ".join(filler(r) for _ in range(200))
            per_sentence = len(tokenizer.encode(sample)) / 200
        sentences = [filler(r) for _ in range(int(target / per_sentence))]
        for nd in needles:
            sentences.insert(r.randrange(len(sentences) // 10, len(sentences) * 9 // 10), nd)
        text = ("Below is a collection of ledger entries. Read them and answer the question after them.\n\n"
                + " ".join(sentences) + f"\n\nQuestion: {q}\n{FINAL}")
        out.append(item("recall", len(out), user(text), answer, check, context_tokens=len(tokenizer.encode(text))))
    return out


# ---- multi-turn ---------------------------------------------------------------------------------------------------
def multiturn_items(r: random.Random) -> list[dict]:
    out = []
    for _ in range(3):                                  # an inventory changed over turns
        boxes = [r.randrange(5, 30) for _ in range(3)]
        give, buy = r.randrange(1, boxes[1]), r.randrange(2, 15)
        total = boxes[0] + buy + boxes[1] - give + boxes[2] * 2
        msgs = [{"role": "user", "content": f"I have three boxes of apples: box 1 holds {boxes[0]}, box 2 holds "
                 f"{boxes[1]} and box 3 holds {boxes[2]}. Please keep track."},
                {"role": "assistant", "content": "Noted: box 1 has {}, box 2 has {} and box 3 has {} apples.".format(
                    *boxes)},
                {"role": "user", "content": f"I gave {give} apples from box 2 to a neighbour and put {buy} newly "
                 "bought apples into box 1."},
                {"role": "assistant", "content": "Understood, I have updated boxes 1 and 2."},
                {"role": "user", "content": f"Now box 3 doubled because my sister added the same amount again. How "
                 f"many apples are there in all three boxes together? {FINAL}"}]
        out.append(item("multiturn", len(out), msgs, total, "number"))
    for _ in range(3):                                  # a schedule moved over turns
        start, dur, shift = r.randrange(8, 12), r.choice([45, 50, 75, 90]), r.choice([30, 45, 60, 90])
        first_end = start * 60 + dur
        second_start = first_end + 15 + shift
        end = second_start + dur
        msgs = [{"role": "user", "content": f"My workshop has two sessions of {dur} minutes each. The first starts at "
                 f"{start:02d}:00 and the second starts 15 minutes after the first ends."},
                {"role": "assistant", "content": "Got it: two sessions, the second 15 minutes after the first."},
                {"role": "user", "content": f"Change of plan: the second session starts {shift} minutes later than "
                 "that. The first session is unchanged."},
                {"role": "assistant", "content": "Okay, I have moved the second session."},
                {"role": "user", "content": f"At what time does the second session end? Use HH:MM on a 24-hour clock. "
                 f"{FINAL}"}]
        out.append(item("multiturn", len(out), msgs, f"{end // 60:02d}:{end % 60:02d}", "text"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="a Qwen3.8-Flash-Next GGUF (its tokenizer sizes the filler)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from holon.core.assembly import resolve_model
    from holon.core.engine import EngineConfig
    from holon.core.plugins import PluginSet, discover
    source, plugin = resolve_model(PluginSet(discover()), EngineConfig(a.tokenizer))
    tokenizer = plugin.bundle(source).tokenizer
    r = random.Random(SEED)
    items = [*math_items(r), *code_items(), *reasoning_items(r), *recall_items(r, tokenizer), *multiturn_items(r)]
    meta = {"name": "fn_tasks", "built": datetime.date.today().isoformat(), "items": len(items), "source": SOURCE}
    pathlib.Path(a.out).write_text("\n".join(json.dumps(x) for x in [meta, *items]) + "\n")
    print(json.dumps(meta), {c: sum(x["category"] == c for x in items) for c in dict.fromkeys(x["category"]
                                                                                              for x in items)})


if __name__ == "__main__":
    main()
