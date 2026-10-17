"""Markdown tables from replay5.json.    python report.py replay5.json"""
import json
import sys

o = json.load(open(sys.argv[1]))
FR = ("10%", "25%", "47%")
MAIN = ("A", "B", "C", "C2", "Bw1", "Bw4", "Bw12", "Bw32", "Bw12_R0", "Bw12_R1", "Bw12_R2", "lru", "lru_pf", "R0", "R1", "R2", "R0_s", "R1_s", "R2_s",
        "R0_all", "R1_all", "R2_all")


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


print("Check (A_m6 / lru_m6 vs M6 replay.py):", all(c["ok"] for c in o["check"].values()), "\n")
print("Traces:", json.dumps(o["traces"]), "\n")
for n in ("agent", "mixed", "chat", "code"):
    h = o["heldout"][n]
    print(f"### Whole trace, held-out {n} (profile {h['profile']}): hit % / fetch MB per token / ms @47.5 / vs A\n")
    rows = []
    for v in MAIN:
        row = [v]
        for f in FR:
            r, a = h["by_fraction"][f][v], h["by_fraction"][f]["A"]
            ms, msa = r["ms_per_token"]["pcie47.5"], a["ms_per_token"]["pcie47.5"]
            row.append(f"{100 * r['hit_rate']:.1f} / {r['bytes_per_token']['fetch'] / 1e6:.1f} / {ms:.2f} / "
                       f"{ms / msa:.3f}")
        rows.append(row)
    table(["variant"] + list(FR), rows)
    print(f"ms per token @25 / @50 (vs A), {n}\n")
    rows = []
    for v in ("A", "B", "C2", "Bw4", "Bw12", "Bw12_R1", "lru", "lru_pf", "R0", "R1", "R2"):
        row = [v]
        for f in FR:
            r, a = h["by_fraction"][f][v], h["by_fraction"][f]["A"]
            row.append(" / ".join(f"{r['ms_per_token'][k]:.2f} ({r['ms_per_token'][k] / a['ms_per_token'][k]:.3f})"
                                  for k in ("pcie25", "pcie50")))
        rows.append(row)
    table(["variant"] + list(FR), rows)

print("### MB per decode token by path, whole trace (hit / fetch / dense / prefill admission), and prefill admission\n")
rows = []
for n in ("agent", "mixed"):
    for f in ("25%", "47%"):
        for v in ("A", "B", "C2", "Bw12", "lru", "lru_pf", "R2"):
            r = o["heldout"][n]["by_fraction"][f][v]
            b = r["bytes_per_token"]
            rows.append([n, f, v, f"{b['hit'] / 1e6:.0f}", f"{b['fetch'] / 1e6:.1f}", f"{b['dense'] / 1e6:.0f}",
                         f"{b['prefill_admission'] / 1e6:.1f}", f"{r['prefill_admission_mb_per_prompt']:.0f}",
                         f"{r['ms_per_token']['pcie47.5']:.2f}", f"{r['ms_per_token_paid']['pcie47.5']:.2f}"])
table(["test", "cached", "variant", "hit", "fetch", "dense", "prefill adm.", "prefill adm. MB per prompt",
       "ms @47.5 free", "ms @47.5 paid"], rows)

print("### Warm-up: decode position in the sequence; hit % / fetch MB per token (whole trace)\n")
rows = []
for n in ("agent", "mixed"):
    for f in ("25%", "47%"):
        for v in ("A", "B", "C2", "Bw12", "lru", "lru_pf", "R0", "R2"):
            w = o["heldout"][n]["by_fraction"][f][v]["warmup"]
            rows.append([n, f, v] + [f"{100 * x['hit_rate']:.1f} / {x['fetch_mb_per_token']:.0f}" for x in w.values()])
        w = o["heldout"][n]["by_fraction"][f]["A"]["warmup"]
rows_n = {n: [x["rows"] for x in o["heldout"][n]["by_fraction"]["47%"]["A"]["warmup"].values()] for n in ("agent", "mixed")}
table(["test", "cached", "variant"] + list(o["heldout"]["agent"]["by_fraction"]["47%"]["A"]["warmup"]), rows)
print("decode rows per bucket:", rows_n, "\n")

print("### Isolated requests (each prompt >= 512 prefill rows replayed alone from the profile seed)\n")
for n in ("agent", "mixed"):
    iso = o["isolated"][n]
    one = iso["47%"]
    print(f"{n}: {one['requests']} requests, prompt rows mean {one['prompt_rows_mean']:.0f}, decode rows mean "
          f"{one['decode_rows_mean']:.0f}\n")
    rows = []
    for v in ("A", "B", "C", "C2", "Bw1", "Bw4", "Bw12", "Bw32", "Bw12_R0", "Bw12_R1", "Bw12_R2", "lru", "lru_pf",
              "R0", "R1", "R2"):
        row = [v]
        for f in FR:
            r, a = iso[f][v], iso[f]["A"]
            ms, msa = r["ms_per_token"]["pcie47.5"], a["ms_per_token"]["pcie47.5"]
            row.append(f"{100 * r['hit_rate']:.1f} / {r['fetch_mb_per_token']:.1f} / {ms:.2f} / {ms / msa:.3f}")
        r = iso["47%"][v]
        paid = r["ms_per_token"]["pcie47.5"] + r["prefill_admission_mb_per_prompt"] / 47.5 / iso["47%"]["decode_rows_mean"]
        row += [f"{r['prefill_admission_mb_per_prompt']:.0f}", f"{paid:.2f} ({paid / iso['47%']['A']['ms_per_token']['pcie47.5']:.3f})"]
        rows.append(row)
    table(["variant"] + list(FR) + ["prefill adm. MB/prompt @47%", "47% ms @47.5 paid (vs A)"], rows)
    rows = []
    for v in ("A", "B", "C2", "Bw12", "Bw12_R1", "lru", "lru_pf", "R0", "R2"):
        w = iso["47%"][v]["warmup"]
        rows.append([v] + [f"{100 * x['hit_rate']:.1f} / {x['fetch_mb_per_token']:.0f}" for x in w.values()])
    table(["47%, warm-up"] + list(iso["47%"]["A"]["warmup"]), rows)

print("### Ring lending (per trace and fraction): ring experts, ring steps, lent, lent restored by the return, kept\n")
rows = []
for n in ("agent", "mixed"):
    for f in FR:
        for v in ("R0", "R1", "R2", "R0_s", "R0_all", "Bw12_R0", "Bw12_R1"):
            r = o["heldout"][n]["by_fraction"][f][v]["ring"]
            rows.append([n, f, v, r["ring_experts"], r["steps"], r["lent"], r["lent_restored"], r["kept"]])
table(["test", "cached", "variant", "ring experts", "ring steps", "lent", "restored", "kept"], rows)
