"""Print the README's markdown tables from results/*.json.  Usage: python report.py [correctness|perf|graph|tiled]"""

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ARCHS = ("sm86", "sm120")


def load(name):
    return {a: json.loads((R / f"{name}_{a}.json").read_text()) for a in ARCHS}


def e(x):  # max_abs / max_rel / fro_rel
    return f"{x['max_abs']:.1e} / {x['max_rel']:.1e} / {x['fro_rel']:.1e}" + ("" if x["finite"] else " **NaN/Inf**")


def walk_finite(obj):
    if isinstance(obj, dict):
        return all(walk_finite(v) for k, v in obj.items() if k != "device") and obj.get("finite", True) is not False
    if isinstance(obj, list):
        return all(walk_finite(v) for v in obj)
    return True


def correctness():
    c = load("correctness")
    for a in ARCHS:
        print(f"\n#### {c[a]['device']['gpu']} ({a}); every tensor finite: {walk_finite(c[a])}\n")
        for m, r in c[a]["models"].items():
            s = r["shape"]
            print(f"\n**{m}** (Hg={s['Hg']}, H={s['H']}) — prefill one shot vs golden; cells: max_abs / max_rel / fro_rel\n")
            print("| T | kernel o | kernel state | HF bf16 o | HF bf16 state | HF fp32 chunk o |")
            print("|---|---|---|---|---|---|")
            for x in r["prefill_one_shot"]:
                print(f"| {x['T']} | {e(x['kernel_o'])} | {e(x['kernel_state'])} | {e(x['hf_bf16_o'])} | "
                      f"{e(x['hf_bf16_state'])} | {e(x['hf_fp32_chunk_o'])} |")
            print("\n| batch-4 varlen T | h0≠0 | o | state |\n|---|---|---|---|")
            for x in r["prefill_batch4_varlen"]:
                print(f"| {x['T']} | {x['h0_nonzero']} | {e(x['o'])} | {e(x['state'])} |")
            print("\n| resume cuts | bitwise = one shot | resumed vs one shot o | resumed o vs golden | resumed state vs golden |")
            print("|---|---|---|---|---|")
            for x in r["prefill_resume"].values():
                print(f"| {x['cuts']} | {x['bitwise_equal']} | {e(x['resumed_vs_one_shot_o'])} | {e(x['resumed_o'])} | "
                      f"{e(x['resumed_state'])} |")
            print("\n| decode kernel (32 steps after prefill) | worst o | worst state | other slots untouched |")
            print("|---|---|---|---|")
            for k, x in r["decode_after_prefill"].items():
                wo = max(x["rows"], key=lambda y: y["o"]["fro_rel"])["o"]
                ws = max(x["rows"], key=lambda y: y["state"]["fro_rel"])["state"]
                print(f"| {k} | {e(wo)} | {e(ws)} | {x['other_slots_untouched']} |")
            v = r["verify_k"]
            a_, b_ = v["checkpoint_slots"]["step_a"], v["checkpoint_slots"]["step_b"]
            print(f"\nverify k={v['k']}, n_accept={v['n_accept']}: checkpoint_slots step A o/slot states bitwise = "
                  f"sequential: {a_['o_bitwise_equal_sequential']}/{a_['slot_states_bitwise_equal_sequential']}; "
                  f"step B (reads slot n_accept-1) bitwise = sequential: {b_['o_bitwise_equal_sequential']}, worst o vs "
                  f"golden {e(max(b_['rows'], key=lambda y: y['o']['fro_rel'])['o'])}; replay: verify o bitwise = "
                  f"checkpoint_slots {v['replay']['verify_o_bitwise_equal_checkpoint_slots']}, committed state bitwise ="
                  f" sequential {v['replay']['committed_state_bitwise_equal_sequential']}.")
            print("\n| layer level T (+16 decode) | prefill ours | prefill HF bf16 | ratio | decode ours | decode HF bf16 | ratio |")
            print("|---|---|---|---|---|---|---|")
            for x in r["layer_level"]:
                p, d = x["prefill"], x["decode"]
                print(f"| {x['T']} | {e(p['ours'])} | {e(p['hf_bf16'])} | {p['fro_ratio_ours_over_hf_bf16']:.2f} | "
                      f"{e(d['ours'])} | {e(d['hf_bf16'])} | {d['fro_ratio_ours_over_hf_bf16']:.2f} |")


def perf():
    p, t = load("perf"), {a: json.loads((R / f"perf_{a}_autotune.json").read_text()) for a in ARCHS}
    print("\n| model | lens | " + " | ".join(f"{a} eager ms/layer (autotuned) | {a} graph ms/layer | {a} Mtok/s/layer eager | {a} ktok/s all GDN layers eager" for a in ARCHS) + " |")
    print("|---|---|" + "---|---|---|---|" * len(ARCHS))
    for m in ("9b", "27b"):
        for i, x in enumerate(p["sm86"]["models"][m]["prefill"]):
            lens = x["lens"][0] if len(x["lens"]) == 1 else f"{len(x['lens'])}x{x['lens'][0]}"
            cells = []
            for a in ARCHS:
                y, z = p[a]["models"][m]["prefill"][i], t[a]["models"][m]["prefill"][i]
                cells.append(f"{y['ms_per_layer']:.3f} ({z['ms_per_layer']:.3f}) | {y['graph_ms_per_layer']:.3f} | {y['tok_s_per_layer'] / 1e6:.2f} | "
                             f"{y['tok_s_all_gdn_layers'] / 1e3:.0f}")
            print(f"| {m} | {lens} | " + " | ".join(cells) + " |")
    for sec in ("decode", "verify"):
        print("\n| model | kernel | Seqs | tok/Seq | " + " | ".join(f"{a} graph µs/layer | {a} eager µs/layer | {a} state GB/s" for a in ARCHS) + " |")
        print("|---|---|---|---|" + "---|---|---|" * len(ARCHS))
        for m in ("9b", "27b"):
            for i, x in enumerate(p["sm86"]["models"][m][sec]):
                cells = []
                for a in ARCHS:
                    y = p[a]["models"][m][sec][i]
                    cells.append(f"{y['graph_us_per_layer']:.1f} | {y['eager_us_per_layer']:.1f} | {y['state_GBps']:.0f}")
                print(f"| {m} | {x['kind']} | {x['n_seqs']} | {x['tokens_per_seq']} | " + " | ".join(cells) + " |")


def graph_det():
    g = load("graph_det")
    print("\n| check | " + " | ".join(ARCHS) + " |\n|---|" + "---|" * len(ARCHS))
    for k in ("graph_decode_fused_recurrent", "graph_decode_fused_sigmoid", "graph_verify_k3_fused_sigmoid"):
        cells = [str(all(r["output_bitwise_equal"] and r["pool_bitwise_equal"] for r in g[a][k]["replays"])) for a in ARCHS]
        print(f"| {k}: 5 replays, new inputs/slots/accepted counts, = eager bitwise | " + " | ".join(cells) + " |")
    print("| graph_chunk_prefill (precomputed chunk indices) = eager bitwise | " + " | ".join(
        str(g[a]["graph_chunk_prefill"].get("output_bitwise_equal")) for a in ARCHS) + " |")
    print("| 10 runs in-process identical (all 5 outputs) | " + " | ".join(
        str(all(g[a]["in_process_10_runs_identical"].values())) for a in ARCHS) + " |")
    print("| 2 fresh processes identical | " + " | ".join(str(g[a]["two_processes_identical"]) for a in ARCHS) + " |")
    print("| autotuned process = pinned | " + " | ".join(
        str(all(g[a]["autotuned_process_equal_to_pinned"].values())) for a in ARCHS) + " |")
    same = {k: g["sm86"]["process_hashes_pinned"][k] == g["sm120"]["process_hashes_pinned"][k] for k in g["sm86"]["process_hashes_pinned"]}
    print(f"| sm_86 hashes = sm_120 hashes | {all(same.values())} | |")


def tiled():
    t = load("tiled_order")
    print("\n| arch | model | knob | core out bitwise | state bitwise | out_proj input bitwise | layer output (max_abs / max_rel / fro_rel) |")
    print("|---|---|---|---|---|---|---|")
    for a in ARCHS:
        for m in t[a]["models"]:
            for knob in ("v_head_tiled=True", "v_head_tiled=False"):
                x = m[knob]
                print(f"| {a} | {m['model']} | {knob} | {x['core_outputs_bitwise_equal']} | {x['final_state_bitwise_equal']} | "
                      f"{x['out_proj_input_bitwise_equal']} | {e(x['layer_output'])} |")
    for a in ARCHS:
        ok = {m: all(v for k in r.values() for v in k.values()) for m, r in t[a]["recurrent_kernels"].items()}
        print(f"\n{a}: fused_recurrent / fused_sigmoid_gating / packed_decode on tiled inputs + knob, 4 decode steps, "
              f"outputs and state bitwise = grouped: {ok}")


if __name__ == "__main__":
    sections = dict(correctness=correctness, perf=perf, graph=graph_det, tiled=tiled)
    for name in sys.argv[1:] or sections:
        sections[name]()
