"""GLM step 7: which stream / event pairs order work (torch 2.11): a producer stream sleeps, a consumer waits by each
fence form, an event recorded on the consumer must not complete before the producer. Prints ok / IGNORED per pair.
    python docs/dev/spikes/glm7/fence_matrix.py"""
import time

import torch

dev = torch.device("cuda")
torch.cuda._sleep(1000)
torch.cuda.synchronize()
def ctx(s):
    return torch.cuda.stream(s) if isinstance(s, torch.cuda.Stream) else s
def check(name, make_a, make_b, fence):
    a, b = make_a(), make_b()
    with ctx(a):
        torch.cuda._sleep(300_000_000)
    fence(a, b)
    with ctx(b):
        e = torch.Event(device=dev)
        e.record()
    time.sleep(0.05)
    print(f"{name:55s} {'IGNORED' if e.query() else 'ok'}")
    torch.cuda.synchronize()
def gen():
    return torch.Stream(device=dev)


def cud():
    return torch.cuda.Stream(dev)
def ev(kind):
    def f(a, b):
        e = torch.cuda.Event() if kind == "cuda" else torch.Event(device=dev)
        with ctx(a):
            e.record()
        b.wait_event(e)
    return f
for an, ma in (("generic", gen), ("cuda", cud)):
    for bn, mb in (("generic", gen), ("cuda", cud)):
        check(f"{an} -> {bn} wait_stream", ma, mb, lambda a, b: b.wait_stream(a))
        check(f"{an} -> {bn} wait_event(torch.cuda.Event)", ma, mb, ev("cuda"))
        check(f"{an} -> {bn} wait_event(torch.Event)", ma, mb, ev("gen"))
