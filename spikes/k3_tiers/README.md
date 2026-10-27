# K step 3: the mixed factor under std.state_tier's dispatch

`bench.py` drives std.state_tier's own I/O path (`iothreads.IoThreads` over `ssd.SsdLevel`, 557,056 B unit-major
units, O_DIRECT) with the tier's dispatch (docs/dev/v1-kv-tiers.md D4 / D11 rev 8.3): while a read is queued, no write
starts except one a queued read depends on. Per cell the driver keeps reads (and, for a share of them, an urgent
dependent write) and background writes `threads` deep, at a target read share of bytes, and compares the bytes moved
with D4's serial lane at the share reached, from pure rates measured just before. A comparison arm (`strict`, not the
tier's) starts no write while a read is queued or running. The GIL cell times a 3–5 ms pure-Python body alone and
beside the threads.

## Predictions (written before the runs)

| | Prediction |
|---|---|
| mixed factor, 2 and 3 threads, 50/50 and 75/25 | 0.85–0.95 (step 2 measured 0.77–0.80 with one writer always in flight) |
| reads beside background writes | within 10% of the pure read rate for their share |
| GIL | the body slows < 3% beside 2 and 3 I/O threads |

## Run

On the RTX 5080 desktop, from the node outside the container (the container's `/tmp` is tmpfs), in a node-local
directory on its NVMe that the script removes:

```bash
python spikes/k3_tiers/bench.py --tag rtx-5080 --dir <node-local dir> --seconds 6 --out <file>
```

## Results (`evidence/k3_mixed_nvme_rtx-5080_2026-10-19.json`)

| | 2 threads | 3 threads |
|---|---|---|
| mixed factor, the tier's dispatch | 0.79–0.82 | 0.73–0.74 |
| mixed factor, strict arm | 0.64–0.77 | 0.62–0.76 |
| reads vs their pure rate, the tier's dispatch | 0.88–0.92 | 0.79–0.82 |
| GIL: the body beside the threads | −3% | −2% |

The factor is the drive's own read/write interference, not the dispatch: serializing the kinds (strict) moves less.
The literal rule bounds writes only while a read waits in the queue; with reads always in flight it rarely does, so
the 75/25 target reached 0.51. Both predictions on the factor and on reads miss; the GIL prediction holds.
