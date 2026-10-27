"""FN step 0, N2 (docs/dev/v1-flash-next.md section 8): the PLE table's two read paths on a bar card, which set D5's
SSD read-ahead depth.

- **Zero-copy row gather** (D5's RAM tier): the IQ4_NL table (`per_layer_token_embd.weight` of the model's second
  shard, 90 B per row) is read into mapped pinned host memory (the core's form: anonymous mmap + cudaHostRegister); one
  kernel per call reads 16 rows per token over PCIe, each row into shared memory and dequantized to fp32 in the
  kernel, for 1-4,096 tokens of uniform random row ids (hashed rows are pseudo-random; repeats that a real prompt
  would dedupe are not modelled, so this is the pessimistic case). Timed with CUDA events, median of 50 calls.
- **O_DIRECT 4 KiB reads** (D5's SSD tier): random page-aligned 4 KiB reads of the same table on the desktop's NVMe
  with Linux AIO, 1-4 submitting threads, queue depth 16-256, 3 s each: IOPS, GB/s and latency percentiles.

    CUDA_VISIBLE_DEVICES=0 python spikes/fn0/n2_rows.py --shard <...-00002-of-00002.gguf> --out <record.json>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import run  # noqa: E402  (bench/run.py: record helpers)

ROW_BYTES, ROW_VALUES, ROWS_PER_TOKEN = 90, 160, 16          # IQ4_NL: 5 blocks of 32 values in 18 B
TOKENS = (1, 4, 16, 64, 256, 1024, 4096)

CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <thread>
#include <vector>

void launch_gather(const uint8_t* table, const int64_t* ids, float* out, int64_t rows, cudaStream_t s);

namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
int io_setup(unsigned n, aio_context_t* c) { return syscall(__NR_io_setup, n, c); }
int io_destroy(aio_context_t c) { return syscall(__NR_io_destroy, c); }
int io_submit(aio_context_t c, long n, iocb** p) { return syscall(__NR_io_submit, c, n, p); }
int io_getevents(aio_context_t c, long lo, long n, io_event* e) {
  return syscall(__NR_io_getevents, c, lo, n, e, nullptr);
}
}

// The file's [offset, offset + bytes) read into anonymous memory registered as mapped pinned memory (the core's host
// pool form: `holon.core.platform`'s anonymous mmap + cudaHostRegister(portable | mapped)).
struct HostTable {
  HostTable(const std::string& path, int64_t offset, int64_t bytes) {
    span_ = bytes;
    base_ = mmap(nullptr, size_t(span_), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TORCH_CHECK(base_ != MAP_FAILED, "mmap failed");
    check(cudaHostRegister(base_, size_t(span_), cudaHostRegisterPortable | cudaHostRegisterMapped),
          "cudaHostRegister");
    const int fd = open(path.c_str(), O_RDONLY);
    TORCH_CHECK(fd >= 0, "open failed");
    for (int64_t done = 0; done < bytes;) {
      const ssize_t n = pread(fd, static_cast<char*>(base_) + done, size_t(std::min<int64_t>(bytes - done, 1 << 30)),
                              offset + done);
      TORCH_CHECK(n > 0, "pread failed");
      done += n;
    }
    close(fd);
    void* d = nullptr;
    check(cudaHostGetDevicePointer(&d, base_, 0), "cudaHostGetDevicePointer");
    dev_ = static_cast<uint8_t*>(d);
  }
  ~HostTable() { cudaHostUnregister(base_); munmap(base_, size_t(span_)); }
  void gather(torch::Tensor ids, torch::Tensor out) {
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt64);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat);
    launch_gather(dev_, ids.data_ptr<int64_t>(), out.data_ptr<float>(), ids.numel(),
                  at::cuda::getCurrentCUDAStream().stream());
    check(cudaGetLastError(), "gather");
  }
  int64_t span_ = 0;
  void* base_ = nullptr;
  uint8_t* dev_ = nullptr;
};

// Random 4 KiB-aligned 4 KiB reads in [offset, offset + bytes) with O_DIRECT, `workers` threads each keeping
// qd / workers in flight, for `seconds`.
std::map<std::string, double> odirect_reads(const std::string& path, int64_t offset, int64_t bytes, int qd, int workers,
                                            double seconds) {
  const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
  TORCH_CHECK(fd >= 0, "open O_DIRECT failed");
  const int64_t page = 4096, first = (offset + page - 1) / page, pages = (offset + bytes) / page - first;
  const int per = std::max(1, qd / workers);
  std::atomic<int64_t> reads{0};
  std::atomic<bool> fail{false};
  std::vector<std::vector<float>> lat(workers);
  const auto t0 = Clock::now(), until = t0 + std::chrono::microseconds(int64_t(seconds * 1e6));
  std::vector<std::thread> ts;
  for (int w = 0; w < workers; ++w)
    ts.emplace_back([&, w] {
      std::mt19937_64 g(17 + w);
      void* mem = nullptr;
      TORCH_CHECK(posix_memalign(&mem, page, size_t(per) * page) == 0);
      aio_context_t ctx = 0;
      if (io_setup(per, &ctx) < 0) { fail = true; free(mem); return; }
      std::vector<iocb> cbs(per);
      std::vector<Clock::time_point> sent(per);
      auto submit = [&](int i) {
        std::memset(&cbs[i], 0, sizeof(iocb));
        cbs[i].aio_fildes = fd;
        cbs[i].aio_lio_opcode = IOCB_CMD_PREAD;
        cbs[i].aio_buf = uint64_t(static_cast<char*>(mem) + i * page);
        cbs[i].aio_nbytes = page;
        cbs[i].aio_offset = (first + int64_t(g() % uint64_t(pages))) * page;
        cbs[i].aio_data = uint64_t(i);
        iocb* p = &cbs[i];
        sent[i] = Clock::now();
        return io_submit(ctx, 1, &p) == 1;
      };
      int inflight = 0;
      for (int i = 0; i < per; ++i) inflight += submit(i) ? 1 : 0;
      std::vector<io_event> ev(per);
      while (inflight > 0) {
        const long n = io_getevents(ctx, 1, per, ev.data());
        if (n < 0) { fail = true; break; }
        const auto now = Clock::now();
        for (long e = 0; e < n; ++e) {
          const int i = int(ev[e].data);
          --inflight;
          if (ev[e].res != page) fail = true;
          lat[w].push_back(float(std::chrono::duration<double, std::micro>(now - sent[i]).count()));
          ++reads;
          if (now < until && !fail && submit(i)) ++inflight;
        }
      }
      io_destroy(ctx);
      free(mem);
    });
  for (auto& t : ts) t.join();
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  close(fd);
  TORCH_CHECK(!fail, "odirect_reads: a read failed");
  std::vector<float> all;
  for (auto& l : lat) all.insert(all.end(), l.begin(), l.end());
  std::sort(all.begin(), all.end());
  auto pct = [&](double p) { return double(all[size_t(p * double(all.size() - 1))]); };
  return {{"kiops", double(reads.load()) / secs / 1e3}, {"gbps", double(reads.load()) * page / secs / 1e9},
          {"lat_p50_us", pct(0.5)}, {"lat_p99_us", pct(0.99)}, {"reads", double(reads.load())}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<HostTable>(m, "HostTable").def(pybind11::init<std::string, int64_t, int64_t>())
      .def("gather", &HostTable::gather);
  m.def("odirect_reads", &odirect_reads, pybind11::call_guard<pybind11::gil_scoped_release>());
}
"""

CUDA = r"""
#include <cuda_fp16.h>
#include <cstdint>
__constant__ int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// One block per row: the 90-byte row read once from host memory into shared memory (45 two-byte loads; rows are
// 2-byte aligned), then 160 threads dequantize one value each.
__global__ void gather_iq4nl(const uint8_t* __restrict__ table, const int64_t* __restrict__ ids,
                             float* __restrict__ out) {
  __shared__ uint16_t row[45];
  const int64_t r = ids[blockIdx.x];
  const uint16_t* src = reinterpret_cast<const uint16_t*>(table + r * 90);
  if (threadIdx.x < 45) row[threadIdx.x] = src[threadIdx.x];
  __syncthreads();
  const uint8_t* b = reinterpret_cast<const uint8_t*>(row);
  const int blk = threadIdx.x / 32, j = threadIdx.x % 32;
  const float d = __half2float(*reinterpret_cast<const __half*>(b + blk * 18));
  const uint8_t q = b[blk * 18 + 2 + (j % 16)];
  out[int64_t(blockIdx.x) * 160 + threadIdx.x] = d * float(kIQ4NL[j < 16 ? (q & 15) : (q >> 4)]);
}

void launch_gather(const uint8_t* table, const int64_t* ids, float* out, int64_t rows, cudaStream_t s) {
  gather_iq4nl<<<unsigned(rows), 160, 0, s>>>(table, ids, out);
}
"""


def ple_extent(shard: str) -> tuple[int, int, int]:
    """(byte offset, bytes, rows) of `per_layer_token_embd.weight` in the shard (gguf-py's reader)."""
    from gguf import GGUFReader
    t = next(t for t in GGUFReader(shard).tensors if t.name == "per_layer_token_embd.weight")
    rows = int(t.n_elements) // ROW_VALUES
    assert t.n_bytes == rows * ROW_BYTES, (t.n_bytes, rows)
    return int(t.data_offset), int(t.n_bytes), rows


def gather_rates(ext, shard: str, offset: int, nbytes: int, rows: int) -> list[dict]:
    table = ext.HostTable(shard, offset, nbytes)
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for t in TOKENS:
        n = t * ROWS_PER_TOKEN
        dst = torch.empty(n, ROW_VALUES, device="cuda")
        times = []
        for i in range(60):
            ids = torch.randint(rows, (n,), device="cuda", generator=g)          # fresh rows: no L2 reuse
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            table.gather(ids, dst)
            b.record()
            b.synchronize()
            if i >= 10:
                times.append(a.elapsed_time(b) * 1e3)
        us = statistics.median(times)
        out.append({"tokens": t, "rows": n, "us": round(us, 2), "us_p10": round(sorted(times)[5], 2),
                    "us_p90": round(sorted(times)[44], 2), "us_per_token": round(us / t, 3),
                    "useful_gbps": round(n * ROW_BYTES / us / 1e3, 3)})
        print(f"gather {t:5d} tokens: {us:9.1f} us ({us / t:.3f} us/token)", flush=True)
    del table
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True, help="the model's second shard (the PLE table) on the local NVMe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    ext = load_inline("fn0_n2", CPP, cuda_sources=CUDA, extra_cflags=["-O2"], extra_cuda_cflags=["-O2"],
                      build_directory=os.environ.get("FN0_BUILD") or None, verbose=False)
    offset, nbytes, rows = ple_extent(a.shard)
    rec = {"name": "fn0_n2_rows", "date": datetime.date.today().isoformat(), "gpu": run.gpu_class(0),
           "commit": run.git("rev-parse", "HEAD"), "command": "python spikes/fn0/n2_rows.py --shard <PLE shard> "
           f"--seconds {a.seconds}", "method": __doc__.split("\n\n")[1].replace("\n", " "),
           "table": {"rows": rows, "bytes": nbytes, "row_bytes": ROW_BYTES, "rows_per_token": ROWS_PER_TOKEN}}
    rec["gather"] = gather_rates(ext, a.shard, offset, nbytes, rows)
    reads = []
    for workers in (1, 2, 4):
        for qd in (16, 32, 64, 128, 256):
            if qd < workers:
                continue
            r = ext.odirect_reads(a.shard, offset, nbytes, qd, workers, a.seconds)
            reads.append({"workers": workers, "qd": qd, **{k: round(v, 3) for k, v in r.items()}})
            print(f"O_DIRECT w={workers} qd={qd}: {r['kiops']:.0f} KIOPS, {r['gbps']:.2f} GB/s, "
                  f"p50 {r['lat_p50_us']:.0f} us, p99 {r['lat_p99_us']:.0f} us", flush=True)
    rec["odirect_4k"] = reads
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
