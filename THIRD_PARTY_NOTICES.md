# Third-party notices

Holon is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). This file lists every piece of third-party
code in this repository — copied, vendored or adapted — with its source, version, license and where it lives.
Libraries Holon only depends on (installed from their own packages: PyTorch, Triton, FlashInfer, transformers,
tokenizers, gguf-py, safetensors, …) are not in this repository; their own licenses apply. Rules for adding an entry:
`AGENTS.md`, "Third-party code"; `scripts/ci/check_provenance.py` checks them.

## In the engine

### llama.cpp / ggml — MIT

Source: https://github.com/ggml-org/llama.cpp at `51ce9c11a6f2dfa895696c0048c4333e8953b728`.
Copyright (c) 2023-2026 The ggml authors. License text below and in
`plugins/holon_plugins/kernels/ggml_cuda/vendor/LICENSE`.

- `plugins/holon_plugins/kernels/ggml_cuda/vendor/` — the quantized CUDA kernels (`mmvq`, `mmq`, `mmid`,
  `get_rows`, `convert` and their headers) and the MoE routing and combine kernels (`topk-moe`,
  `moe-weighted-reduction`), unmodified upstream files fetched by
  `plugins/holon_plugins/kernels/ggml_cuda/vendor.sh`, then modified by
  `plugins/holon_plugins/kernels/ggml_cuda/vendor.patch` (bf16 activations and output, no environment reads, top-k
  ids at row stride k, a bf16 combine); `plugins/holon_plugins/kernels/ggml_cuda/SHA256SUMS` pins the result.
- `plugins/holon_plugins/kernels/ggml_cuda/csrc/shim.cu` — adapted: ggml's type traits, tensor helpers and device
  table, after `ggml/src/ggml.c` and `ggml/src/ggml-cuda/ggml-cuda.cu`.
- `plugins/holon_plugins/kernels/ggml_cuda/csrc/api.cu` — adapted: llama.cpp's matmul dispatch (dense and
  `MUL_MAT_ID`), fused SwiGLU gate, scratch pool and mmq schedule, after `ggml/src/ggml-cuda/ggml-cuda.cu`,
  `mmq.cuh` and `mmq.cu`.

### Qwen tokenizer — Apache-2.0

Source: the `tokenizer.json` of Qwen2 / Qwen3 (https://huggingface.co/Qwen), Copyright Alibaba Cloud (Qwen team).

- `std/holon_std/weights/gguf/tokenizer.py` — copied: the `qwen2` pre-tokenizer split regex.

## In spikes (research code, not part of the engine)

Each spike fetches its third-party code with a script at a pinned commit; the fetched trees are not committed. The
tracked scripts and patches below contain or modify upstream lines.

- `spikes/s2_gguf/scripts/vendor_kernels.sh`, `spikes/s2_gguf/patches/0001-iq3_s-grid.patch`,
  `spikes/s2_gguf/csrc/compat.h` — SGLang's GGUF CUDA kernels (https://github.com/sgl-project/sglang at
  `968726f3`, Apache-2.0; derived via vLLM from llama.cpp, MIT), and llama.cpp's IQ3_S grid (MIT).
- `spikes/s2_gguf/patches/llama.cpp-s2-perf-cases.patch` — llama.cpp test cases (MIT), at the commit above.
- `spikes/s3_gdn/vendor.sh`, `spikes/s3_gdn/vendor.patch` — vLLM's copy of flash-linear-attention
  (https://github.com/vllm-project/vllm at `6dd813e0`, Apache-2.0; flash-linear-attention, MIT).

## License texts

### MIT (llama.cpp / ggml)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
