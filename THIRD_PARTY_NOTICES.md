# Third-party notices

Holon is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). This file lists every piece of third-party
code in this repository — copied, vendored or adapted — with its source, version, license and where it lives.
Libraries Holon only depends on (installed from their own packages: PyTorch, Triton, FlashInfer, transformers,
tokenizers, gguf-py, safetensors, xgrammar, …) are not in this repository; their own licenses apply. Rules for adding an entry:
`AGENTS.md`, "Third-party code"; `scripts/ci/check_provenance.py` checks them.

## In the engine

### llama.cpp / ggml — MIT

Source: https://github.com/ggml-org/llama.cpp at `cbe28bb3cb0758f0c566a9a3561766821aeb0748`.
Copyright (c) 2023-2026 The ggml authors. License text below and in
`plugins/holon_plugins/kernels/ggml_cuda/vendor/LICENSE`.

- `plugins/holon_plugins/kernels/ggml_cuda/vendor/` — the quantized CUDA kernels (`mmvq`, `mmq`, `mmid`,
  `get_rows`, `convert` and their headers) and the MoE routing and combine kernels (`topk-moe`,
  `moe-weighted-reduction`), unmodified upstream files fetched by
  `plugins/holon_plugins/kernels/ggml_cuda/vendor.sh`, then modified by
  `plugins/holon_plugins/kernels/ggml_cuda/vendor.patch` (bf16 activations and output, also with mmvq's fused
  scale, no environment reads, top-k ids at row stride k, a bf16 combine, mmq's src1 precision exported, an IQ1_M
  mmq tile loader after IQ2_S's with its own template instance);
  `plugins/holon_plugins/kernels/ggml_cuda/SHA256SUMS` pins the result.
- `plugins/holon_plugins/kernels/ggml_cuda/csrc/shim.cu` — adapted: ggml's type traits, tensor helpers and device
  table, after `ggml/src/ggml.c` and `ggml/src/ggml-cuda/ggml-cuda.cu`.
- `plugins/holon_plugins/kernels/ggml_cuda/csrc/api.cu` — adapted: llama.cpp's matmul dispatch (dense and
  `MUL_MAT_ID`), fused SwiGLU gate and NVFP4 scales, scratch pool and mmq schedule, after
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmq.cuh` and `mmq.cu`.
- `plugins/holon_plugins/kernels/ggml_cuda/plugin.py` (`repack`, `tail_bytes`) — adapted: the byte layouts of
  llama.cpp's HF converter (`conversion/base.py` `_nvfp4_pack`, `repack_mxfp4_blocks`) and its CUDA buffers' row
  padding (`ggml-cuda.cu`).
- `plugins/holon_plugins/kernels/ggml_cpu/vendor/` — ggml-cpu's activation quantizers and dot products (`quants.c`,
  its x86 `arch/x86/quants.c`, `ggml-quants.c` and their headers), unmodified upstream files fetched by
  `plugins/holon_plugins/kernels/ggml_cpu/vendor.sh`; `plugins/holon_plugins/kernels/ggml_cpu/SHA256SUMS` pins them.
- `plugins/holon_plugins/kernels/ggml_cpu/csrc/shim.c` — adapted: ggml's block sizes, `ggml_abort` and the CPU
  backend's FP4 scale tables, after `ggml/src/ggml.c` and `ggml/src/ggml-cpu/ggml-cpu.c`.
- `plugins/holon_plugins/kernels/ggml_cpu/csrc/cpu_experts.cpp` (`traits`) — adapted: the CPU backend's dot product
  and activation type per weight type, after `ggml/src/ggml-cpu/ggml-cpu.c` (`type_traits_cpu`).

### flash-linear-attention, as copied into vLLM — MIT and Apache-2.0

Source: https://github.com/vllm-project/vllm at `6dd813e0b7ca8314fed28c13a73cfb5fc5310163`
(`vllm/third_party/flash_linear_attention`; its files are Apache-2.0, Copyright contributors to the vLLM project,
and contain code copied from https://github.com/fla-org/flash-linear-attention, MIT, Copyright (c) 2023-2025,
Songlin Yang, Yu Zhang). License text below and in `plugins/holon_plugins/kernels/fla/vendor/LICENSE`.

- `plugins/holon_plugins/kernels/fla/vendor/` — the Gated DeltaNet Triton kernels (`chunk_gated_delta_rule` and its
  parts, `fused_sigmoid_gating`, l2norm, index and utility helpers) from the directory's `ops/` and its `LICENSE`,
  unmodified upstream files fetched by `plugins/holon_plugins/kernels/fla/vendor.sh`, then modified by
  `plugins/holon_plugins/kernels/fla/vendor.patch` (no vLLM imports or environment reads, one pinned config per
  kernel, the `v_head_tiled` order, the recurrent kernel's skip sentinel at slot < 0, has-initial-state flags);
  `plugins/holon_plugins/kernels/fla/SHA256SUMS` pins the upstream files.

### Qwen tokenizer — Apache-2.0

Source: the `tokenizer.json` of Qwen2 / Qwen3 (https://huggingface.co/Qwen), Copyright Alibaba Cloud (Qwen team).

- `std/holon_std/weights/gguf/tokenizer.py` — copied: the `qwen2` pre-tokenizer split regex.

## In tests

### JSON-Schema-Test-Suite — MIT

Source: https://github.com/json-schema-org/JSON-Schema-Test-Suite at `8568bcc477cc3aaeed5bc397c9c8648e631317fc`.
Copyright (c) 2012 Julian Berman. License text below and in `plugins/tests/fixtures/jsts/LICENSE`.

- `plugins/tests/fixtures/jsts/` — the test files of drafts 3, 4, 6, 7, 2019-09 and 2020-12 (the grammar plugin's
  model-free oracle, `plugins/tests/test_grammar.py`), unmodified upstream files fetched by
  `plugins/tests/fixtures/jsts.sh`; `plugins/tests/fixtures/jsts.SHA256SUMS` pins them.

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

### MIT (flash-linear-attention)

```
MIT License

Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

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

### MIT (JSON-Schema-Test-Suite)

```
Copyright (c) 2012 Julian Berman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
