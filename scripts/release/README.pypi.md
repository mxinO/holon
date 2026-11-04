# Holon

Holon is an LLM inference engine for one consumer NVIDIA GPU. `pip install holonlm` brings it with its native
modules prebuilt, so no CUDA toolkit is needed: `holon serve <model>` serves the OpenAI and Anthropic APIs, starting
from the lock tuned and verified for your card. Source, documentation and the supported models:
<https://github.com/mxinO/holon>.

## Supported GPUs

Holon ships its native modules for three GPU architectures (SMs). Each model's lock, tuned and verified, exists for
these tested cards:

| SM | tested cards (a tuned lock) | other cards of the same SM (not tested) |
|---|---|---|
| 86 | RTX 3090 | RTX 30 series; RTX A2000 to A6000; A10, A40 |
| 89 | RTX 4090 | RTX 40 series; RTX 6000 Ada; L4, L40, L40S |
| 120 | RTX 5080, RTX 5090 | RTX 50 series; RTX PRO Blackwell |

Another card of these SMs starts from the closest tested card's lock of its SM, marked unverified and fitted to its
memory, or from automatic defaults for a model without one. A GPU of another SM installs but is refused at start,
naming why: it needs a source install with CUDA 13's nvcc.

## Requirements

- Linux x86-64, and an NVIDIA driver for CUDA 13.0 (R580 or newer).
- CPython 3.12 with its headers, and a C compiler: Triton builds a small launcher per kernel at first use.
- About 3.2 GB to download (torch and its CUDA 13 libraries) and 6 GB on disk.

## Install

On Ubuntu 24.04:

```bash
sudo apt install python3.12-venv python3.12-dev build-essential
python3.12 -m venv ~/holon && . ~/holon/bin/activate
pip install holonlm
holon serve unsloth/Qwen3-8B-GGUF
```

On any Linux with a C compiler, a uv-managed Python brings its own headers (and serves hosts whose Python is not 3.12):

```bash
uv tool install --python 3.12 holonlm
holon serve unsloth/Qwen3-8B-GGUF
```

Use a venv without extra package indexes: `holonlm` pins torch 2.11.0 for CUDA 13.0 (PyPI's) and every other
package to the versions the locks were verified with.

If the venv got another torch build (a `+cu128` or `+cpu` one, from an extra index in `pip.conf`), `holon serve` refuses
it; replace it with the one Holon pins:

```bash
pip install --force-reinstall --no-deps "torch==2.11.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

## What fails, and how

- **No C compiler or Python headers:** the first start is refused at once with the apt line above.
- **Another torch build** (a `+cu128` or `+cpu` torch from another index): refused, naming the reinstall command above.
- **A GPU of another SM:** refused at start, naming the source install.
- **Python other than 3.12, aarch64, macOS, Windows:** pip finds no matching distribution.
