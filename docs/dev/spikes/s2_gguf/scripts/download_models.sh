#!/usr/bin/env bash
# Download the S2 GGUF files into the default HF cache.
set -euo pipefail
hf download unsloth/Qwen3-8B-GGUF \
  Qwen3-8B-Q8_0.gguf Qwen3-8B-Q4_K_M.gguf Qwen3-8B-IQ4_XS.gguf \
  Qwen3-8B-UD-IQ3_XXS.gguf Qwen3-8B-UD-Q3_K_XL.gguf
hf download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-UD-IQ3_XXS.gguf
