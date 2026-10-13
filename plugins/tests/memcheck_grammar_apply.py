"""grammar.xgrammar's `apply` reads and writes only inside its tensors (docs/m05-grammar.md §5, L0-mask, per class):

    cd plugins
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m tests.memcheck_grammar_apply

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size, so a read past the mask's
last word (a vocabulary that is not a multiple of 32, the last block of words) or a write past a logits row is
outside any allocation. Qwen3's and Qwen3.5's vocabularies and an odd one, with padding rows.
"""
import torch
from holon_plugins.grammar.xgrammar import kernel

from tests.test_grammar import _mask_case


def main() -> None:
    for vocab in (151936, 248320, 1000):
        kernel.apply(*_mask_case(vocab, "cuda", 0))
    torch.cuda.synchronize()
    print(f"memcheck run: apply over vocabularies 151936, 248320, 1000 on sm {torch.cuda.get_device_capability()}")


if __name__ == "__main__":
    main()
