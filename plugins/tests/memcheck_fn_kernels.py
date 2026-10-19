"""No kernel of Qwen3.8-Flash-Next's step 4 (kernels.qsa_triton, kernels.hc_triton, kernels.ple_triton)
reads or writes past its allocations (docs/v1-flash-next.md acceptance 3, run once per class):

    cd plugins && PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
        python -m tests.memcheck_fn_kernels

Without torch's caching allocator every tensor is its own cudaMalloc of exactly its size. The L0 cases run as they
are: the index write and pooling in two chunks with padding; the selection at 32K (8,192 blocks) and below the budget,
sub-batched too; the sparse attention over every layout this card selects (q8_0's scales are the end of each unit's
byte segment, so a scale read past the last unit is outside the allocation; the last unit is in use); HC mix and
combine at 0 to 77 tokens; PLE rows, and the PLE block one shot and in chunks (NaN state of rows without has_init).
"""
import torch

from tests import test_hc_ple_kernels as hp
from tests import test_qsa_kernels as qsa
from tests.test_attention_kv_quant import fp8_device


def main() -> None:
    qsa.test_index(torch.bfloat16)
    qsa.test_select_ties()
    layouts = [("kv.hnd", torch.bfloat16), ("kv.hnd", torch.float16), ("kv.hnd.q8_0", torch.bfloat16)]
    for layout, dtype in layouts + ([("kv.hnd.fp8", torch.bfloat16)] if fp8_device() else []):
        qsa.test_attend(layout, dtype)
    for inject in (True, False):
        hp.test_hc(inject)
    hp.test_ple_rows()
    hp.test_ple_block()
    hp.test_ple_block_chunks()
    torch.cuda.synchronize()
    print(f"memcheck run: qsa index/select/attend ({len(layouts) + fp8_device()} layouts), hc mix/combine, "
          f"ple rows/block on sm {torch.cuda.get_device_capability()}")


if __name__ == "__main__":
    main()
