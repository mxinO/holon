"""The S2 GGUF files (in the default HF cache)."""
from huggingface_hub import hf_hub_download

FILES = {
    "8B-Q8_0": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf"),
    "8B-Q4_K_M": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
    "8B-IQ4_XS": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-IQ4_XS.gguf"),
    "8B-UD-IQ3_XXS": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-UD-IQ3_XXS.gguf"),
    "8B-UD-Q3_K_XL": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-UD-Q3_K_XL.gguf"),
    "30B-A3B-UD-IQ3_XXS": ("unsloth/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-UD-IQ3_XXS.gguf"),
}


def path(key: str) -> str:
    repo, name = FILES[key]
    return hf_hub_download(repo, name, local_files_only=True)


def find_tensor(type_name: str, prefer: tuple[str, ...] = ()) -> tuple[str, "gguf.ReaderTensor"]:
    """First tensor of the given ggml type across the S2 files (files in `prefer` are searched first)."""
    import gguf

    keys = list(prefer) + [k for k in FILES if k not in prefer]
    for key in keys:
        for t in gguf.GGUFReader(path(key)).tensors:
            if t.tensor_type.name == type_name and "token_embd" not in t.name:
                return key, t
    raise KeyError(type_name)
