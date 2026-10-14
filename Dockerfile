# Holon installed from source, for hosts without a CUDA toolkit (release design D1). Build it from a clean tree, the
# release tag's archive, never from a working directory:
#
#   git archive --format=tar <tag> | docker build -t holon:<tag> -
#   docker run --gpus all -p 8000:8000 -v <model dir>:/models holon:<tag> serve /models/<file> --host 0.0.0.0
#
# The environment's identity is the plan's `uv_lock` hash (the runtime closure) plus the base image digest below.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce \
     /uv /usr/local/bin/uv

# Every cache at a fixed path, whoever runs the container: Python and the packages (uv), the prebuilt GGUF kernels
# (XDG), FlashInfer's modules, Triton's kernels and the Hugging Face hub.
ENV UV_PYTHON_INSTALL_DIR=/opt/holon/python \
    UV_CACHE_DIR=/opt/holon/uv-cache \
    XDG_CACHE_HOME=/opt/holon/cache \
    FLASHINFER_WORKSPACE_BASE=/opt/holon/cache \
    TRITON_CACHE_DIR=/opt/holon/cache/triton \
    HF_HOME=/opt/holon/hf \
    PATH=/opt/holon/src/.venv/bin:$PATH

WORKDIR /opt/holon/src
COPY . .
RUN uv sync --frozen && rm -rf "$UV_CACHE_DIR"
# Native code for every supported SM (no GPU needed to build): the GGUF kernels and FlashInfer's modules.
RUN holon prebuild --sm 86 89 120

EXPOSE 8000
ENTRYPOINT ["holon"]
