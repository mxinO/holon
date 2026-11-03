# Holon from its wheels, for hosts without a CUDA toolkit (docs/dev/v1-pip.md D12): Ubuntu 24.04's Python 3.12, its
# headers and gcc (Triton's launchers), and Holon's wheels with their pins. No toolkit and no prebuild: the native
# modules come prebuilt. It is also the GPU acceptance's container.
#
#   docker build -f Dockerfile -t holon <a directory holding the release's wheels>     # the tested artifacts
#   docker build --build-arg HOLON=holonlm==<version> -t holon - < Dockerfile          # from PyPI, once published
#   docker run --gpus all -p 8000:8000 -v <model dir>:/models holon serve /models/<file> --host 0.0.0.0
#
# `--build-arg HEADERS=` leaves out Python's headers (the acceptance's negative run: the start must refuse at once).
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

ARG HEADERS=python3.12-dev
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv $HEADERS build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Every cache at a fixed path, whoever runs the container: Holon's (the weights' hashes, the files locks name),
# FlashInfer's workspace (it writes generated sources there), Triton's and CuTe DSL's kernels, the hub.
ENV HOME=/opt/holon/user \
    HF_HOME=/opt/holon/hf \
    TRITON_CACHE_DIR=/opt/holon/cache/triton \
    CUTE_DSL_CACHE_DIR=/opt/holon/cache/cute \
    PATH=/opt/holon/venv/bin:$PATH

ARG HOLON=holonlm
COPY . /opt/holon/wheels/
RUN python3.12 -m venv /opt/holon/venv \
    && wheels=$(ls /opt/holon/wheels/*.whl 2> /dev/null || true) \
    && /opt/holon/venv/bin/pip install --no-cache-dir ${wheels:-$HOLON} \
    && rm -rf /opt/holon/wheels \
    && mkdir -p "$HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$CUTE_DSL_CACHE_DIR" \
    && chmod -R a+rwX "$HOME" "$HF_HOME" /opt/holon/cache

EXPOSE 8000
ENTRYPOINT ["holon"]
