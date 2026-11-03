# The overlays' build image (docs/dev/v1-pip.md D4): glibc 2.28 (manylinux_2_28), the locks' nvcc (CUDA 13.0.0) and
# gcc-toolset-13 for every host compile (nvcc's host compiler, torch's cpp_extension, FlashInfer's ninja).
FROM nvidia/cuda:13.0.0-devel-rockylinux8@sha256:c229277e7f9a61e77fc2726c252e731ccdf8d5de89f723995c75a7daa7c5774a
RUN dnf install -y -q gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ gcc-toolset-13-binutils which && dnf clean all
COPY --from=ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce \
     /uv /uvx /usr/local/bin/
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
