# The overlays' build image (docs/dev/v1-pip.md D4): glibc 2.28 (manylinux_2_28), the locks' nvcc (CUDA 13.0.0) and
# gcc-toolset-13 for every host compile (nvcc's host compiler, torch's cpp_extension, FlashInfer's ninja).
FROM nvidia/cuda:13.0.0-devel-rockylinux8@sha256:c229277e7f9a61e77fc2726c252e731ccdf8d5de89f723995c75a7daa7c5774a
# The compilers pinned (docs/dev/v1-pip.md D4): a new release of them is a deliberate change, gated like any build.
RUN dnf install -y -q gcc-toolset-13-gcc-13.3.1-2.2.el8_10 gcc-toolset-13-gcc-c++-13.3.1-2.2.el8_10 \
        gcc-toolset-13-binutils-2.40-21.el8 which-2.21-21.el8_10 && dnf clean all
COPY --from=ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce \
     /uv /uvx /usr/local/bin/
# Ubuntu's gcc defines _FORTIFY_SOURCE=3 by default, and nvcc preprocesses device code with the host compiler's
# headers: glibc's fortify wrappers change the device code of some kernels (FlashInfer's rope-quantize on sm_86). Every
# nvcc run here gets it too, so the SASS is the locks' (docs/dev/v1-pip.md D4, §16).
ENV NVCC_APPEND_FLAGS="-D_FORTIFY_SOURCE=3"
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
