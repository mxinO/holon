# The reference build's image (docs/dev/v1-pip.md D10.4): the locks' own environment (CUDA 13.0.0 on Ubuntu 24.04,
# its gcc 13), so the SASS gate compares the overlays' modules with what the locks were verified with.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce \
     /uv /uvx /usr/local/bin/
ENV CUDA_HOME=/usr/local/cuda UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
