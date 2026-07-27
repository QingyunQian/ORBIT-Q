FROM python:3.11-slim

ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
ARG APT_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ARG PIP_EXTRA_INDEX=https://pypi.org/simple
ARG RUST_CRATES_INDEX_URL=https://rsproxy.cn/index/

ENV DEBIAN_FRONTEND=noninteractive \
    JAX_ENABLE_X64=0 \
    NUMBA_DISABLE_JIT=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX} \
    PYTHONUNBUFFERED=1

RUN for file in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
        if [ -f "$file" ]; then \
            sed -i \
                -e "s|http://deb.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
                -e "s|https://deb.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
                -e "s|http://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
                -e "s|https://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
                -e "s|http://deb.debian.org/debian|${APT_MIRROR}|g" \
                -e "s|https://deb.debian.org/debian|${APT_MIRROR}|g" \
                "$file"; \
        fi; \
    done \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cargo \
        git \
        pkg-config \
        rustc \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.cargo \
    && printf \
        '[source.crates-io]\nreplace-with = "benchmark-mirror"\n\n[source.benchmark-mirror]\nregistry = "sparse+%s"\n' \
        "${RUST_CRATES_INDEX_URL}" \
        > /root/.cargo/config.toml

COPY envs/tensorcircuit-py311/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir "maturin==1.14.1" \
    && pip install --no-cache-dir --no-build-isolation -r /tmp/requirements.lock

WORKDIR /bench

