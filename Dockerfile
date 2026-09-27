# syntax=docker/dockerfile:1.7
# Von System One server image for Unraid, one tag per target: cpu or nvidia.
# Build from this folder:
#   docker build --build-arg TARGET=cpu --build-arg VON_VERSION=1.2.3 -t pikkonmg/von-system-one:cpu .
ARG PYTHON_IMAGE=python:3.12-slim-trixie

FROM ${PYTHON_IMAGE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TARGET
ARG VON_VERSION
ARG TORCH_VERSION=2.14.0
# CUDA 12.6 runs on NVIDIA drivers from the 560 series onward.
ARG TORCH_NVIDIA_INDEX=cu126

# The cpu image adds OpenVINO, which runs the Von encoder faster than PyTorch on a CPU.
RUN test -n "$VON_VERSION" || { echo "VON_VERSION is required" >&2; exit 1; } \
    && case "$TARGET" in \
        cpu) echo cpu > /tmp/torch_index; echo "von-sdk[intel]==${VON_VERSION}" > /tmp/von_requirement ;; \
        nvidia) echo "$TORCH_NVIDIA_INDEX" > /tmp/torch_index; echo "von-sdk==${VON_VERSION}" > /tmp/von_requirement ;; \
        *) echo "TARGET must be cpu or nvidia" >&2; exit 1 ;; \
    esac

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Fail the build when the index serves a different PyTorch build than requested.
RUN torch_index="$(cat /tmp/torch_index)" \
    && pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${torch_index}" \
    && python -c "import sys, torch; expected = sys.argv[1]; sys.exit(0 if torch.__version__ == expected else 'Expected PyTorch ' + expected + ', installed ' + torch.__version__)" "${TORCH_VERSION}+${torch_index}"

RUN pip install "$(cat /tmp/von_requirement)" && pip check

# Triton only compiles kernels for torch.compile and the native JIT, which this
# image turns off with TORCH_DISABLE_NATIVE_JIT. Dropping it saves about 0.9 GB.
RUN if pip show triton >/dev/null 2>&1; then pip uninstall -y triton; fi

FROM ${PYTHON_IMAGE} AS runtime

ARG TARGET
ARG VON_VERSION

LABEL org.opencontainers.image.title="Von System One" \
      org.opencontainers.image.description="Von System One decision model server for Unraid" \
      org.opencontainers.image.source="https://github.com/wfzyx/von" \
      org.opencontainers.image.version="${VON_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"

# torch 2.14 compiles Triton kernels on the first CUDA inference unless native JIT is
# off, and this image carries no C compiler.
ENV PATH="/opt/venv/bin:$PATH" \
    TORCH_DISABLE_NATIVE_JIT=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    VON_TARGET=${TARGET} \
    VON_HOST=0.0.0.0 \
    VON_PORT=8000 \
    VON_PRELOAD=1 \
    VON_LOG_LEVEL=info \
    PUID=99 \
    PGID=100 \
    HOME=/data \
    HF_HOME=/data/huggingface

COPY --from=build /opt/venv /opt/venv
COPY start.sh serve.py /opt/von/

VOLUME /data
EXPOSE 8000

# With preload on, the server loads the model before it listens, so the first
# start can take several minutes while the weights download.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10m \
    CMD ["python3", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['VON_PORT'] + '/health', timeout=4)"]

ENTRYPOINT ["/opt/von/start.sh"]
CMD ["python", "/opt/von/serve.py"]
