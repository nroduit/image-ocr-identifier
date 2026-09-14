# syntax=docker/dockerfile:1.7

# Model names shared across stages.
ARG DET_MODEL=PP-OCRv6_medium_det
ARG REC_MODEL=PP-OCRv6_medium_rec

# =============================================================================
# Stage 1 - OCR models
# Downloads the PaddleOCR detection and recognition models from HuggingFace at
# build time so the image is self-contained and starts without network access.
# huggingface_hub.snapshot_download is layout-agnostic and resilient (retries,
# resume, integrity checks); allow_patterns fetches only the files that exist,
# so it works across model versions regardless of whether they ship config.json
# (PP-OCRv6 does not).
# =============================================================================
FROM python:3.14-slim AS models

ARG DET_MODEL
ARG REC_MODEL
# Pin per-model revisions for fully reproducible builds.
ARG DET_REVISION=8e0f56fb2ef86b461d99cfc7ac5c137738985f61
ARG REC_REVISION=e5a92bcbc5cc1b494628e458d267778f0704fd7c
# Point at an internal HuggingFace mirror for air-gapped/enterprise networks.
ARG HF_ENDPOINT=https://huggingface.co

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=${HF_ENDPOINT} \
    DET_MODEL=${DET_MODEL} \
    REC_MODEL=${REC_MODEL} \
    DET_REVISION=${DET_REVISION} \
    REC_REVISION=${REC_REVISION}

RUN pip install --no-cache-dir "huggingface_hub>=0.25,<1.0"

RUN python3 <<'PY'
import glob
import os
import shutil

from huggingface_hub import snapshot_download

patterns = ["inference.json", "inference.pdiparams", "inference.yml", "config.json"]
jobs = (
    ("detection", os.environ["DET_MODEL"], os.environ["DET_REVISION"]),
    ("recognition", os.environ["REC_MODEL"], os.environ["REC_REVISION"]),
)
for kind, name, revision in jobs:
    snapshot_download(
        repo_id=f"PaddlePaddle/{name}",
        revision=revision,
        allow_patterns=patterns,
        local_dir=f"/models/{kind}/{name}",
    )

for cache in glob.glob("/models/**/.cache", recursive=True):
    shutil.rmtree(cache, ignore_errors=True)
PY

# =============================================================================
# Stage 2 - Python dependencies
# Resolves and installs runtime dependencies into a self-contained virtualenv
# using the committed poetry.lock for reproducible builds.
# Pinned to linux/amd64: paddlepaddle only publishes x86_64 Linux wheels.
# =============================================================================
FROM --platform=linux/amd64 python:3.14-slim AS builder


ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1

RUN pip install "poetry>=2.1,<3.0" poetry-plugin-export

WORKDIR /app

# Only the files needed to resolve dependencies, so this layer is cached as
# long as the lockfile does not change.
COPY pyproject.toml poetry.lock README.md ./

# Export the locked main dependencies and install them with pip into a relocatable
# virtualenv. pip is used instead of "poetry install" because the dependency tree
# pulls both opencv-python and opencv-contrib-python, whose files overlap in cv2/;
# poetry's installer aborts on the collision while pip resolves it cleanly.
RUN poetry export --only main --extras paddle --without-hashes -f requirements.txt -o /tmp/requirements.txt \
    && python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 3 - Runtime
# Minimal image running as a non-root user with only the shared libraries
# required by OpenCV and PaddlePaddle.
# Pinned to linux/amd64: paddlepaddle only publishes x86_64 Linux wheels.
# =============================================================================
FROM --platform=linux/amd64 python:3.14-slim AS runtime

ARG DET_MODEL
ARG REC_MODEL

LABEL org.opencontainers.image.title="image-ocr-identifier" \
      org.opencontainers.image.description="OCR-based DICOM burned-in text deidentification API" \
      org.opencontainers.image.source="https://github.com/nroduit/image-ocr-identifier"

# Native libraries required by the OpenCV wheels (full, non-headless build) and
# PaddlePaddle's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    HOME=/home/app \
    MODEL_DATA_PATH=/app/data \
    OCR_DET_MODEL=${DET_MODEL} \
    OCR_REC_MODEL=${REC_MODEL} \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PORT=8000 \
    LOG_LEVEL=INFO
# WORKERS defaults to the host CPU count. Each worker fully loads the OCR models
# into memory, so set WORKERS explicitly when sizing the container.

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --from=models --chown=app:app /models /app/data/models

# Debug image output (DEBUG_IMAGES=1) is written under /app/output
RUN mkdir -p /app/output && chown app:app /app/output

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=3).status==200 else 1)"]

CMD ["python", "-m", "image_ocr_identifier.main"]
