# syntax=docker/dockerfile:1.7

# Model names shared across stages.
ARG DET_MODEL=PP-OCRv5_mobile_det
ARG REC_MODEL=latin_PP-OCRv5_mobile_rec

# =============================================================================
# Stage 1 - OCR models
# Downloads the PaddleOCR detection and recognition models from HuggingFace at
# build time so the image is self-contained and starts without network access.
# Models are small and version-pinned; baking them in keeps startup fast and
# reproducible.
# =============================================================================
FROM alpine:3.20 AS models

ARG HF_BASE=https://huggingface.co/PaddlePaddle
ARG DET_MODEL
ARG REC_MODEL

RUN set -eux; \
    for spec in "detection:${DET_MODEL}" "recognition:${REC_MODEL}"; do \
        kind="${spec%%:*}"; name="${spec##*:}"; \
        dest="/models/${kind}/${name}"; \
        mkdir -p "${dest}"; \
        for f in config.json inference.json inference.pdiparams inference.yml; do \
            curl -fSL --retry 5 --retry-delay 2 --retry-all-errors \
                -o "${dest}/${f}" \
                "${HF_BASE}/${name}/resolve/main/${f}"; \
        done; \
    done

# =============================================================================
# Stage 2 - Python dependencies
# Resolves and installs runtime dependencies into a self-contained virtualenv
# using the committed poetry.lock for reproducible builds.
# Pinned to linux/amd64: paddlepaddle only publishes x86_64 Linux wheels.
# =============================================================================
FROM --platform=linux/amd64 python:3.12-slim AS builder


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
RUN poetry export --only main --without-hashes -f requirements.txt -o /tmp/requirements.txt \
    && python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 3 - Runtime
# Minimal image running as a non-root user with only the shared libraries
# required by OpenCV and PaddlePaddle.
# Pinned to linux/amd64: paddlepaddle only publishes x86_64 Linux wheels.
# =============================================================================
FROM --platform=linux/amd64 python:3.12-slim AS runtime

ARG DET_MODEL
ARG REC_MODEL

LABEL org.opencontainers.image.title="deidentification-karnak" \
      org.opencontainers.image.description="OCR-based DICOM burned-in text deidentification API" \
      org.opencontainers.image.source="https://git.hisaas.hcuge.ch/projects/VNAI/repos/image-ocr-identifier"

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

USER app

EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
#     CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=3).status==200 else 1)"]

CMD ["python", "-m", "image_ocr_identifier.main"]
