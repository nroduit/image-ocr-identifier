[![License](https://img.shields.io/badge/License-EPL%202.0-blue.svg)](https://opensource.org/licenses/EPL-2.0) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Image OCR Identifier

A FastAPI service that detects and masks sensitive text burned into DICOM images. It uses OCR to locate text regions, matches them against a provided list of sensitive DICOM tag values using fuzzy matching, and returns mask coordinates with background color information so the caller (e.g. [Karnak](https://github.com/OsiriX-Foundation/karnak)) can apply pixel-level deidentification.

Two OCR backends are available, selectable via the `OCR_BACKEND` environment variable:

| Backend | Engine | Use case |
|---|---|---|
| `paddle` (default) | PaddleOCR | Server deployment, best accuracy, GPU support |
| `rapid` | RapidOCR (ONNX) | Portable standalone executable, lightweight |


## Table of Contents

1. [Language Support](#language-support)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
   - [Server (PaddleOCR backend)](#server-paddleocr-backend)
   - [Portable (RapidOCR/ONNX backend)](#portable-rapidocronnx-backend)
   - [Development (both backends)](#development-both-backends)
5. [OCR Models](#ocr-models)
   - [Paddle backend models](#paddle-backend-models)
   - [Rapid backend models (ONNX)](#rapid-backend-models-onnx)
6. [Configuration](#configuration)
   - [Common variables](#common-variables)
   - [Paddle backend variables](#paddle-backend-variables-ocr_backendpaddle)
   - [Rapid backend variables](#rapid-backend-variables-ocr_backendrapid)
7. [Running the Service](#running-the-service)
   - [Server (development)](#server-development)
   - [Docker (CPU)](#docker-cpu)
   - [Docker (GPU)](#docker-gpu)
8. [Building the Portable Executable](#building-the-portable-executable)
9. [API Reference](#api-reference)
   - [Health Check](#health-check)
   - [Deidentify Image](#deidentify-image)
   - [Reporting](#reporting)
10. [How It Works](#how-it-works)
11. [Running Tests](#running-tests)
12. [Project Structure](#project-structure)
13. [Integration with Karnak](#integration-with-karnak)
14. [License](#license)


## Language Support

> **Important:** Only **Latin** and **CJ** (Chinese, Japanese) scripts are supported by the OCR engine. Text in other scripts (Arabic, Cyrillic, Devanagari, Korean, Thai, etc.) will **not** be detected or deidentified correctly.

## Features

- OCR-based detection of burned-in text in DICOM images
- Fuzzy matching of detected text against sensitive DICOM tag values
- Background color detection for seamless mask application
- Decoding of raw and compressed DICOM pixel data (JPEG, JPEG2000, JPEG-LS, RLE) with Modality LUT, VOI windowing, and palette color support
- Reporting endpoint to audit which sensitive tags appear without computing masks
- API versioning via `Accept` header
- GPU acceleration support (NVIDIA CUDA, paddle backend only)
- Standalone portable build via PyInstaller (rapid backend)

## Requirements

- Python >= 3.12, < 3.14 (portable/rapid backend requires < 3.13)
- [Poetry](https://python-poetry.org/) >= 2.0
- OCR model files (see below)

---

## Installation

### Server (PaddleOCR backend)

```bash
git clone https://github.com/nroduit/image-ocr-identifier
cd deidentification-karnak

# Install base + paddle backend
poetry install --extras paddle
```

### Portable (RapidOCR/ONNX backend)

```bash
# Install base + rapid backend + dev tools (pyinstaller)
poetry install --extras rapid
```

### Development (both backends)

```bash
# Install everything for local development and testing
poetry install --all-extras
```

---

## OCR Models

### Paddle backend models

Download PaddleOCR detection and recognition models from [HuggingFace/PaddlePaddle](https://huggingface.co/PaddlePaddle).

Available model combinations:

| Detection Model | Recognition Model | Notes |
|---|---|---|
| `PP-OCRv6_medium_det` | `PP-OCRv6_medium_rec` | (**Recommended**) Best accuracy, multilingual |
| `PP-OCRv6_small_det` | `PP-OCRv6_small_rec` | Small, multilingual |
| `PP-OCRv6_tiny_det` | `PP-OCRv6_tiny_rec` | Tiny, multilingual |
| `PP-OCRv5_mobile_det` | `latin_PP-OCRv5_mobile_rec` | Lightweight, Latin-only |

Place them under:
```
data/models/detection/<model_name>/
data/models/recognition/<model_name>/
```

### Rapid backend models (ONNX)

ONNX models go under `data/models/onnx/`. Supported layouts:

```
data/models/onnx/PP-OCRv5_mobile/       # v5 style: det.onnx, rec.onnx, keys.txt
data/models/onnx/PP-OCRv6_medium_det_onnx/  # v6 style: inference.onnx
data/models/onnx/PP-OCRv6_medium_rec_onnx/  # v6 style: inference.onnx + keys.txt
```

---

## Configuration

Create a `.env` file at the project root (see [.env.example](.env.example)):

### Common variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server listening port |
| `WORKERS` | CPU count | Number of Uvicorn workers |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_DATA_PATH` | `./data` | Base path for model and data files |
| `OCR_BACKEND` | `paddle` | OCR engine: `paddle` or `rapid` |
| `DEBUG_IMAGES` | (unset) | Set to `1`/`true`/`yes` to save debug images |

### Paddle backend variables (`OCR_BACKEND=paddle`)

| Variable | Default | Description |
|---|---|---|
| `OCR_DET_MODEL` | `PP-OCRv6_medium_det` | Detection model name |
| `OCR_REC_MODEL` | `PP-OCRv6_medium_rec` | Recognition model name |
| `OCR_DET_MODEL_DIR` | deduced from model name | Detection model directory (override) |
| `OCR_REC_MODEL_DIR` | deduced from model name | Recognition model directory (override) |
| `OCR_DEVICE` | (unset = CPU) | Device: `cpu`, `gpu`, `gpu:0` |

### Rapid backend variables (`OCR_BACKEND=rapid`)

| Variable | Default | Description |
|---|---|---|
| `OCR_MODEL` | (none) | ONNX model name under `models/onnx/` (e.g. `PP-OCRv6_medium`) |
| `OCR_DET_MODEL_PATH` | deduced from `OCR_MODEL` | Path to `det.onnx` (override) |
| `OCR_REC_MODEL_PATH` | deduced from `OCR_MODEL` | Path to `rec.onnx` (override) |
| `OCR_REC_KEYS_PATH` | deduced from `OCR_MODEL` | Path to `keys.txt` (override) |

### Example `.env` for server with PP-OCRv6_medium

```dotenv
OCR_BACKEND=paddle
PORT=8000
WORKERS=2
LOG_LEVEL=INFO
OCR_DET_MODEL=PP-OCRv6_medium_det
OCR_REC_MODEL=PP-OCRv6_medium_rec
```

### Example `.env` for portable with PP-OCRv5_mobile (ONNX)

```dotenv
OCR_BACKEND=rapid
PORT=8000
OCR_MODEL=PP-OCRv5_mobile
```

---

## Running the Service

### Server (development)

```bash
poetry run uvicorn image_ocr_identifier.main:app --reload --port 8000
```

Or directly:

```bash
poetry run python -m image_ocr_identifier.main
```

### Docker (CPU)

```bash
docker build -f docker/Dockerfile -t deidentification-karnak .
docker run -p 8000:8000 deidentification-karnak
```

Build with custom models:

```bash
docker build -f docker/Dockerfile \
  --build-arg DET_MODEL=PP-OCRv6_medium_det \
  --build-arg REC_MODEL=PP-OCRv6_medium_rec \
  -t deidentification-karnak .
```

### Docker (GPU)

Requires an NVIDIA GPU with drivers installed and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -f docker/Dockerfile.gpu -t deidentification-karnak:gpu .
docker run --gpus all -p 8000:8000 deidentification-karnak:gpu
```

---

## Building the Portable Executable

The portable build produces a standalone directory (no Python required on the target machine). It uses PyInstaller with the RapidOCR/ONNX backend.

### Prerequisites

```bash
# Install with rapid + dev (pyinstaller)
poetry install --extras rapid
```

### Configuration

Set `OCR_MODEL` in your `.env` to select which ONNX model to bundle:

```dotenv
OCR_MODEL=PP-OCRv5_mobile
```

The `.spec` reads this value to include only the selected model in the bundle.

### Build

```bash
poetry run pyinstaller image_ocr_identifier.spec --noconfirm
```

The output is in `dist/deidentify-karnak/`. Run it:

```bash
./dist/deidentify-karnak/deidentify-karnak
```

The executable:
- Sets `OCR_BACKEND=rapid` automatically
- Auto-detects the bundled ONNX model
- Listens on `127.0.0.1:8000` by default (override with `PORT` and `HOST` env vars)
- Runs a single worker (no multiprocessing)

### What is excluded from the bundle

PaddlePaddle, PyTorch, and other heavy DL frameworks are excluded via the `.spec` file. Only the ONNX runtime (~50 Mo) is bundled.

---

## API Reference

Base URL: `http://localhost:8000`

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok"}
```

### Deidentify Image

```
POST /deidentify-image
Content-Type: multipart/form-data
Accept: application/json; version=1
```

**Form fields:**

The `image` field carries the raw pixel data extracted from a DICOM file (or compressed pixel data with the appropriate transfer syntax). This is **not** a standard image file - it is the binary content of the DICOM `PixelData` (7FE0,0010) element. The DICOM metadata fields below are required for the service to decode and process the pixel data correctly.

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | Yes | Raw DICOM pixel data (binary content of the PixelData element) |
| `sensitive_data_list` | string | Yes | JSON object mapping DICOM tag names to values |
| `rows` | integer | Yes | Image height in pixels (DICOM tag `Rows`) |
| `columns` | integer | Yes | Image width in pixels (DICOM tag `Columns`) |
| `bits_allocated` | integer | Yes | Bits per pixel component (DICOM tag `BitsAllocated`) |
| `samples_per_pixel` | integer | Yes | Number of channels (DICOM tag `SamplesPerPixel`) |
| `transfer_syntax_uid` | string | Yes* | DICOM Transfer Syntax UID (required for compressed data) |
| `photometric_interpretation` | string | Yes* | DICOM Photometric Interpretation (e.g. `MONOCHROME2`, `RGB`, `YBR_FULL_422`) |
| `sop_instance_uid` | string | No | SOP Instance UID, echoed back in the response |
| `rescale_slope` | number | No | Modality LUT rescale slope |
| `rescale_intercept` | number | No | Modality LUT rescale intercept |
| `window_center` | number | No | VOI LUT window center |
| `window_width` | number | No | VOI LUT window width |
| `is_monochrome1` | boolean | No | True if photometric interpretation is MONOCHROME1 |
| `palette_color_lut` | string | No | JSON with "red", "green", "blue" palette arrays |

\* `transfer_syntax_uid` is required when the pixel data is compressed. `photometric_interpretation` is needed for correct color space handling.

**Example request:**

```bash
curl -X POST http://localhost:8000/deidentify-image \
  -H "Accept: application/json; version=1" \
  -F "image=@pixel_data.raw;type=application/octet-stream" \
  -F 'sensitive_data_list={"PatientID":"962738","PatientName":"Doe^John","PatientBirthDate":"19280303"}' \
  -F "rows=512" \
  -F "columns=512" \
  -F "bits_allocated=16" \
  -F "samples_per_pixel=1" \
  -F "transfer_syntax_uid=1.2.840.10008.1.2.4.70" \
  -F "photometric_interpretation=MONOCHROME2" \
  -F "window_center=400" \
  -F "window_width=1500"
```

**Example response (sensitive data detected):**

```json
{
  "masks": [
    {
      "stationName": "*",
      "color": "14181a",
      "rectangles": [
        "78 3 161 28",
        "246 7 102 21"
      ]
    }
  ],
  "message": "3 sensitive data detected"
}
```

**Example response (no sensitive data detected):**

```json
{
  "message": "No sensitive data detected"
}
```

Each rectangle is formatted as `"x y width height"` in pixel coordinates. The `color` field is the detected background color as a 6-character hex string (RRGGBB, no `#` prefix), allowing the caller to paint over detected regions seamlessly.

### Reporting

```
POST /reporting
Content-Type: multipart/form-data
Accept: application/json; version=1
```

Same input as `/deidentify-image`. Returns a list of DICOM tag names detected in the image without computing mask rectangles.

**Example response:**

```json
{
  "detected_tags": ["PatientName", "PatientAge", "PatientBirthDate"],
  "message": "3 sensitive tags detected"
}
```

## How It Works

1. **Image decoding** - Receives raw DICOM pixel data along with the required metadata (`rows`, `columns`, `bits_allocated`, `samples_per_pixel`). Handles compressed transfer syntaxes (JPEG, JPEG2000, JPEG-LS, RLE) via pydicom, and applies Modality LUT, VOI windowing, and palette color as needed to produce a displayable image.
2. **Preprocessing** - Upscales the image and applies CLAHE/unsharp masking to improve OCR accuracy on low-contrast medical images.
3. **OCR** - The selected backend (`paddle` or `rapid`) detects and recognizes text regions in the image.
4. **Sensitive data detection** - Fuzzy string matching (rapidfuzz) compares OCR results against the provided sensitive data list. Uses `partial_ratio > 85` and `token_ratio > 90` thresholds.
5. **Background color detection** - For each matched bounding box, the dominant background color is sampled.
6. **Response** - Returns mask rectangles grouped by background color, ready for the caller to apply.

## Running Tests

```bash
# Unit tests only (default, excludes integration tests)
poetry run pytest

# Include integration tests only (requires DICOM test data in data/integration_test/)
poetry run pytest -m integration
```

## Project Structure

```
src/image_ocr_identifier/
  main.py                     - FastAPI app entry point, /health endpoint
  routers/deidentify_image.py - POST /deidentify-image endpoint
  routers/reporting.py        - POST /reporting endpoint
  pipeline.py                 - Orchestration (preprocess, OCR, detect, color)
  image_processing.py         - OCR call (via backend) + text block splitting
  ocr/                        - OCR backend abstraction
    __init__.py               - Factory: get_ocr_backend() (lazy imports)
    _base.py                  - OcrBackend protocol (interface contract)
    _paddle.py                - PaddleOCR backend (server)
    _rapid.py                 - RapidOCR/ONNX backend (portable)
  sensitive_data_detection.py - Fuzzy matching against sensitive data
  color_detection.py          - Background color analysis
  dicom_decode.py             - DICOM pixel decode (decompression, LUT, VOI)
  preprocessing.py            - Image preprocessing (upscale, CLAHE, unsharp)
  models/response.py          - Pydantic response models
  utils.py                    - Coordinate transforms, color conversion
  debug.py                    - Debug image output
server_portable.py            - Entry point for the PyInstaller portable build
deidentify-karnak.spec        - PyInstaller build specification
tests/
  test_router.py              - Router integration tests
  test_sensitive_data_detection.py
  test_image_processing.py
  test_color_detection.py
  test_dicom_decode.py
  test_utils.py
  integration/                - End-to-end tests with real DICOM data
docker/
  Dockerfile                  - Production CPU image
  Dockerfile.gpu              - Production GPU image (NVIDIA CUDA)
```

## Integration with Karnak

This service is designed to work with [Karnak](https://github.com/OsiriX-Foundation/karnak), a DICOM gateway for clinical research deidentification. Karnak sends DICOM pixel data along with the sensitive tag values to this service, then uses the returned mask coordinates and colors to overwrite burned-in text directly in the DICOM pixel data before forwarding the de-identified image.

## License

See [LICENSE](LICENSE) for details.
