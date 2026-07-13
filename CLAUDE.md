# Image OCR Identifier

FastAPI service that detects and masks sensitive text burned into DICOM images using OCR (PaddleOCR), then returns mask coordinates for Karnak to apply deidentification.

## Project Structure

```
src/image_ocr_identifier/
  main.py                  - FastAPI app, /health endpoint
  routers/deidentify_image.py - POST /deidentify-image endpoint (HTTP layer, validation)
  pipeline.py              - Deidentification orchestration (preprocess, OCR, detect, color)
  models/response.py       - Pydantic response models (MaskGroup, DeidentificationResponse)
  image_processing.py      - PaddleOCR text detection and recognition, OCR block splitting
  sensitive_data_detection.py - Fuzzy matching OCR results against sensitive data list
  color_detection.py       - Background color analysis for bounding boxes
  dicom_decode.py          - DICOM pixel decode (decompression, LUT, VOI, palette)
  draw_image.py            - Debug visualization (draw masks on images)
  preprocessing.py         - Image preprocessing for OCR (upscale, CLAHE, unsharp)
  utils.py                 - Coordinate transforms and color conversion helpers
  debug.py                 - Debug output (controlled by DEBUG_IMAGES env var)
tests/
  conftest.py              - Shared fixtures
  test_router.py           - Router integration tests
  test_sensitive_data_detection.py - Fuzzy matching tests
  test_image_processing.py - OCR processing tests
  test_color_detection.py  - Color detection tests
  test_dicom_decode.py     - DICOM decode tests
  test_utils.py            - Utility function tests
```

## Commands

- Run server: `uvicorn image_ocr_identifier.main:app`
- Run tests: `pytest`
- Install: `poetry install`

## Environment Variables

- `LOG_LEVEL` - Logging level (default: INFO)
- `DEBUG_IMAGES` - Set to "1"/"true"/"yes" to save debug images to disk

## Key Technical Details

- Python >=3.12, <3.14
- Build system: Poetry (poetry-core masonry backend)
- OCR engine: PaddleOCR (with PaddlePaddle backend)
- Fuzzy matching: rapidfuzz (partial_ratio > 85, token_ratio > 90)
- DICOM handling: pydicom + python-gdcm for decompression
- Image processing: OpenCV
- API versioning via Accept header: `application/json; version=N`

## Approach

- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

## Output

- Return code first. Explanation after, only if non-obvious.
- No inline prose. Use comments sparingly - only where logic is unclear.
- No boilerplate unless explicitly requested.

## Code Rules

- Simplest working solution. No over-engineering.
- No abstractions for single-use operations.
- No speculative features or "you might also want..."
- Read the file before modifying it. Never edit blind.
- No docstrings or type annotations on code not being changed.
- No error handling for scenarios that cannot happen.
- Three similar lines is better than a premature abstraction.

## Review Rules

- State the bug. Show the fix. Stop.
- No suggestions beyond the scope of the review.
- No compliments on the code before or after the review.

## Debugging Rules

- Never speculate about a bug without reading the relevant code first.
- State what you found, where, and the fix. One pass.
- If cause is unclear: say so. Do not guess.

## Simple Formatting

- No em dashes, smart quotes, or decorative Unicode symbols.
- Plain hyphens and straight quotes only.
- Natural language characters (accented letters, CJK, etc.) are fine when the content requires them.
- Code output must be copy-paste safe.
