import os
import sys
from unittest.mock import MagicMock

# Inject a mock for paddleocr before any test module triggers the import
# of image_processing.py, which calls PaddleOCR(...) at module level.
# This must run at conftest load time (before pytest collects test files).
_real_ocr = bool(os.environ.get("DEIDENT_REAL_OCR")) or any(
    "integration" in arg for arg in sys.argv
)
if not _real_ocr:
    sys.modules["paddleocr"] = MagicMock()
