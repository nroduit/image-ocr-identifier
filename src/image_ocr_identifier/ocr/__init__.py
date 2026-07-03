import os

from image_ocr_identifier.ocr._base import OcrBackend

_backend: OcrBackend | None = None


def get_ocr_backend() -> OcrBackend:
    """Return the OCR backend selected by the ``OCR_BACKEND`` env var.

    Supported values: ``paddle`` (default, server) and ``rapid`` (portable,
    ONNX). The backend module is imported lazily so only the selected engine's
    dependencies are required at runtime.
    """
    global _backend
    if _backend is not None:
        return _backend

    name = os.environ.get("OCR_BACKEND", "paddle").strip().lower()
    if name == "paddle":
        from image_ocr_identifier.ocr._paddle import PaddleBackend

        _backend = PaddleBackend()
    elif name == "rapid":
        from image_ocr_identifier.ocr._rapid import RapidBackend

        _backend = RapidBackend()
    else:
        raise ValueError(f"Unknown OCR_BACKEND: {name!r}. Use 'paddle' or 'rapid'.")
    return _backend
