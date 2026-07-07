import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Detection thresholds kept identical to the PaddleOCR configuration.
_DET_THRESH = 0.2  # was text_det_thresh
_DET_BOX_THRESH = 0.4  # was text_det_box_thresh

# Black border added around the image before detection. RapidOCR's text
# detector misses text that is flush against the image edge (PaddleOCR did
# not); padding moves such text off the border so burned-in PHI at the very
# top/edge of the frame is still detected. Box coordinates are shifted back.
_OCR_BORDER_PAD = 32

# RapidOCR downscales any image whose largest side exceeds ``max_side_len``
# (default 2000). The preprocessing step deliberately upscales images to a
# 1500px minimum side, which pushes the larger side past that default and
# triggers a counter-productive downscale that drops small/edge text. Raise
# the limit so our upscaled (and padded) images are detected at full size.
_OCR_MAX_SIDE_LEN = 3000

_ocr = None


def _get_base_data_path() -> Path:
    """
    Resolve base data path with proper support for PyInstaller frozen bundles.
    Called lazily to ensure MODEL_DATA_PATH environment variable is set first.
    """
    if getattr(sys, "frozen", False):
        return Path(
            os.environ.get("MODEL_DATA_PATH", os.path.join(sys._MEIPASS, "data"))
        )
    return Path(
        os.environ.get("MODEL_DATA_PATH", str(Path(__file__).parents[3] / "data"))
    )


def _resolve_model_dir_paths(onnx_dir: Path, model_name: str) -> tuple[str, str, str]:
    """Resolve det/rec/keys paths from an OCR_MODEL name.

    Supported layouts:
      - PP-OCRv5_mobile: single folder with det.onnx, rec.onnx, keys.txt
      - PP-OCRv6_{size}: separate _det_onnx/_rec_onnx folders
        with inference.onnx + keys.txt
    """
    model_dir = onnx_dir / model_name
    if model_dir.is_dir():
        # Single-folder layout (v5 style)
        return (
            str(model_dir / "det.onnx"),
            str(model_dir / "rec.onnx"),
            str(model_dir / "keys.txt"),
        )
    # v6 style: PP-OCRv6_medium -> PP-OCRv6_medium_det_onnx / PP-OCRv6_medium_rec_onnx
    det_dir = onnx_dir / f"{model_name}_det_onnx"
    rec_dir = onnx_dir / f"{model_name}_rec_onnx"
    return (
        str(det_dir / "inference.onnx"),
        str(rec_dir / "inference.onnx"),
        str(rec_dir / "keys.txt"),
    )


def _get_model_paths() -> tuple[str, str, str]:
    """Return (det_model_path, rec_model_path, rec_keys_path), resolved at call time."""
    onnx_dir = _get_base_data_path() / "models" / "onnx"

    # High-level model selector: set OCR_MODEL to switch models easily.
    # Examples: PP-OCRv5_mobile, PP-OCRv6_medium, PP-OCRv6_small, PP-OCRv6_tiny
    ocr_model = os.environ.get("OCR_MODEL")
    if ocr_model:
        det, rec, keys = _resolve_model_dir_paths(onnx_dir, ocr_model)
    else:
        det = str(onnx_dir / "det.onnx")
        rec = str(onnx_dir / "rec.onnx")
        keys = str(onnx_dir / "keys.txt")

    # Individual path overrides still take priority.
    det_model_path = os.environ.get("OCR_DET_MODEL_PATH", det)
    rec_model_path = os.environ.get("OCR_REC_MODEL_PATH", rec)
    rec_keys_path = os.environ.get("OCR_REC_KEYS_PATH", keys)
    return det_model_path, rec_model_path, rec_keys_path


def _get_ocr():
    """Get or initialize the RapidOCR (ONNX runtime) instance with model paths."""
    global _ocr
    if _ocr is not None:
        return _ocr

    from rapidocr_onnxruntime import RapidOCR

    det_model_path, rec_model_path, rec_keys_path = _get_model_paths()
    for name, path in zip(
        ("detection", "recognition", "keys"),
        (det_model_path, rec_model_path, rec_keys_path),
        strict=False,
    ):
        if not Path(path).is_file():
            raise RuntimeError(f"OCR {name} file not found: {path}")

    logger.info(
        "Loading RapidOCR: det=%s rec=%s keys=%s",
        det_model_path,
        rec_model_path,
        rec_keys_path,
    )

    _ocr = RapidOCR(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        rec_keys_path=rec_keys_path,
        use_det=True,
        use_cls=False,
        use_rec=True,
        det_thresh=_DET_THRESH,
        det_box_thresh=_DET_BOX_THRESH,
        text_score=0.0,
        max_side_len=_OCR_MAX_SIDE_LEN,
    )
    return _ocr


class RapidBackend:
    def detect(self, image: np.ndarray) -> tuple[list[str], list[list[int]]]:
        import cv2

        pad = _OCR_BORDER_PAD
        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        result, _ = _get_ocr()(padded, use_det=True, use_cls=False, use_rec=True)
        if not result:
            return [], []

        height, width = image.shape[:2]
        texts: list[str] = []
        boxes: list[list[int]] = []
        for polygon, text, _score in result:
            # Shift coordinates back from the padded image to the original frame
            # and clip to the image bounds.
            xs = [point[0] - pad for point in polygon]
            ys = [point[1] - pad for point in polygon]
            x_min = max(0, min(int(min(xs)), width))
            y_min = max(0, min(int(min(ys)), height))
            x_max = max(0, min(int(max(xs)), width))
            y_max = max(0, min(int(max(ys)), height))
            boxes.append([x_min, y_min, x_max, y_max])
            texts.append(text)
        return texts, boxes
