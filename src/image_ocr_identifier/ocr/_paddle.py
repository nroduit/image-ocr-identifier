import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_BASE_DATA_PATH = Path(
    os.environ.get("MODEL_DATA_PATH", str(Path(__file__).parents[3] / "data"))
)

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr

    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    det_model_name = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
    rec_model_name = os.environ.get("OCR_REC_MODEL", "latin_PP-OCRv5_mobile_rec")
    det_model_dir = os.environ.get(
        "OCR_DET_MODEL_DIR",
        str(_BASE_DATA_PATH / "models" / "detection" / det_model_name),
    )
    rec_model_dir = os.environ.get(
        "OCR_REC_MODEL_DIR",
        str(_BASE_DATA_PATH / "models" / "recognition" / rec_model_name),
    )

    for name, directory in (
        ("detection", det_model_dir),
        ("recognition", rec_model_dir),
    ):
        if not Path(directory).is_dir():
            raise RuntimeError(f"OCR {name} model directory not found: {directory}")

    # Device selection: "cpu", "gpu", "gpu:0", ... Unset means the PaddleOCR
    # default (CPU).
    device = os.environ.get("OCR_DEVICE") or None

    logger.info(
        "Loading PaddleOCR: det=%s rec=%s device=%s",
        det_model_dir,
        rec_model_dir,
        device,
    )

    _ocr = PaddleOCR(
        use_textline_orientation=False,
        text_detection_model_name=det_model_name,
        text_detection_model_dir=det_model_dir,
        text_recognition_model_name=rec_model_name,
        text_recognition_model_dir=rec_model_dir,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_det_thresh=0.2,
        text_det_box_thresh=0.4,
        enable_mkldnn=False,
        device=device,
    )
    return _ocr


class PaddleBackend:
    def detect(self, image: np.ndarray) -> tuple[list[str], list[list[int]]]:
        result = _get_ocr().predict(image, return_word_box=False)
        if not result:
            return [], []
        texts = list(result[0]["rec_texts"])
        boxes = [[int(v) for v in box] for box in result[0]["rec_boxes"]]
        return texts, boxes
