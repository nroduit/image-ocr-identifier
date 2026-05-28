import os
from datetime import datetime
from pathlib import Path

import numpy as np

DEBUG_IMAGES = os.environ.get("DEBUG_IMAGES", "").lower() in ("1", "true", "yes")

_BASE_PATH = Path(__file__).parent.parent.parent
OUTPUT_DIR = _BASE_PATH / "output" / "debug_images"


class DebugSession:
    """Groups all debug outputs for a single input image into one unique folder."""

    def __init__(self, image_name: str):
        image_stem = Path(image_name).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.folder = OUTPUT_DIR / f"{image_stem}_{timestamp}"
        self._created = False

    def _ensure_folder(self):
        if not self._created:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._created = True


def create_debug_session(image_name: str) -> DebugSession | None:
    if not DEBUG_IMAGES:
        return None
    return DebugSession(image_name)


def save_debug_image(
    image: np.ndarray, color_to_boxes: dict, session: DebugSession | None
) -> None:
    if session is None:
        return
    from deidentification_karnak.draw_image import draw_masks_on_image

    session._ensure_folder()
    draw_masks_on_image(image.copy(), color_to_boxes, output_folder=session.folder)


def save_debug_ocr(ocr_result_raw, session: DebugSession | None) -> None:
    if session is None:
        return

    session._ensure_folder()
    for res in ocr_result_raw:
        res.save_to_img(str(session.folder))


_SPLIT_BOX_COLORS = [
    (255, 0, 0),
    (0, 200, 0),
    (0, 0, 255),
    (255, 165, 0),
    (128, 0, 128),
    (0, 200, 200),
    (200, 0, 200),
    (200, 200, 0),
]


def save_debug_split_boxes(
    image: np.ndarray, ocr_result: dict[str, list], session: DebugSession | None
) -> None:
    if session is None:
        return
    import cv2

    session._ensure_folder()

    img = image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    texts = ocr_result["texts"]
    boxes = ocr_result["boxes"]

    for idx, (text, box) in enumerate(zip(texts, boxes)):
        color = _SPLIT_BOX_COLORS[idx % len(_SPLIT_BOX_COLORS)]
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.putText(
            img,
            text,
            (x_min, max(y_min - 2, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path = session.folder / "split_boxes.png"
    cv2.imwrite(str(output_path), img)


def save_debug_preprocessed(image: np.ndarray, session: DebugSession | None) -> None:
    if session is None:
        return
    import cv2

    session._ensure_folder()
    cv2.imwrite(str(session.folder / "preprocessed.png"), image)
