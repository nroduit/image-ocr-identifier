import os

import numpy as np

DEBUG_IMAGES = os.environ.get("DEBUG_IMAGES", "").lower() in ("1", "true", "yes")


def save_debug_image(image: np.ndarray, color_to_boxes: dict, image_name: str) -> None:
    if not DEBUG_IMAGES:
        return
    from deidentification_karnak.draw_image import draw_masks_on_image

    draw_masks_on_image(image.copy(), color_to_boxes, image_name=image_name)


def save_debug_ocr(ocr_result_raw, image_name: str) -> None:
    if not DEBUG_IMAGES:
        return
    from pathlib import Path
    from deidentification_karnak.draw_image import OUTPUT_DIR

    image_stem = Path(image_name).stem
    output_folder = OUTPUT_DIR / image_stem
    output_folder.mkdir(exist_ok=True)

    for res in ocr_result_raw:
        res.save_to_img(str(output_folder))


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
    image: np.ndarray, ocr_result: dict[str, list], image_name: str
) -> None:
    if not DEBUG_IMAGES:
        return
    import cv2
    from pathlib import Path
    from deidentification_karnak.draw_image import OUTPUT_DIR

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

    image_stem = Path(image_name).stem
    output_folder = OUTPUT_DIR / image_stem
    output_folder.mkdir(exist_ok=True)
    output_path = output_folder / f"{image_stem}_split_boxes.png"
    cv2.imwrite(str(output_path), img)
