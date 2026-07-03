"""Background color detection for OCR bounding boxes.

Detects the background color inside text bounding boxes by filtering out
text pixels using statistical distance from the median color.
"""

import numpy as np

from image_ocr_identifier.dicom_decode import decode_image_bytes

QUANTIZATION_STEP = 1
TEXT_DISTANCE_THRESHOLD = 30


def _decode_image(image: bytes | np.ndarray) -> np.ndarray | None:
    if isinstance(image, bytes):
        return decode_image_bytes(image)
    return image


def _polygon_to_bbox(polygon: list | np.ndarray) -> tuple[int, int, int, int]:
    """Convert a polygon or flat coordinate list to (x_min, y_min, x_max, y_max)."""
    pts = np.asarray(polygon)
    if pts.ndim == 1:
        if len(pts) == 8:
            pts = pts.reshape(4, 2)
        elif len(pts) == 4:
            return int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        else:
            raise ValueError(f"Unexpected box format with {len(pts)} elements")
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _background_color(pixels: np.ndarray) -> tuple[int, int, int]:
    """Return the background color from an (N, 3) pixel array, excluding text.

    Uses the median as a robust initial estimate (text < 50% of pixels),
    then filters out pixels far from the median (text pixels) and returns
    the mean of the remaining background pixels, quantized for grouping.
    """
    if len(pixels) == 0:
        return (0, 0, 0)

    median = np.median(pixels.astype(np.float64), axis=0)
    distances = np.linalg.norm(pixels.astype(np.float64) - median, axis=1)
    mask = distances <= TEXT_DISTANCE_THRESHOLD
    bg_pixels = pixels[mask]

    if len(bg_pixels) == 0:
        bg_pixels = pixels

    mean = bg_pixels.mean(axis=0)
    q = (mean // QUANTIZATION_STEP * QUANTIZATION_STEP).astype(np.uint8)
    return int(q[0]), int(q[1]), int(q[2])


def get_colors(
    image: bytes | np.ndarray, boxes: list | np.ndarray
) -> dict[tuple[int, int, int], list[list]]:
    """Return a dict mapping background colors (BGR) to their corresponding boxes.

    Extracts pixels inside each bounding box, filters out text pixels using
    statistical distance from the median, and returns the mean background color.
    """
    if boxes is None or len(boxes) == 0:
        return {}

    image = _decode_image(image)
    if image is None:
        return {}

    h, w = image.shape[:2]
    color_to_boxes: dict[tuple[int, int, int], list[list]] = {}

    for box in boxes:
        try:
            x0, y0, x1, y1 = _polygon_to_bbox(box)
        except (ValueError, IndexError):
            continue

        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = image[y0:y1, x0:x1]
        pixels = roi.reshape(-1, 3)
        color = _background_color(pixels)

        box_list = box.tolist() if isinstance(box, np.ndarray) else box
        color_to_boxes.setdefault(color, []).append(box_list)

    return color_to_boxes
