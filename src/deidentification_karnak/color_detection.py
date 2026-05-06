"""Background color detection for OCR bounding boxes.

Detects the background color behind text regions by sampling pixels from
the border ring surrounding each bounding box, then groups boxes by their
dominant surrounding color.
"""

import numpy as np

from deidentification_karnak.utils import decode_image_bytes

QUANTIZATION_STEP = 1
BORDER_MARGIN = 1


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


def _collect_surrounding_pixels(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    margin: int,
) -> np.ndarray:
    """Return an (N, 3) array of BGR pixels from the border ring around a box.

    Collects four non-overlapping strips (top, bottom, left, right) from the
    margin-wide ring outside [x0, y0, x1, y1], clamped to image bounds.
    """
    image_height, image_width = image.shape[:2]

    # Outer rectangle (expanded by margin, clamped)
    outer_x0, outer_y0 = max(0, x0 - margin), max(0, y0 - margin)
    outer_x1, outer_y1 = min(image_width, x1 + margin), min(image_height, y1 + margin)

    strips: list[np.ndarray] = []

    if outer_y0 < y0:  # top
        strips.append(image[outer_y0:y0, outer_x0:outer_x1].reshape(-1, 3))
    if y1 < outer_y1:  # bottom
        strips.append(image[y1:outer_y1, outer_x0:outer_x1].reshape(-1, 3))
    if outer_x0 < x0:  # left (between top and bottom)
        strips.append(image[y0:y1, outer_x0:x0].reshape(-1, 3))
    if x1 < outer_x1:  # right (between top and bottom)
        strips.append(image[y0:y1, x1:outer_x1].reshape(-1, 3))

    if not strips:
        return np.empty((0, 3), dtype=np.uint8)

    return np.concatenate(strips)


def _dominant_color(pixels: np.ndarray) -> tuple[int, int, int]:
    """Return the most frequent quantized BGR color from an (N, 3) pixel array.

    Colors are bucketed by QUANTIZATION_STEP to absorb compression noise,
    then packed into a single uint32 for fast counting with np.unique.
    """
    if len(pixels) == 0:
        return (0, 0, 0)

    q = (pixels // QUANTIZATION_STEP * QUANTIZATION_STEP).astype(np.uint32)
    packed = q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16)

    values, counts = np.unique(packed, return_counts=True)
    winner = values[counts.argmax()]
    return int(winner & 0xFF), int((winner >> 8) & 0xFF), int((winner >> 16) & 0xFF)


def get_colors(
    image: bytes | np.ndarray, boxes: list | np.ndarray
) -> dict[tuple[int, int, int], list[list]]:
    """Return a dict mapping background colors (BGR) to their corresponding boxes.

    The function samples the border pixels around each bounding box (where text
    is unlikely to appear) to determine the background color of that box,
    then groups boxes by their detected background color.

    Args:
        image: BGR image as bytes or numpy array.
        boxes: List of bounding boxes. Supports both formats:
               - Polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
               - Axis-aligned: [x_min, y_min, x_max, y_max]

    Returns:
        A dict where keys are (B, G, R) color tuples and values are lists of
        boxes that have that background color. Returns empty dict if boxes is
        empty or None.

    Example:
        {
            (0, 0, 0): [[[10, 10], [50, 10], [50, 30], [10, 30]]],
            (255, 255, 255): [[[60, 10], [100, 10], [100, 30], [60, 30]],
                              [[120, 10], [160, 10], [160, 30], [120, 30]]]
        }
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

        # Clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        pixels = _collect_surrounding_pixels(image, x0, y0, x1, y1, BORDER_MARGIN)
        color = _dominant_color(pixels)

        box_list = box.tolist() if isinstance(box, np.ndarray) else box
        color_to_boxes.setdefault(color, []).append(box_list)

    return color_to_boxes
