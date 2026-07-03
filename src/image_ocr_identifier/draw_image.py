from pathlib import Path

import cv2
import numpy as np

from image_ocr_identifier.dicom_decode import decode_image_bytes


def _polygon_to_bbox(polygon: list | np.ndarray) -> tuple[int, int, int, int]:
    """Convert a polygon (4 corner points) to an axis-aligned bounding box."""
    polygon = np.array(polygon)

    if polygon.ndim == 1:
        if len(polygon) == 8:
            polygon = polygon.reshape(4, 2)
        elif len(polygon) == 4:
            return (int(polygon[0]), int(polygon[1]), int(polygon[2]), int(polygon[3]))
        else:
            raise ValueError(f"Unexpected box format with {len(polygon)} elements")

    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max()),
        int(y_coords.max()),
    )


def draw_rectangle(img, box, color=(0, 0, 0), thickness=-1):
    try:
        x_min, y_min, x_max, y_max = _polygon_to_bbox(box)
    except (ValueError, IndexError):
        return
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)


def draw_masks_on_image(
    image: bytes | np.ndarray,
    color_to_boxes: dict[tuple[int, int, int], list],
    output_folder: Path | None = None,
) -> Path:
    """Draw masks on image with each box's corresponding background color.

    Args:
        image: The source image as bytes or a decoded numpy array.
        color_to_boxes: Dict mapping BGR color tuples to lists of boxes.
        output_folder: Directory to save the output image.

    Returns:
        Path to the saved masked image.
    """
    if isinstance(image, bytes):
        image = decode_image_bytes(image)
    if image is None:
        return None

    for color, boxes in color_to_boxes.items():
        for box in boxes:
            draw_rectangle(image, box, color=color)

    output_path = output_folder / "masked.png"
    cv2.imwrite(str(output_path), image)

    return output_path
