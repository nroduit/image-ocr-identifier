from pathlib import Path
import cv2
import numpy as np

from deidentification_karnak.utils import decode_image_bytes

_BASE_PATH = Path(__file__).parent.parent.parent
OUTPUT_DIR = _BASE_PATH / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_DET_MODEL_NAME = "PP-OCRv3_mobile_det"
_REC_MODEL_NAME = "latin_PP-OCRv3_mobile_rec"


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
    image_name: str = "image",
) -> Path:
    """Draw masks on image with each box's corresponding background color.

    Args:
        image: The source image as bytes or a decoded numpy array.
        color_to_boxes: Dict mapping BGR color tuples to lists of boxes.
        image_name: Name used for the output file.

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

    image_stem = Path(image_name).stem
    output_filename = (
        f"PaddleOCR_{image_stem}_{_DET_MODEL_NAME}_{_REC_MODEL_NAME}_masked.png"
    )
    output_folder = OUTPUT_DIR / image_stem
    output_folder.mkdir(exist_ok=True)
    output_path = output_folder / output_filename
    cv2.imwrite(str(output_path), image)

    return output_path
