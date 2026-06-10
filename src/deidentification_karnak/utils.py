import logging
import cv2

logger = logging.getLogger(__name__)


def expand_boxes(
    boxes: list[list[int | float]], margin: int = 2
) -> list[list[int | float]]:
    """Expand each bounding box by a fixed margin on all sides."""
    expanded = []
    for box in boxes:
        if len(box) != 4:
            expanded.append(box)
            continue
        x_min, y_min, x_max, y_max = box
        expanded.append(
            [
                max(0, x_min - margin),
                max(0, y_min - margin),
                x_max + margin,
                y_max + margin,
            ]
        )
    return expanded


def numpy_to_python_type(obj):
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return obj


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"{r:02x}{g:02x}{b:02x}"


def format_boxes(boxes: list) -> list[str]:
    formatted = []
    for box in boxes:
        formatted.append(cv2_to_karnak_coord(box))
    return formatted


def cv2_to_karnak_coord(box: list[int | float]) -> list[int]:
    if (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(coord, (int, float)) for coord in box)
    ):
        x_min, y_min, x_max, y_max = box
        return f"{int(x_min)} {int(y_min)} {int(x_max - x_min)} {int(y_max - y_min)}"
    else:
        logger.debug("Warning: Skipping invalid box format: %s", box)
        return ""


def convert_upscaled_boxes(
    boxes: list[list[int | float]], scale_factor: float
) -> list[list[int | float]]:
    """Convert boxes from upscaled coordinates back to original image coordinates."""
    return [[coord / scale_factor for coord in box] for box in boxes]
