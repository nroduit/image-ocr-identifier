import os
from pathlib import Path
import numpy as np

from deidentification_karnak.debug import save_debug_ocr

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR

_BASE_DATA_PATH = Path(
    os.environ.get("MODEL_DATA_PATH", str(Path(__file__).parent.parent.parent / "data"))
)

MODEL_TEXT_DETECTION_V3 = str(
    _BASE_DATA_PATH / "models" / "detection" / "en_PP-OCRv3_det"
)
MODEL_TEXT_RECOGNITION_V3 = str(
    _BASE_DATA_PATH / "models" / "recognition" / "latin_PP-OCRv3_mobile_rec"
)

MODEL_TEXT_DETECTION_MOBILE_V5 = str(
    _BASE_DATA_PATH / "models" / "detection" / "PP-OCRv5_mobile_det"
)
MODEL_TEXT_DETECTION_SERVER_V5 = str(
    _BASE_DATA_PATH / "models" / "detection" / "PP-OCRv5_server_det"
)
MODEL_TEXT_RECOGNITION_V5 = str(
    _BASE_DATA_PATH / "models" / "recognition" / "latin_PP-OCRv5_mobile_rec"
)

SPACE_WEIGHT = 0.2

_det_model_name = "PP-OCRv3_mobile_det"
_rec_model_name = "latin_PP-OCRv3_mobile_rec"
# _det_model_name = "PP-OCRv5_server_det"
# _rec_model_name = "latin_PP-OCRv5_mobile_rec"

# Initialize OCR
_ocr = PaddleOCR(
    use_angle_cls=False,
    lang="latin",
    text_detection_model_name=_det_model_name,
    text_detection_model_dir=MODEL_TEXT_DETECTION_V3,
    text_recognition_model_name=_rec_model_name,
    text_recognition_model_dir=MODEL_TEXT_RECOGNITION_V3,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)


def process_image_with_ocr(
    image: np.ndarray, image_name: str = "image"
) -> dict[str, list]:
    result = _ocr.predict(image, return_word_box=False)
    save_debug_ocr(result, image_name=image_name)
    texts, boxes = result[0]["rec_texts"], result[0]["rec_boxes"]
    return {"texts": texts, "boxes": boxes}


def split_ocr_blocks(ocr_result: dict[str, list]) -> dict[str, list]:
    """Split OCR text blocks into sub-blocks based on delimiters.

    For each text+box pair, splits the text on whitespace, commas, or
    space-surrounded slashes (e.g. " / "). Slashes without surrounding
    spaces (e.g. "27/10/2020") are NOT treated as delimiters.
    Sub-boxes are computed by proportional x-coordinate interpolation within
    the original bounding box.

    Blocks with a single token (no delimiter) are kept as-is.
    """
    texts = ocr_result["texts"]
    boxes = ocr_result["boxes"]

    split_texts = []
    split_boxes = []

    for text, box in zip(texts, boxes):
        sub_blocks = _split_single_block(text, box)
        for sub_text, sub_box in sub_blocks:
            split_texts.append(sub_text)
            split_boxes.append(sub_box)

    return {"texts": split_texts, "boxes": split_boxes}


def _split_single_block(
    text: str, box: list[int | float]
) -> list[tuple[str, list[int]]]:
    """Split one text+box into sub-blocks based on delimiter positions.

    Adds a small padding to each sub-box to compensate for variable
    character widths in the original image (proportional split is an
    approximation assuming monospace).
    """
    if len(box) != 4 or isinstance(box[0], (list, tuple)):
        return [(text, box)]

    parts_with_positions = _find_parts(text)
    if len(parts_with_positions) <= 1:
        return [(text, box)]

    x_min, y_min, x_max, y_max = (
        int(box[0]),
        int(box[1]),
        int(box[2]),
        int(box[3]),
    )
    total_len = sum(
        [SPACE_WEIGHT if c == " " else 1.0 for c in text]
    )  # Treat spaces as 0.3 characters for width estimation
    if total_len == 0:
        return [(text, box)]

    print(parts_with_positions)
    print(x_min, y_min, x_max, y_max)
    print("---------")

    box_width = x_max - x_min
    pixel_per_char = box_width / total_len
    # Padding: 90% of a character width on each side, minimum 3px
    padding = max(3, int(pixel_per_char * 0.9))

    results = []
    prev_end_idx = None
    for part, start_idx, end_idx in parts_with_positions:
        visual_start_idx = prev_end_idx if prev_end_idx is not None else start_idx
        sub_x_min = max(
            x_min, int(x_min + _effective_pos(text, visual_start_idx) * pixel_per_char)
        )
        sub_x_max = min(
            x_max, int(x_min + _effective_pos(text, end_idx) * pixel_per_char) + padding
        )
        results.append((part, [sub_x_min, y_min, sub_x_max, y_max]))
        prev_end_idx = end_idx + 1

    return results


def _effective_pos(text: str, idx: int, space_weight: float = SPACE_WEIGHT) -> float:
    return sum(space_weight if c == " " else 1.0 for c in text[:idx])


def _find_parts(text: str) -> list[tuple[str, int, int]]:
    """Find non-delimiter parts with their start and end character indices.

    Delimiters are: commas and non-date slashes. Whitespace is NOT a delimiter
    so that text dates like "2021 Jan 26" or "26 janvier 2021" stay as one block.
    Slashes between numeric segments (e.g. "27/10/2020", "1928 / 03 / 03") are
    also preserved.
    """
    parts = []
    i = 0
    n = len(text)
    while i < n:
        # Skip commas, slashes (at boundary level), and whitespace between tokens
        if text[i] in (",", " ", "\t"):
            i += 1
            continue
        # Slash at token boundary is a delimiter
        if text[i] == "/":
            i += 1
            continue
        # Start of a token
        start = i
        while i < n:
            if text[i] == ",":
                break
            if text[i] == "/":
                # Preserve slash if it separates numeric segments (date-like)
                if _is_date_slash(text, i):
                    i += 1
                    # Skip any spaces between the slash and the next digit
                    while i < n and text[i] in (" ", "\t"):
                        i += 1
                    continue
                else:
                    break
            i += 1
        # Strip trailing whitespace from the token
        token = text[start:i].rstrip()
        if token:
            parts.append((token, start, start + len(token)))
    return parts


def _is_date_slash(text: str, slash_idx: int) -> bool:
    """Return True if the slash at slash_idx separates numeric segments (date-like).

    Allows optional spaces before and after the slash, e.g. "03/ 03" or "1928 / 03".
    When there is a space before the slash, the segment immediately preceding it
    (back to the last space or start) must contain no letters, otherwise it is a
    field separator.
    """
    # Nearest non-space character before slash must be a digit
    j = slash_idx - 1
    while j >= 0 and text[j] in (" ", "\t"):
        j -= 1
    if j < 0 or not text[j].isdigit():
        return False
    # Verify the preceding segment (back to last separator) has no letters
    k = j
    while k >= 0 and text[k] not in (" ", "\t", ",", "/"):
        if text[k] == "." and k > 0 and text[k - 1].isalpha():
            break  # abbreviation dot acts as segment boundary
        if text[k].isalpha():
            return False
        k -= 1
    # Next non-space character after slash must be a digit
    j = slash_idx + 1
    while j < len(text) and text[j] in (" ", "\t"):
        j += 1
    return j < len(text) and text[j].isdigit()
