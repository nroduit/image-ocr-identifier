import os
from pathlib import Path
import numpy as np

from deidentification_karnak.debug import DebugSession, save_debug_ocr

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR

_BASE_DATA_PATH = Path(
    os.environ.get("MODEL_DATA_PATH", str(Path(__file__).parent.parent.parent / "data"))
)

SPACE_WEIGHT = 0.2

_det_model_name = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
_rec_model_name = os.environ.get("OCR_REC_MODEL", "latin_PP-OCRv5_mobile_rec")

_det_model_dir = os.environ.get(
    "OCR_DET_MODEL_DIR", str(_BASE_DATA_PATH / "models" / "detection" / _det_model_name)
)
_rec_model_dir = os.environ.get(
    "OCR_REC_MODEL_DIR",
    str(_BASE_DATA_PATH / "models" / "recognition" / _rec_model_name),
)

# It should fail during start if the model is absent
for _name, _dir in (("detection", _det_model_dir), ("recognition", _rec_model_dir)):
    if not Path(_dir).is_dir():
        raise RuntimeError(f"OCR {_name} model directory not found: {_dir}")

# Device selection: "cpu", "gpu", "gpu:0", ... Unset means the PaddleOCR default
# (CPU). Set OCR_DEVICE=gpu to run on an NVIDIA GPU
_device = os.environ.get("OCR_DEVICE") or None

# Initialize OCR
# enable_mkldnn=False: the oneDNN backend crashes under Paddle 3.x's PIR executor
# with PP-OCRv5 (NotImplementedError: ConvertPirAttribute2RuntimeAttribute) on
# x86_64 Linux. It is unused on macOS arm64 and on GPU, so disabling it is safe.
_ocr = PaddleOCR(
    use_textline_orientation=False,
    text_detection_model_name=_det_model_name,
    text_detection_model_dir=_det_model_dir,
    text_recognition_model_name=_rec_model_name,
    text_recognition_model_dir=_rec_model_dir,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_det_thresh=0.2,
    text_det_box_thresh=0.4,
    enable_mkldnn=False,
    device=_device,
)


def process_image_with_ocr(
    image: np.ndarray, debug_session: DebugSession | None = None
) -> dict[str, list]:
    result = _ocr.predict(image, return_word_box=False)
    save_debug_ocr(result, debug_session)
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
    groups = []

    for group_id, (text, box) in enumerate(zip(texts, boxes)):
        sub_blocks = _split_single_block(text, box)
        for sub_text, sub_box in sub_blocks:
            split_texts.append(sub_text)
            split_boxes.append(sub_box)
            groups.append(group_id)

    return {"texts": split_texts, "boxes": split_boxes, "groups": groups}


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

    box_width = x_max - x_min
    pixel_per_char = box_width / total_len
    # Padding: 90% of a character width on each side, minimum 3px
    padding = max(3, int(pixel_per_char * 0.9))

    results = []
    prev_end_px = None
    for part, start_idx, end_idx in parts_with_positions:
        start_px = x_min + _effective_pos(text, start_idx) * pixel_per_char
        end_px = x_min + _effective_pos(text, end_idx) * pixel_per_char
        # Pad the left edge by `padding`, but never past the previous token's
        # padded right edge. Anchoring on the previous token's end (the old
        # behaviour) overshot left for multi-character delimiters (e.g. " / "),
        # extending the box into the previous token.
        left_pad = padding
        if prev_end_px is not None:
            gap = start_px - (prev_end_px + padding)
            left_pad = max(0.0, min(padding, gap))
        sub_x_min = max(x_min, int(start_px - left_pad))
        sub_x_max = min(x_max, int(end_px) + padding)
        results.append((part, [sub_x_min, y_min, sub_x_max, y_max]))
        prev_end_px = end_px

    return results


def _effective_pos(text: str, idx: int, space_weight: float = SPACE_WEIGHT) -> float:
    return sum(space_weight if c == " " else 1.0 for c in text[:idx])


def _find_parts(text: str) -> list[tuple[str, int, int]]:
    """Find non-delimiter parts with their start and end character indices.

    Delimiters are: commas, non-date slashes, consecutive dots (e.g. "..."),
    and single dots between long numeric segments. Whitespace is NOT a delimiter
    so that text dates like "2021 Jan 26" or "26 janvier 2021" stay as one block.
    Slashes between numeric segments (e.g. "27/10/2020", "1928 / 03 / 03") are
    also preserved. Dots between short numeric segments (e.g. "27.10.2020") are
    preserved as date-like patterns.
    """
    parts = []
    i = 0
    n = len(text)
    while i < n:
        # Skip commas, slashes (at boundary level), and whitespace between tokens
        if text[i] in (",", " ", "\t"):
            i += 1
            continue
        # Slash at token boundary is a delimiter so we skip it
        if text[i] == "/":
            i += 1
            continue
        # Consecutive dots (2+) are delimiters (OCR truncation artifact)
        # but only at token boundaries after meaningful content
        if text[i] == "." and i + 1 < n and text[i + 1] == ".":
            while i < n and text[i] == ".":
                i += 1
            continue
        # Dot as field delimiter (preserved only for dates and alphanumeric tokens)
        if text[i] == "." and _is_dot_delimiter(text, i):
            i += 1
            continue
        # Start of a token
        start = i
        while i < n:
            if text[i] == ",":
                break
            # Consecutive dots break the token only if enough content precedes them
            if text[i] == "." and i + 1 < n and text[i + 1] == ".":
                alnum_before = sum(1 for c in text[start:i] if c.isalnum())
                if alnum_before >= 4:
                    break
            if text[i] == "/":
                # Preserve slash if it separates numeric segments (date-like)
                if _is_date_slash(text, i):
                    i += 1
                    # Skip spaces and stray repeated slashes (OCR artifact,
                    # e.g. "09/ /09") between the slash and the next digit
                    while i < n and text[i] in (" ", "\t", "/"):
                        i += 1
                    continue
                else:
                    break
            if text[i] == ".":
                if _is_dot_delimiter(text, i):
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
        if not text[k].isalnum() and text[k] != ".":
            break  # non-alphanumeric boundary (e.g. parenthesis)
        if text[k] == "." and k > 0 and text[k - 1].isalpha():
            break  # abbreviation dot acts as segment boundary
        if text[k].isalpha():
            return False
        k -= 1
    # Next significant character after slash must be a digit. Skip spaces and
    # stray repeated slashes (OCR artifact, e.g. "09/ /09/1947").
    j = slash_idx + 1
    while j < len(text) and text[j] in (" ", "\t", "/"):
        j += 1
    return j < len(text) and text[j].isdigit()


def _is_dot_delimiter(text: str, dot_idx: int) -> bool:
    """Return True if the dot at dot_idx should act as a field delimiter.

    A dot is preserved (not a delimiter) when:
    - Adjacent character on either side is not alphanumeric (e.g. "(..20")
    - It's inside an alphanumeric token (e.g. "IM1.3", "ITm1.0")
    - It's between short numeric segments (date-like, e.g. "27.10.2020")
    """
    # Dot is only a delimiter if both adjacent characters are alphanumeric
    if dot_idx == 0 or dot_idx >= len(text) - 1:
        return False
    if not text[dot_idx - 1].isalnum() or not text[dot_idx + 1].isalnum():
        return False

    # Acronym pattern: a single letter immediately before the dot, followed by
    # a letter (e.g. "H.U.G") -> preserve so initials stay joined as one token.
    if text[dot_idx - 1].isalpha() and text[dot_idx + 1].isalpha():
        prev = dot_idx - 2
        if prev < 0 or text[prev] in (" ", "\t", ",", ".", "/"):
            return False

    # Count consecutive digits immediately before the dot
    j = dot_idx - 1
    while j >= 0 and text[j].isdigit():
        j -= 1
    digits_before = dot_idx - 1 - j

    # Count consecutive digits immediately after the dot
    k = dot_idx + 1
    while k < len(text) and text[k].isdigit():
        k += 1
    digits_after = k - dot_idx - 1

    # If both sides have digits adjacent to the dot
    if digits_before > 0 and digits_after > 0:
        # Inside an alphanumeric token (letter before the digit run) -> preserve
        if j >= 0 and text[j].isalpha() and digits_before <= 7 and digits_after <= 7:
            return False
        # Both digit runs are short -> date-like pattern -> preserve
        if digits_before <= 4 and digits_after <= 4:
            return False
        # Long numeric segments -> delimiter
        return True

    # Letter directly before the dot followed by a short digit run -> date-like
    # (e.g. "Nov.08", "Jan.2019"). Long digit runs (IDs) still split.
    if digits_before == 0 and 0 < digits_after <= 4:
        return False

    # For the reverse case : (e.g. "08.Nov")
    if digits_after == 0 and 0 < digits_before <= 4:
        return False

    # Alphabetic or mixed segments on both sides (e.g. "JEAN.ALBERT") -> delimiter
    return True
