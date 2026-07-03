import numpy as np

from image_ocr_identifier.color_detection import get_colors
from image_ocr_identifier.debug import (
    create_debug_session,
    save_debug_image,
    save_debug_preprocessed,
    save_debug_split_boxes,
)
from image_ocr_identifier.image_processing import (
    process_image_with_ocr,
    split_ocr_blocks,
)
from image_ocr_identifier.preprocessing import preprocess_image_for_ocr
from image_ocr_identifier.models.response import (
    DeidentificationResponse,
    MaskGroup,
    ReportingResponse,
)
from image_ocr_identifier.sensitive_data_detection import (
    detect_sensitive_data,
    detect_sensitive_keys,
)
from image_ocr_identifier.utils import (
    bgr_to_hex,
    convert_upscaled_boxes,
    expand_boxes,
    format_boxes,
)


def _run_ocr_pipeline(decoded_image: np.ndarray, image_name: str):
    """Preprocess, OCR, and split the image into line boxes.

    Returns ``(ocr_result, debug_session)`` with boxes rescaled back to the
    original image, or ``(None, debug_session)`` when OCR finds no text.
    """
    preprocessed_image, scale_factor = preprocess_image_for_ocr(decoded_image)
    debug_session = create_debug_session(image_name)
    save_debug_preprocessed(preprocessed_image, debug_session)

    ocr_result = process_image_with_ocr(
        preprocessed_image,
        debug_session=debug_session,
    )

    if not ocr_result["texts"]:
        return None, debug_session

    ocr_result = split_ocr_blocks(ocr_result)
    save_debug_split_boxes(preprocessed_image, ocr_result, debug_session)
    ocr_result["boxes"] = convert_upscaled_boxes(ocr_result["boxes"], scale_factor)
    return ocr_result, debug_session


def run_deidentification(
    decoded_image: np.ndarray,
    sensitive_data: dict[str, str],
    sop_instance_uid: str | None,
    image_name: str,
) -> DeidentificationResponse:

    ocr_result, debug_session = _run_ocr_pipeline(decoded_image, image_name)
    if ocr_result is None:
        return DeidentificationResponse(
            message="No sensitive data detected", sop_instance_uid=sop_instance_uid
        )

    masks = detect_sensitive_data(ocr_result, sensitive_data)
    # Expand boxes a little bit to cover text border pixels
    masks["boxes"] = expand_boxes(masks["boxes"], margin=2)

    color_to_boxes = get_colors(decoded_image, masks["boxes"])
    save_debug_image(decoded_image, color_to_boxes, debug_session)

    mask_groups = [
        MaskGroup(
            color=bgr_to_hex(color),
            rectangles=format_boxes(boxes),
        )
        for color, boxes in color_to_boxes.items()
    ]
    total = sum(len(boxes) for boxes in color_to_boxes.values())

    return DeidentificationResponse(
        masks=mask_groups if mask_groups else None,
        message=(
            f"{total} sensitive data detected"
            if mask_groups
            else "No sensitive data detected"
        ),
        sop_instance_uid=sop_instance_uid,
    )


def run_reporting(
    decoded_image: np.ndarray,
    sensitive_data: dict[str, str],
    sop_instance_uid: str | None,
    image_name: str,
) -> ReportingResponse:

    ocr_result, _ = _run_ocr_pipeline(decoded_image, image_name)
    if ocr_result is None:
        return ReportingResponse(
            message="No sensitive data detected", sop_instance_uid=sop_instance_uid
        )

    detected_tags = detect_sensitive_keys(ocr_result, sensitive_data)

    return ReportingResponse(
        detected_tags=detected_tags,
        message=(
            f"{len(detected_tags)} sensitive tags detected"
            if detected_tags
            else "No sensitive data detected"
        ),
        sop_instance_uid=sop_instance_uid,
    )
