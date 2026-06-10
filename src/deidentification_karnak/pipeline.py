import numpy as np

from deidentification_karnak.color_detection import get_colors
from deidentification_karnak.debug import (
    create_debug_session,
    save_debug_image,
    save_debug_preprocessed,
    save_debug_split_boxes,
)
from deidentification_karnak.image_processing import (
    process_image_with_ocr,
    split_ocr_blocks,
)
from deidentification_karnak.preprocessing import preprocess_image_for_ocr
from deidentification_karnak.models.response import DeidentificationResponse, MaskGroup
from deidentification_karnak.sensitive_data_detection import detect_sensitive_data
from deidentification_karnak.utils import (
    bgr_to_hex,
    convert_upscaled_boxes,
    expand_boxes,
    format_boxes,
)


def run_deidentification(
    decoded_image: np.ndarray,
    sensitive_data: dict[str, str],
    sop_instance_uid: str | None,
    image_name: str,
) -> DeidentificationResponse:

    # Preprocessing
    preprocessed_image, scale_factor = preprocess_image_for_ocr(decoded_image)
    debug_session = create_debug_session(image_name)
    save_debug_preprocessed(preprocessed_image, debug_session)

    # OCR and sensitive data detection
    ocr_result = process_image_with_ocr(
        preprocessed_image,
        debug_session=debug_session,
    )

    if not ocr_result["texts"]:
        return DeidentificationResponse(
            message="No sensitive data detected", sop_instance_uid=sop_instance_uid
        )

    ocr_result = split_ocr_blocks(ocr_result)
    save_debug_split_boxes(preprocessed_image, ocr_result, debug_session)
    ocr_result["boxes"] = convert_upscaled_boxes(ocr_result["boxes"], scale_factor)

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
