import asyncio
import json
import re

from fastapi import APIRouter, Form, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from deidentification_karnak.color_detection import get_colors
from deidentification_karnak.dicom_decode import decode_image_bytes
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
from deidentification_karnak.models.response import (
    DeidentificationResponse,
    MaskGroup,
)
from deidentification_karnak.preprocessing import preprocess_image_for_ocr
from deidentification_karnak.sensitive_data_detection import detect_sensitive_data
from deidentification_karnak.utils import (
    bgr_to_hex,
    convert_upscaled_boxes,
    expand_boxes,
    format_boxes,
)

SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jp2",
    "image/jpeg2000",
    "image/x-raw",
    "image/raw",
    "application/octet-stream",
}

SUPPORTED_VERSIONS = {1}
DEFAULT_VERSION = 1

_VERSION_RE = re.compile(r"version\s*=\s*(\d+)")

router = APIRouter()


def version_dep(request: Request) -> int:
    """Extract API version from the Accept header.

    Expects ``application/json; version=N``.  Falls back to the default
    version when the parameter is absent.
    """
    accept = request.headers.get("accept", "")
    match = _VERSION_RE.search(accept)
    if match:
        version = int(match.group(1))
        if version not in SUPPORTED_VERSIONS:
            raise HTTPException(
                status_code=406,
                detail=f"Unsupported API version {version}. Supported: {sorted(SUPPORTED_VERSIONS)}",
            )
        return version
    return DEFAULT_VERSION


def _versioned_response(data: DeidentificationResponse, version: int) -> JSONResponse:
    return JSONResponse(
        content=data.model_dump(by_alias=True, exclude_none=True),
        media_type=f"application/json; version={version}",
    )


@router.post("/deidentify-image", response_model=DeidentificationResponse)
async def deidentify_image(
    image: UploadFile,
    sensitive_data_list: str = Form(
        ...,
        description='JSON object mapping DICOM tag names to their string values, e.g. {"PatientID": "12345"}',
    ),
    sop_instance_uid: str | None = Form(None, description="DICOM SOP Instance UID"),
    rows: int | None = Form(
        None, description="Image height in pixels (raw pixel data only)"
    ),
    columns: int | None = Form(
        None, description="Image width in pixels (raw pixel data only)"
    ),
    bits_allocated: int | None = Form(
        None, description="Bits per pixel component (raw pixel data only)"
    ),
    samples_per_pixel: int | None = Form(
        None, description="Number of channels (raw pixel data only)"
    ),
    rescale_slope: float | None = Form(
        None, description="Modality LUT rescale slope (raw pixel data only)"
    ),
    rescale_intercept: float | None = Form(
        None, description="Modality LUT rescale intercept (raw pixel data only)"
    ),
    window_center: float | None = Form(
        None, description="VOI LUT window center (raw pixel data only)"
    ),
    window_width: float | None = Form(
        None, description="VOI LUT window width (raw pixel data only)"
    ),
    is_monochrome1: bool = Form(
        False,
        description="True if photometric interpretation is MONOCHROME1 (raw pixel data only)",
    ),
    palette_color_lut: str | None = Form(
        None,
        description='JSON object with "red", "green", "blue" arrays for palette color LUT (raw pixel data only)',
    ),
    transfer_syntax_uid: str | None = Form(
        None,
        description="DICOM Transfer Syntax UID for compressed pixel data (e.g. 1.2.840.10008.1.2.4.70)",
    ),
    photometric_interpretation: str | None = Form(
        None,
        description="DICOM Photometric Interpretation (e.g. MONOCHROME1, MONOCHROME2, RGB, YBR_FULL_422)",
    ),
    version: int = Depends(version_dep),
):
    if image.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        sensitive_data: dict[str, str] = json.loads(sensitive_data_list)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format for the sensitive data."
        ) from exc

    if not isinstance(sensitive_data, dict):
        raise HTTPException(
            status_code=400, detail="sensitive_data_list must be a JSON object."
        )

    no_sensitive = DeidentificationResponse(
        message="No sensitive data list provided", sop_instance_uid=sop_instance_uid
    )

    if not sensitive_data:
        return _versioned_response(no_sensitive, version)

    image_bytes = await image.read()

    parsed_palette = None
    if palette_color_lut:
        try:
            parsed_palette = json.loads(palette_color_lut)
        except (json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="Invalid JSON format for palette_color_lut."
            ) from exc

    decoded_image = await asyncio.to_thread(
        decode_image_bytes,
        image_bytes,
        rows=rows,
        columns=columns,
        bits_allocated=bits_allocated,
        samples_per_pixel=samples_per_pixel,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        window_center=window_center,
        window_width=window_width,
        is_monochrome1=is_monochrome1,
        palette_color_lut=parsed_palette,
        transfer_syntax_uid=transfer_syntax_uid,
        photometric_interpretation=photometric_interpretation,
    )
    if decoded_image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Provide rows, columns, bits_allocated, samples_per_pixel, transfer_syntax_uid, and photometric_interpretation.",
        )

    # Preprocessing
    preprocessed_image, scale_factor = preprocess_image_for_ocr(decoded_image)
    debug_session = create_debug_session(image.filename or "image")
    save_debug_preprocessed(preprocessed_image, debug_session)

    # OCR and sensitive data detection
    ocr_result = await asyncio.to_thread(
        process_image_with_ocr,
        preprocessed_image,
        debug_session=debug_session,
    )

    if not ocr_result["texts"]:
        return _versioned_response(no_sensitive, version)

    ocr_result = split_ocr_blocks(ocr_result)
    save_debug_split_boxes(preprocessed_image, ocr_result, debug_session)

    ocr_result["boxes"] = convert_upscaled_boxes(ocr_result["boxes"], scale_factor)

    masks = await asyncio.to_thread(detect_sensitive_data, ocr_result, sensitive_data)

    # Expand boxes a little bit to cover text border pixels
    masks["boxes"] = expand_boxes(masks["boxes"], margin=2)

    color_to_boxes = await asyncio.to_thread(get_colors, decoded_image, masks["boxes"])

    save_debug_image(decoded_image, color_to_boxes, debug_session)

    mask_groups = [
        MaskGroup(
            color=bgr_to_hex(color),
            rectangles=format_boxes(boxes),
        )
        for color, boxes in color_to_boxes.items()
    ]

    total = sum(len(boxes) for boxes in color_to_boxes.values())

    result = DeidentificationResponse(
        masks=mask_groups if mask_groups else None,
        message=(
            f"{total} sensitive data detected"
            if mask_groups
            else "No sensitive data detected"
        ),
        sop_instance_uid=sop_instance_uid,
    )
    return _versioned_response(result, version)
