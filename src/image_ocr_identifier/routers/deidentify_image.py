import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from image_ocr_identifier.models.response import DeidentificationResponse
from image_ocr_identifier.pipeline import run_deidentification
from image_ocr_identifier.routers.image_request import (
    ImageRequest,
    decode_image,
    parse_image_request,
    version_dep,
    versioned_response,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/deidentify-image", response_model=DeidentificationResponse)
async def deidentify_image(
    request: ImageRequest = Depends(parse_image_request),
    version: int = Depends(version_dep),
):
    if not request.sensitive_data:
        no_sensitive = DeidentificationResponse(
            message="No sensitive data list provided",
            sop_instance_uid=request.sop_instance_uid,
        )
        return versioned_response(no_sensitive, version)

    decoded_image = await decode_image(request)
    if decoded_image is None:
        logger.warning(
            "Rejected request: failed to decode image (filename=%r, decode_kwargs=%r)",
            request.filename,
            request.decode_kwargs,
        )
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Provide rows, columns, bits_allocated, samples_per_pixel, transfer_syntax_uid, and photometric_interpretation.",
        )

    result = await asyncio.to_thread(
        run_deidentification,
        decoded_image,
        request.sensitive_data,
        request.sop_instance_uid,
        request.filename,
    )
    return versioned_response(result, version)
