import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from deidentification_karnak.models.response import ReportingResponse
from deidentification_karnak.pipeline import run_reporting
from deidentification_karnak.routers.image_request import (
    ImageRequest,
    decode_image,
    parse_image_request,
    version_dep,
    versioned_response,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reporting", response_model=ReportingResponse)
async def reporting(
    request: ImageRequest = Depends(parse_image_request),
    version: int = Depends(version_dep),
):
    if not request.sensitive_data:
        no_sensitive = ReportingResponse(
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
        run_reporting,
        decoded_image,
        request.sensitive_data,
        request.sop_instance_uid,
        request.filename,
    )
    return versioned_response(result, version)
