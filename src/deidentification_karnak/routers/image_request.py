"""Shared request parsing, image decoding, and API versioning for the routers.

Both ``/deidentify-image`` and ``/reporting`` accept the same multipart form
(an image plus DICOM metadata) and negotiate the API version the same way, so
that logic lives here once and is consumed via FastAPI dependencies.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass

import numpy as np
from fastapi import Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from deidentification_karnak.dicom_decode import decode_image_bytes

logger = logging.getLogger(__name__)

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


def versioned_response(data: BaseModel, version: int) -> JSONResponse:
    return JSONResponse(
        content=data.model_dump(by_alias=True, exclude_none=True),
        media_type=f"application/json; version={version}",
    )


@dataclass
class ImageRequest:
    image_bytes: bytes
    sensitive_data: dict[str, str]
    sop_instance_uid: str | None
    filename: str
    palette_color_lut: str | None
    decode_kwargs: dict


async def parse_image_request(
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
) -> ImageRequest:
    if image.content_type not in SUPPORTED_CONTENT_TYPES:
        logger.warning(
            "Rejected request: unsupported content type %r (filename=%r)",
            image.content_type,
            image.filename,
        )
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        sensitive_data = json.loads(sensitive_data_list)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "Rejected request: invalid JSON for sensitive_data_list (filename=%r): %s",
            image.filename,
            exc,
        )
        raise HTTPException(
            status_code=400, detail="Invalid JSON format for the sensitive data."
        ) from exc

    if not isinstance(sensitive_data, dict):
        logger.warning(
            "Rejected request: sensitive_data_list is not a JSON object (filename=%r, type=%s)",
            image.filename,
            type(sensitive_data).__name__,
        )
        raise HTTPException(
            status_code=400, detail="sensitive_data_list must be a JSON object."
        )

    image_bytes = await image.read()

    return ImageRequest(
        image_bytes=image_bytes,
        sensitive_data=sensitive_data,
        sop_instance_uid=sop_instance_uid,
        filename=image.filename or "image",
        palette_color_lut=palette_color_lut,
        decode_kwargs=dict(
            rows=rows,
            columns=columns,
            bits_allocated=bits_allocated,
            samples_per_pixel=samples_per_pixel,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
            window_center=window_center,
            window_width=window_width,
            is_monochrome1=is_monochrome1,
            transfer_syntax_uid=transfer_syntax_uid,
            photometric_interpretation=photometric_interpretation,
        ),
    )


async def decode_image(request: ImageRequest) -> np.ndarray | None:
    parsed_palette = None
    if request.palette_color_lut:
        try:
            parsed_palette = json.loads(request.palette_color_lut)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Rejected request: invalid JSON for palette_color_lut (filename=%r): %s",
                request.filename,
                exc,
            )
            raise HTTPException(
                status_code=400, detail="Invalid JSON format for palette_color_lut."
            ) from exc

    return await asyncio.to_thread(
        decode_image_bytes,
        request.image_bytes,
        palette_color_lut=parsed_palette,
        **request.decode_kwargs,
    )
