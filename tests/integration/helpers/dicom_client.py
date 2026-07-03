"""Read a DICOM file, build the API request, and call the in-process service.

Mirrors what Karnak sends to /deidentify-image: the (possibly compressed) pixel
data of a single frame plus the metadata tags needed to decode it, and the list
of sensitive tag values to look for in the burned-in text.
"""

import json

import numpy as np
import pydicom
from pydicom.encaps import get_frame

# Tags Karnak forwards as the sensitive-data list.
SENSITIVE_TAGS = [
    "AccessionNumber",
    "InstitutionName",
    "OperatorsName",
    "PatientAge",
    "PatientBirthDate",
    "PatientID",
    "PatientName",
    "PatientSex",
    "PerformingPhysicianName",
    "PerformedProcedureStepID",
    "ReferringPhysicianName",
    "StudyDate",
]


def build_request(dcm_path: str) -> tuple[bytes, dict[str, str]]:
    """Extract the first-frame pixel bytes and form fields from a DICOM file."""
    ds = pydicom.dcmread(dcm_path)
    transfer_syntax = str(ds.file_meta.TransferSyntaxUID)
    number_of_frames = int(ds.get("NumberOfFrames") or 1)

    if pydicom.uid.UID(transfer_syntax).is_compressed:
        frame_bytes = get_frame(ds.PixelData, 0, number_of_frames=number_of_frames)
    else:
        frame_size = len(ds.PixelData) // number_of_frames
        frame_bytes = bytes(ds.PixelData[:frame_size])

    sensitive = {
        tag: str(ds.get(tag, "")) for tag in SENSITIVE_TAGS if str(ds.get(tag, ""))
    }

    data = {
        "sensitive_data_list": json.dumps(sensitive),
        "sop_instance_uid": str(ds.get("SOPInstanceUID", "")),
        "rows": str(ds.Rows),
        "columns": str(ds.Columns),
        "bits_allocated": str(ds.BitsAllocated),
        "samples_per_pixel": str(ds.SamplesPerPixel),
        "transfer_syntax_uid": transfer_syntax,
        "photometric_interpretation": str(ds.PhotometricInterpretation),
    }

    if not pydicom.uid.UID(transfer_syntax).is_compressed:
        data.update(
            {
                "rescale_slope": str(ds.get("RescaleSlope", "")),
                "rescale_intercept": str(ds.get("RescaleIntercept", "")),
                "window_center": str(ds.get("WindowCenter", "")),
                "window_width": str(ds.get("WindowWidth", "")),
            }
        )

    palette_json = build_palette_color_lut_json(ds)
    if palette_json is not None:
        data["palette_color_lut"] = palette_json

    return frame_bytes, data


def _extract_lut_data(ds, data_tag: str, descriptor_tag: str) -> list[int] | None:
    """Decode one channel's palette LUT into a list of ints.

    The descriptor is [numberOfEntries, firstStoredPixelValue, bitsPerEntry].
    8-bit entries are read as bytes, 16-bit entries as little-endian unsigned
    shorts (matching the common DICOM Little Endian encoding).
    """
    descriptor = ds.get(descriptor_tag)
    data = ds.get(data_tag)
    if descriptor is None or data is None:
        return None
    bits_per_entry = int(descriptor[2])
    raw = bytes(data)
    if bits_per_entry == 8:
        return list(raw)
    return np.frombuffer(raw, dtype="<u2").tolist()


def build_palette_color_lut_json(ds) -> str | None:
    """Build the palette_color_lut JSON string Karnak sends, or None.

    Returns a JSON object with "red", "green", "blue" integer arrays when the
    image is PALETTE COLOR and all LUT descriptors/data are present.
    """
    if str(ds.get("PhotometricInterpretation", "")) != "PALETTE COLOR":
        return None

    red = _extract_lut_data(
        ds, "RedPaletteColorLookupTableData", "RedPaletteColorLookupTableDescriptor"
    )
    green = _extract_lut_data(
        ds, "GreenPaletteColorLookupTableData", "GreenPaletteColorLookupTableDescriptor"
    )
    blue = _extract_lut_data(
        ds, "BluePaletteColorLookupTableData", "BluePaletteColorLookupTableDescriptor"
    )
    if red is None or green is None or blue is None:
        return None

    return json.dumps({"red": red, "green": green, "blue": blue})


def deidentify(client, dcm_path: str) -> dict:
    """POST a DICOM frame to the in-process API and return the parsed JSON."""
    frame_bytes, data = build_request(dcm_path)
    response = client.post(
        "/deidentify-image",
        files={"image": ("frame.bin", frame_bytes, "application/octet-stream")},
        data=data,
        headers={"Accept": "application/json; version=1"},
    )
    response.raise_for_status()
    return response.json()


def extract_rectangles(response: dict) -> list[str]:
    """Flatten the per-color mask groups into a single list of rectangles."""
    masks = response.get("masks") or []
    return [rect for group in masks for rect in group["rectangles"]]
