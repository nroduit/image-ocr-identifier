import json
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from image_ocr_identifier.main import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)
_JPEG_CONTENT_TYPE = "image/jpeg"
_VALID_SENSITIVE_DATA = json.dumps({"PatientName": "Smith^John"})

# Patch targets. decode_image_bytes is bound in the shared image_request module;
# the OCR, color and debug functions are bound in the pipeline module after the
# service extraction.
_PATCH_DECODE = "image_ocr_identifier.routers.image_request.decode_image_bytes"
_PATCH_OCR = "image_ocr_identifier.pipeline.process_image_with_ocr"
_PATCH_COLORS = "image_ocr_identifier.pipeline.get_colors"
_PATCH_DEBUG = "image_ocr_identifier.pipeline.save_debug_image"


def _post(
    sensitive_data_list=_VALID_SENSITIVE_DATA,
    content_type=_JPEG_CONTENT_TYPE,
    accept="application/json; version=1",
    extra_data=None,
    filename="test.jpg",
):
    data = {"sensitive_data_list": sensitive_data_list}
    if extra_data:
        data.update(extra_data)
    return client.post(
        "/deidentify-image",
        files={"image": (filename, b"fake", content_type)},
        data=data,
        headers={"Accept": accept},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Input validation -- 400 errors
# ---------------------------------------------------------------------------


def test_unsupported_content_type_returns_400():
    response = _post(content_type="text/plain")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_invalid_json_in_sensitive_data_list_returns_400():
    response = _post(sensitive_data_list="not json")
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_sensitive_data_list_not_a_dict_returns_400():
    response = _post(sensitive_data_list='["a", "b"]')
    assert response.status_code == 400
    assert "JSON object" in response.json()["detail"]


def test_invalid_palette_color_lut_returns_400():
    with patch(_PATCH_DECODE, return_value=_FAKE_IMAGE), patch(_PATCH_DEBUG):
        response = _post(extra_data={"palette_color_lut": "not-json"})
    assert response.status_code == 400
    assert "palette_color_lut" in response.json()["detail"]


def test_image_decode_failure_returns_400():
    with patch(_PATCH_DECODE, return_value=None):
        response = _post()
    assert response.status_code == 400
    assert "Failed to decode image" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Version negotiation -- 406
# ---------------------------------------------------------------------------


def test_unsupported_version_returns_406():
    response = _post(accept="application/json; version=99")
    assert response.status_code == 406
    assert "Unsupported API version" in response.json()["detail"]


def test_no_version_header_uses_default():
    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value={"texts": [], "boxes": []}),
        patch(_PATCH_DEBUG),
    ):
        response = _post(accept="application/json")
    assert response.status_code == 200
    assert "version=1" in response.headers["content-type"]


def test_explicit_version_1_accepted():
    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value={"texts": [], "boxes": []}),
        patch(_PATCH_DEBUG),
    ):
        response = _post(accept="application/json; version=1")
    assert response.status_code == 200
    assert "version=1" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Early returns (200)
# ---------------------------------------------------------------------------


def test_empty_sensitive_data_list_returns_early():
    response = _post(sensitive_data_list="{}")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "No sensitive data list provided"
    assert body.get("masks") is None


def test_sop_instance_uid_propagated_in_early_return():
    response = _post(
        sensitive_data_list="{}",
        extra_data={"sop_instance_uid": "1.2.3.4"},
    )
    assert response.status_code == 200
    assert response.json()["sop_instance_uid"] == "1.2.3.4"


def test_no_ocr_texts_returns_early():
    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value={"texts": [], "boxes": []}),
        patch(_PATCH_DEBUG),
    ):
        response = _post()
    assert response.status_code == 200
    body = response.json()
    assert body.get("masks") is None


# ---------------------------------------------------------------------------
# Happy path -- sensitive data detected
# ---------------------------------------------------------------------------


def test_sensitive_data_found_returns_masks():
    # "john" will fuzzy-match the PatientName "Smith^John" lookup entry "john"
    ocr = {"texts": ["john"], "boxes": [[2, 2, 8, 8]]}
    colors = {(255, 255, 255): [[1, 1, 9, 9]]}

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
        patch(_PATCH_COLORS, return_value=colors),
        patch(_PATCH_DEBUG),
    ):
        response = _post()

    assert response.status_code == 200
    body = response.json()
    assert body["masks"] is not None
    assert len(body["masks"]) == 1
    mask = body["masks"][0]
    assert mask["color"] == "ffffff"
    assert mask["rectangles"] == ["1 1 8 8"]
    assert "1 sensitive data detected" in body["message"]


def test_sensitive_data_found_includes_sop_instance_uid():
    ocr = {"texts": ["john"], "boxes": [[2, 2, 8, 8]]}
    colors = {(255, 255, 255): [[1, 1, 9, 9]]}

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
        patch(_PATCH_COLORS, return_value=colors),
        patch(_PATCH_DEBUG),
    ):
        response = _post(extra_data={"sop_instance_uid": "1.2.3"})

    assert response.json()["sop_instance_uid"] == "1.2.3"


def test_multiple_boxes_same_color_in_one_mask_group():
    ocr = {"texts": ["john", "smith"], "boxes": [[0, 0, 4, 4], [5, 5, 9, 9]]}
    colors = {(0, 0, 0): [[0, 0, 4, 4], [5, 5, 9, 9]]}

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
        patch(_PATCH_COLORS, return_value=colors),
        patch(_PATCH_DEBUG),
    ):
        response = _post()

    assert response.status_code == 200
    body = response.json()
    assert len(body["masks"]) == 1
    assert len(body["masks"][0]["rectangles"]) == 2


def test_multiple_boxes_different_colors_produce_multiple_mask_groups():
    ocr = {"texts": ["john", "smith"], "boxes": [[0, 0, 4, 4], [5, 5, 9, 9]]}
    colors = {(255, 255, 255): [[0, 0, 4, 4]], (0, 0, 0): [[5, 5, 9, 9]]}

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
        patch(_PATCH_COLORS, return_value=colors),
        patch(_PATCH_DEBUG),
    ):
        response = _post()

    assert response.status_code == 200
    assert len(response.json()["masks"]) == 2


# ---------------------------------------------------------------------------
# No sensitive data detected (OCR found text but no match)
# ---------------------------------------------------------------------------


def test_no_sensitive_data_detected_returns_message():
    # "zzzzzzzzz" won't match anything from PatientName "Smith^John"
    ocr = {"texts": ["zzzzzzzzz"], "boxes": [[0, 0, 5, 5]]}

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
        patch(_PATCH_DEBUG),
    ):
        response = _post()

    assert response.status_code == 200
    body = response.json()
    assert "mask" not in body
    assert body["message"] == "No sensitive data detected"


# ---------------------------------------------------------------------------
# Reporting endpoint
# ---------------------------------------------------------------------------


def _post_reporting(
    sensitive_data_list=_VALID_SENSITIVE_DATA,
    content_type=_JPEG_CONTENT_TYPE,
    accept="application/json; version=1",
    extra_data=None,
    filename="test.jpg",
):
    data = {"sensitive_data_list": sensitive_data_list}
    if extra_data:
        data.update(extra_data)
    return client.post(
        "/reporting",
        files={"image": (filename, b"fake", content_type)},
        data=data,
        headers={"Accept": accept},
    )


def test_reporting_empty_sensitive_data_list_returns_early():
    response = _post_reporting(sensitive_data_list="{}")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "No sensitive data list provided"
    assert body["detected_tags"] == []


def test_reporting_unsupported_content_type_returns_400():
    response = _post_reporting(content_type="text/plain")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_reporting_image_decode_failure_returns_400():
    with patch(_PATCH_DECODE, return_value=None):
        response = _post_reporting()
    assert response.status_code == 400
    assert "Failed to decode image" in response.json()["detail"]


def test_reporting_no_ocr_texts_returns_empty_tags():
    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value={"texts": [], "boxes": []}),
    ):
        response = _post_reporting()
    assert response.status_code == 200
    body = response.json()
    assert body["detected_tags"] == []
    assert body["message"] == "No sensitive data detected"


def test_reporting_returns_detected_tags():
    sensitive = json.dumps(
        {
            "PatientName": "Smith^John",
            "PatientBirthDate": "19640101",
            "PatientID": "12345",
            "AccessionNumber": "ZZZ999",
        }
    )
    # OCR finds the name, the birth date, and the patient id, but not the
    # accession number, so only those three tags are reported.
    ocr = {
        "texts": ["john", "01.01.1964", "12345"],
        "boxes": [[0, 0, 4, 4], [0, 5, 9, 9], [0, 10, 9, 14]],
    }

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
    ):
        response = _post_reporting(sensitive_data_list=sensitive)

    assert response.status_code == 200
    body = response.json()
    assert body["detected_tags"] == ["PatientName", "PatientBirthDate", "PatientID"]
    assert "3 sensitive tags detected" in body["message"]


def test_reporting_propagates_sop_instance_uid():
    ocr = {"texts": ["john"], "boxes": [[2, 2, 8, 8]]}

    with (
        patch(_PATCH_DECODE, return_value=_FAKE_IMAGE),
        patch(_PATCH_OCR, return_value=ocr),
    ):
        response = _post_reporting(extra_data={"sop_instance_uid": "1.2.3"})

    assert response.status_code == 200
    body = response.json()
    assert body["sop_instance_uid"] == "1.2.3"
    assert body["detected_tags"] == ["PatientName"]
