"""Call the /reporting endpoint with a DICOM frame and read back the tags."""

from . import dicom_client


def report(client, dcm_path: str) -> dict:
    """POST a DICOM frame to the in-process /reporting API and return the JSON."""
    frame_bytes, data = dicom_client.build_request(dcm_path)
    response = client.post(
        "/reporting",
        files={"image": ("frame.bin", frame_bytes, "application/octet-stream")},
        data=data,
        headers={"Accept": "application/json; version=1"},
    )
    response.raise_for_status()
    return response.json()


def extract_tags(response: dict) -> list[str]:
    """Return the detected DICOM tag names from a reporting response."""
    return list(response.get("detected_tags") or [])
