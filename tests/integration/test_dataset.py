"""End-to-end test: process each DICOM through the real OCR pipeline and compare
the masked regions against a per-file golden answer.

Goldens are tolerant: a golden region must be covered by the union of the
detected masks with a coverage fraction >= THRESHOLD, so one mask may cover
several golden regions and several masks may jointly cover one. Missing a golden
region (false negative) fails the test; extra masking (false positive) is
reported as a warning only.
"""

import json

import pytest

from .conftest import DATASET_DIR, DICOM_FILES, GOLDEN_DIR
from .helpers import compare, dicom_client

pytestmark = pytest.mark.integration

COVERAGE_THRESHOLD = 0.9


def _golden_path(rel_name: str):
    """Map a dataset-relative DICOM path to its golden file (flat, unique)."""
    stem = rel_name.rsplit(".", 1)[0] if rel_name.endswith(".dcm") else rel_name
    safe = stem.replace("/", "__").replace("\\", "__")
    return GOLDEN_DIR / f"{safe}.json"


@pytest.mark.parametrize("dcm_name", DICOM_FILES)
def test_dicom_masks_match_golden(client, update_golden, dcm_name):
    dcm_path = DATASET_DIR / dcm_name
    assert dcm_path.exists(), f"Missing DICOM file: {dcm_path}"

    response = dicom_client.deidentify(client, str(dcm_path))
    detected = dicom_client.extract_rectangles(response)

    golden_path = _golden_path(dcm_name)

    if update_golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        if not golden_path.exists():
            golden_path.write_text(
                json.dumps(
                    {
                        "sop_instance_uid": response.get("sop_instance_uid"),
                        "rectangles": detected,
                    },
                    indent=2,
                )
            )
            pytest.skip(f"Golden generated: {golden_path.name}")
        else:
            pytest.skip(f"Golden already exists: {golden_path.name}")

    assert golden_path.exists(), (
        f"No golden for {dcm_name}. Generate it with --update-golden."
    )
    golden = json.loads(golden_path.read_text())["rectangles"]

    result = compare.compare(golden, detected, threshold=COVERAGE_THRESHOLD)

    if result.false_positives:
        print(
            f"\n{dcm_name}: {len(result.false_positives)} extra masks (FP): "
            f"{result.false_positives}"
        )

    assert result.ok, (
        f"{dcm_name}: {len(result.false_negatives)} golden region(s) not masked "
        f"(coverage < {COVERAGE_THRESHOLD}):\n"
        + "\n".join(
            f"  {rect} coverage={cov:.2%}" for rect, cov in result.false_negatives
        )
    )
