"""End-to-end test: run each DICOM through the real /reporting pipeline and
compare the reported sensitive tags against a per-file golden set.

Run with the real OCR pipeline:

    pytest -m integration tests/integration/test_reporting_dataset.py

Regenerate the reporting goldens from the current pipeline output:

    pytest -m integration tests/integration/test_reporting_dataset.py --update-golden
"""

import json

import pytest

from .conftest import DATASET_DIR, DICOM_FILES, REPORTING_GOLDEN_DIR
from .helpers import reporting_client, tag_compare

pytestmark = pytest.mark.integration


def _golden_path(rel_name: str):
    """Map a dataset-relative DICOM path to its golden file (flat, unique)."""
    stem = rel_name.rsplit(".", 1)[0] if rel_name.endswith(".dcm") else rel_name
    safe = stem.replace("/", "__").replace("\\", "__")
    return REPORTING_GOLDEN_DIR / f"{safe}.json"


@pytest.mark.parametrize("dcm_name", DICOM_FILES)
def test_dicom_tags_match_golden(client, update_golden, dcm_name):
    dcm_path = DATASET_DIR / dcm_name
    assert dcm_path.exists(), f"Missing DICOM file: {dcm_path}"

    response = reporting_client.report(client, str(dcm_path))
    detected = reporting_client.extract_tags(response)

    golden_path = _golden_path(dcm_name)

    if update_golden:
        REPORTING_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        if not golden_path.exists():
            golden_path.write_text(
                json.dumps(
                    {
                        "sop_instance_uid": response.get("sop_instance_uid"),
                        "detected_tags": detected,
                    },
                    indent=2,
                )
            )
            pytest.skip(f"Golden generated: {golden_path.name}")
        else:
            pytest.skip(f"Golden already exists: {golden_path.name}")

    assert (
        golden_path.exists()
    ), f"No golden for {dcm_name}. Generate it with --update-golden."
    golden = json.loads(golden_path.read_text())["detected_tags"]

    result = tag_compare.compare(golden, detected)

    if result.false_positives:
        print(f"\n{dcm_name}: extra tags reported (FP): {result.false_positives}")

    assert result.ok, (
        f"{dcm_name}: {len(result.false_negatives)} golden tag(s) no longer "
        f"reported (false negatives):\n  {result.false_negatives}"
    )
