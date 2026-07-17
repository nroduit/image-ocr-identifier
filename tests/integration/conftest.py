"""Fixtures and options for the end-to-end DICOM dataset tests.

Run with the real OCR pipeline:

    pytest -m integration tests/integration

Regenerate golden files from the current pipeline output (review the diff
before committing):

    pytest -m integration tests/integration --update-golden
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from image_ocr_identifier.main import app

DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "integration_test"
GOLDEN_DIR = DATASET_DIR / "golden"
REPORTING_GOLDEN_DIR = DATASET_DIR / "golden_reporting"


def _is_dicom_file(path: Path) -> bool:
    """A file is DICOM if it ends in .dcm or has the DICM magic at offset 128."""
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(132)[128:132] == b"DICM"
    except OSError:
        return False


def discover_dicom_files() -> list[str]:
    """All DICOM files under DATASET_DIR (recursively), excluding the golden dir.

    Returns paths relative to DATASET_DIR, as strings, sorted for stable test ids.
    """
    files = [
        p
        for p in DATASET_DIR.rglob("*")
        if p.is_file()
        and GOLDEN_DIR not in p.parents
        and REPORTING_GOLDEN_DIR not in p.parents
        and not p.name.startswith(".")
        and _is_dicom_file(p)
    ]
    return sorted(str(p.relative_to(DATASET_DIR)) for p in files)


DICOM_FILES = discover_dicom_files()


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Write current pipeline output as golden files instead of asserting.",
    )


@pytest.fixture(scope="session")
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def update_golden(request) -> bool:
    return request.config.getoption("--update-golden")
