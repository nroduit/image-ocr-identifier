#!/usr/bin/env python3
"""Test specific DICOM files from the integration dataset and show results.

Usage:
    # Single file (path relative to data/integration_test/)
    poetry run python scripts/test_single.py "25.09.2024/1.2.840.113619.2.391.20423.1727257700.4920.60.512"

    # Multiple files
    poetry run python scripts/test_single.py "25.09.2024/1.2.840.113619.2.391.20423.1727257700.4920.60.512" "16.09.2024/4018fe95.dcm"

    # With debug images saved to disk
    DEBUG_IMAGES=1 poetry run python scripts/test_single.py "16.09.2024/4018fe95.dcm"

    # Grep pattern (runs all files whose path contains the substring)
    poetry run python scripts/test_single.py --grep "25.09.2024"

    # Update golden for specific files
    poetry run python scripts/test_single.py --update-golden "16.09.2024/4018fe95.dcm"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from image_ocr_identifier.main import app
from tests.integration.conftest import DATASET_DIR, GOLDEN_DIR
from tests.integration.helpers import compare, dicom_client

COVERAGE_THRESHOLD = 0.9


def golden_path(rel_name: str) -> Path:
    stem = rel_name.rsplit(".", 1)[0] if rel_name.endswith(".dcm") else rel_name
    safe = stem.replace("/", "__").replace("\\", "__")
    return GOLDEN_DIR / f"{safe}.json"


def resolve_files(patterns: list[str], grep: bool) -> list[str]:
    """Resolve patterns to actual dataset-relative DICOM paths."""
    if not grep:
        return patterns

    all_files = [
        str(p.relative_to(DATASET_DIR))
        for p in DATASET_DIR.rglob("*")
        if p.is_file() and GOLDEN_DIR not in p.parents and not p.name.startswith(".")
    ]
    matched = []
    for pattern in patterns:
        matched.extend(f for f in all_files if pattern in f)
    return sorted(set(matched))


def run(files: list[str], update_golden: bool) -> int:
    client = TestClient(app, raise_server_exceptions=False)
    failures = 0

    for dcm_name in files:
        dcm_path = DATASET_DIR / dcm_name
        if not dcm_path.exists():
            print(f"  MISSING: {dcm_path}")
            failures += 1
            continue

        print(f"\n{'=' * 70}")
        print(f"  {dcm_name}")
        print(f"{'=' * 70}")

        response = dicom_client.deidentify(client, str(dcm_path))
        detected = dicom_client.extract_rectangles(response)

        gp = golden_path(dcm_name)

        if update_golden:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            gp.write_text(
                json.dumps(
                    {
                        "sop_instance_uid": response.get("sop_instance_uid"),
                        "rectangles": detected,
                    },
                    indent=2,
                )
            )
            print(f"  GOLDEN WRITTEN: {gp.name}")
            continue

        print(f"  Detected masks: {len(detected)}")
        for r in detected:
            print(f"    {r}")

        if not gp.exists():
            print("  NO GOLDEN (run with --update-golden to create)")
            continue

        golden = json.loads(gp.read_text())["rectangles"]
        print(f"  Golden regions: {len(golden)}")

        result = compare.compare(golden, detected, threshold=COVERAGE_THRESHOLD)

        if result.matched:
            print(f"  MATCHED: {len(result.matched)}")
        if result.false_negatives:
            print(f"  FALSE NEGATIVES (missed): {len(result.false_negatives)}")
            for rect, cov in result.false_negatives:
                print(f"    {rect} coverage={cov:.2%}")
            failures += 1
        if result.false_positives:
            print(f"  FALSE POSITIVES (extra): {len(result.false_positives)}")
            for fp in result.false_positives:
                print(f"    {fp}")

        if result.ok:
            print("  PASS")
        else:
            print("  FAIL")

    print(f"\n{'=' * 70}")
    print(f"  {len(files)} file(s), {failures} failure(s)")
    print(f"{'=' * 70}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(
        description="Test specific DICOM files from the integration dataset."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="DICOM paths relative to data/integration_test/, or substrings with --grep",
    )
    parser.add_argument(
        "--grep",
        action="store_true",
        help="Treat arguments as substrings and match all files containing them",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Overwrite golden files with current pipeline output",
    )
    args = parser.parse_args()

    files = resolve_files(args.files, args.grep)
    if not files:
        print("No matching files found.")
        return 1

    print(f"Testing {len(files)} file(s)...")
    return run(files, args.update_golden)


if __name__ == "__main__":
    sys.exit(main())
