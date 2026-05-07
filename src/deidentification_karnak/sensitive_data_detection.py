from datetime import datetime
import logging
import re
from typing import Dict, Union
import unicodedata
import numpy as np
from rapidfuzz import fuzz

from deidentification_karnak.utils import numpy_to_python_type

logger = logging.getLogger(__name__)


# Compare OCR results with sensitive_data_list and return only the data that needs to be masked
def detect_sensitive_data(
    ocr_result: Dict[str, Union[list, np.ndarray]], sensitive_data_list: Dict[str, str]
) -> Dict[str, list]:
    ocr_texts = ocr_result["texts"]
    ocr_boxes = ocr_result["boxes"]

    # Build lookup set from sensitive data
    lookup = build_sensitive_lookup(sensitive_data_list)

    filtered_texts = []
    filtered_boxes = []

    for text, box in zip(ocr_texts, ocr_boxes):
        if is_sensitive(text, lookup) or token_sensitive(text, lookup):
            filtered_texts.append(text)
            filtered_boxes.append(numpy_to_python_type(box))
            logger.debug("Sensitive data detected: '%s' at box %s", text, box)

    return {
        "texts": filtered_texts,
        "boxes": filtered_boxes,
    }


def is_sensitive(
    ocr_text: str,
    lookup: set[str],
    threshold: int = 85,
    min_length_ratio: float = 0.5,
    min_length: int = 3,
) -> bool:
    text = normalize_text(ocr_text)

    if len(text) < min_length:
        return False

    for term in lookup:
        if len(term) == 0:
            continue
        # Skip when either side is far shorter than the other: a short
        # string (e.g. "f", "0") would always score 100 with partial_ratio
        # against any longer string that contains it as a substring.
        if len(text) / len(term) < min_length_ratio:
            continue
        if len(term) / len(text) < min_length_ratio:
            continue
        if fuzz.partial_ratio(text, term) > threshold:
            logger.debug(
                "Partial Match found: '%s' matches '%s' with score %d",
                text,
                term,
                fuzz.partial_ratio(text, term),
            )
            return True
    return False


def token_sensitive(text, lookup):
    tokens = normalize_text(text).split()

    for token in tokens:
        for term in lookup:
            if fuzz.ratio(token, term) > 90:
                logger.debug(
                    " Token Match found: '%s' matches '%s' with score %d",
                    text,
                    term,
                    fuzz.ratio(token, term),
                )
                return True
    return False


def normalize_text_list(text_list: list[str]) -> list[str]:
    return [normalize_text(text) for text in text_list]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Replace multiple whitespace with a single space and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


SPECIAL_KEYS = {
    "PatientName",
    "PatientSex",
    "PatientBirthDate",
    "StudyDate",
    "PatientAge",
}


def build_sensitive_lookup(data: dict[str, str]) -> set[str]:
    lookup = set()

    # Add all non-empty values from data (except special keys) to lookup
    lookup |= {data[key] for key in data if key not in SPECIAL_KEYS and data[key] != ""}

    # Special cases
    if "PatientName" in data:
        lookup |= expand_patient_name(data["PatientName"])
    if "PatientSex" in data:
        lookup |= expand_sex(data["PatientSex"])
    if "PatientBirthDate" in data:
        lookup |= expand_date(data["PatientBirthDate"])
    if "StudyDate" in data:
        lookup |= expand_date(data["StudyDate"])
    if "PatientAge" in data:
        lookup |= expand_age(data["PatientAge"])

    return {normalize_text(x) for x in lookup}


# Patient Name
def expand_patient_name(name: str) -> set[str]:
    parts = name.split("^")
    last = normalize_text(parts[0]) if len(parts) > 0 else ""
    first = normalize_text(parts[1]) if len(parts) > 1 else ""

    variants = set()

    if first:
        variants.add(first)
    if last:
        variants.add(last)
    if first and last:
        variants.add(f"{first} {last}")
        variants.add(f"{last} {first}")

    return variants


# Patient Sex
SEX_MAP = {
    "f": {"female", "femme", "fille", "feminin"},
    "m": {"male", "homme", "garcon", "masculin"},
    "o": {"other", "autre", "non-binary", "nonbinary", "non binaire"},
}


def expand_sex(sex: str) -> set[str]:
    sex = sex.lower()
    return SEX_MAP.get(sex, {sex})


# Dates
MONTHS_EN = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
MONTHS_EN_SHORT = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]
MONTHS_FR = [
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
]
MONTHS_FR_SHORT = [
    "jan",
    "fev",
    "mar",
    "avr",
    "mai",
    "jun",
    "jul",
    "aou",
    "sep",
    "oct",
    "nov",
    "dec",
]


def _text_month_variants(dt: datetime) -> set[str]:
    m = dt.month - 1  # 0-indexed
    d = dt.day
    y = dt.year
    variants = set()
    for month_name in (
        MONTHS_EN[m],
        MONTHS_EN_SHORT[m],
        MONTHS_FR[m],
        MONTHS_FR_SHORT[m],
    ):
        # "26 Jan 2021", "26 janvier 2021", "Jan 26 2021", etc.
        variants.update(
            {
                f"{d} {month_name} {y}",
                f"{d:02d} {month_name} {y}",
                f"{month_name} {d} {y}",
                f"{month_name} {d:02d} {y}",
                f"{y} {month_name} {d}",
                f"{y} {month_name} {d:02d}",
                f"{y}{month_name}{d}",
                f"{y}{month_name}{d:02d}",
                f"{y}{month_name} {d}",
                f"{y}{month_name} {d:02d}",
                f"{y} {month_name}{d}",
                f"{y} {month_name}{d:02d}",
            }
        )
    return variants


def expand_date(dicom_date: str) -> set[str]:
    dt = datetime.strptime(dicom_date, "%Y%m%d")

    numeric_formats = [
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
    ]

    variants = {dt.strftime(fmt) for fmt in numeric_formats}
    variants |= _text_month_variants(dt)
    return variants


# Patient Age
def expand_age(age: str) -> set[str]:
    if age == "" or len(age) < 4:
        return set()

    num = int(age[:3])
    unit = age[3].lower()

    return {
        str(num),
        f"{num}{unit}",
        f"{num:03}{unit}",
    }
