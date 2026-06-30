from datetime import datetime
import logging
import re
from typing import Dict
import unicodedata
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from deidentification_karnak.utils import numpy_to_python_type

logger = logging.getLogger(__name__)


# Compare OCR results with sensitive_data_list and return only the data that needs to be masked
def detect_sensitive_data(
    ocr_result: Dict[str, list | np.ndarray], sensitive_data_list: Dict[str, str]
) -> Dict[str, list]:
    ocr_texts = ocr_result["texts"]
    ocr_boxes = ocr_result["boxes"]
    groups = ocr_result.get("groups")

    # Build lookup set from sensitive data
    lookup = build_sensitive_lookup(sensitive_data_list)
    name_lookup = {
        normalize_text(x)
        for x in expand_patient_name(sensitive_data_list.get("PatientName", ""))
    }

    sensitive_indices = _detect_sensitive_indices(
        ocr_texts, ocr_boxes, groups, lookup, name_lookup
    )

    filtered_texts = []
    filtered_boxes = []
    for i in sorted(sensitive_indices):
        filtered_texts.append(ocr_texts[i])
        filtered_boxes.append(numpy_to_python_type(ocr_boxes[i]))
        logger.debug(
            "Sensitive data detected: '%s' at box %s", ocr_texts[i], ocr_boxes[i]
        )

    return {
        "texts": filtered_texts,
        "boxes": filtered_boxes,
    }


# Return the DICOM tag names whose values are detected in the OCR text, without
# computing masks. Each key is tested independently against only its own value
# variants, so the result reports exactly which tags appear in the image.
def detect_sensitive_keys(
    ocr_result: Dict[str, list | np.ndarray], sensitive_data_list: Dict[str, str]
) -> list[str]:
    ocr_texts = ocr_result["texts"]
    ocr_boxes = ocr_result["boxes"]
    groups = ocr_result.get("groups")

    lookup_by_key = build_sensitive_lookup_by_key(sensitive_data_list)
    name_lookup = {
        normalize_text(x)
        for x in expand_patient_name(sensitive_data_list.get("PatientName", ""))
    }

    detected = []
    # Iterate in the caller's key order so the report mirrors the input.
    for key in sensitive_data_list:
        lookup = lookup_by_key.get(key)
        if not lookup:
            continue
        key_name_lookup = name_lookup if key == "PatientName" else set()
        if _detect_sensitive_indices(
            ocr_texts, ocr_boxes, groups, lookup, key_name_lookup
        ):
            detected.append(key)
    return detected


def _detect_sensitive_indices(
    ocr_texts: list[str],
    ocr_boxes: list,
    groups: list | None,
    lookup: set[str],
    name_lookup: set[str],
) -> set[int]:
    sensitive_indices = set()

    for i, text in enumerate(ocr_texts):
        if (
            is_sensitive(text, lookup)
            or token_sensitive(text, lookup)
            or name_token_sensitive(text, name_lookup)
        ):
            sensitive_indices.add(i)

    # Second pass: for siblings in the same group as a match, retry with relaxed threshold
    if groups:
        sensitive_groups = {groups[i] for i in sensitive_indices}
        for i, g in enumerate(groups):
            if i not in sensitive_indices and g in sensitive_groups:
                if is_sensitive(ocr_texts[i], lookup, threshold=63):
                    sensitive_indices.add(i)

    # Third pass: reconstruct terms that PaddleOCR split across several adjacent
    # boxes on the same line (e.g. "HUG - Gynécologie" as "HUG -"/"Gynécologie",
    # or a birth date "1928 / 03 / 03" as "1928"/"03"/"03"). Eligible terms are
    # those that can appear split: multi-word terms (spaces) and numeric
    # multi-segment terms (dates, IDs) whose separators are dropped by
    # normalization so they have no space but still span several boxes.
    splittable_terms = {
        term
        for term in lookup
        if " " in term or (len(term) >= 6 and any(c.isdigit() for c in term))
    }
    if splittable_terms:
        sensitive_indices |= detect_split_terms(
            ocr_texts, ocr_boxes, splittable_terms, member_terms=lookup | name_lookup
        )

    return sensitive_indices


def detect_split_terms(
    texts: list[str],
    boxes: list,
    terms: set[str],
    member_terms: set[str] | None = None,
    max_window: int = 5,
) -> set[int]:
    """Find boxes whose text, when concatenated with adjacent same-line boxes,
    reconstructs a multi-word sensitive term.

    Returns the indices of every box that contributes to such a reconstruction.
    Matching is anchored on the whole run being (nearly) equal to the term, so
    unrelated neighbours do not get flagged.

    ``member_terms`` is the full set of sensitive terms used to decide whether an
    individual box in a matched run is a genuine fragment. A box is kept only if
    its own text resembles some sensitive term: this drops stray letters or a
    manufacturer logo dragged onto the line, while still keeping a fragment that
    belongs to a *different* term than the one the run matched.
    """
    if member_terms is None:
        member_terms = terms
    matched: set[int] = set()
    normalized = [normalize_text(t) for t in texts]

    valid = [
        i
        for i, box in enumerate(boxes)
        if len(box) == 4 and not isinstance(box[0], (list, tuple))
    ]

    for start in valid:
        if not normalized[start]:
            continue
        run = [start]
        combined = normalized[start]
        current = start
        # Grow the run by following the nearest right-hand neighbour that sits
        # on the same text line (vertical overlap), then re-test against terms.
        # Single boxes are not "splits" (handled by the earlier passes), so a
        # run must contain at least two boxes before it can match.
        for _ in range(max_window - 1):
            nxt = _next_box_on_line(current, valid, boxes)
            if nxt is None:
                break
            run.append(nxt)
            if normalized[nxt]:
                combined = f"{combined} {normalized[nxt]}".strip()
            current = nxt
            if len(combined) >= 4 and any(
                _run_matches_term(combined, term) for term in terms
            ):
                # Flag only boxes whose own text is a genuine fragment of some
                # sensitive term. Date/ID/name fragments resemble a term and are
                # kept (including fragments of a different name part than the one
                # the run matched); stray letters or a manufacturer logo dragged
                # onto the line resemble nothing and are dropped.
                matched.update(
                    idx
                    for idx in run
                    if normalized[idx]
                    and _box_in_any_term(normalized[idx], member_terms)
                )
                break
    return matched


def _box_in_any_term(box_text: str, terms: set[str]) -> bool:
    return any(_box_in_term(box_text, term) for term in terms)


def _box_in_term(box_text: str, term: str) -> bool:
    if fuzz.partial_ratio(box_text, term) >= 85:
        return True
    # Mirror the run-level handling: numeric segments lose their separators in
    # normalization, so a box like "03" still matches a term like "19280303".
    stripped = box_text.replace(" ", "")
    if stripped != box_text and fuzz.partial_ratio(stripped, term) >= 85:
        return True
    # A box like "date naiss 1928" contains a 4+ digit year that is a substring
    # of a date term like "19280303". Accept it when contextually part of a run.
    for token in box_text.split():
        if len(token) >= 4 and token.isdigit() and token in term:
            return True
    return False


def _run_matches_term(combined: str, term: str) -> bool:
    # The concatenated boxes must approximate the *entire* term, so a short
    # fragment (e.g. "gyn") cannot match a long term via substring coincidence.
    if len(combined) < len(term) * 0.6:
        return False
    # Whole-run similarity: the concatenated boxes must approximate the entire
    # term, which prevents random adjacent boxes from matching.
    if fuzz.ratio(combined, term) >= 85:
        return True
    # Numeric multi-segment terms (e.g. dates "19280303") lose their separators
    # in normalization, so the joined run "1928 03 03" only matches once the
    # word-joining spaces are removed too.
    stripped = combined.replace(" ", "")
    if stripped != combined and fuzz.ratio(stripped, term) >= 85:
        return True
    # Allow small OCR noise (extra chars) while bounding the run length so a
    # short term cannot match inside a long unrelated concatenation.
    if len(combined) <= len(term) * 1.3 and fuzz.partial_ratio(combined, term) >= 90:
        return True
    # Try word-boundary suffixes to handle label prefixes in OCR boxes
    if _suffix_matches_term(combined, term):
        return True
    return False


def _suffix_matches_term(combined: str, term: str) -> bool:
    words = combined.split()
    if len(words) <= 1:
        return False
    for i in range(1, len(words)):
        suffix = " ".join(words[i:])
        if len(suffix) < len(term) * 0.6:
            break
        if fuzz.ratio(suffix, term) >= 85:
            return True
        stripped = suffix.replace(" ", "")
        # Require the stripped suffix to be close in length to the term so
        # an incomplete fragment (e.g. "192803" vs "19280303") doesn't match
        # prematurely before remaining boxes are added to the run.
        if (
            stripped != suffix
            and len(stripped) >= len(term) * 0.85
            and fuzz.ratio(stripped, term) >= 85
        ):
            return True
    return False


def _next_box_on_line(
    current: int,
    candidates: list[int],
    boxes: list,
    min_vertical_overlap: float = 0.4,
    max_gap_ratio: float = 2.0,
) -> int | None:
    """Return the nearest box to the right of ``current`` on the same text line.

    Same line is decided by vertical span overlap (robust to small angle/offset
    shifts between boxes), not by a global line anchor. The horizontal gap must
    not exceed ``max_gap_ratio`` times the current box height.
    """
    cx_min, cy_min, cx_max, cy_max = (float(boxes[current][i]) for i in range(4))
    c_height = cy_max - cy_min
    best = None
    best_x = None
    for j in candidates:
        if j == current:
            continue
        jx_min, jy_min, _, jy_max = (float(boxes[j][i]) for i in range(4))
        if jx_min <= cx_min:
            continue
        overlap = min(cy_max, jy_max) - max(cy_min, jy_min)
        min_height = min(c_height, jy_max - jy_min)
        if min_height <= 0 or overlap / min_height < min_vertical_overlap:
            continue
        gap = jx_min - cx_max
        if gap > max_gap_ratio * c_height:
            continue
        if best_x is None or jx_min < best_x:
            best = j
            best_x = jx_min
    return best


def is_sensitive(
    ocr_text: str,
    lookup: set[str],
    threshold: int = 85,
    min_length_ratio: float = 0.45,
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
        score = fuzz.partial_ratio(text, term)
        if score > threshold:
            # For short texts, partial_ratio gives inflated scores from
            # substring coincidences. Require full-string similarity too.
            if len(text) <= 5 and fuzz.ratio(text, term) < 70:
                continue
            logger.debug(
                "Partial Match found: '%s' matches '%s' with score %d",
                text,
                term,
                score,
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


def name_token_sensitive(
    ocr_text: str,
    name_lookup: set[str],
    min_length: int = 3,
) -> bool:
    """Match a single OCR token against single-word PatientName terms.

    Tolerates small OCR errors via edit distance: 1 edit for length <=5,
    2 edits for 6-9, 3 edits for >=10. Length difference between text and
    term must not exceed the allowed edits, which prevents short substrings
    of long names from matching.
    """
    text = normalize_text(ocr_text)
    if len(text) < min_length:
        return False
    for term in name_lookup:
        if " " in term or len(term) < min_length:
            continue
        max_edits = _max_name_edits(min(len(text), len(term)))
        if abs(len(text) - len(term)) > max_edits:
            continue
        distance = Levenshtein.distance(text, term)
        if distance <= max_edits:
            logger.debug(
                "Name token match: '%s' matches '%s' with distance %d",
                text,
                term,
                distance,
            )
            return True
    return False


def _max_name_edits(length: int) -> int:
    if length <= 3:
        return 0
    if length <= 5:
        return 1
    if length <= 9:
        return 2
    return 3


def normalize_text_list(text_list: list[str]) -> list[str]:
    return [normalize_text(text) for text in text_list]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove non-alphanumeric, non-letter characters (keep Unicode letters/digits)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"[_]", "", text)
    # Replace multiple whitespace with a single space and trim
    text = re.sub(r"\s+", " ", text).strip()
    # Fix common OCR confusion: digit 0 between letters is likely 'o'
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text)
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
        lookup |= expand_age(
            data["PatientAge"],
            data.get("PatientBirthDate", ""),
            data.get("StudyDate", ""),
        )

    normalized = {normalize_text(x) for x in lookup}
    # Add space-stripped variants for matching OCR text that omits spaces
    normalized |= {t.replace(" ", "") for t in normalized if " " in t}
    return normalized


def build_sensitive_lookup_by_key(data: dict[str, str]) -> dict[str, set[str]]:
    """Like ``build_sensitive_lookup`` but keep each tag's variants separate.

    Returns a mapping of DICOM tag name to its normalized lookup set. Empty
    values and keys that expand to nothing are omitted so callers can treat a
    present key as one that can actually be matched.
    """
    variants_by_key: dict[str, set[str]] = {}

    for key, value in data.items():
        if key in SPECIAL_KEYS or value == "":
            continue
        variants_by_key[key] = {value}

    if data.get("PatientName"):
        variants_by_key["PatientName"] = expand_patient_name(data["PatientName"])
    if data.get("PatientSex"):
        variants_by_key["PatientSex"] = expand_sex(data["PatientSex"])
    if data.get("PatientBirthDate"):
        variants_by_key["PatientBirthDate"] = expand_date(data["PatientBirthDate"])
    if data.get("StudyDate"):
        variants_by_key["StudyDate"] = expand_date(data["StudyDate"])
    if data.get("PatientAge"):
        variants_by_key["PatientAge"] = expand_age(
            data["PatientAge"],
            data.get("PatientBirthDate", ""),
            data.get("StudyDate", ""),
        )

    lookup_by_key: dict[str, set[str]] = {}
    for key, variants in variants_by_key.items():
        normalized = {normalize_text(x) for x in variants}
        normalized |= {t.replace(" ", "") for t in normalized if " " in t}
        normalized.discard("")
        if normalized:
            lookup_by_key[key] = normalized
    return lookup_by_key


# Patient Name
def expand_patient_name(name: str) -> set[str]:
    parts = name.split("^")
    last = normalize_text(parts[0]) if len(parts) > 0 else ""
    firsts = normalize_text_list(parts[1].split(",")) if len(parts) > 1 else []

    variants = set()

    if firsts:
        variants.update(firsts)
    if last:
        variants.add(last)
    if firsts and last:
        for first in firsts:
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
def expand_age(age: str, birth_date: str = "", study_date: str = "") -> set[str]:
    if age == "" or len(age) < 4:
        return set()

    num = int(age[:3])
    unit = age[3].lower()

    variants = {
        f"{num}{unit}",
        f"{num:03}{unit}",
    }

    if unit == "y":
        variants.add(f"{num}ans")
        months = _age_months_remainder(birth_date, study_date)
        if months is not None:
            variants.add(f"{num}y{months}m")
            variants.add(f"{num}a{months}m")

    return variants


def _age_months_remainder(birth_date: str, study_date: str) -> int | None:
    if not birth_date or not study_date:
        return None
    birth = datetime.strptime(birth_date, "%Y%m%d")
    study = datetime.strptime(study_date, "%Y%m%d")
    months = (study.year - birth.year) * 12 + (study.month - birth.month)
    if study.day < birth.day:
        months -= 1
    return months % 12
