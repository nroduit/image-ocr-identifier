import pytest

from image_ocr_identifier.image_processing import split_ocr_blocks
from image_ocr_identifier.sensitive_data_detection import (
    build_sensitive_lookup,
    detect_sensitive_data,
    expand_age,
    expand_date,
    expand_patient_name,
    expand_sex,
    is_sensitive,
    normalize_text,
    normalize_text_list,
    token_sensitive,
)

# --- normalize_text ---


def test_normalize_text_lowercase():
    assert normalize_text("HELLO") == "hello"


def test_normalize_text_removes_special_chars():
    assert normalize_text("hello@world!") == "helloworld"


def test_normalize_text_removes_accents():
    assert normalize_text("àcafé") == "acafe"


def test_normalize_text_extra_whitespaces():
    assert normalize_text("  hello   world   ") == "hello world"


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_only_special_chars():
    assert normalize_text("@#$%") == ""


def test_normalize_patient_name():
    assert normalize_text("Anonyme^Fille") == "anonymefille"


# --- normalize_text_list ---


def test_normalize_text_list_multiple_items():
    assert normalize_text_list(["HELLO", "WORLD"]) == ["hello", "world"]


def test_normalize_text_list_empty():
    assert normalize_text_list([]) == []


def test_normalize_text_list_with_special_chars():
    assert normalize_text_list(["Hello@123", "World!"]) == ["hello123", "world"]


# --- is_sensitive ---


def test_is_sensitive_exact_match():
    assert is_sensitive("John Doe", {"john doe"}) is True


def test_is_sensitive_fuzzy_match():
    # Single-char OCR transposition should still match above the 85 threshold
    assert is_sensitive("Jhon Doe", {"john doe"}) is True


def test_is_sensitive_no_match():
    # Similar-length strings that share no significant substrings
    assert is_sensitive("hello world", {"zzzzz yyyyy"}) is False


def test_is_sensitive_text_too_short():
    # Text normalizes to fewer than min_length (3) chars -> early return False
    assert is_sensitive("ab", {"ab"}) is False


def test_is_sensitive_empty_term_is_skipped():
    # Empty term must be skipped to avoid division by zero
    assert is_sensitive("hello world", {""}) is False


def test_is_sensitive_text_much_shorter_than_term():
    # len(text)/len(term) = 3/12 = 0.25 < min_length_ratio (0.4) -> term skipped
    assert is_sensitive("abc", {"abcdefghijkl"}) is False


def test_is_sensitive_term_much_shorter_than_text():
    # len(term)/len(text) = 3/15 = 0.2 < min_length_ratio (0.4) -> term skipped
    assert is_sensitive("hello world foo", {"abc"}) is False


# --- token_sensitive ---


def test_token_sensitive_matching_token():
    # "john" is an exact token in the text and appears in the lookup
    assert token_sensitive("Patient John Smith", {"john"}) is True


def test_token_sensitive_no_match():
    assert token_sensitive("hello world", {"xyz"}) is False


def test_token_sensitive_empty_text():
    # No tokens to iterate -> always False
    assert token_sensitive("", {"john"}) is False


# --- expand_patient_name ---


def test_expand_patient_name_full_name():
    assert expand_patient_name("Doe^John") == {"doe", "john", "john doe", "doe john"}


def test_expand_patient_name_last_only():
    assert expand_patient_name("Smith") == {"smith"}


def test_expand_patient_name_first_only():
    # Leading ^ means empty last name, only first is added
    assert expand_patient_name("^Jane") == {"jane"}


def test_expand_patient_name_empty():
    assert expand_patient_name("") == set()


# --- expand_sex ---


@pytest.mark.parametrize(
    "code, expected",
    [
        ("F", {"female", "femme", "fille", "feminin"}),
        ("M", {"male", "homme", "garcon", "masculin"}),
        ("O", {"other", "autre", "non-binary", "nonbinary", "non binaire"}),
    ],
)
def test_expand_sex_known_codes(code, expected):
    assert expand_sex(code) == expected


def test_expand_sex_unknown_code():
    # Unknown code should fall back to lowercase
    assert expand_sex("X") == {"x"}


# --- expand_date ---


def test_expand_date_numeric_formats():
    result = expand_date("20230115")
    assert "15/01/2023" in result
    assert "01/15/2023" in result
    assert "2023-01-15" in result
    assert "15.01.2023" in result
    assert "15-01-2023" in result


def test_expand_date_text_month_variants():
    result = expand_date("20230115")
    assert "15 january 2023" in result
    assert "15 janvier 2023" in result
    assert "january 15 2023" in result
    assert "15 jan 2023" in result


# --- expand_age ---


def test_expand_age_normal():
    assert expand_age("045Y") == {"45y", "045y", "45ans"}


def test_expand_age_months_from_dates():
    result = expand_age("045Y", "19800115", "20250320")
    assert "45y2m" in result
    assert "45a2m" in result
    assert "45y" in result
    assert "045y" in result


def test_expand_age_leading_zeros():
    assert expand_age("001D") == {"1d", "001d"}


def test_expand_age_empty_string():
    assert expand_age("") == set()


def test_expand_age_string_too_short():
    # Age string must be at least 4 characters (e.g. "004Y") to be valid
    # Below that, it is considered invalid and returns an empty set.
    assert expand_age("04") == set()


# --- build_sensitive_lookup ---


def test_build_sensitive_lookup_plain_value_normalized():
    result = build_sensitive_lookup({"InstitutionName": "General Hospital"})
    assert "general hospital" in result


def test_build_sensitive_lookup_empty_value_excluded():
    assert build_sensitive_lookup({"InstitutionName": ""}) == set()


def test_build_sensitive_lookup_patient_name():
    result = build_sensitive_lookup({"PatientName": "Doe^John"})
    assert "doe" in result
    assert "john" in result
    assert "john doe" in result


def test_build_sensitive_lookup_patient_sex():
    result = build_sensitive_lookup({"PatientSex": "F"})
    assert "female" in result
    assert "femme" in result


def test_build_sensitive_lookup_patient_birth_date():
    result = build_sensitive_lookup({"PatientBirthDate": "20230115"})
    assert "15012023" in result
    assert "15 january 2023" in result


def test_build_sensitive_lookup_study_date():
    result = build_sensitive_lookup({"StudyDate": "20230115"})
    assert "15012023" in result
    assert "15 january 2023" in result


def test_build_sensitive_lookup_patient_age():
    result = build_sensitive_lookup({"PatientAge": "045Y"})
    assert "45y" in result


def test_build_sensitive_lookup_all_special_keys():
    data = {
        "PatientName": "Doe^John",
        "PatientSex": "M",
        "PatientBirthDate": "19900315",
        "StudyDate": "20230115",
        "PatientAge": "033Y",
        "InstitutionName": "City Hospital",
    }
    result = build_sensitive_lookup(data)
    assert "doe" in result
    assert "male" in result
    assert "15031990" in result
    assert "15012023" in result
    assert "33y" in result
    assert "city hospital" in result


# --- detect_sensitive_data ---


def test_detect_sensitive_data_returns_structure():
    ocr_result = {"texts": ["test"], "boxes": [[0, 0, 10, 10]]}
    result = detect_sensitive_data(ocr_result, {"key": "value"})
    assert "texts" in result
    assert "boxes" in result


def test_detect_sensitive_data_no_match_returns_empty():
    ocr_result = {
        "texts": ["hello", "world"],
        "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
    }
    sensitive_data = {"name": "test"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert not result["texts"]
    assert not result["boxes"]


def test_detect_sensitive_data_with_matches():
    ocr_result = {
        "texts": ["hello", "world", "test"],
        "boxes": [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
    }
    sensitive_data = {"name": "world"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert result["texts"] == ["world"]
    assert result["boxes"] == [[20, 20, 30, 30]]


def test_detect_sensitive_data_with_matches_tricky_case():
    ocr_result = {
        "texts": [
            "DUPONTMOR...12345678",
            "DU PONT MORAND...12345678",
            "DUP0NTM0R...12345678",
            "DUPONTTOR...12345678",
            "E'Lat",
        ],
        "boxes": [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
            [60, 60, 70, 70],
            [80, 80, 90, 90],
        ],
    }

    ocr_result = split_ocr_blocks(ocr_result)

    sensitive_data = {
        "name": "DU PONT MORAND^DURAND,MARIE",
        "id": "12345678",
    }

    result = detect_sensitive_data(ocr_result, sensitive_data)

    assert "DUPONTMOR" in result["texts"]
    assert "12345678" in result["texts"]
    assert "DU PONT MORAND" in result["texts"]
    assert "DUP0NTM0R" in result["texts"]  # les deux O remplacés par des 0
    assert "DUPONTTOR" in result["texts"]  # erreur OCR sur une lettre
    assert "E'Lat" not in result["texts"]


def test_detect_sensitive_data_with_matches_tricky_case_2():
    ocr_result = {
        "texts": ["MARTIN12345678.20101016", "4URNND,TEST"],
        "boxes": [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
    }

    ocr_result = split_ocr_blocks(ocr_result)

    sensitive_data = {
        "PatientID": "12345678",
        "PatientName": "MARTIN^DURAND,TEST",
        "PatientBirthDate": "20101016",
    }

    result = detect_sensitive_data(ocr_result, sensitive_data)

    assert "MARTIN12345678" in result["texts"]
    assert "20101016" in result["texts"]
    assert "4URNND" in result["texts"]
    assert "TEST" in result["texts"]


def test_detect_sensitive_data_with_matches_tricky_case_3():
    ocr_result = {
        "texts": ["AnOnYmE.", "FICLE"],
        "boxes": [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
    }
    ocr_result = split_ocr_blocks(ocr_result)
    sensitive_data = {
        "PatientName": "ANONYME^FILLE",
    }
    result = detect_sensitive_data(ocr_result, sensitive_data)

    assert "AnOnYmE." in result["texts"]
    assert "FICLE" in result["texts"]


def test_detect_sensitive_data_with_matches_tricky_case_3():
    ocr_result = {
        "texts": ["AnOnYmE.", "FICLE"],
        "boxes": [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
    }
    ocr_result = split_ocr_blocks(ocr_result)
    sensitive_data = {
        "PatientName": "ANONYME^FILLE",
    }
    result = detect_sensitive_data(ocr_result, sensitive_data)

    assert "AnOnYmE." in result["texts"]
    assert "FICLE" in result["texts"]


def test_detect_sensitive_data_with_date_matches():
    ocr_result = {
        "texts": [
            "hello",
            "world",
            "test",
            "2019Nov08",
            "2019Nov 08",
            "Né(..20/11/1880",
            "08-Nov-19",
        ],
        "boxes": [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
            [60, 60, 70, 70],
            [80, 80, 90, 90],
            [100, 100, 110, 110],
            [120, 120, 130, 130],
        ],
    }
    sensitive_data = {
        "name": "world",
        "StudyDate": "20191108",
        "PatientBirthDate": "18801120",
    }
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert result["texts"] == [
        "world",
        "2019Nov08",
        "2019Nov 08",
        "Né(..20/11/1880",
        "08-Nov-19",
    ]
    assert result["boxes"] == [
        [20, 20, 30, 30],
        [60, 60, 70, 70],
        [80, 80, 90, 90],
        [100, 100, 110, 110],
        [120, 120, 130, 130],
    ]


def test_detect_sensitive_data_term_split_across_boxes():
    # PaddleOCR emitted one phrase as two separate same-line boxes. Only the
    # second box partially matches on its own; the merge pass must flag both.
    ocr_result = {
        "texts": ["ABC -", "Gynecology"],
        "boxes": [[0, 0, 40, 20], [45, 0, 130, 20]],
    }
    sensitive_data = {"institution": "ABC - Gynecology"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert "ABC -" in result["texts"]
    assert "Gynecology" in result["texts"]


def test_detect_sensitive_data_term_split_three_boxes():
    ocr_result = {
        "texts": ["ABC", "-", "Gynecology"],
        "boxes": [[0, 0, 30, 20], [33, 0, 40, 20], [45, 0, 130, 20]],
    }
    sensitive_data = {"institution": "ABC - Gynecology"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert "ABC" in result["texts"]
    assert "Gynecology" in result["texts"]


def test_detect_split_terms_does_not_merge_different_lines():
    # Same horizontal neighbours but on different lines must not be merged.
    ocr_result = {
        "texts": ["ABC -", "Gynecology"],
        "boxes": [[0, 0, 40, 20], [45, 200, 130, 220]],
    }
    sensitive_data = {"institution": "ABC - Gynecology"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    # Only the box that matches on its own is flagged, not the distant "ABC -".
    assert "ABC -" not in result["texts"]


def test_detect_split_terms_ignores_unrelated_neighbours():
    ocr_result = {
        "texts": ["Patient", "weight 70kg"],
        "boxes": [[0, 0, 60, 20], [65, 0, 160, 20]],
    }
    sensitive_data = {"institution": "ABC - Gynecology"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert not result["texts"]


def test_detect_split_terms_real_geometry_with_neighbour_boxes():
    # Real ultrasound layout: the two halves of "ABC - Gynecology" share a line
    # with several other boxes (Machine, MDL2-A17-UT, ...). A fixed-anchor line
    # grouping wrongly split the halves into separate lines; same-line detection
    # by vertical overlap must still merge them.
    ocr_result = {
        "texts": ["Machine", "ABC -", "Gynecology", "MDL2-A17-UT", "GYN"],
        "boxes": [
            [4, 1, 104, 28],
            [1089, 14, 1207, 49],
            [1195, 14, 1403, 55],
            [1806, 5, 1994, 39],
            [1915, 43, 1995, 76],
        ],
    }
    sensitive_data = {"institution": "ABC - Gynecology"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert "ABC -" in result["texts"]
    assert "Gynecology" in result["texts"]
    # Short boxes whose normalized text is a substring of the term must not leak.
    assert "GYN" not in result["texts"]
    assert "Machine" not in result["texts"]


def test_detect_split_terms_numeric_date_across_boxes():
    # A birth date split into "1928" and "/ 03 / 03" on the same line. The
    # numeric term "19280303" loses its separators in normalization, so the
    # joined run only matches after spaces are stripped. The "Date naiss."
    # label must not be flagged.
    ocr_result = {
        "texts": ["Date naiss.", "1928", "/ 03 / 03"],
        "boxes": [
            [98, 731, 403, 778],
            [382, 726, 512, 777],
            [552, 728, 821, 779],
        ],
    }
    sensitive_data = {"PatientBirthDate": "19280303"}
    result = detect_sensitive_data(ocr_result, sensitive_data)
    assert "1928" in result["texts"]
    assert "/ 03 / 03" in result["texts"]
    assert "Date naiss." not in result["texts"]
