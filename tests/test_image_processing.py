from image_ocr_identifier.image_processing import (
    _find_parts,
    _split_single_block,
    split_ocr_blocks,
)

# --- _find_parts ---


def test_find_parts_space_separated():
    # Spaces are no longer delimiters - stays as one block
    assert _find_parts("16:20:15 27.10.2020") == [("16:20:15 27.10.2020", 0, 19)]


def test_find_parts_comma_separated():
    assert _find_parts("ANONYME,FILLE") == [
        ("ANONYME", 0, 7),
        ("FILLE", 8, 13),
    ]


def test_find_parts_slash_with_spaces():
    # Non-numeric slash with spaces: still a delimiter
    assert _find_parts("12345 / DUPONT") == [
        ("12345", 0, 5),
        ("DUPONT", 8, 14),
    ]


def test_find_parts_slash_no_spaces():
    # Slash between non-numeric content is still a delimiter
    assert _find_parts("12345/DUPONT") == [
        ("12345", 0, 5),
        ("DUPONT", 6, 12),
    ]


def test_find_parts_date_with_slash_not_split():
    assert _find_parts("27/10/2020") == [("27/10/2020", 0, 10)]


def test_find_parts_date_slash_in_context():
    # Space no longer splits, so whole string is one block
    assert _find_parts("16:20:15 27/10/2020") == [("16:20:15 27/10/2020", 0, 19)]


def test_find_parts_date_with_space_before_slash():
    # "1928 / 03/ 03" -> single date token (slash between numerics, spaces allowed)
    assert _find_parts("1928 / 03/ 03") == [("1928 / 03/ 03", 0, 13)]


def test_find_parts_date_with_spaces_all_around_slashes():
    assert _find_parts("1928 / 03 / 03") == [("1928 / 03 / 03", 0, 14)]


def test_find_parts_date_space_after_slash_only():
    assert _find_parts("27/ 10/ 2020") == [("27/ 10/ 2020", 0, 12)]


def test_find_parts_spaced_date_in_context():
    # Without space splitting, the whole string is one token
    assert _find_parts("16:20:15 1928 / 03 / 03") == [
        ("16:20:15 1928 / 03 / 03", 0, 23)
    ]


def test_find_parts_metric_before_date_splits():
    # "ITm1.0" contains letters -> slash is a field separator, not a date slash
    assert _find_parts("ITm1.0 / 02-09-2021") == [
        ("ITm1.0", 0, 6),
        ("02-09-2021", 9, 19),
    ]


def test_find_parts_full_dicom_line_splits_correctly():
    text = "CA1-7A / Abdomen / IPS9 / IM1.3 / ITm1.0 / 02-09-2021 12:15:36"
    parts = _find_parts(text)
    tokens = [p[0] for p in parts]
    assert tokens == [
        "CA1-7A",
        "Abdomen",
        "IPS9",
        "IM1.3",
        "ITm1.0",
        "02-09-2021 12:15:36",
    ]


def test_find_parts_full_dicom_line_without_space_splits_correctly():
    text = "CA1-7A/Abdomen/IPS9/IM1.3/ITm1.0/02-09-2021 12:15:36"
    parts = _find_parts(text)
    tokens = [p[0] for p in parts]
    assert tokens == [
        "CA1-7A",
        "Abdomen",
        "IPS9",
        "IM1.3",
        "ITm1.0",
        "02-09-2021 12:15:36",
    ]


def test_find_parts_names_id_date_mixed():
    text = "JEAN,ALBERTDUPONT,97658223.20190116"
    parts = _find_parts(text)
    tokens = [p[0] for p in parts]
    assert tokens == ["JEAN", "ALBERTDUPONT", "97658223", "20190116"]


def test_find_parts_names_id_date_mixed_dot():
    text = "JEAN.ALBERTDUPONT.97658223.20190116"
    parts = _find_parts(text)
    tokens = [p[0] for p in parts]
    assert tokens == ["JEAN", "ALBERTDUPONT", "97658223", "20190116"]


def test_find_parts_not_split_date_begin_with_dot():
    text = "Né(..20/11/1880"
    parts = _find_parts(text)
    tokens = [p[0] for p in parts]
    assert tokens == ["Né(..20/11/1880"]


def test_find_parts_date_with_stray_double_slash_not_split():
    # OCR artifact "/ /" (stray repeated slash) inside a date must not split it
    text = "Né(…09/ /09/1947 F"
    parts = _find_parts(text)
    tokens = [p[0] for p in parts]
    assert tokens == ["Né(…09/ /09/1947 F"]


def test_find_parts_acronym_dots_not_split():
    # Dots between single letters are part of an acronym, not delimiters
    assert _find_parts("H.U.G. 3C") == [("H.U.G. 3C", 0, 9)]


def test_find_parts_acronym_alone():
    assert _find_parts("H.U.G") == [("H.U.G", 0, 5)]


def test_find_parts_acronym_trailing_dot():
    assert _find_parts("U.S.A.") == [("U.S.A.", 0, 6)]


def test_find_parts_acronym_in_comma_context():
    parts = _find_parts("DUPONT,H.U.G")
    tokens = [p[0] for p in parts]
    assert tokens == ["DUPONT", "H.U.G"]


def test_find_parts_name_dot_still_splits():
    # Multi-letter segments around a dot remain a field delimiter
    parts = _find_parts("JEAN.ALBERT")
    tokens = [p[0] for p in parts]
    assert tokens == ["JEAN", "ALBERT"]


def test_find_parts_pure_numeric_spaced_slash_preserved():
    # All-numeric segment before slash: kept as date
    assert _find_parts("1928 / 03 / 03") == [("1928 / 03 / 03", 0, 14)]


def test_find_parts_multiple_spaces():
    # Spaces inside a token are included - no split on spaces alone
    assert _find_parts("hello   world") == [("hello   world", 0, 13)]


def test_find_parts_no_delimiter():
    assert _find_parts("SIEMENS") == [("SIEMENS", 0, 7)]


def test_find_parts_empty():
    assert _find_parts("") == []


# --- _split_single_block ---


def test_split_single_block_datetime():
    # Spaces no longer split, so stays as one block
    text = "16:20:15 27.10.2020"
    box = [0, 0, 190, 20]
    result = _split_single_block(text, box)
    assert len(result) == 1
    assert result[0] == (text, box)


def test_split_single_block_comma():
    text = "ANONYME,FILLE"
    box = [10, 5, 140, 25]
    result = _split_single_block(text, box)
    assert len(result) == 2
    assert result[0][0] == "ANONYME"
    assert result[1][0] == "FILLE"
    # pixel_per_char = 130/13 = 10, padding = max(3, int(10*0.9)) = 9
    # ANONYME: start_px=10, end_px=80 -> [max(10, 10-9)=10, min(140, 80+9)=89]
    assert result[0][1] == [10, 5, 89, 25]
    # FILLE: start_px=90, gap=90-(80+9)=1, left_pad=1 -> [89, min(140,140+9)=140]
    assert result[1][1] == [89, 5, 140, 25]


def test_split_single_block_multichar_delimiter_no_left_overshoot():
    # " / " is a 3-char delimiter. A token's left edge must not extend back
    # into the previous token (regression: left edge was anchored on the
    # previous token's end index, overshooting for multi-char delimiters).
    text = "AB / CD"
    box = [0, 0, 70, 20]
    result = _split_single_block(text, box)
    assert len(result) == 2
    assert result[0][0] == "AB"
    assert result[1][0] == "CD"
    # CD left edge must not extend past AB right edge (no overlap)
    assert result[1][1][0] >= result[0][1][2]


def test_split_single_block_no_split():
    text = "SIEMENS"
    box = [100, 50, 200, 70]
    result = _split_single_block(text, box)
    assert len(result) == 1
    assert result[0] == ("SIEMENS", [100, 50, 200, 70])


def test_split_single_block_empty_text():
    text = ""
    box = [0, 0, 100, 20]
    result = _split_single_block(text, box)
    assert len(result) == 1
    assert result[0] == ("", [0, 0, 100, 20])


def test_split_single_block_non_standard_box():
    # Polygon box (not 4-element) should be returned as-is
    text = "hello world"
    box = [[0, 0], [100, 0], [100, 20], [0, 20]]
    result = _split_single_block(text, box)
    assert len(result) == 1
    assert result[0] == (text, box)


# --- split_ocr_blocks ---


def test_split_ocr_blocks_mixed():
    ocr_result = {
        "texts": ["16:20:15 27.10.2020", "SIEMENS", "ANONYME,FILLE", "2019Nov.08"],
        "boxes": [
            [0, 0, 190, 20],
            [300, 0, 400, 20],
            [0, 30, 130, 50],
            [0, 60, 100, 80],
        ],
    }
    result = split_ocr_blocks(ocr_result)
    # Space no longer splits, so datetime stays as one block
    assert result["texts"] == [
        "16:20:15 27.10.2020",
        "SIEMENS",
        "ANONYME",
        "FILLE",
        "2019Nov.08",
    ]
    assert len(result["boxes"]) == 5


def test_split_ocr_blocks_empty():
    ocr_result = {"texts": [], "boxes": []}
    result = split_ocr_blocks(ocr_result)
    assert result == {"texts": [], "boxes": [], "groups": []}


def test_split_ocr_blocks_preserves_unsplittable():
    ocr_result = {
        "texts": ["GEN", "2D", "100%"],
        "boxes": [[0, 0, 30, 20], [50, 0, 70, 20], [100, 0, 140, 20]],
    }
    result = split_ocr_blocks(ocr_result)
    assert result["texts"] == ["GEN", "2D", "100%"]
    assert result["boxes"] == [[0, 0, 30, 20], [50, 0, 70, 20], [100, 0, 140, 20]]
