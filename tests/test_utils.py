import numpy as np
import pytest

from image_ocr_identifier.utils import (
    bgr_to_hex,
    cv2_to_karnak_coord,
    expand_boxes,
    format_boxes,
    numpy_to_python_type,
)

# --- expand_boxes ---


def test_expand_boxes_normal():
    assert expand_boxes([[10, 10, 20, 20]], margin=2) == [[8, 8, 22, 22]]


def test_expand_boxes_clamped_at_origin():
    assert expand_boxes([[1, 1, 5, 5]], margin=2) == [[0, 0, 7, 7]]


def test_expand_boxes_at_zero():
    assert expand_boxes([[0, 0, 5, 5]], margin=2) == [[0, 0, 7, 7]]


def test_expand_boxes_default_margin():
    assert expand_boxes([[10, 10, 20, 20]]) == [[8, 8, 22, 22]]


def test_expand_boxes_wrong_length_passthrough():
    assert expand_boxes([[1, 2, 3]]) == [[1, 2, 3]]


def test_expand_boxes_empty():
    assert expand_boxes([]) == []


# --- bgr_to_hex ---


@pytest.mark.parametrize(
    "bgr, expected",
    [
        ((0, 0, 255), "ff0000"),  # B=0,G=0,R=255 => RRGGBB = ff0000
        ((0, 255, 0), "00ff00"),
        ((255, 0, 0), "0000ff"),  # B=255,G=0,R=0 => RRGGBB = 0000ff
        ((0, 0, 0), "000000"),
        ((255, 255, 255), "ffffff"),
    ],
)
def test_bgr_to_hex(bgr, expected):
    assert bgr_to_hex(bgr) == expected


# --- cv2_to_karnak_coord ---


def test_cv2_to_karnak_coord_normal():
    # [x_min, y_min, x_max, y_max] => "x y width height"
    assert cv2_to_karnak_coord([10, 20, 50, 80]) == "10 20 40 60"


def test_cv2_to_karnak_coord_float_coords_truncated():
    assert cv2_to_karnak_coord([10.7, 20.3, 50.9, 80.1]) == "10 20 40 59"


def test_cv2_to_karnak_coord_zero_origin():
    assert cv2_to_karnak_coord([0, 0, 30, 30]) == "0 0 30 30"


def test_cv2_to_karnak_coord_invalid_box_returns_empty():
    assert cv2_to_karnak_coord([1, 2, 3]) == ""


# --- format_boxes ---


def test_format_boxes_multiple():
    assert format_boxes([[10, 20, 50, 80], [0, 0, 30, 30]]) == [
        "10 20 40 60",
        "0 0 30 30",
    ]


def test_format_boxes_empty():
    assert format_boxes([]) == []


def test_format_boxes_invalid_box():
    assert format_boxes([[1, 2, 3]]) == [""]


# --- numpy_to_python_type ---


def test_numpy_to_python_type_array():
    result = numpy_to_python_type(np.array([1, 2, 3]))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_numpy_to_python_type_scalar():
    result = numpy_to_python_type(np.float32(3.0))
    assert isinstance(result, float)


def test_numpy_to_python_type_plain_int():
    assert numpy_to_python_type(42) == 42


def test_numpy_to_python_type_plain_string():
    assert numpy_to_python_type("hello") == "hello"
