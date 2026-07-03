import numpy as np
import numpy.testing as npt
import pytest

from image_ocr_identifier.color_detection import (
    _background_color,
    _decode_image,
    _polygon_to_bbox,
    get_colors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    """Return an (h, w, 3) uint8 BGR image filled with a single color."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# ---------------------------------------------------------------------------
# _decode_image
# ---------------------------------------------------------------------------


def test_decode_image_passthrough_ndarray():
    img = _solid_image(4, 4, (255, 255, 255))
    result = _decode_image(img)
    npt.assert_array_equal(result, img)


def test_decode_image_bytes_without_metadata_returns_none():
    # decode_image_bytes needs rows/columns/etc; raw bytes without them -> None
    result = _decode_image(b"raw bytes")
    assert result is None


# ---------------------------------------------------------------------------
# _polygon_to_bbox
# ---------------------------------------------------------------------------


def test_polygon_to_bbox_2d_array():
    poly = [[10, 20], [50, 20], [50, 80], [10, 80]]
    assert _polygon_to_bbox(poly) == (10, 20, 50, 80)


def test_polygon_to_bbox_flat_8_element():
    flat = [10, 20, 50, 20, 50, 80, 10, 80]
    assert _polygon_to_bbox(flat) == (10, 20, 50, 80)


def test_polygon_to_bbox_flat_4_element_axis_aligned():
    # 4-element flat: [x_min, y_min, x_max, y_max] passthrough
    assert _polygon_to_bbox([5, 10, 30, 40]) == (5, 10, 30, 40)


def test_polygon_to_bbox_numpy_2d():
    poly = np.array([[10, 20], [50, 20], [50, 80], [10, 80]])
    assert _polygon_to_bbox(poly) == (10, 20, 50, 80)


def test_polygon_to_bbox_numpy_flat_8():
    flat = np.array([10, 20, 50, 20, 50, 80, 10, 80])
    assert _polygon_to_bbox(flat) == (10, 20, 50, 80)


def test_polygon_to_bbox_invalid_length_raises():
    with pytest.raises(ValueError):
        _polygon_to_bbox([1, 2, 3])


# ---------------------------------------------------------------------------
# _background_color
# ---------------------------------------------------------------------------


def test_background_color_single_color():
    pixels = np.full((100, 3), fill_value=128, dtype=np.uint8)
    assert _background_color(pixels) == (128, 128, 128)


def test_background_color_excludes_text_pixels():
    # 80% white background, 20% black text -> should return white
    bg = np.full((80, 3), 255, dtype=np.uint8)
    text = np.zeros((20, 3), dtype=np.uint8)
    pixels = np.vstack([bg, text])
    assert _background_color(pixels) == (255, 255, 255)


def test_background_color_empty_returns_black():
    assert _background_color(np.empty((0, 3), dtype=np.uint8)) == (0, 0, 0)


def test_background_color_colored_background_with_text():
    # Blue background (BGR) with some red text pixels
    bg = np.array([[255, 0, 0]] * 70, dtype=np.uint8)
    text = np.array([[0, 0, 255]] * 30, dtype=np.uint8)
    pixels = np.vstack([bg, text])
    result = _background_color(pixels)
    assert result == (255, 0, 0)


# ---------------------------------------------------------------------------
# get_colors
# ---------------------------------------------------------------------------


def test_get_colors_empty_boxes_returns_empty():
    img = _solid_image(10, 10, (255, 255, 255))
    assert get_colors(img, []) == {}


def test_get_colors_none_boxes_returns_empty():
    img = _solid_image(10, 10, (255, 255, 255))
    assert get_colors(img, None) == {}


def test_get_colors_image_bytes_decode_failure_returns_empty():
    # Bytes without metadata -> _decode_image returns None -> empty dict
    assert get_colors(b"not an image", [[0, 0, 5, 5]]) == {}


def test_get_colors_single_box_axis_aligned():
    img = _solid_image(20, 20, (100, 150, 200))
    boxes = [[5, 5, 15, 15]]
    result = get_colors(img, boxes)
    assert len(result) == 1
    color = next(iter(result))
    assert result[color] == boxes


def test_get_colors_groups_boxes_by_color():
    # Top half white, bottom half black; one box in each half
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:10, :] = 255  # top half white
    box_top = [2, 2, 8, 8]  # interior is white
    box_bottom = [2, 12, 8, 18]  # interior is black
    result = get_colors(img, [box_top, box_bottom])
    assert len(result) == 2


def test_get_colors_out_of_bounds_box_skipped():
    img = _solid_image(10, 10, (0, 0, 0))
    # Box completely outside image bounds -> x1 <= x0 after clamping -> skipped
    result = get_colors(img, [[20, 20, 30, 30]])
    assert result == {}


def test_get_colors_invalid_box_format_skipped():
    img = _solid_image(10, 10, (0, 0, 0))
    valid = [2, 2, 8, 8]
    invalid = [1, 2, 3]  # 3-element flat -> ValueError in _polygon_to_bbox
    result = get_colors(img, [invalid, valid])
    # invalid box is skipped; valid box produces one entry
    assert len(result) == 1


def test_get_colors_numpy_box_converted_to_list():
    img = _solid_image(20, 20, (50, 50, 50))
    box = np.array([2, 2, 10, 10])
    result = get_colors(img, [box])
    color = next(iter(result))
    # Box stored as plain list, not numpy array
    assert isinstance(result[color][0], list)


def test_get_colors_polygon_box():
    img = _solid_image(20, 20, (80, 80, 80))
    poly = [[5, 5], [15, 5], [15, 15], [5, 15]]
    result = get_colors(img, [poly])
    assert len(result) == 1
