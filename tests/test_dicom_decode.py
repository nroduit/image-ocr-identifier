from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from image_ocr_identifier.dicom_decode import (
    _apply_lut,
    _apply_palette,
    _decode_raw_pixels,
    _decompress_with_pydicom,
    _to_bgr,
    decode_image_bytes,
)

# --- _apply_lut ---


def test_apply_lut_auto_minmax_stretch():
    arr = np.array([[0.0, 100.0]])
    result = _apply_lut(arr, None, None, None, None, False)
    npt.assert_array_equal(result, [[0, 255]])


def test_apply_lut_uniform_image_returns_zeros():
    arr = np.array([[50.0, 50.0]])
    result = _apply_lut(arr, None, None, None, None, False)
    npt.assert_array_equal(result, [[0, 0]])


def test_apply_lut_windowing():
    # window_center=128, window_width=256 -> lower=0, upper=256
    arr = np.array([[0.0, 256.0]])
    result = _apply_lut(arr, None, None, 128.0, 256.0, False)
    npt.assert_array_equal(result, [[0, 255]])


def test_apply_lut_windowing_clips_out_of_range():
    arr = np.array([[-100.0, 500.0]])
    result = _apply_lut(arr, None, None, 128.0, 256.0, False)
    npt.assert_array_equal(result, [[0, 255]])


def test_apply_lut_slope_intercept():
    arr = np.array([[0.0, 50.0]])
    # slope=2 -> [0., 100.]; auto min-max -> [0, 255]
    result = _apply_lut(arr, 2.0, 0.0, None, None, False)
    npt.assert_array_equal(result, [[0, 255]])


def test_apply_lut_monochrome1_inverts():
    arr = np.array([[0.0, 256.0]])
    # windowed: [0, 255]; inverted: [255, 0]
    result = _apply_lut(arr, None, None, 128.0, 256.0, True)
    npt.assert_array_equal(result, [[255, 0]])


# --- _apply_palette ---


def test_apply_palette_8bit():
    palette = {"red": [255, 0], "green": [0, 255], "blue": [0, 0]}
    pixel_array = np.array([[0, 1]])
    result = _apply_palette(pixel_array, palette)
    expected = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    npt.assert_array_equal(result, expected)


def test_apply_palette_16bit_normalized_to_8bit():
    palette = {"red": [65535, 0], "green": [0, 0], "blue": [0, 0]}
    result = _apply_palette(np.array([[0]]), palette)
    assert result[0, 0, 0] == 255
    assert result[0, 0, 1] == 0
    assert result[0, 0, 2] == 0


def test_apply_palette_missing_key_returns_none():
    assert _apply_palette(np.array([[0]]), {"red": [255]}) is None


def test_apply_palette_none_input_returns_none():
    assert _apply_palette(np.array([[0]]), None) is None


# --- _to_bgr ---


def test_to_bgr_grayscale_2d():
    gray = np.array([[100, 200]], dtype=np.uint8)
    result = _to_bgr(gray)
    assert result.shape == (1, 2, 3)
    npt.assert_array_equal(result[:, :, 0], gray)
    npt.assert_array_equal(result[:, :, 1], gray)
    npt.assert_array_equal(result[:, :, 2], gray)


def test_to_bgr_rgb_3d():
    rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red in RGB
    result = _to_bgr(rgb)
    # R and B channels swapped: (255,0,0) RGB -> (0,0,255) BGR
    npt.assert_array_equal(result, [[[0, 0, 255]]])


def test_to_bgr_other_shape_passthrough():
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    result = _to_bgr(arr)
    npt.assert_array_equal(result, arr)


# --- _decode_raw_pixels ---


def test_decode_raw_pixels_grayscale():
    raw = np.array([[0, 128], [64, 255]], dtype=np.uint8).tobytes()
    result = _decode_raw_pixels(
        raw, rows=2, columns=2, bits_allocated=8, samples_per_pixel=1
    )
    assert result is not None
    assert result.shape == (2, 2, 3)


def test_decode_raw_pixels_rgb():
    raw = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
    result = _decode_raw_pixels(
        raw, rows=2, columns=2, bits_allocated=8, samples_per_pixel=3
    )
    assert result is not None
    assert result.shape == (2, 2, 3)


def test_decode_raw_pixels_16bit():
    raw = np.zeros((2, 2), dtype=np.uint16).tobytes()
    result = _decode_raw_pixels(
        raw, rows=2, columns=2, bits_allocated=16, samples_per_pixel=1
    )
    assert result is not None
    assert result.shape == (2, 2, 3)


def test_decode_raw_pixels_size_mismatch_returns_none():
    result = _decode_raw_pixels(
        b"short", rows=2, columns=2, bits_allocated=8, samples_per_pixel=1
    )
    assert result is None


def test_decode_raw_pixels_with_valid_palette():
    raw = np.array([[0, 1]], dtype=np.uint8).tobytes()
    palette = {"red": [255, 0], "green": [0, 255], "blue": [0, 0]}
    result = _decode_raw_pixels(
        raw,
        rows=1,
        columns=2,
        bits_allocated=8,
        samples_per_pixel=1,
        palette_color_lut=palette,
    )
    assert result is not None
    assert result.shape == (1, 2, 3)
    npt.assert_array_equal(result[0, 0], [0, 0, 255])  # (255,0,0) RGB -> (0,0,255) BGR
    npt.assert_array_equal(result[0, 1], [0, 255, 0])  # (0,255,0) RGB -> (0,255,0) BGR


def test_decode_raw_pixels_invalid_palette_falls_back_to_lut():
    raw = np.zeros((2, 2), dtype=np.uint8).tobytes()
    result = _decode_raw_pixels(
        raw,
        rows=2,
        columns=2,
        bits_allocated=8,
        samples_per_pixel=1,
        palette_color_lut={"bad": "palette"},
    )
    assert result is not None
    assert result.shape == (2, 2, 3)


# --- _decompress_with_pydicom ---


def test_decompress_with_pydicom_invalid_bytes_returns_none():
    # Invalid bytes cause pydicom to raise; the function catches and returns None
    result = _decompress_with_pydicom(
        b"not valid compressed data",
        rows=4,
        columns=4,
        bits_allocated=8,
        samples_per_pixel=1,
        transfer_syntax_uid="1.2.840.10008.1.2.4.70",  # JPEG Lossless
        photometric_interpretation="MONOCHROME2",
    )
    assert result is None


# --- decode_image_bytes ---


def test_decode_image_bytes_all_params_missing():
    assert decode_image_bytes(b"data") is None


def test_decode_image_bytes_partial_params():
    assert decode_image_bytes(b"data", rows=4, columns=4) is None


def test_decode_image_bytes_raw_grayscale():
    raw = np.zeros((4, 4), dtype=np.uint8).tobytes()
    result = decode_image_bytes(
        raw, rows=4, columns=4, bits_allocated=8, samples_per_pixel=1
    )
    assert result is not None
    assert result.shape == (4, 4, 3)


def test_decode_image_bytes_non_compressed_transfer_syntax_falls_through_to_raw():
    # Explicit VR Little Endian (1.2.840.10008.1.2.1) is uncompressed
    # -> falls through to raw path
    raw = np.zeros((4, 4), dtype=np.uint8).tobytes()
    result = decode_image_bytes(
        raw,
        rows=4,
        columns=4,
        bits_allocated=8,
        samples_per_pixel=1,
        transfer_syntax_uid="1.2.840.10008.1.2.1",
    )
    assert result is not None
    assert result.shape == (4, 4, 3)


def test_decode_image_bytes_compressed_success():
    fake_pixel_array = np.zeros((4, 4), dtype=np.uint8)
    with patch(
        "image_ocr_identifier.dicom_decode._decompress_with_pydicom",
        return_value=fake_pixel_array,
    ):
        result = decode_image_bytes(
            b"fake compressed",
            rows=4,
            columns=4,
            bits_allocated=8,
            samples_per_pixel=1,
            transfer_syntax_uid="1.2.840.10008.1.2.4.70",  # JPEG Lossless (compressed)
        )
    assert result is not None
    assert result.shape == (4, 4, 3)


def test_decode_image_bytes_compressed_decompression_failure():
    with patch(
        "image_ocr_identifier.dicom_decode._decompress_with_pydicom",
        return_value=None,
    ):
        result = decode_image_bytes(
            b"fake compressed",
            rows=4,
            columns=4,
            bits_allocated=8,
            samples_per_pixel=1,
            transfer_syntax_uid="1.2.840.10008.1.2.4.70",
        )
    assert result is None
