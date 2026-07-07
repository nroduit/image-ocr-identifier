import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def decode_image_bytes(
    image_bytes: bytes,
    rows: int | None = None,
    columns: int | None = None,
    bits_allocated: int | None = None,
    samples_per_pixel: int | None = None,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    window_center: float | None = None,
    window_width: float | None = None,
    is_monochrome1: bool = False,
    palette_color_lut: dict | None = None,
    transfer_syntax_uid: str | None = None,
    photometric_interpretation: str | None = None,
) -> np.ndarray | None:
    """Decode image bytes to a BGR numpy array.

    Uses pydicom to decompress compressed transfer syntaxes.
    For uncompressed data, interprets the bytes as raw pixels with optional
    modality LUT, VOI windowing, or palette color LUT.
    """
    if not (rows and columns and bits_allocated and samples_per_pixel):
        return None

    if transfer_syntax_uid:
        from pydicom.uid import UID

        if UID(transfer_syntax_uid).is_compressed:
            pixel_array = _decompress_with_pydicom(
                image_bytes,
                rows,
                columns,
                bits_allocated,
                samples_per_pixel,
                transfer_syntax_uid,
                photometric_interpretation,
            )
            if pixel_array is not None:
                return _pixel_array_to_bgr(
                    pixel_array,
                    palette_color_lut,
                    rescale_slope,
                    rescale_intercept,
                    window_center,
                    window_width,
                    is_monochrome1,
                )
            return None

    return _decode_raw_pixels(
        image_bytes,
        rows,
        columns,
        bits_allocated,
        samples_per_pixel,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        window_center=window_center,
        window_width=window_width,
        is_monochrome1=is_monochrome1,
        palette_color_lut=palette_color_lut,
    )


def _pixel_array_to_bgr(
    pixel_array: np.ndarray,
    palette_color_lut: dict | None,
    rescale_slope: float | None,
    rescale_intercept: float | None,
    window_center: float | None,
    window_width: float | None,
    is_monochrome1: bool,
) -> np.ndarray:
    if palette_color_lut:
        rgb = _apply_palette(pixel_array, palette_color_lut)
        if rgb is not None:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pixel_array = _apply_lut(
        pixel_array,
        rescale_slope,
        rescale_intercept,
        window_center,
        window_width,
        is_monochrome1,
    )
    return _to_bgr(pixel_array)


def _decompress_with_pydicom(
    compressed_bytes: bytes,
    rows: int,
    columns: int,
    bits_allocated: int,
    samples_per_pixel: int,
    transfer_syntax_uid: str,
    photometric_interpretation: str,
) -> np.ndarray | None:
    """Decompress pixel data using pydicom + GDCM."""
    from pydicom.dataset import Dataset
    from pydicom.encaps import encapsulate
    from pydicom.uid import UID

    ds = Dataset()
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = samples_per_pixel
    ds.NumberOfFrames = 1
    ds.PhotometricInterpretation = photometric_interpretation
    if samples_per_pixel != 1:
        ds.PlanarConfiguration = 0

    ds.PixelData = encapsulate([compressed_bytes])
    ds["PixelData"].is_undefined_length = True
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = UID(transfer_syntax_uid)

    try:
        return ds.pixel_array
    except Exception:
        logger.debug("pydicom decompression failed for %s", transfer_syntax_uid)
        return None


def _decode_raw_pixels(
    raw_bytes: bytes,
    rows: int,
    columns: int,
    bits_allocated: int,
    samples_per_pixel: int,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    window_center: float | None = None,
    window_width: float | None = None,
    is_monochrome1: bool = False,
    palette_color_lut: dict | None = None,
) -> np.ndarray | None:
    """Interpret raw pixel bytes as an image using the provided DICOM-style metadata."""
    dtype = np.uint8 if bits_allocated <= 8 else np.uint16
    expected_size = rows * columns * samples_per_pixel * (bits_allocated // 8)

    if len(raw_bytes) != expected_size:
        logger.debug(
            "Raw pixel size mismatch: got %d bytes, expected %d (%dx%d, %dbit, %dch)",
            len(raw_bytes),
            expected_size,
            columns,
            rows,
            bits_allocated,
            samples_per_pixel,
        )
        return None

    pixel_array = np.frombuffer(raw_bytes, dtype=dtype).reshape(
        (rows, columns, samples_per_pixel) if samples_per_pixel > 1 else (rows, columns)
    )

    return _pixel_array_to_bgr(
        pixel_array,
        palette_color_lut,
        rescale_slope,
        rescale_intercept,
        window_center,
        window_width,
        is_monochrome1,
    )


def _apply_palette(
    pixel_array: np.ndarray,
    palette: dict,
) -> np.ndarray | None:
    """Map indexed pixel values to RGB using palette color LUT tables.

    Expects palette to have "red", "green", "blue" keys, each an array of
    integer values. Entries can be 8-bit (0-255) or 16-bit (0-65535).
    """
    try:
        red = np.asarray(palette["red"])
        green = np.asarray(palette["green"])
        blue = np.asarray(palette["blue"])
    except (KeyError, TypeError):
        logger.debug("Invalid palette_color_lut: missing red/green/blue keys")
        return None

    # Normalize 16-bit palette entries to 8-bit
    for lut in (red, green, blue):
        if lut.max() > 255:
            break
    else:
        lut = None
    if lut is not None and lut.max() > 255:
        red = (red / 65535 * 255).astype(np.uint8)
        green = (green / 65535 * 255).astype(np.uint8)
        blue = (blue / 65535 * 255).astype(np.uint8)
    else:
        red = red.astype(np.uint8)
        green = green.astype(np.uint8)
        blue = blue.astype(np.uint8)

    indices = pixel_array.ravel().astype(np.intp)
    max_index = len(red) - 1
    indices = np.clip(indices, 0, max_index)

    rgb = np.stack([red[indices], green[indices], blue[indices]], axis=-1)
    return rgb.reshape((*pixel_array.shape, 3))


def _apply_lut(
    pixel_array: np.ndarray,
    rescale_slope: float | None,
    rescale_intercept: float | None,
    window_center: float | None,
    window_width: float | None,
    is_monochrome1: bool,
) -> np.ndarray:
    """Apply modality LUT, VOI windowing,
    and photometric inversion to produce a uint8 array."""
    arr = pixel_array.astype(np.float64)

    # Modality LUT: stored pixel value -> modality unit (e.g. Hounsfield)
    if rescale_slope is not None and rescale_intercept is not None:
        arr = arr * rescale_slope + rescale_intercept

    # VOI LUT: modality value -> display value (0-255)
    if window_center is not None and window_width is not None:
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        arr = np.clip((arr - lower) / (upper - lower) * 255, 0, 255)
    else:
        min_val, max_val = float(arr.min()), float(arr.max())
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val) * 255
        else:
            arr = np.zeros_like(arr)

    if is_monochrome1:
        arr = 255 - arr

    return arr.astype(np.uint8)


def _to_bgr(pixel_array: np.ndarray) -> np.ndarray:
    """Convert a uint8 pixel array to BGR for OpenCV."""
    if pixel_array.ndim == 2:
        return cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
    elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
        return cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
    return pixel_array
