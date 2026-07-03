import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_image_for_ocr(image: np.ndarray) -> tuple[np.ndarray, float]:
    # Apply the preprocess pipeline
    # Upscaling
    image_processed, scale_factor = upscale_image(image, min_side=1500)
    # CLAHE
    image_processed = apply_clahe(image_processed, clip_limit=2.0, tile_size=8)
    # Unsharp masking
    image_processed = apply_unsharp_mask(image_processed, sigma=1.0, strength=1.5)
    return image_processed, scale_factor


def upscale_image(image: np.ndarray, min_side: int = 1500) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if min(height, width) >= min_side:
        logger.debug("Image is already large enough, skipping upscaling.")
        return image, 1.0
    scale_factor = min_side / min(height, width)
    logger.debug(
        "Upscaling image with scale factor %.2f to meet minimum side length of %d",
        scale_factor,
        min_side,
    )
    return (
        cv2.resize(
            image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
        ),
        scale_factor,
    )


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    image: np.ndarray, sigma: float = 1.0, strength: float = 1.5
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
