from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import CLASS_PALETTE, INTERNAL_PALETTE

IntGrid = NDArray[np.int_]
RgbImage = NDArray[np.uint8]


def grid_to_image(grid: IntGrid, *, use_class_palette: bool = False, scale: int = 12) -> RgbImage:
    palette = CLASS_PALETTE if use_class_palette else INTERNAL_PALETTE
    image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    for value, color in palette.items():
        image[grid == value] = np.array(color, dtype=np.uint8)
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return image


def confidence_to_image(confidence: NDArray[np.float64], *, scale: int = 12) -> RgbImage:
    normalized = np.clip(confidence, 0.0, 1.0)
    image = np.zeros((confidence.shape[0], confidence.shape[1], 3), dtype=np.uint8)
    image[..., 0] = (255 * (1.0 - normalized)).astype(np.uint8)
    image[..., 1] = (180 * normalized + 40).astype(np.uint8)
    image[..., 2] = (120 * normalized + 40).astype(np.uint8)
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return image


def mask_to_image(mask: NDArray[np.bool_], *, scale: int = 12) -> RgbImage:
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    image[mask] = np.array([91, 186, 109], dtype=np.uint8)
    image[~mask] = np.array([49, 55, 61], dtype=np.uint8)
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return image
