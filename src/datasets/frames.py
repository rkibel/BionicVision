"""Frame discovery and image I/O utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as grayscale uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_gray(path: str | Path, image: np.ndarray) -> Path:
    """Save a grayscale-compatible array as uint8."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, normalize_to_uint8(image))
    return output_path


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """Normalize any numeric array into the 0-255 uint8 display range."""

    array = np.asarray(values)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    min_value = float(array[finite].min())
    max_value = float(array[finite].max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
