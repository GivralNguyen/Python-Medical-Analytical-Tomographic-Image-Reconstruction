from __future__ import annotations

import numpy as np
from skimage.util import img_as_float


def as_float_image(image) -> np.ndarray:
    """Image as float; integer/bool images are rescaled to [0, 1] like skimage's `img_as_float`."""
    image = np.asarray(image)
    if image.dtype == np.float16:
        return image.astype(np.float32)
    return img_as_float(image)


def as_float_data(data) -> np.ndarray:
    """Measured data (e.g. a sinogram) as float, keeping its values."""
    data = np.asarray(data)
    if data.dtype.char not in "df":
        data = data.astype(np.float32 if data.dtype == np.float16 else float)
    return data


def default_theta(n_angles: int) -> np.ndarray:
    """`n_angles` equally spaced angles in [0, 180) degrees."""
    return np.linspace(0, 180, n_angles, endpoint=False)


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Centred size x size crop of a square image (centre pixel at index shape // 2)."""
    start = image.shape[0] // 2 - size // 2
    return image[start:start + size, start:start + size]
