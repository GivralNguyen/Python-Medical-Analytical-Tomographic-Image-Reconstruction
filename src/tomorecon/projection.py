"""Forward model: the 2-D parallel-beam Radon transform."""
from __future__ import annotations

from warnings import warn

import numpy as np
from skimage.transform import warp

from ._utils import as_float_image, default_theta

__all__ = ["radon"]


def _pad_to_square(image: np.ndarray, circle: bool) -> np.ndarray:
    if circle:
        # Object must lie inside the inscribed circle; crop to the largest centred square.
        size = min(image.shape)
        radius = size // 2
        rows, cols = np.ogrid[:image.shape[0], :image.shape[1]]
        dist2 = (rows - image.shape[0] // 2) ** 2 + (cols - image.shape[1] // 2) ** 2
        if np.any(image[dist2 > radius ** 2]):
            warn("radon: image must be zero outside the reconstruction circle", stacklevel=3)
        starts = [int(np.ceil((s - size) / 2)) for s in image.shape]
        return image[starts[0]:starts[0] + size, starts[1]:starts[1] + size]

    # Pad so the whole image fits inside the inscribed circle of the padded square.
    diagonal = np.sqrt(2) * max(image.shape)
    pad = [int(np.ceil(diagonal - s)) for s in image.shape]
    before = [(s + p) // 2 - s // 2 for s, p in zip(image.shape, pad)]
    return np.pad(image, [(b, p - b) for b, p in zip(before, pad)], mode="constant")


def radon(image: np.ndarray, theta: np.ndarray | None = None, circle: bool = True) -> np.ndarray:
    """
    Radon transform: line integrals of `image` along parallel rays at each angle.

    For each angle the image is rotated about its centre (bilinear interpolation) and
    summed along the columns. The detector coordinate is t = x cos(theta) - y sin(theta)
    with x = column and y = row, both relative to the centre pixel.

    Parameters
    ----------
    image : 2-D array
    theta : 1-D array, optional
        Projection angles in degrees. Default: 0, 1, ..., 179.
    circle : bool
        If True (default), assume the object lies inside the circle inscribed in the image;
        the sinogram then has as many detector bins as the image side. If False, the image
        is zero-padded so that nothing is cut off (sinogram length ~ sqrt(2) * side).

    Returns
    -------
    sinogram : ndarray of shape (n_detectors, len(theta))
        One projection per column; the rotation centre is at row n_detectors // 2.
    """
    image = as_float_image(image)
    if image.ndim != 2:
        raise ValueError("radon: image must be 2-D")
    theta = default_theta(180) if theta is None else np.asarray(theta, dtype=float)

    padded = _pad_to_square(image, circle)
    center = padded.shape[0] // 2
    sinogram = np.zeros((padded.shape[0], len(theta)), dtype=padded.dtype)
    for i, angle in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                             [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                             [0, 0, 1]])
        sinogram[:, i] = warp(padded, rotation, clip=False).sum(axis=0)
    return sinogram
