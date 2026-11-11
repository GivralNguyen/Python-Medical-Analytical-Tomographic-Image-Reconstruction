"""Test objects: the Shepp-Logan phantom, a disk, or any image file."""
from __future__ import annotations

from importlib.resources import as_file, files

import imageio.v3 as iio
import numpy as np
from skimage.transform import rescale, resize

__all__ = ["shepp_logan", "disk", "load_image"]


def shepp_logan(size: int = 256) -> np.ndarray:
    """
    Modified Shepp-Logan head phantom (bundled with the package).

    Parameters
    ----------
    size : int
        Side length of the returned square image.

    Returns
    -------
    ndarray of shape (size, size), float, values in [0, 1]
    """
    with as_file(files("tomorecon") / "data" / "shepp_logan_phantom.png") as path:
        image = iio.imread(path, mode="L")
    return rescale(image, scale=size / image.shape[0], mode="reflect", channel_axis=None)


def disk(shape: tuple[int, int] = (120, 120), radius: float = 20,
         center: tuple[float, float] = (100, 60)) -> np.ndarray:
    """
    Binary disk, useful for seeing how a single object traces a sine curve in the sinogram.

    Parameters
    ----------
    shape : (rows, cols)
    radius : float
    center : (x, y)
        Disk centre as (column, row).

    Returns
    -------
    ndarray of shape `shape`, float, 1 inside the disk and 0 outside
    """
    y, x = np.mgrid[:shape[0], :shape[1]]
    return (((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius ** 2).astype(float)


def load_image(path, size: int = 256) -> np.ndarray:
    """
    Load an image file as grayscale and resize it to (size, size).

    Returns
    -------
    ndarray of shape (size, size), float, values in [0, 1]
    """
    image = iio.imread(path, mode="L")
    return resize(image, (size, size), mode="reflect", anti_aliasing=True)
