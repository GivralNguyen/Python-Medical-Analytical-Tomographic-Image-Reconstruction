"""Backprojection-based reconstruction: BP, FBP, CBP and BPF."""
from __future__ import annotations

import numpy as np

from ._utils import as_float_data, default_theta
from .filters import cone_filter, convolve_sinogram, filter_sinogram

__all__ = ["backproject", "fbp", "cbp", "bpf"]


def _angles(sinogram, theta):
    if theta is None:
        return default_theta(sinogram.shape[1])
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (sinogram.shape[1],):
        raise ValueError(f"theta has {theta.size} angles but the sinogram has {sinogram.shape[1]} columns")
    return theta


def backproject(sinogram: np.ndarray, theta: np.ndarray | None = None,
                output_size: int | None = None, interval: int = 1) -> np.ndarray:
    """
    Unfiltered backprojection: smear every projection back across the image plane.

    Parameters
    ----------
    sinogram : ndarray of shape (n_detectors, n_angles)
        As returned by `radon`; rotation centre at row n_detectors // 2.
    theta : 1-D array, optional
        Projection angles in degrees. Default: n_angles equally spaced angles in [0, 180).
    output_size : int, optional
        Side of the reconstructed image. Default: n_detectors.
    interval : int
        Use only every `interval`-th projection (to show the effect of fewer angles).

    Returns
    -------
    ndarray of shape (output_size, output_size)
        Scaled by pi / (2 * angles used), so that backprojecting a ramp-filtered
        sinogram reconstructs the object at its original intensity.
    """
    sinogram = as_float_data(sinogram)
    theta = _angles(sinogram, theta)
    n = sinogram.shape[0]
    output_size = output_size or n

    rows, cols = np.mgrid[:output_size, :output_size] - output_size // 2
    detector = np.arange(n) - n // 2
    image = np.zeros((output_size, output_size))
    used = range(0, len(theta), interval)
    for i in used:
        angle = np.deg2rad(theta[i])
        t = cols * np.cos(angle) - rows * np.sin(angle)
        image += np.interp(t, detector, sinogram[:, i], left=0, right=0)
    return image * np.pi / (2 * len(used))


def fbp(sinogram: np.ndarray, theta: np.ndarray | None = None, filter: str | None = "ramp",
        output_size: int | None = None) -> np.ndarray:
    """
    Filtered backprojection: filter every projection in the frequency domain, then backproject.

    `filter` is one of `FILTERS` ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann');
    smoother windows trade resolution for less noise. See `backproject` for the other arguments.
    """
    return backproject(filter_sinogram(sinogram, filter), theta, output_size)


def cbp(sinogram: np.ndarray, theta: np.ndarray | None = None, filter: str | None = "ramp",
        output_size: int | None = None) -> np.ndarray:
    """
    Convolve-backproject: like `fbp`, but the filter is applied as a spatial convolution
    of each projection. The result is identical to `fbp` up to floating-point error.
    """
    return backproject(convolve_sinogram(sinogram, filter), theta, output_size)


def bpf(sinogram: np.ndarray, theta: np.ndarray | None = None, filter: str | None = "ramp",
        output_size: int | None = None, grid_factor: int = 2) -> np.ndarray:
    """
    Backproject-filter: backproject first, then remove the 1/r blur with the 2-D cone filter.

    Parameters
    ----------
    grid_factor : int
        Backproject onto a grid `grid_factor` times larger than `output_size` before
        filtering, because the 1/r blur extends far outside the object (1 gives visible
        errors; 2 is enough in practice).

    See `backproject` and `fbp` for the other arguments.
    """
    output_size = output_size or np.asarray(sinogram).shape[0]
    blurred = backproject(sinogram, theta, grid_factor * output_size)
    return cone_filter(blurred, filter, output_size)[1]
