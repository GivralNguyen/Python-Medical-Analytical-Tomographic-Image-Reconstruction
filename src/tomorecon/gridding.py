"""Direct Fourier reconstruction (Fourier gridding) via the Fourier slice theorem."""
from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftshift, ifft2, ifftshift
from scipy.interpolate import RegularGridInterpolator

from ._utils import as_float_data, center_crop, default_theta

__all__ = [
    "frequency_axis",
    "projection_spectra",
    "polar_grid",
    "cartesian_grid",
    "polar_to_cartesian",
    "inverse_fft2",
    "fourier_gridding",
]


def frequency_axis(n: int) -> np.ndarray:
    """Angular frequencies (rad/sample) of an fftshift-ed length-n FFT; zero at index n // 2."""
    return (np.arange(n) - n // 2) * (2 * np.pi / n)


def projection_spectra(sinogram: np.ndarray, oversample: int = 1) -> np.ndarray:
    """
    1-D Fourier transform of every projection (along the detector axis).

    Parameters
    ----------
    sinogram : ndarray of shape (n_detectors, n_angles)
    oversample : int
        Zero-pad each projection to `oversample * n_detectors` first. This samples the
        Fourier slices more finely and greatly reduces gridding interpolation error
        (1 gives strong artifacts; 4 is about as accurate as FBP).

    Returns
    -------
    ndarray of shape (oversample * n_detectors, n_angles), complex
        fftshift-ed: zero frequency at row N // 2, frequencies given by `frequency_axis(N)`.
    """
    sinogram = as_float_data(sinogram)
    n0 = sinogram.shape[0]
    n = n0 * oversample
    padded = np.zeros((n, sinogram.shape[1]))
    start = n // 2 - n0 // 2                      # keep the rotation centre at row n // 2
    padded[start:start + n0] = sinogram
    return fftshift(fft(ifftshift(padded, axes=0), axis=0), axes=0)


def polar_grid(n: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theta_grid [rad], radius_grid [rad/sample]), both (n, len(theta)): where the slices are sampled."""
    return np.meshgrid(np.deg2rad(theta), frequency_axis(n))


def cartesian_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(kx, ky), both (n, n): kx varies along columns, ky along rows (rad/sample)."""
    return np.meshgrid(frequency_axis(n), frequency_axis(n))


def polar_to_cartesian(spectra: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Interpolate the polar samples of the 2-D Fourier transform onto the Cartesian grid.

    Fourier slice convention of `radon` (t = x cos(theta) - y sin(theta)):
        P_theta(w) = F(kx = w cos(theta), ky = -w sin(theta)).
    Each Cartesian point maps to theta = atan2(-ky, kx), w = |k|; angles below 0 use
    P_{theta+pi}(w) = P_theta(-w). Interpolation is bilinear in (w, theta), with the slice
    at 180 degrees appended so the 179 -> 0 degree wrap-around is handled.

    Parameters
    ----------
    spectra : ndarray of shape (N, n_angles)
        From `projection_spectra`.
    theta : 1-D array
        Projection angles in degrees, ascending, within [0, 180).

    Returns
    -------
    ndarray of shape (N, N), complex, fftshift-ed (DC at [N // 2, N // 2])
    """
    n = spectra.shape[0]
    w = frequency_axis(n)
    theta = np.deg2rad(np.asarray(theta, dtype=float))

    # slice at pi = slice at 0 mirrored (w -> -w maps row k to row 2*(n//2) - k)
    mirrored = spectra[(2 * (n // 2) - np.arange(n)) % n, 0]
    values = np.concatenate([spectra, mirrored[:, None]], axis=1)
    angles = np.append(theta, np.pi)

    kx, ky = cartesian_grid(n)
    angle = np.arctan2(-ky, kx)
    radius = np.hypot(kx, ky)
    negative = angle < 0
    angle[negative] += np.pi
    radius[negative] *= -1

    interpolate = RegularGridInterpolator((w, angles), values, bounds_error=False, fill_value=0)
    return interpolate(np.column_stack((radius.ravel(), angle.ravel()))).reshape(n, n)


def inverse_fft2(spectrum: np.ndarray, output_size: int | None = None) -> np.ndarray:
    """Real image from an fftshift-ed 2-D spectrum, optionally cropped to the centred output_size."""
    image = np.real(fftshift(ifft2(ifftshift(spectrum))))
    return image if output_size is None else center_crop(image, output_size)


def fourier_gridding(sinogram: np.ndarray, theta: np.ndarray | None = None,
                     output_size: int | None = None, oversample: int = 4) -> np.ndarray:
    """
    Direct Fourier reconstruction: 1-D FFT of each projection, polar-to-Cartesian
    interpolation of the slices, inverse 2-D FFT.

    Parameters
    ----------
    sinogram : ndarray of shape (n_detectors, n_angles)
    theta : 1-D array, optional
        Projection angles in degrees. Default: n_angles equally spaced angles in [0, 180).
    output_size : int, optional
        Side of the reconstructed image. Default: n_detectors.
    oversample : int
        Zero-padding factor of the projections, see `projection_spectra`.
    """
    theta = default_theta(np.shape(sinogram)[1]) if theta is None else theta
    spectrum = polar_to_cartesian(projection_spectra(sinogram, oversample), theta)
    return inverse_fft2(spectrum, output_size or np.shape(sinogram)[0])
