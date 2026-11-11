"""Reconstruction filters: 1-D ramp-type filters, their spatial kernels, and the 2-D cone filter."""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, fft2, fftfreq, fftshift, ifft, ifft2

from ._utils import as_float_data, center_crop

__all__ = [
    "FILTERS",
    "padded_size",
    "fourier_filter",
    "spatial_kernel",
    "filter_sinogram",
    "convolve_sinogram",
    "cone_filter",
]

FILTERS = ("ramp", "shepp-logan", "cosine", "hamming", "hann")


def _check(name):
    if name is not None and name not in FILTERS:
        raise ValueError(f"Unknown filter {name!r}; choose one of {FILTERS} or None")


def padded_size(n: int) -> int:
    """Smallest power of two >= 2n (and >= 64): FFT filtering at this length is a linear convolution."""
    return max(64, int(2 ** np.ceil(np.log2(2 * n))))


def fourier_filter(size: int, name: str | None = "ramp") -> np.ndarray:
    """
    Frequency response of a reconstruction filter, in FFT order (DC first).

    The ramp is built as the DFT of the band-limited spatial ramp kernel
    (h[0] = 1/4, h[odd n] = -1/(pi n)^2), which avoids the DC error of a sampled |f|.
    It equals 2|f| (f in cycles/sample), i.e. 1 at Nyquist. Other filters are the ramp
    times an apodizing window.

    Parameters
    ----------
    size : int
        Filter length (use `padded_size(n_detectors)`).
    name : {'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'} or None
        None returns an all-pass (unity) filter.

    Returns
    -------
    ndarray of shape (size,)
    """
    _check(name)
    if name is None:
        return np.ones(size)

    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    h = np.zeros(size)
    h[0] = 0.25
    h[1::2] = -1 / (np.pi * n) ** 2
    response = 2 * np.real(fft(h))

    if name == "shepp-logan":
        omega = np.pi * fftfreq(size)[1:]
        response[1:] *= np.sin(omega) / omega
    elif name == "cosine":
        response *= fftshift(np.sin(np.linspace(0, np.pi, size, endpoint=False)))
    elif name == "hamming":
        response *= fftshift(np.hamming(size))
    elif name == "hann":
        response *= fftshift(np.hanning(size))
    return response


def spatial_kernel(n_detectors: int, name: str | None = "ramp") -> np.ndarray:
    """
    Spatial convolution kernel equivalent to `fourier_filter` (for convolve-backproject).

    Covers lags -(n-1) ... (n-1), every lag that occurs when convolving a projection of
    length n, so convolving with it equals FFT filtering with zero-padding.

    Returns
    -------
    ndarray of shape (2 * n_detectors - 1,), lag 0 at index n_detectors - 1
    """
    size = padded_size(n_detectors)
    h = np.real(ifft(fourier_filter(size, name)))
    lags = np.arange(-(n_detectors - 1), n_detectors)
    return h[lags % size]


def filter_sinogram(sinogram: np.ndarray, name: str | None = "ramp") -> np.ndarray:
    """
    Filter every projection (column) in the frequency domain.

    Projections are zero-padded to `padded_size` so the product of FFTs is a linear,
    not circular, convolution.
    """
    sinogram = as_float_data(sinogram)
    n = sinogram.shape[0]
    size = padded_size(n)
    spectrum = fft(sinogram, n=size, axis=0) * fourier_filter(size, name)[:, None]
    return np.real(ifft(spectrum, axis=0))[:n]


def convolve_sinogram(sinogram: np.ndarray, name: str | None = "ramp") -> np.ndarray:
    """
    Filter every projection (column) by spatial convolution with `spatial_kernel`.
    Gives the same result as `filter_sinogram` (to floating-point precision).
    """
    sinogram = as_float_data(sinogram)
    n = sinogram.shape[0]
    kernel = spatial_kernel(n, name)
    # full convolution has length 3n-2; output sample i sits at index i + (n-1)
    return np.stack([np.convolve(column, kernel)[n - 1:2 * n - 1] for column in sinogram.T], axis=1)


def _window(freq: np.ndarray, name: str | None) -> np.ndarray:
    """Apodizing window at |freq| (cycles/sample); beyond Nyquist it is held at its Nyquist value."""
    _check(name)
    f = np.minimum(np.abs(freq), 0.5)
    if name in ("ramp", None):
        return np.ones_like(f)
    if name == "shepp-logan":
        return np.sinc(f)
    if name == "cosine":
        return np.cos(np.pi * f)
    if name == "hamming":
        return 0.54 + 0.46 * np.cos(2 * np.pi * f)
    return 0.5 + 0.5 * np.cos(2 * np.pi * f)  # hann


def cone_filter(image: np.ndarray, name: str | None = "ramp",
                output_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the 2-D cone filter 2|rho| (optionally apodized) to an unfiltered backprojection.

    Unfiltered backprojection is the object convolved with 1/r, whose 2-D Fourier transform
    is 1/|rho|; multiplying by |rho| undoes it. The factor 2 matches the scaling of the
    1-D ramp in `fourier_filter`.

    Parameters
    ----------
    image : square 2-D array
        Unfiltered backprojection. The 1/r blur reaches far outside the object, so compute it
        on a larger grid than needed (e.g. 2x) and crop with `output_size`.
    name : filter name, see `FILTERS`
    output_size : int, optional
        Side of the centred crop to return. Default: full size.

    Returns
    -------
    spectrum : ndarray
        Filtered 2-D spectrum, fftshift-ed (DC in the centre), for display.
    filtered : ndarray
        Filtered image, cropped to `output_size`.
    """
    size = image.shape[0]
    u, v = np.meshgrid(fftfreq(size), fftfreq(size))
    rho = np.hypot(u, v)
    spectrum = fft2(image) * (2 * rho * _window(rho, name))
    filtered = np.real(ifft2(spectrum))
    return fftshift(spectrum), center_crop(filtered, output_size or size)
