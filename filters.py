import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift


def _padded_size(size):
    """Smallest power of two >= 2*size (and >= 64), so FFT-based filtering is a linear convolution."""
    return max(64, int(2 ** np.ceil(np.log2(2 * size))))


def _frequency_window(freq, filter_name):
    """
    Apodizing window evaluated at frequency `freq` (cycles/sample, |freq| <= 0.5),
    matching the 1-D windows used in `_get_fourier_filter`. Frequencies beyond Nyquist
    (corners of a 2-D spectrum) are clamped to the Nyquist value.
    """
    f = np.minimum(np.abs(freq), 0.5)
    if filter_name in ("ramp", None):
        return np.ones_like(f)
    if filter_name == "shepp-logan":
        return np.sinc(f)                       # sin(pi f) / (pi f)
    if filter_name == "cosine":
        return np.cos(np.pi * f)
    if filter_name == "hamming":
        return 0.54 + 0.46 * np.cos(2 * np.pi * f)
    if filter_name == "hann":
        return 0.5 + 0.5 * np.cos(2 * np.pi * f)
    raise ValueError(f"Unknown filter: {filter_name}")


def cone_filter(backprojected, output_size=None, filter_name="ramp"):
    """
    Backproject-filter: apply the 2-D cone filter |rho| to an unfiltered backprojection.

    Unfiltered backprojection equals f convolved with 1/r, whose 2-D Fourier transform
    is F(rho)/|rho|. Multiplying the spectrum of the backprojected image by |rho|
    (optionally apodized by a window) therefore recovers f.

    Parameters:
    - backprojected (numpy.ndarray): Unfiltered backprojection (square). Because the 1/r blur
      extends outside the object, it should be computed on a grid larger than the desired
      output (e.g. 2x) and cropped afterwards via `output_size`.
    - output_size (int, optional): Size of the centred crop returned. Default: full size.
    - filter_name (str, optional): Apodizing window: 'ramp' (none), 'shepp-logan', 'cosine',
      'hamming', 'hann'.

    Returns:
    - modified_spectrum (numpy.ndarray): Filtered 2-D spectrum, fftshift-ed (DC in the centre) for display.
    - reconstructed_image (numpy.ndarray): Filtered image, cropped to `output_size`.

    Example:
    >>> _, bp = backprojection(sinogram, theta, filter=None, output_size=2 * n)
    >>> spectrum, image = cone_filter(bp, output_size=n)
    """
    M = backprojected.shape[0]
    if output_size is None:
        output_size = M

    # 2-D frequency grid in cycles/sample, in the same (unshifted) layout as fft2's output
    u, v = np.meshgrid(fftfreq(M), fftfreq(M))
    rho = np.hypot(u, v)
    # factor 2 matches the scaling of the 1-D ramp in `_get_fourier_filter` (2|f|)
    cone = 2 * rho * _frequency_window(rho, filter_name)

    spectrum = fft2(backprojected) * cone
    image = np.real(ifft2(spectrum))

    c0 = M // 2 - output_size // 2
    return fftshift(spectrum), image[c0:c0 + output_size, c0:c0 + output_size]


def _get_fourier_filter(size, filter_name):
    """
    Generate a Fourier filter for image processing.

    Parameters:
    - size (int): Size of the filter.
    - filter_name (str): Name of the filter. Options: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None.

    Returns:
    - fourier_filter (numpy.ndarray): 1D array representing the Fourier filter.

    Notes:
    - The function generates a Fourier filter based on the specified name.
    - The available filter options are 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', and None.
    - If None is provided, a unity filter is returned.
    - The generated filter is suitable for use in image processing operations.

    Example:
    >>> size = 256
    >>> filter_name = 'ramp'
    >>> fourier_filter = _get_fourier_filter(size, filter_name)
    """
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(fft(f))         # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]


def _get_inverse_fourier_filter(size, filter_name):
    """
    Spatial-domain convolution kernel equivalent to `_get_fourier_filter` (for Convolve-Backproject).

    The kernel is obtained by inverse-FFT of the filter at the zero-padded size used by
    `backprojection`, and covers lags -(size-1) ... +(size-1), which is every lag that can
    occur when convolving a projection of length `size`.

    Returns:
    - kernel (numpy.ndarray): 1-D kernel of length 2*size-1; lag 0 is at index size-1.
    """
    padded_size = _padded_size(size)
    h = np.real(ifft(_get_fourier_filter(padded_size, filter_name).ravel()))
    lags = np.arange(-(size - 1), size)
    return h[lags % padded_size]


def convolve_sinogram(sinogram, filter_name):
    """
    Filter every projection (column of the sinogram) by spatial convolution with the
    kernel from `_get_inverse_fourier_filter`. Identical to the FFT filtering in
    `backprojection(..., filter=filter_name)`, but done in the spatial domain.
    """
    size = sinogram.shape[0]
    kernel = _get_inverse_fourier_filter(size, filter_name)
    # full convolution has length 3*size-2; output sample i sits at index i + (size-1)
    return np.stack([np.convolve(sinogram[:, j], kernel)[size - 1:2 * size - 1]
                     for j in range(sinogram.shape[1])], axis=1)
