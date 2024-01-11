import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from numpy.fft import ifftshift
def cone_filter(frequency_spectrum, u, v, window_size=15, angular_weighting=None):
    """
    Apply a cone filter to the frequency spectrum of an image.

    Parameters:
    - frequency_spectrum (numpy.ndarray): 2D array representing the frequency spectrum of an image.
    - u (numpy.ndarray): 2D array representing the x-axis frequencies.
    - v (numpy.ndarray): 2D array representing the y-axis frequencies.
    - window_size (int, optional): Size of the apodizing lowpass filter. Default is 15.
    - angular_weighting (function, optional): Custom angular weighting function. Default is None.

    Returns:
    - modified_spectrum (numpy.ndarray): Modified frequency spectrum after applying the cone filter.
    - reconstructed_image (numpy.ndarray): Reconstructed image from the modified frequency spectrum.

    Notes:
    - The cone filter is designed to emphasize certain frequencies in the frequency spectrum.
    - The apodizing lowpass filter and windowing function are applied to the spectrum.
    - Optionally, a custom angular weighting function can be applied.
    - The modified frequency spectrum is obtained by multiplying the original spectrum with the filter components.
    - The reconstructed image is obtained by inverse Fourier transform.

    Example:
    >>> image = read_image("example.jpg")
    >>> frequency_spectrum, u, v = compute_frequency_spectrum(image)
    >>> modified_spectrum, reconstructed_image = cone_filter(frequency_spectrum, u, v, window_size=15)
    """
    # Compute angle in radians
    angle = np.angle(np.pi * (u + 1j * v))

    # Apodizing lowpass filter
    # std_deviation = 0.25  # Adjust this value as needed
    # A = np.exp(-(u**2 + v**2) / (2 * (std_deviation**2)))
    a = 10  # Adjust this value as needed
    A = np.exp(-a * np.sqrt(u**2 + v**2))
    # Windowing function
    w = np.sinc(angle / np.pi)

    # Angular weighting function
    if angular_weighting is not None:
        w *= angular_weighting(angle)

    # Frequency domain modification
    modified_spectrum = A * np.sqrt(u**2 + v**2) * w * frequency_spectrum

    # Inverse Fourier transform to get the reconstructed image
    reconstructed_image = np.real(np.fft.ifft2(modified_spectrum))

    return modified_spectrum, reconstructed_image

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
    # Use the _get_fourier_filter function to get the original filter
    fourier_filter = _get_fourier_filter(size, filter_name)
    
    # Get the frequencies corresponding to each element in the filter
    freq = fftfreq(size)
    
    # Invert the filter in the frequency domain (performing the inverse Fourier transform)
    inverse_filter = ifftshift(np.fft.ifft(fourier_filter.flatten())).real
    
    return inverse_filter[:, np.newaxis]