from skimage._shared.utils import convert_to_float
from scipy.interpolate import interp1d
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift

def backprojection(sinogram,theta = None, filter = None, output_size=None, interval=1):
    if theta is None:
        theta = np.linspace(0, 180, sinogram.shape[1], endpoint=False)
    
    sinogram = convert_to_float(sinogram, preserve_range = True)

    if output_size is None:
        output_size = int(np.floor(np.sqrt((sinogram.shape[0]) ** 2 / 2.0))) #shape of output
    img_shape = sinogram.shape[0]

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter not in filter_types:
        raise ValueError(f"Unknown filter: {filter}")
    if filter is not None:
        fourier_filter = _get_fourier_filter(img_shape, filter)
        projection = fft(sinogram, axis=0) * fourier_filter
        sinogram = np.real(ifft(projection, axis=0)[:img_shape, :])

    reconstructed = np.zeros((output_size, output_size),
                             dtype=sinogram.dtype)
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2
    

    for col_idx, angle in zip(range(0, len(theta), interval), np.deg2rad(theta[::interval])):
        col = sinogram[:, col_idx]
        t = ypr * np.cos(angle) - xpr * np.sin(angle)

        interpolant = interp1d(x, col, kind='cubic',
                                bounds_error=False, fill_value=0)
        reconstructed += interpolant(t)
    
    return sinogram, reconstructed * np.pi / (2 * len(theta))


def _get_fourier_filter(size, filter_name):

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