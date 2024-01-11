from skimage._shared.utils import convert_to_float
from scipy.interpolate import interp1d
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from filters import _get_fourier_filter

def backprojection(sinogram,theta = None, filter = None, output_size=None, interval=1):
    """
    Perform backprojection to reconstruct an image from a sinogram.

    Parameters:
    - sinogram (numpy.ndarray): Input sinogram data.
    - theta (numpy.ndarray, optional): Projection angles in degrees. If None, evenly spaced angles between 0 and 180 are used.
    - filter (str, optional): Filter type to be applied during backprojection. Options: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None.
    - output_size (int, optional): Size of the reconstructed image. If None, it is set to floor(sqrt((sinogram.shape[0]) ** 2 / 2.0)).
    - interval (int, optional): Interval for selecting projection angles.

    Returns:
    - sinogram (numpy.ndarray): Filtered Sinogram data (Original if Filter is None).
    - reconstructed (numpy.ndarray): Reconstructed image.

    Notes:
    - The backprojection is performed using linear interpolation.
    - Optionally, a filter can be applied to the sinogram before backprojection.
    - The output size of the reconstructed image is determined based on the input sinogram shape.
    - The sinogram and reconstructed image are returned.

    Example:

    >>> filtered_sinogram, reconstructed = backprojection(sinogram, theta, filter='ramp', output_size=512)
    """
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
    
    """
    for col_idx, angle in zip(range(0, len(theta), interval), np.deg2rad(theta[::interval])):: 
    This loop iterates over columns of the sinogram with a specified interval of projection angles. 
    It uses the range function to create a sequence of column indices, and zip pairs these indices with 
    the corresponding projection angles converted to radians.

    col = sinogram[:, col_idx]: Extracts the column of the sinogram corresponding to the current angle.

    t = ypr * np.cos(angle) - xpr * np.sin(angle): Computes a linear combination of ypr and xpr (presumably 
    representing pixel coordinates in the reconstructed image) using the cosine and sine of the current projection angle.

    interpolant = interp1d(x, col, kind='linear', bounds_error=False, fill_value=0): Uses interp1d from SciPy to create 
    a linear interpolation function (interpolant) based on the values in the sinogram column (col) with respect to the coordinates x.

    reconstructed += interpolant(t): Evaluates the interpolation function at the coordinates t and adds the result to the 
    reconstructed variable. This suggests that the reconstructed image is being built up by 
    interpolating values from the sinogram at the specified coordinates.
    """
    for col_idx, angle in zip(range(0, len(theta), interval), np.deg2rad(theta[::interval])):
        col = sinogram[:, col_idx]
        t = ypr * np.cos(angle) - xpr * np.sin(angle)

        interpolant = interp1d(x, col, kind='linear',
                                bounds_error=False, fill_value=0)
        reconstructed += interpolant(t)
    
    return sinogram, reconstructed * np.pi / (2 * len(theta))


