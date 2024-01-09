import numpy as np

def FT_1D(sinogram):
    """
    Perform 1D Fourier Transform on a given sinogram along the first dimension (projection angle).

    Parameters:
    - sinogram (numpy.ndarray): Input sinogram data with shape (projection_angles, projection_positions).

    Returns:
    - N (int): Length of the Fourier Transform.
    - proj_fft (numpy.ndarray): 1D Fourier Transform of the input sinogram.

    Notes:
    - The input sinogram is assumed to be the result of Radon Transform.
    - The zero-frequency component is shifted to the center along the first dimension.
    - The result is shifted back to have the zero-frequency component at the corners.
    - The input sinogram should have a shape (projection_angles, projection_positions).

    Example:
    >>> sinogram = np.random.rand(180, 256)
    >>> N, proj_fft = FT_1D(sinogram)
    """

    # Shift zero-frequency component to the center along the first dimension
    shifted_sinogram = np.fft.ifftshift(sinogram, axes=0)

    # Compute the FFT along the first dimension (projection angle)
    proj_fft = np.fft.fft(shifted_sinogram, axis=0)

    # Get the length of the FFT
    N = proj_fft.shape[0]
    # Shift the zero-frequency component back to the corner
    proj_fft = np.fft.fftshift(proj_fft, axes=0)

    return N, proj_fft
