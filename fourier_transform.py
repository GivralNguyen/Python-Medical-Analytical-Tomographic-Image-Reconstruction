import numpy as np

def FT_1D(sinogram, oversample=1):
    """
    Perform 1D Fourier Transform of every projection of a sinogram (along the detector axis).

    Parameters:
    - sinogram (numpy.ndarray): Input sinogram with shape (projection_positions, projection_angles),
      as returned by `radon`. The rotation centre is at row index shape[0] // 2.
    - oversample (int, optional): Zero-pad each projection to `oversample * N` samples before the FFT.
      This samples the Fourier slices more finely, which greatly reduces the interpolation
      error in Fourier gridding (the reconstructed field of view grows by the same factor
      and is cropped back afterwards).

    Returns:
    - N (int): Length of the Fourier Transform (= oversample * projection length).
    - proj_fft (numpy.ndarray): 1D Fourier Transform of each projection, shape (N, projection_angles),
      zero frequency at row N // 2.

    Example:
    >>> sinogram = radon(center, padded_image, theta=np.arange(180))   # shape (256, 180)
    >>> N, proj_fft = FT_1D(sinogram, oversample=4)
    """
    n0 = sinogram.shape[0]
    N = n0 * oversample
    # Zero-pad symmetrically so the rotation centre (row n0//2) lands on row N//2
    start = N // 2 - n0 // 2
    padded = np.zeros((N, sinogram.shape[1]), dtype=sinogram.dtype)
    padded[start:start + n0] = sinogram

    # Move the rotation centre to index 0, FFT, then move zero frequency to the centre
    proj_fft = np.fft.fft(np.fft.ifftshift(padded, axes=0), axis=0)
    proj_fft = np.fft.fftshift(proj_fft, axes=0)

    return N, proj_fft
