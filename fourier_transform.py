import numpy as np

def FT_1D(sinogram):
    # Shift zero-frequency component to the center along the first dimension
    shifted_sinogram = np.fft.ifftshift(sinogram, axes=0)

    # Compute the FFT along the first dimension (projection angle)
    proj_fft = np.fft.fft(shifted_sinogram, axis=0)

    # Get the length of the FFT
    N = proj_fft.shape[0]
    # Shift the zero-frequency component back to the corner
    proj_fft = np.fft.fftshift(proj_fft, axes=0)

    return N, proj_fft
