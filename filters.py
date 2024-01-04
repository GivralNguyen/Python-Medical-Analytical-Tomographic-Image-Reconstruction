import numpy as np

def cone_filter(frequency_spectrum, u, v, window_size=15, angular_weighting=None):
    # Compute angle in radians
    angle = np.angle(np.pi * (u + 1j * v))

    # Apodizing lowpass filter
    std_deviation = 0.25  # Adjust this value as needed
    A = np.exp(-(u**2 + v**2) / (2 * (std_deviation**2)))
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