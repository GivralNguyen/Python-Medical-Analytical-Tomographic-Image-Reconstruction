import numpy as np
from scipy.interpolate import RegularGridInterpolator
from numpy.fft import fftshift, ifft2, ifftshift


def _frequency_axis(N):
    """
    Angular frequencies of an fftshift-ed length-N FFT: index k <-> (k - N//2) * 2*pi/N.
    The zero frequency is exactly at index N//2 for both even and odd N.
    """
    return (np.arange(N) - N // 2) * (2 * np.pi / N)


def create_radial_grid(N, theta):
    """
    Create the radial (polar) grid on which the Fourier slices of the projections are sampled.

    Parameters:
    - N (int): Length of the Fourier Transform.
    - theta (numpy.ndarray): Projection angles in degrees.

    Returns:
    - theta_grid (numpy.ndarray): (N, len(theta)) array of slice angles in radians.
    - r_grid (numpy.ndarray): (N, len(theta)) array of signed radial frequencies (rad/sample).

    Example:
    >>> theta_grid, r_grid = create_radial_grid(256, np.arange(0, 180))
    """
    omega_r = _frequency_axis(N)
    omega_theta = np.deg2rad(theta)

    theta_grid, r_grid = np.meshgrid(omega_theta, omega_r)
    return theta_grid, r_grid


def create_cartesian_grid(N):
    """
    Create the Cartesian frequency grid of the 2D Fourier Transform.

    Parameters:
    - N (int): Length of the Fourier Transform.

    Returns:
    - omega_grid_x (numpy.ndarray): (N, N) array, frequency along image columns (varies along axis 1).
    - omega_grid_y (numpy.ndarray): (N, N) array, frequency along image rows (varies along axis 0).

    Example:
    >>> omega_grid_x, omega_grid_y = create_cartesian_grid(256)
    """
    omega_xy = _frequency_axis(N)
    omega_grid_x, omega_grid_y = np.meshgrid(omega_xy, omega_xy)
    return omega_grid_x, omega_grid_y


def radial_to_cartesian_interpolation(omega_grid_x, omega_grid_y, r_grid, theta_grid, proj_fft):
    """
    Interpolate the polar samples of the 2D Fourier Transform (Fourier slice theorem) onto a Cartesian grid.

    Fourier slice convention of `radon` in this repo: the projection at angle theta is
    p(t) with t = x*cos(theta) - y*sin(theta)  (x = column, y = row), hence
        P_theta(w) = F(kx = w*cos(theta), ky = -w*sin(theta)).

    Each Cartesian point (kx, ky) is therefore mapped to theta = atan2(-ky, kx) and w = |k|;
    points with theta < 0 use the symmetry P_{theta+pi}(w) = P_theta(-w).
    Bilinear interpolation is done on the regular (w, theta) grid, with theta extended to
    pi so that the 179 deg -> 0 deg wrap-around is interpolated correctly.

    Parameters:
    - omega_grid_x, omega_grid_y (numpy.ndarray): Cartesian grid from `create_cartesian_grid`.
    - r_grid, theta_grid (numpy.ndarray): Polar grid from `create_radial_grid`.
    - proj_fft (numpy.ndarray): (N, n_angles) Fourier slices from `FT_1D`. Angles must be
      ascending and lie in [0, 180) degrees.

    Returns:
    - FFT2 (numpy.ndarray): (N, N) interpolated 2D spectrum, fftshift-ed (DC at [N//2, N//2]).
    """
    omega_r = r_grid[:, 0]
    omega_theta = theta_grid[0, :]

    # Periodic extension: the slice at theta = pi is the slice at theta = 0 with w -> -w.
    # On the grid (k - N//2), w -> -w maps row k to row 2*(N//2) - k.
    N = proj_fft.shape[0]
    mirror = (2 * (N // 2) - np.arange(N)) % N
    slice_pi = proj_fft[mirror, 0]
    if N % 2 == 0:
        slice_pi[0] = proj_fft[0, 0]  # -N/2 has no positive partner on the grid; Nyquist, negligible
    values = np.concatenate([proj_fft, slice_pi[:, None]], axis=1)
    thetas = np.append(omega_theta, np.pi)

    # Cartesian -> (signed radius, angle in [0, pi])
    angle = np.arctan2(-omega_grid_y, omega_grid_x)
    radius = np.hypot(omega_grid_x, omega_grid_y)
    negative = angle < 0
    angle[negative] += np.pi
    radius[negative] *= -1

    interp = RegularGridInterpolator((omega_r, thetas), values, method='linear',
                                     bounds_error=False, fill_value=0)
    FFT2 = interp(np.column_stack((radius.ravel(), angle.ravel())))
    return FFT2.reshape(omega_grid_x.shape)


# Backwards-compatible alias for the original (misspelt) name
radial_to_cartesian_inperpolation = radial_to_cartesian_interpolation


def inverse_2D_FT(FFT2, output_size=None):
    """
    Inverse 2D Fourier Transform of an fftshift-ed spectrum, returning a centred image.

    Parameters:
    - FFT2 (numpy.ndarray): (N, N) spectrum with DC at [N//2, N//2].
    - output_size (int, optional): If given, return only the centred output_size x output_size
      crop (use this when the projections were zero-padded / oversampled in `FT_1D`).

    Returns:
    - I (numpy.ndarray): Reconstructed image (real part; the imaginary part is only numerical noise).
    """
    N = FFT2.shape[0]
    I = np.real(fftshift(ifft2(ifftshift(FFT2))))
    if output_size is not None:
        c0 = N // 2 - output_size // 2
        I = I[c0:c0 + output_size, c0:c0 + output_size]
    return I
