import numpy as np
from scipy.interpolate import griddata
from numpy.fft import fftshift, ifft2, ifftshift

def create_radial_grid(N,theta):
    """
    Create a radial grid for 2D Fourier Transform based on specified parameters.

    Parameters:
    - N (int): Length of the Fourier Transform.
    - theta (numpy.ndarray): Array of angles in degrees for the radial grid.

    Returns:
    - theta_grid (numpy.ndarray): 2D array representing the radial angles in radians.
    - r_grid (numpy.ndarray): 2D array representing the radial coordinates.

    Notes:
    - The radial grid is created for the 2D Fourier Transform.
    - The angular values are converted from degrees to radians.
    - The radial values are defined based on the length of the Fourier Transform.
    - The resulting grids can be used to evaluate the 2D Fourier Transform.

    Example:
    >>> N = 256
    >>> theta = np.arange(0, 180)
    >>> theta_grid, r_grid = create_radial_grid(N, theta)
    """
    if N % 2 == 1:
        omega_r = np.arange(-N/2, (N+1)/2) * (2 * np.pi / N)  # define x axis of the FFT,
    else:
        omega_r = np.arange(-(N-1)/2, (N+1)/2) * (2 * np.pi / N)

    omega_theta = theta * np.pi / 180 # angles of the radial grid of FFT in rad

    theta_grid, r_grid = np.meshgrid(omega_theta, omega_r) #create a radial grid for the FFT values that we have

    return theta_grid, r_grid

def create_cartesian_grid(N):
    """
    Create a Cartesian grid for 2D Fourier Transform based on specified parameters.

    Parameters:
    - N (int): Length of the Fourier Transform.

    Returns:
    - omega_grid_x (numpy.ndarray): 2D array representing the x-axis coordinates.
    - omega_grid_y (numpy.ndarray): 2D array representing the y-axis coordinates.

    Notes:
    - The Cartesian grid is created for the 2D Fourier Transform.
    - The resulting grids can be used to evaluate the 2D Fourier Transform.
    - The x and y coordinates are defined based on the length of the Fourier Transform.

    Example:
    >>> N = 256
    >>> omega_grid_x, omega_grid_y = create_cartesian_grid(N)
    """
    if N % 2 == 1:
        omega_xy = np.arange(-N/2, (N+1)/2) * (2 * np.pi / N)  # define x axis of the FFT,
    else:
        omega_xy = np.arange(-(N-1)/2, (N+1)/2) * (2 * np.pi / N)
    omega_grid_x, omega_grid_y = np.meshgrid(omega_xy, omega_xy) 

    return omega_grid_x, omega_grid_y

def radial_to_cartesian_inperpolation(omega_grid_x, omega_grid_y, r_grid, theta_grid, proj_fft):
    """
    Perform radial-to-cartesian interpolation on a 2D Fourier Transform.

    Parameters:
    - omega_grid_x (numpy.ndarray): 2D array representing the x-axis coordinates.
    - omega_grid_y (numpy.ndarray): 2D array representing the y-axis coordinates.
    - r_grid (numpy.ndarray): 2D array representing the radial coordinates.
    - theta_grid (numpy.ndarray): 2D array representing the angular coordinates in radians.
    - proj_fft (numpy.ndarray): 2D array representing the Fourier Transform coefficients.

    Returns:
    - FFT2_flipped (numpy.ndarray): Result of the radial-to-cartesian interpolation.

    Notes:
    - The input coordinates are transformed from Cartesian to polar.
    - Bilinear interpolation is performed using griddata.
    - The result is reshaped to match the shape of the input coordinates.

    Example:
    >>> N = 256
    >>> theta = np.arange(0, 180)
    >>> theta_grid, r_grid = create_radial_grid(N, theta)
    >>> omega_grid_x, omega_grid_y = create_cartesian_grid(N)
    >>> proj_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))  # Example Fourier Transform
    >>> result = radial_to_cartesian_inperpolation(omega_grid_x, omega_grid_y, r_grid, theta_grid, proj_fft)
    """
    # MATLAB: FFT2 = interp2(theta_grid,r_grid,proj_fft,coord_th_fft2,coord_r_fft2,'bilinear',(0+1i.*0)); %   
    coord_th_fft2, coord_r_fft2 = np.angle(omega_grid_x + 1j * omega_grid_y), np.abs(omega_grid_x + 1j * omega_grid_y) #transform cartesian to polar coordinates

    coord_r_fft2 *= np.sign(coord_th_fft2) #if theta>pi

    coord_th_fft2 = coord_th_fft2 + np.pi / 2
    coord_th_fft2[coord_th_fft2 < 0] += np.pi

    points = np.column_stack((r_grid.flatten(), theta_grid.flatten()))
    values = proj_fft.flatten()
    # Define points where interpolation is needed
    points_q = np.column_stack((coord_r_fft2.flatten(), coord_th_fft2.flatten()))
    # Perform bilinear interpolation
    FFT2 = griddata(points, values, points_q, method='nearest', fill_value=0+1j*0) #interpolate coefficients that we have to the grid points Fx, Fy

    # Reshape the result to match the shape of coord_th_fft2 and coord_r_fft2
    FFT2 = FFT2.reshape(coord_r_fft2.shape, )
    # FFT2_flipped = np.flipud(FFT2))
    FFT2_flipped = np.flipud(np.rot90(FFT2, k=3))
    return FFT2_flipped

def inverse_2D_FT(FFT2):
    """
    Perform the inverse 2D Fourier Transform.

    Parameters:
    - FFT2 (numpy.ndarray): 2D array representing the Fourier Transform coefficients.

    Returns:
    - I (numpy.ndarray): Result of the inverse 2D Fourier Transform.

    Notes:
    - The input FFT2 is expected to be in the frequency domain.
    - The inverse Fourier Transform is applied using ifft2.
    - The result is shifted back to the spatial domain using ifftshift and fftshift.

    Example:
    >>> N = 256
    >>> omega_grid_x, omega_grid_y = create_cartesian_grid(N)
    >>> proj_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))  # Example Fourier Transform
    >>> result = inverse_2D_FT(proj_fft)
    """

    I = ifftshift(ifft2(fftshift(FFT2)))
    return I