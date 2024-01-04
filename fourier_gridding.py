import numpy as np
from scipy.interpolate import griddata
from numpy.fft import fftshift, ifft2, ifftshift

def create_radial_grid(N,theta):
    if N % 2 == 1:
        omega_r = np.arange(-N/2, (N+1)/2) * (2 * np.pi / N)  # define x axis of the FFT,
    else:
        omega_r = np.arange(-(N-1)/2, (N+1)/2) * (2 * np.pi / N)

    omega_theta = theta * np.pi / 180 # angles of the radial grid of FFT in rad

    theta_grid, r_grid = np.meshgrid(omega_theta, omega_r) #create a radial grid for the FFT values that we have

    return theta_grid, r_grid

def create_cartesian_grid(N):
    if N % 2 == 1:
        omega_xy = np.arange(-N/2, (N+1)/2) * (2 * np.pi / N)  # define x axis of the FFT,
    else:
        omega_xy = np.arange(-(N-1)/2, (N+1)/2) * (2 * np.pi / N)
    omega_grid_x, omega_grid_y = np.meshgrid(omega_xy, omega_xy) 

    return omega_grid_x, omega_grid_y

def radial_to_cartesian_inperpolation(omega_grid_x, omega_grid_y, r_grid, theta_grid, proj_fft):
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
    FFT2 = griddata(points, values, points_q, method='cubic', fill_value=0+1j*0) #interpolate coefficients that we have to the grid points Fx, Fy

    # Reshape the result to match the shape of coord_th_fft2 and coord_r_fft2
    FFT2 = FFT2.reshape(coord_r_fft2.shape, )
    # FFT2_flipped = np.flipud(FFT2))
    FFT2_flipped = np.flipud(np.rot90(FFT2, k=3))
    return FFT2_flipped

def inverse_2D_FT(FFT2):
    I = ifftshift(ifft2(fftshift(FFT2)))
    return I