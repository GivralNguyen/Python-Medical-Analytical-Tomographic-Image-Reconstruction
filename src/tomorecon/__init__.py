"""
tomorecon: analytical tomographic image reconstruction.

Radon transform, Fourier gridding, (filtered) backprojection, backproject-filter
and convolve-backproject for 2-D parallel-beam tomography.

Example
-------
>>> import tomorecon as tr
>>> phantom = tr.shepp_logan()
>>> sinogram = tr.radon(phantom)          # 180 angles, 0..179 degrees
>>> image = tr.fbp(sinogram)              # ramp-filtered backprojection
"""

from .backprojection import backproject, bpf, cbp, fbp
from .filters import (
    FILTERS,
    cone_filter,
    convolve_sinogram,
    filter_sinogram,
    fourier_filter,
    padded_size,
    spatial_kernel,
)
from .gridding import (
    cartesian_grid,
    fourier_gridding,
    frequency_axis,
    inverse_fft2,
    polar_grid,
    polar_to_cartesian,
    projection_spectra,
)
from .phantoms import disk, load_image, shepp_logan
from .projection import radon

__version__ = "0.1.0"

__all__ = [
    # data
    "shepp_logan", "disk", "load_image",
    # forward model
    "radon",
    # reconstruction
    "backproject", "fbp", "cbp", "bpf", "fourier_gridding",
    # filters
    "FILTERS", "fourier_filter", "spatial_kernel", "filter_sinogram", "convolve_sinogram",
    "cone_filter", "padded_size",
    # Fourier gridding steps
    "projection_spectra", "polar_to_cartesian", "inverse_fft2", "polar_grid", "cartesian_grid",
    "frequency_axis",
]
