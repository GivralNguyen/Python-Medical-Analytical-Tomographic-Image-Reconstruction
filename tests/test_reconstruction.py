"""Every reconstruction method must reproduce the Shepp-Logan phantom (RMSE inside the FOV circle)."""
import numpy as np
import pytest
from skimage.transform import iradon
from skimage.transform import radon as sk_radon

import tomorecon as tr

THETA = np.arange(180)


@pytest.fixture(scope="module")
def phantom():
    return tr.shepp_logan()


@pytest.fixture(scope="module")
def sinogram(phantom):
    return tr.radon(phantom, THETA)


@pytest.fixture(scope="module")
def inside(phantom):
    n = phantom.shape[0]
    rows, cols = np.mgrid[:n, :n] - n // 2
    return rows ** 2 + cols ** 2 < (n // 2 - 2) ** 2


@pytest.fixture(scope="module")
def rmse(phantom, inside):
    return lambda image: float(np.sqrt(np.mean((image[inside] - phantom[inside]) ** 2)))


def test_phantom(phantom):
    assert phantom.shape == (256, 256)
    assert 0 <= phantom.min() and phantom.max() <= 1
    assert tr.shepp_logan(128).shape == (128, 128)


def test_disk():
    d = tr.disk((50, 60), radius=5, center=(30, 20))
    assert d.shape == (50, 60) and d[20, 30] == 1 and d[0, 0] == 0


def test_radon_matches_skimage(phantom, sinogram):
    np.testing.assert_allclose(sinogram, sk_radon(phantom, THETA), atol=1e-10)


def test_radon_no_circle_matches_skimage(phantom):
    np.testing.assert_allclose(tr.radon(phantom, THETA[::10], circle=False),
                               sk_radon(phantom, THETA[::10], circle=False), atol=1e-10)


def test_fbp_matches_skimage(sinogram, inside):
    ref = iradon(sinogram, THETA, filter_name="ramp", circle=True)
    np.testing.assert_allclose(tr.fbp(sinogram, THETA)[inside], ref[inside], atol=1e-8)


@pytest.mark.parametrize("name", tr.FILTERS)
def test_fbp(sinogram, rmse, name):
    assert rmse(tr.fbp(sinogram, THETA, filter=name)) < 0.06


@pytest.mark.parametrize("name", ["ramp", "cosine"])
def test_cbp_equals_fbp(sinogram, name):
    np.testing.assert_allclose(tr.cbp(sinogram, THETA, name), tr.fbp(sinogram, THETA, name), atol=1e-10)


def test_bpf(sinogram, rmse):
    assert rmse(tr.bpf(sinogram, THETA)) < 0.05


def test_fourier_gridding(sinogram, rmse):
    assert rmse(tr.fourier_gridding(sinogram, THETA)) < 0.04
    # without oversampling the interpolation error is much larger
    assert rmse(tr.fourier_gridding(sinogram, THETA, oversample=1)) > 0.06


def test_fourier_slice_theorem(phantom, sinogram):
    # theta = 0: the FT of the projection is the ky = 0 row of the 2-D FT
    spectra = tr.projection_spectra(sinogram)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phantom)))
    n = spectra.shape[0]
    np.testing.assert_allclose(spectra[:, 0], F[n // 2, :], atol=1e-6 * np.abs(F).max())


def test_default_theta(sinogram):
    np.testing.assert_allclose(tr.fbp(sinogram), tr.fbp(sinogram, THETA))


def test_theta_length_mismatch(sinogram):
    with pytest.raises(ValueError):
        tr.fbp(sinogram, THETA[:10])


def test_interval_normalisation(sinogram):
    full = tr.backproject(sinogram, THETA).mean()
    assert abs(tr.backproject(sinogram, THETA, interval=18).mean() / full - 1) < 0.02


def test_spatial_kernel_is_centred():
    k = tr.spatial_kernel(256)
    assert k.size == 511 and np.argmax(k) == 255
    np.testing.assert_allclose(k, k[::-1], atol=1e-12)


def test_filter_validation():
    np.testing.assert_array_equal(tr.fourier_filter(64, None), np.ones(64))
    with pytest.raises(ValueError):
        tr.fourier_filter(64, "nope")
