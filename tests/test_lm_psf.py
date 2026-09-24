from __future__ import annotations

import numpy as np
import pytest

from lm.recon import psf_fwhm
from utils.scanner import PSF_FWHM_MM, PSF_XY_MM, PSF_Z_MM

FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def test_the_default_is_ges_psf_as_a_list():
    f = psf_fwhm()
    assert f == [PSF_XY_MM, PSF_XY_MM, PSF_Z_MM] == list(PSF_FWHM_MM)
    assert type(f) is list
    assert all(type(v) is float for v in f)


@pytest.mark.parametrize("psf, want", [
    (6.4, [6.4, 6.4, 6.4]),
    ([6.4], [6.4, 6.4, 6.4]),
    ((4.87, 4.45), [4.87, 4.87, 4.45]),
    (np.array([4.87, 4.45]), [4.87, 4.87, 4.45]),
    ([3.0, 4.0, 5.0], [3.0, 4.0, 5.0]),
])
def test_one_two_or_three_values(psf, want):
    f = psf_fwhm(psf)
    assert f == want
    assert type(f) is list


@pytest.mark.parametrize("psf", [None, 0, 0.0, [0.0], [0.0, 0.0], [0, 0, 0]])
def test_zero_everywhere_switches_it_off(psf):
    assert psf_fwhm(psf) is None


@pytest.mark.parametrize("psf", [[4.87, 0.0], [1.0, 0.0, 1.0], [-1.0], [1, 2, 3, 4]])
def test_a_partial_zero_or_a_wrong_length_is_refused(psf):
    with pytest.raises(ValueError):
        psf_fwhm(psf)


def test_gaussian_filter_takes_it_per_axis():
    pytest.importorskip("torch")
    filters = pytest.importorskip("pytomography.transforms.shared.filters")
    g = filters.GaussianFilter(psf_fwhm((4.87, 4.45)))
    assert np.allclose(np.array(g.sigma) * FWHM_PER_SIGMA, [4.87, 4.87, 4.45])
    wrong = filters.GaussianFilter(tuple(PSF_FWHM_MM))
    assert np.ndim(wrong.sigma[0]) == 1
