from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytomography")
pytest.importorskip("parallelproj")

import torch  # noqa: E402

from sino.projector import SinogramSystemMatrix, object_meta  # noqa: E402
from utils import attn_proj  # noqa: E402
from utils.binmap import BinMap  # noqa: E402
from utils.scanner import DR_MM, PLANE_MM  # noqa: E402

XY = 48


@pytest.fixture(scope="module")
def binmap(mini_hs):
    return BinMap(mini_hs)


def meta(binmap):
    return object_meta(XY, 2 * binmap.nrings - 1)


def system(binmap, **kw):
    kw.setdefault("psf", 0)
    return SinogramSystemMatrix(binmap, meta=meta(binmap), device="cpu", **kw)


def blob(binmap, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((XY, XY, 2 * binmap.nrings - 1)).astype(np.float32)
    c = (np.arange(XY) - (XY - 1) / 2) * DR_MM
    r = np.hypot(*np.meshgrid(c, c, indexing="ij"))
    return torch.from_numpy(x * (r < 40.0)[:, :, None].astype(np.float32))


def test_the_sinogram_has_the_bin_maps_shape(binmap):
    sm = system(binmap)
    y = sm.forward(blob(binmap))
    assert tuple(y.shape) == binmap.shape


@pytest.mark.parametrize("psf", [0, [4.87, 4.87, 4.45]])
def test_back_projection_is_the_adjoint(binmap, psf):
    sm = system(binmap, psf=psf)
    x = blob(binmap, 1)
    y = torch.from_numpy(np.random.default_rng(2).random(binmap.shape)
                         .astype(np.float32))
    lhs = float((sm.forward(x).double() * y.double()).sum())
    rhs = float((x.double() * sm.backward(y).double()).sum())
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_subsets_partition_the_views(binmap):
    sm = system(binmap)
    y = torch.from_numpy(np.random.default_rng(3).random(binmap.shape)
                         .astype(np.float32))
    full = sm.backward(y)
    sm.set_n_subsets(4)
    parts = sum(sm.backward(sm.get_projection_subset(y, k), k) for k in range(4))
    assert torch.allclose(parts, full, rtol=1e-4, atol=1e-4)
    x = blob(binmap, 4)
    yf = sm.forward(x)
    for k in range(4):
        assert torch.allclose(sm.forward(x, k),
                              sm.get_projection_subset(yf, k), rtol=1e-5, atol=1e-5)


def test_chunking_does_not_change_the_answer(binmap):
    x = blob(binmap, 5)
    a = system(binmap).forward(x)
    b = system(binmap, max_rays=binmap.n_view * binmap.n_tang).forward(x)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_the_geometry_is_the_attenuation_projectors(binmap, mini_hs):
    mu = np.zeros((2 * binmap.nrings - 1, XY, XY), np.float32)
    c = (np.arange(XY) - (XY - 1) / 2) * DR_MM
    yy, xx = np.meshgrid(c, c, indexing="ij")
    mu[:, np.hypot(xx - 12.0, yy + 7.0) < 30.0] = 0.096
    mu[3:6] *= 1.7
    af = attn_proj.factors(mu, mini_hs, plane_mm=PLANE_MM, device="cpu",
                           out=lambda *_: None)

    img = torch.from_numpy(attn_proj.image_for_projector(mu)[0])
    line = system(binmap).forward(img).numpy()
    one = binmap.mult == 1
    assert one.any() and (~one).any()
    assert np.allclose(np.exp(-line[one]), af[one], rtol=1e-5, atol=1e-6)
    assert np.allclose(np.exp(-line[~one]), af[~one], rtol=0, atol=2e-3)


def test_the_normalisation_is_the_back_projected_sensitivity(binmap):
    s = np.random.default_rng(6).random(binmap.shape).astype(np.float32)
    sm = system(binmap, sensitivity=torch.from_numpy(s))
    sm.set_n_subsets(3)
    got = sum(sm.compute_normalization_factor(k) for k in range(3))
    want = sm.backward(torch.from_numpy(s))
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_osem_recovers_a_noiseless_object(binmap):
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood

    rng = np.random.default_rng(7)
    s = (0.5 + rng.random(binmap.shape)).astype(np.float32)
    b = (0.2 * rng.random(binmap.shape)).astype(np.float32)
    sm = system(binmap, sensitivity=torch.from_numpy(s))
    x_true = blob(binmap, 8) * 10
    y = torch.from_numpy(s) * sm.forward(x_true) + torch.from_numpy(b)

    ll = PoissonLogLikelihood(sm, y, additive_term=torch.from_numpy(b / s))
    x0 = (x_true > 0).float()
    err = []
    for n_it in (1, 20):
        x = OSEM(ll, object_initial=x0.clone())(n_iters=n_it, n_subsets=4)
        err.append(float(((sm.forward(x) - sm.forward(x_true)).norm()
                          / sm.forward(x_true).norm())))
    assert err[1] < err[0] * 0.5
    assert err[1] < 0.05
