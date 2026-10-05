from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytomography")
pytest.importorskip("parallelproj")

import pytomography  # noqa: E402
import torch  # noqa: E402

from fbsem.bed import Bed  # noqa: E402
from fbsem.model import FBSEMNet, n_params  # noqa: E402
from sino.projector import SinogramSystemMatrix, object_meta  # noqa: E402
from utils.binmap import BinMap  # noqa: E402
from utils.scanner import DR_MM  # noqa: E402

XY = 48


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(pytomography, "device", torch.device("cpu"))


@pytest.fixture(scope="module")
def binmap(mini_hs):
    return BinMap(mini_hs)


def meta(binmap):
    return object_meta(XY, 2 * binmap.nrings - 1)


def blob(binmap):
    nz = 2 * binmap.nrings - 1
    c = (np.arange(XY) - (XY - 1) / 2) * DR_MM
    r = np.hypot(*np.meshgrid(c, c, indexing="ij"))
    x = np.where(r < 30.0, 4.0, 0.0) + np.where(r < 10.0, 6.0, 0.0)
    return torch.from_numpy(np.repeat(x[:, :, None], nz, 2).astype(np.float32))


def problem(binmap, seed=0):
    sm = SinogramSystemMatrix(binmap, meta=meta(binmap), device="cpu", psf=0)
    rng = np.random.default_rng(seed)
    S = rng.uniform(0.5, 1.0, binmap.shape).astype(np.float32)
    ybar = S * sm.forward(blob(binmap)).numpy() + 0.2
    y = rng.poisson(ybar).astype(np.float32)
    add = (0.2 / S).astype(np.float32)
    return y, S, add


def bed(binmap, n_sub, seed=0, psf=0):
    y, S, add = problem(binmap, seed)
    shape = (XY, XY, 2 * binmap.nrings - 1)
    return Bed(binmap, y, S, add, n_sub, psf=psf, xy=XY, meta=meta(binmap),
               mask=np.ones(shape, np.float32), device="cpu")


@pytest.mark.parametrize("psf", [0, [4.87, 4.87, 4.45]])
@pytest.mark.parametrize("n_sub", [1, 4])
def test_gamma_zero_is_pytomography_osem(binmap, n_sub, psf):
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood

    y, S, add = problem(binmap)
    sm = SinogramSystemMatrix(binmap, sensitivity=torch.from_numpy(S),
                              meta=meta(binmap), device="cpu", psf=psf)
    ll = PoissonLogLikelihood(sm, torch.from_numpy(y),
                              additive_term=torch.from_numpy(add))
    shape = (XY, XY, 2 * binmap.nrings - 1)
    ref = OSEM(ll, object_initial=torch.ones(shape))(n_iters=2, n_subsets=n_sub)

    net = FBSEMNet(depth=2, kernels=4).eval()
    with torch.no_grad():
        net.gamma.zero_()
        x = net(bed(binmap, n_sub, psf=psf), 2)
    err = float((x - ref).abs().max() / ref.abs().max())
    assert err < 1e-5


def test_fusion_is_the_root_of_the_surrogate():
    torch.manual_seed(0)
    net = FBSEMNet(depth=2, kernels=2).double()
    n = 4096
    x_em = torch.rand(n, dtype=torch.float64) * 5 + 1e-3
    x_reg = torch.rand(n, dtype=torch.float64) * 5
    s = torch.rand(n, dtype=torch.float64) * 3 + 0.05
    for g in (0.01, 1.0, 50.0):
        with torch.no_grad():
            net.gamma.fill_(g)
            net.set_units(1.0, 2.0)
            x = net.fuse(x_em, x_reg, s)
        d = g * 2.0 / s
        res = d * x * x + (1 - d * x_reg) * x - x_em
        assert bool((x > 0).all())
        assert float((res.abs() / x_em).max()) < 1e-9
        assert bool(((1 - d * x_reg) < 0).any()) or g < 1
    with torch.no_grad():
        x0 = net.fuse(x_em, x_reg, torch.zeros_like(s))
    assert torch.equal(x0, x_em)


def test_gradient_reaches_every_parameter(binmap):
    torch.manual_seed(1)
    net = FBSEMNet(depth=3, kernels=4).train()
    with torch.no_grad():
        net.gamma.fill_(0.5)
    b = bed(binmap, 4)
    x = net(b, 1)
    loss = torch.nn.functional.mse_loss(x, blob(binmap))
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        assert float(p.grad.abs().sum()) > 0, name


def test_parameter_count():
    assert n_params(FBSEMNet(depth=3, kernels=32)) == 29_572


def test_gamma_reset():
    net = FBSEMNet()
    with torch.no_grad():
        net.gamma.fill_(-1.0)
    net.clamp_gamma()
    assert net.gamma.item() == pytest.approx(0.01)
