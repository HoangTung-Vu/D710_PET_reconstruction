from __future__ import annotations

import os

import numpy as np
import pytest

import lmmini as M

from cases import decoded_beds


@pytest.fixture(scope="module")
def torch():
    return M.torch()


@pytest.fixture(scope="module")
def sm():
    return M.build(4000)


def test_forward_is_differentiable(torch, sm):
    from lmnet.project import LMForward

    x = M.blob().requires_grad_(True)
    y = LMForward.apply(sm, x)
    assert y.shape == (4000,)
    y.sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().max()) > 0


def test_adjoint(torch, sm):
    from lmnet.project import LMBackward, LMForward

    rng = np.random.default_rng(7)
    x = M.blob(seed=5)
    y = torch.from_numpy(rng.random(4000).astype(np.float32))

    px = LMForward.apply(sm, x)
    pty = LMBackward.apply(sm, y)

    lhs = float(px.double() @ y.double())
    rhs = float((pty.double() * x.double()).sum())
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rel < 1e-6, (lhs, rhs, rel)


def test_backward_is_the_adjoint_of_forward(torch, sm):
    from lmnet.project import LMForward

    rng = np.random.default_rng(11)
    x = M.blob(seed=6).requires_grad_(True)
    g = torch.from_numpy(rng.random(4000).astype(np.float32))

    (LMForward.apply(sm, x) * g).sum().backward()

    lhs = float((x.grad.double() * x.detach().double()).sum())
    rhs = float(LMForward.apply(sm, x.detach()).double() @ g.double())
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 1e-6


def test_finite_difference(torch, sm):
    from lmnet.project import LMForward

    rng = np.random.default_rng(13)
    g = torch.from_numpy(rng.random(4000).astype(np.float32))
    x0 = M.blob(seed=8, scale=100.0)

    def loss(x):
        return 0.5 * ((LMForward.apply(sm, x) - g) ** 2).sum()

    x = x0.clone().requires_grad_(True)
    loss(x).backward()
    grad = x.grad.clone()

    eps = 1.0
    scale = float(grad.abs().max())
    idx = torch.nonzero(grad.abs() > 0.05 * scale)
    idx = idx[torch.linspace(0, len(idx) - 1, 12).long()]

    worst = 0.0
    for i in idx.tolist():
        i = tuple(i)
        xp, xm = x0.clone(), x0.clone()
        xp[i] += eps
        xm[i] -= eps
        fd = float(loss(xp) - loss(xm)) / (2 * eps)
        worst = max(worst, abs(fd - float(grad[i])) / scale)
    assert worst < 1e-2, worst


def test_contiguity_of_the_incoming_gradient(torch, sm):
    from lmnet.project import LMBackward

    rng = np.random.default_rng(17)
    h = torch.from_numpy(rng.random(4000).astype(np.float32))
    h.requires_grad_(True)
    img = LMBackward.apply(sm, h)
    (img.transpose(0, 2) * 2.0).sum().backward()
    assert h.grad is not None
    assert float(h.grad.abs().max()) > 0


def _toy_loop(sm, n_phase, wrap: bool):
    import torch as t
    import torch.nn as nn

    from lmnet.project import LMBackward, LMForward

    t.manual_seed(0)
    duals = nn.ModuleList(nn.Linear(3, 1) for _ in range(n_phase))
    n = sm.proj_meta.detector_ids.shape[0]
    a = t.full((n, 1), 0.3)
    h = t.zeros((n, 1))
    u = t.zeros((M.XY, M.XY, M.NPLANE))

    for k in range(n_phase):
        fwd = LMForward.apply(sm, u) if wrap else sm.forward(u.detach())
        h = h + duals[k](t.cat((h, fwd.unsqueeze(1), a), dim=1))
        col = h[:, 0].contiguous()
        bp = LMBackward.apply(sm, col) if wrap else sm.backward(col.detach())
        u = u + 1e-3 * bp

    return u, duals


def test_toy_loop_dual_gradients_are_nonzero(torch, sm):
    u, duals = _toy_loop(sm, 3, wrap=True)
    assert u.requires_grad
    u.sum().backward()
    norms = [float(sum(p.grad.pow(2).sum() for p in d.parameters()) ** 0.5)
             for d in duals]
    assert all(n > 0 for n in norms), norms


def test_toy_loop_unwrapped_kills_the_dual_gradients(torch, sm):
    u, duals = _toy_loop(sm, 3, wrap=False)
    assert not u.requires_grad
    assert all(p.grad is None for d in duals for p in d.parameters())


def _one_bed():
    beds = decoded_beds()
    if not beds:
        pytest.skip("no decoded bed with vendor terms under $D710_OUT; "
                    "run `d710 exam` first")
    want = os.environ.get("D710_CASE")
    for b in beds:
        if not want or b["case"] == want:
            return b
    return beds[0]


def test_full_grid_adjoint_on_a_real_bed():
    torch = M.torch()
    M.pytomography()
    b = _one_bed()

    from lm import events as ev
    from lm import geom, recon
    from lmnet import sens as senscache
    from lmnet.project import LMBackward, LMForward
    from utils.paths import case as get_case
    from utils.scanner import NSEG0, PSF_FWHM_MM, XY

    C = get_case(b["case"])
    npy = C.decoded / f"bed{b['bed']}.lm.npy"
    if not npy.exists():
        pytest.skip(f"no {npy}")

    n_keep = 100_000
    e = ev.load(npy)
    step = max(1, len(e) // n_keep)
    e = np.asarray(e[::step])[:n_keep]

    binmap = geom.BinMap(C.prompt(b["bed"]))
    s = senscache.get(C, b["bed"], binmap)
    ids = ev.detector_ids(e, geom.N_TOF_RAW, 1)
    keep = ev.bins(e, binmap) >= 0
    sm = recon.build_sm(ids[keep], geom.N_TOF_RAW, xy=XY, n_plane=NSEG0,
                        psf=PSF_FWHM_MM, sensitivity=torch.from_numpy(s))

    rng = np.random.default_rng(3)
    n = int(keep.sum())
    x = torch.from_numpy(
        np.ascontiguousarray(rng.random((XY, XY, NSEG0)), np.float32))
    y = torch.from_numpy(rng.random(n).astype(np.float32))

    lhs = float(LMForward.apply(sm, x).double() @ y.double())
    rhs = float((LMBackward.apply(sm, y).double() * x.double()).sum())
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 1e-5
