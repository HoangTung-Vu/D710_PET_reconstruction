from __future__ import annotations

import numpy as np
import pytest

import lmmini as M

DUAL = (8, 8)

PRIMAL = (4, 8, 4)

N_EVENTS = 10_000


@pytest.fixture(scope="module")
def torch():
    return M.torch()


@pytest.fixture(scope="module")
def sm():
    return M.build(N_EVENTS)


@pytest.fixture(scope="module")
def inputs(torch, sm):
    from lmnet import sens

    rng = np.random.default_rng(23)
    s_np = sens.image(sm.norm_BP.cpu().numpy())
    s_np[:3] = 0.0
    s = torch.from_numpy(np.ascontiguousarray(s_np, np.float32))
    a = torch.from_numpy(
        np.ascontiguousarray(0.1 + rng.random((N_EVENTS, 1)), np.float32))
    kappa = sens.kappa(N_EVENTS, s_np)
    return a, s, kappa


def net(torch, **kw):
    from lmnet.model import LMPDNet3D

    torch.manual_seed(0)
    kw.setdefault("n_phase", 3)
    kw.setdefault("amp", False)
    return LMPDNet3D(dual_widths=DUAL, primal_widths=PRIMAL, **kw)


def test_shapes_support_and_nonnegativity(torch, sm, inputs):
    a, s, kappa = inputs
    m = net(torch, ckpt_phase=False)
    out = m(sm, a, s, kappa).detach()

    assert out.shape == (M.XY, M.XY, M.NPLANE)
    assert float(out.min()) >= 0.0
    assert torch.all(out[s <= 0] == 0)
    assert float(out.abs().max()) > 0


def _absorbed_by_batchnorm(name: str) -> bool:
    return name.startswith("primal.") and name.endswith(".0.bias")


def test_every_parameter_gets_a_gradient(torch, sm, inputs):
    a, s, kappa = inputs
    for ckpt in (False, True):
        m = net(torch, ckpt_phase=ckpt)
        m(sm, a, s, kappa).pow(2).sum().backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, (ckpt, missing)
        dead = [n for n, p in m.named_parameters()
                if float(p.grad.abs().max()) == 0.0
                and not _absorbed_by_batchnorm(n)]
        assert not dead, (ckpt, dead)


def test_conv_bias_before_batchnorm_carries_no_signal(torch, sm, inputs):
    a, s, kappa = inputs
    m = net(torch, ckpt_phase=False)
    m(sm, a, s, kappa).pow(2).sum().backward()

    g = dict(m.named_parameters())
    checked = 0
    for n, p in g.items():
        if not _absorbed_by_batchnorm(n):
            continue
        w = g[n[: -len("bias")] + "weight"]
        assert float(p.grad.abs().max()) < 1e-6 * float(w.grad.abs().max()), n
        checked += 1
    assert checked == m.n_phase * (len(PRIMAL) + 1)


def test_checkpointing_does_not_change_the_loss(torch, sm, inputs):
    a, s, kappa = inputs
    m = net(torch, ckpt_phase=False)
    ref = float(m(sm, a, s, kappa).pow(2).sum().detach())

    m.set_checkpointing(phase=True)
    got = float(m(sm, a, s, kappa).pow(2).sum().detach())
    assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref))

    m.set_checkpointing(phase=True, block=True)
    got = float(m(sm, a, s, kappa).pow(2).sum().detach())
    assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref))


def test_checkpointing_does_not_change_the_gradients(torch, sm, inputs):
    a, s, kappa = inputs

    def grads(**kw):
        m = net(torch, **kw)
        m(sm, a, s, kappa).pow(2).sum().backward()
        return {n: p.grad.clone() for n, p in m.named_parameters()}

    def rel(x, y):
        return {n: float((x[n] - y[n]).abs().max())
                / max(float(x[n].abs().max()), 1e-12) for n in x}

    off = grads(ckpt_phase=False)
    repeat = rel(off, grads(ckpt_phase=False))
    on = rel(off, grads(ckpt_phase=True, ckpt_block=True))

    for n in off:
        if _absorbed_by_batchnorm(n):
            continue
        assert on[n] <= max(4 * repeat[n], 1e-4), (n, on[n], repeat[n])


def test_dual_and_primal_both_learn(torch, sm, inputs):
    a, s, kappa = inputs
    m = net(torch, ckpt_phase=True)
    m(sm, a, s, kappa).pow(2).sum().backward()

    for k in range(m.n_phase):
        d = sum(float(p.grad.abs().sum()) for p in m.dual[k].parameters())
        p_ = sum(float(p.grad.abs().sum()) for p in m.primal[k].parameters())
        assert d > 0, k
        assert p_ > 0, k


def test_phase_count_and_size(torch):
    from lmnet.model import LMPDNet3D, n_parameters

    m = LMPDNet3D(n_phase=8)
    assert len(m.dual) == len(m.primal) == 8
    assert n_parameters(m) > 1_000_000


def test_amp_is_a_noop_on_cpu(torch, sm, inputs):
    a, s, kappa = inputs
    off = net(torch, ckpt_phase=False, amp=False)(sm, a, s, kappa)
    on = net(torch, ckpt_phase=False, amp=True)(sm, a, s, kappa)
    rel = float((off - on).abs().max()) / max(float(off.abs().max()), 1e-12)
    assert rel < 1e-5, rel


def test_dualnet_is_residual(torch):
    from lmnet.model import DualNet

    torch.manual_seed(1)
    d = DualNet(3, DUAL)
    for p in d.block[-1].parameters():
        torch.nn.init.zeros_(p)
    h = torch.randn(5, 1)
    out = d(h, torch.randn(5, 1), torch.randn(5, 1))
    assert torch.allclose(out, h)


def test_primalnet_is_residual(torch):
    from lmnet.model import PrimalNet3D

    torch.manual_seed(1)
    p = PrimalNet3D(3, PRIMAL).eval()
    for q in p.blocks[-1].parameters():
        torch.nn.init.zeros_(q)
    u = torch.randn(1, 1, 5, 5, 3)
    out = p(u, torch.randn(1, 2, 5, 5, 3))
    assert torch.allclose(out, u, atol=1e-6)
