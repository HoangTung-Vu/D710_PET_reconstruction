from __future__ import annotations

import numpy as np
import pytest

import lmmini as M


@pytest.fixture(scope="module")
def torch():
    return M.torch()


@pytest.fixture(scope="module")
def computed():
    return M.build(2000)


def test_grid_and_sensitivity_shape(computed):
    assert tuple(computed.norm_BP.shape) == (M.XY, M.XY, M.NPLANE)
    assert float(computed.norm_BP.max()) > 0


def test_tof_is_on(computed):
    assert computed.TOF is True
    assert computed.proj_meta.detector_ids.shape[1] == 3


def test_non_tof_drops_the_tof_column():
    sm = M.build(500, n_tof=1)
    assert sm.TOF is False
    assert sm.proj_meta.detector_ids.shape[1] == 2


def test_injected_sensitivity_matches_computed(torch, computed):
    sens = computed.norm_BP.clone()
    injected = M.build(2000, sensitivity=sens, with_sens_lors=False)

    assert torch.equal(injected.norm_BP, computed.norm_BP)

    x = M.blob()
    assert torch.equal(injected.forward(x), computed.forward(x))

    y = computed.forward(x)
    assert torch.allclose(injected.backward(y), computed.backward(y),
                          rtol=1e-6, atol=0.0)


def test_forward_is_bit_deterministic(torch, computed):
    x = M.blob()
    assert torch.equal(computed.forward(x), computed.forward(x))


def test_backprojection_is_not_bit_deterministic_but_is_tight(torch, computed):
    rng = np.random.default_rng(29)
    n = computed.proj_meta.detector_ids.shape[0]
    g = torch.from_numpy(rng.random(n).astype(np.float32))
    a, b = computed.backward(g), computed.backward(g)
    rel = float((a - b).abs().max() / a.abs().max())
    assert rel < 1e-6, rel


def test_injected_sensitivity_is_not_written_into(torch):
    probe = torch.zeros((M.XY, M.XY, M.NPLANE), dtype=torch.float32)
    sm = M.build(200, sensitivity=probe, with_sens_lors=False)

    assert float(sm.norm_BP.min()) == 1e7
    assert float(probe.abs().max()) == 0.0


def test_injected_sensitivity_is_idempotent(torch, computed):
    once = M.build(200, sensitivity=computed.norm_BP.clone(),
                   with_sens_lors=False)
    twice = M.build(200, sensitivity=once.norm_BP.clone(),
                    with_sens_lors=False)
    assert torch.equal(once.norm_BP, twice.norm_BP)


def test_psf_changes_the_sensitivity(torch):
    with_psf = M.build(200)
    without = M.build(200, psf=0.0)
    assert not torch.allclose(with_psf.norm_BP, without.norm_BP)
    assert len(with_psf.obj2obj_transforms) == 1
    assert len(without.obj2obj_transforms) == 0


def test_build_sm_accepts_torch_ids(torch):
    from lm import recon

    ids = torch.from_numpy(M.events(300))
    sens_ids, sens_w = M.sens_lors()
    sm = recon.build_sm(ids, M.NTOF, xy=M.XY, n_plane=M.NPLANE, psf=M.PSF,
                        n_splits=2, sens_ids=sens_ids, sens_w=sens_w,
                        lut=M.lut(), tof=M.tof_meta())
    assert sm.proj_meta.detector_ids.shape == (300, 3)


def test_sens_image_zeroes_the_sentinel(computed):
    from lmnet import sens

    raw = computed.norm_BP.cpu().numpy().copy()
    raw[0, 0, 0] = 1e7
    img = sens.image(raw)
    assert img[0, 0, 0] == 0.0
    assert float(img.max()) < 1e7


def test_sens_kappa():
    from lmnet import sens

    s = np.full((4, 4, 2), 0.5, np.float32)
    assert sens.kappa(64, s) == pytest.approx(64.0 / 16.0)
