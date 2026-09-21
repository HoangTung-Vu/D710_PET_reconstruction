from __future__ import annotations

import numpy as np
import pytest

RINGS, NDET = 4, 32

R_MM = 60.0

XY, NPLANE = 17, 7

PSF = 6.4

NTOF = 5

TOF_RANGE_MM = 2 * R_MM

TOF_FWHM_MM = 24.0


def torch():
    return pytest.importorskip("torch")


def pytomography():
    return pytest.importorskip("pytomography")


def lut():
    from utils.geometry import crystal_positions
    from utils.scanner import RING_PITCH_MM

    return np.ascontiguousarray(
        crystal_positions(RINGS, NDET, R_MM, RING_PITCH_MM, 0.0, True),
        np.float32)


def tof_meta(n_bins: int = NTOF, n_sigmas: float = 3.0):
    from pytomography.metadata.PET import PETTOFMeta

    return PETTOFMeta(num_bins=n_bins, tof_range=TOF_RANGE_MM,
                      fwhm=TOF_FWHM_MM, n_sigmas=n_sigmas)


def opposite_pairs():
    r1, d1 = np.meshgrid(np.arange(RINGS), np.arange(NDET), indexing="ij")
    r1, d1 = r1.ravel(), d1.ravel()
    d2 = (d1 + NDET // 2) % NDET
    r2 = (RINGS - 1) - r1
    a = r1 * NDET + d1
    b = r2 * NDET + d2
    keep = a < b
    return np.stack([a[keep], b[keep]], 1).astype(np.int32)


def events(n: int, n_tof: int = NTOF, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = opposite_pairs()
    i = rng.integers(0, len(pairs), n)
    t = rng.integers(0, n_tof, n).astype(np.int32)
    return np.ascontiguousarray(
        np.concatenate([pairs[i], t[:, None]], 1), np.int32)


def sens_lors(seed: int = 1):
    ids = opposite_pairs()
    rng = np.random.default_rng(seed)
    w = (0.5 + rng.random(len(ids))).astype(np.float32)
    return ids, w


def build(n_events: int = 2000, n_tof: int = NTOF, psf: float = PSF,
          sensitivity=None, with_sens_lors: bool = True, seed: int = 0,
          n_splits: int = 2):
    from lm import recon

    pytomography()
    ids = events(n_events, n_tof, seed)
    sens_ids = sens_w = None
    if with_sens_lors:
        sens_ids, sens_w = sens_lors()
    return recon.build_sm(ids, n_tof, xy=XY, n_plane=NPLANE, psf=psf,
                          n_splits=n_splits, sens_ids=sens_ids, sens_w=sens_w,
                          sensitivity=sensitivity, lut=lut(),
                          tof=tof_meta(n_tof) if n_tof > 1 else None)


def blob(seed: int = 3, scale: float = 1.0):
    t = torch()
    rng = np.random.default_rng(seed)
    x = np.zeros((XY, XY, NPLANE), np.float32)
    c = XY // 2
    x[c - 3:c + 4, c - 3:c + 4, 2:5] = 1.0
    x += 0.05 * rng.random(x.shape).astype(np.float32)
    return t.from_numpy(np.ascontiguousarray(x * scale))


def sensitivity_image(sm, n_tang: int | None = None):
    from lmnet import sens

    return sens.image(sm.norm_BP.cpu().numpy())
