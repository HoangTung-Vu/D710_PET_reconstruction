from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pydicom")
pytest.importorskip("scipy")

from tools import ct_nifti, dicom_suv


def _affine(px, dz, x0, y0, z0):
    return np.array([[-px, 0, 0, -x0], [0, -px, 0, -y0], [0, 0, dz, z0], [0, 0, 0, 1.0]])


def _world(affine, shape):
    i, j, k = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    w = affine @ np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    return [c.reshape(shape) for c in w[:3]]


def _ramp(x, y, z):
    return 0.5 * x - 0.25 * y + 2.0 * z + 10.0


def test_resample_like_is_exact_on_a_linear_ramp():
    ct_aff, ct_shape = _affine(1.25, 3.0, -80.0, -80.0, -30.0), (128, 128, 21)
    pet_aff, pet_shape = _affine(2.5, 3.0, -70.0, -70.0, -24.0), (56, 56, 15)
    ct = _ramp(*_world(ct_aff, ct_shape)).astype(np.float32)
    got = ct_nifti.resample_like(ct, ct_aff, pet_aff, pet_shape)
    assert got.shape == pet_shape and got.dtype == np.float32
    assert np.allclose(got, _ramp(*_world(pet_aff, pet_shape)), atol=1e-3)


def test_resample_like_fills_air_outside_the_ct():
    ct_aff, ct_shape = _affine(2.0, 3.0, -20.0, -20.0, 0.0), (21, 21, 5)
    pet_aff, pet_shape = _affine(2.0, 3.0, -60.0, -60.0, 0.0), (61, 61, 5)
    got = ct_nifti.resample_like(np.zeros(ct_shape, np.float32), ct_aff, pet_aff, pet_shape)
    assert got[0, 0, 2] == ct_nifti.AIR_HU
    assert got[30, 30, 2] == 0.0


def test_resample_like_keeps_the_edge_slices_when_the_grids_share_them():
    ct_aff, ct_shape = _affine(1.0, 3.27, -10.0, -10.0, -1014.46), (21, 21, 299)
    pet_aff, pet_shape = _affine(2.0, 3.2700043, -9.5, -9.5, -1014.46), (10, 10, 299)
    ct = np.full(ct_shape, 40.0, np.float32)
    got = ct_nifti.resample_like(ct, ct_aff, pet_aff, pet_shape)
    assert np.all(got[:, :, 0] == 40.0) and np.all(got[:, :, -1] == 40.0)


def test_resample_like_on_its_own_grid_is_the_identity():
    aff, shape = _affine(2.0, 3.27, -30.0, -30.0, -10.0), (31, 31, 7)
    vol = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    assert np.allclose(ct_nifti.resample_like(vol, aff, aff, shape), vol, atol=1e-5)


def test_bqml_writes_the_dicom_volume_unchanged(tmp_path, monkeypatch):
    vol = np.random.default_rng(1).uniform(0, 5e4, size=(4, 6, 6)).astype(np.float32)
    meta = {"ds": None, "n": 4, "shape": vol.shape, "x0": -7.5, "y0": -7.5,
            "z0": -100.0, "px": 3.0, "py": 3.0, "dz": 3.27, "decay": "START",
            "recon": "VPFXS", "desc": "PET WB"}
    monkeypatch.setattr(dicom_suv, "read_series", lambda d: (vol, meta))
    out = tmp_path / "x_ge_bqml.nii.gz"
    info = dicom_suv.convert("unused", str(out), unit="bqml")
    img = nib.load(out)
    assert np.array_equal(np.asarray(img.dataobj), np.transpose(vol, (2, 1, 0)))
    assert np.allclose(np.diag(img.affine)[:3], [-3.0, -3.0, 3.27])
    assert info["unit"] == "bqml" and info["decay_correction"] == "START"
