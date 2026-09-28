from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pytest

import estimate
import synth_ct
from utils import attenuation, attn, nifti

N_SLICES, N = 60, 64
PX, DZ, Z0 = synth_ct.DEFAULT_PIXEL_MM, synth_ct.DEFAULT_DZ, -100.0
X0 = Y0 = -(N // 2) * PX
XY, DR_MM = 32, PX


def lps_image():
    hu = synth_ct.volume(N_SLICES, N, PX)
    aff = np.array([[-PX, 0, 0, -X0], [0, -PX, 0, -Y0], [0, 0, DZ, Z0], [0, 0, 0, 1.0]])
    return np.ascontiguousarray(hu.transpose(2, 1, 0)), aff


def flip(arr, aff, axis):
    out = aff.copy()
    out[:3, 3] = aff[:3, 3] + aff[:3, axis] * (arr.shape[axis] - 1)
    out[:3, axis] = -aff[:3, axis]
    return np.flip(arr, axis), out


def permute(arr, aff, order):
    out = aff.copy()
    out[:3, :3] = aff[:3, list(order)]
    return np.transpose(arr, order), out


def variant(name):
    arr, aff = lps_image()
    if name == "LAS":
        arr, aff = flip(arr, aff, 1)
    elif name == "RAS":
        arr, aff = flip(*flip(arr, aff, 0), 1)
    elif name == "PLS":
        arr, aff = permute(arr, aff, (1, 0, 2))
    elif name == "SLP":
        arr, aff = permute(*flip(arr, aff, 2), (2, 0, 1))
    return arr, aff


def save(path, arr, aff, dtype=np.int16, qform_only=False, cast=True):
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(arr, dtype if cast else None), None, hdr)
    img.set_qform(aff, code=1)
    img.set_sform(None if qform_only else aff, code=0 if qform_only else 2)
    nib.save(img, str(path))
    return str(path)


@pytest.mark.parametrize("name,dtype,qform_only", [
    ("ct.nii.gz", np.int16, False),
    ("ct.nii", np.float32, False),
    ("ct.nii.gz", np.int32, True),
])
def test_reader_matches_nibabel(tmp_path, name, dtype, qform_only):
    arr, aff = variant("LAS")
    p = save(tmp_path / name, arr, aff, dtype, qform_only)
    got, got_aff = nifti.read(p)
    ref = nib.load(p)
    if qform_only:
        assert ref.header["sform_code"] == 0 and ref.header["pixdim"][0] == -1
    assert np.array_equal(got, np.asarray(ref.dataobj))
    assert np.allclose(got_aff, ref.affine, atol=1e-5)


def test_reader_applies_the_scale(tmp_path):
    arr, aff = lps_image()
    p = save(tmp_path / "scaled.nii.gz", arr.astype(np.float32) * 1.5 + 0.25, aff, np.int16,
             cast=False)
    ref = nib.load(p)
    assert ref.header["scl_slope"] not in (0.0, 1.0)
    got, _ = nifti.read(p)
    assert np.allclose(got, np.asarray(ref.dataobj), atol=1e-3)


@pytest.mark.parametrize("orient", ["LPS", "LAS", "RAS", "PLS", "SLP"])
def test_nifti_ct_is_the_dicom_ct(ct_dir, tmp_path, orient):
    d = attenuation.load(ct_dir)
    n = attenuation.load(save(tmp_path / f"{orient}.nii.gz", *variant(orient)))
    assert np.array_equal(n.hu, d.hu)
    assert np.allclose(n.z, d.z, atol=1e-3)
    assert n.x0 == pytest.approx(d.x0, abs=1e-3)
    assert n.y0 == pytest.approx(d.y0, abs=1e-3)
    assert n.pixel_mm == pytest.approx(d.pixel_mm, rel=1e-5)
    assert n.dz == pytest.approx(d.dz, rel=1e-5)
    assert n.meta["format"] == "nifti" and d.meta["format"] == "dicom"


def test_mu_map_is_the_same_from_nifti_and_dicom(ct_dir, tmp_path):
    d = attenuation.load(ct_dir)
    n = attenuation.load(save(tmp_path / "ct.nii.gz", *variant("RAS")))
    span = (attenuation.PLANES_PER_BED - 1) * attenuation.PLANE_MM
    table = float(d.z[0] + (d.z[-1] - d.z[0] - span) / 2)
    a = attenuation.mu_map(d, table, XY, DR_MM)
    b = attenuation.mu_map(n, table, XY, DR_MM)
    assert a.max() > 0
    assert np.allclose(a, b, atol=1e-5)


def test_refuses_a_tilted_volume(tmp_path):
    arr, aff = lps_image()
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    aff[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ aff[:3, :3]
    with pytest.raises(SystemExit, match="tilted"):
        attenuation.load(save(tmp_path / "tilt.nii.gz", arr, aff))


def test_refuses_rectangular_pixels(tmp_path):
    arr, aff = lps_image()
    aff[1, 1] *= 1.5
    with pytest.raises(SystemExit, match="square"):
        attenuation.load(save(tmp_path / "rect.nii.gz", arr, aff))


def test_refuses_a_volume_with_no_air(tmp_path):
    arr, aff = lps_image()
    with pytest.raises(SystemExit, match="no air"):
        attenuation.load(save(tmp_path / "pet.nii.gz", arr + 1100, aff))


def test_refuses_a_time_series(tmp_path):
    arr, aff = lps_image()
    with pytest.raises(SystemExit, match="3-D"):
        attenuation.load(save(tmp_path / "4d.nii.gz", np.stack([arr, arr], -1), aff))


def test_refuses_a_file_without_orientation(tmp_path):
    arr, aff = lps_image()
    img = nib.Nifti1Image(arr.astype(np.int16), None)
    img.set_qform(None, code=0)
    img.set_sform(None, code=0)
    nib.save(img, str(tmp_path / "bare.nii.gz"))
    with pytest.raises(SystemExit, match="neither an sform nor a qform"):
        attenuation.load(str(tmp_path / "bare.nii.gz"))


def test_refuses_a_file_that_is_not_nifti(tmp_path):
    p = tmp_path / "junk.nii"
    p.write_bytes(os.urandom(1024))
    with pytest.raises(SystemExit, match="not a NIfTI"):
        attenuation.load(str(p))


def test_same_exam_check_is_skipped_only_for_nifti(ct_dir, tmp_path):
    hdr = {"sop_instance_uid": "1.2.3"}
    attn.check_same_exam(attenuation.load(save(tmp_path / "ct.nii.gz", *lps_image())), hdr)
    with pytest.raises(SystemExit, match="same exam"):
        attn.check_same_exam(attenuation.load(ct_dir), hdr)


def test_ct_mount(ct_dir, tmp_path):
    assert estimate.ct_mount(ct_dir) == ((ct_dir, "/ct", "ro"), "/ct", False)
    p = save(tmp_path / "ct.nii.gz", *lps_image())
    assert estimate.ct_mount(p) == ((str(tmp_path), "/ct", "ro"), "/ct/ct.nii.gz", True)
