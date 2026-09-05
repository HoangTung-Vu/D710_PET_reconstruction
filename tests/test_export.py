"""`export.py` -- the reconstructed block out to NIfTI and DICOM.

The whole file is one geometric claim: a voxel that came from a given place in
the CT must go back to that same place in the exported series.  There is
exactly one flip in the chain (`attenuation.to_radiological`, applied on the
way in) and the export has to undo it, so the tests follow a marked voxel all
the way round rather than checking the flip in isolation.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pytest

from utils import attenuation
from utils import export
from utils import quant
from utils.paths import case as get_case
import synth_ct

VX = VY = 2.1306
VZ = 3.2699997

#: The RDF header fields `write_dicom` reads.  Real values, no PHI.
HDR = {
    "bed_start_time": dt.datetime(2026, 7, 28, 3, 45, 0,
                                  tzinfo=dt.timezone.utc).timestamp(),
    "radiopharm_start_datetime": "20260728024500.00",
    "frame_duration_ms": 90000,
    "half_life_s": 6586.2002,
    "positron_fraction": 0.967,
    "dose_mbq": 185.0,
    "residual_dose_mbq": 3.7,
    "patient_weight_kg": 25.0,
    "patient_height_m": 1.18,
    "radiopharmaceutical": "FDG -- fluorodeoxyglucose",
    "manufacturer": "GE MEDICAL SYSTEMS",
    "model_name": "Discovery 710",
    "institution": "SYNTHETIC",
    "study_description": "FDG PET/CT",
    "accession_number": "TEST0001",
    "patient_name": "TEST^SYNTHETIC",
    "patient_id": "000000",
    "patient_birth_date": "20180101",
    "sop_instance_uid": synth_ct.FRAME_OF_REFERENCE,
    "study_instance_uid": "1.2.826.0.1.3680043.10.1338.99.2",
}


def marked_volume(nz=6, xy=16):
    """A STIR-order block with one bright voxel off-centre in every axis."""
    v = np.zeros((nz, xy, xy), dtype=np.float32)
    v[1, 3, 5] = 1000.0
    return v


# ------------------------------------------------------------- the one flip

def test_to_dicom_order_undoes_the_mu_map_flip():
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert (export.to_dicom_order(attenuation.to_radiological(a)) == a).all()


def test_to_dicom_order_touches_neither_z_nor_x():
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    d = export.to_dicom_order(a)
    assert (d[:, ::-1, :] == a).all()


def test_to_dicom_order_returns_a_contiguous_copy():
    """`PixelData` is taken with `.tobytes()`, which a reversed view reorders."""
    assert export.to_dicom_order(marked_volume()).flags["C_CONTIGUOUS"]


def test_grid_origin_puts_the_scanner_axis_at_zero():
    x0, y0 = export.grid_origin(328, 328, VX, VY)
    assert x0 == pytest.approx(-164 * VX)
    assert y0 == pytest.approx(-164 * VY)
    # The same rule `attenuation.mu_image` samples on: (arange(n) - n//2) * v.
    c = (np.arange(328) - 328 // 2) * VX
    assert x0 == pytest.approx(c[0])


def test_grid_origin_asks_each_axis_for_its_own_size():
    """A non-square grid must not take its y origin from the x count."""
    x0, y0 = export.grid_origin(64, 32, 2.0, 4.0)
    assert (x0, y0) == (-64.0, -64.0)
    x0, y0 = export.grid_origin(32, 64, 2.0, 4.0)
    assert (x0, y0) == (-32.0, -128.0)


# ------------------------------------------------------------------ NIfTI

def test_nifti_affine_is_ras(tmp_path):
    import nibabel as nib

    vol = marked_volume()
    p = export.write_nifti(vol, str(tmp_path / "a.nii.gz"), VX, VY, VZ, -394.94)
    img = nib.load(p)
    assert img.shape == (16, 16, 6)                # (i, j, k) = (x, y, z)

    d = export.to_dicom_order(vol)
    k, j, i = np.argwhere(d == d.max())[0]
    ras = nib.affines.apply_affine(img.affine, [i, j, k])
    x0, y0 = export.grid_origin(16, 16, VX, VY)
    lps = np.array([x0 + i * VX, y0 + j * VY, -394.94 + k * VZ])
    assert ras == pytest.approx([-lps[0], -lps[1], lps[2]], rel=1e-5)


def test_nifti_keeps_the_voxel_values(tmp_path):
    import nibabel as nib

    vol = marked_volume()
    p = export.write_nifti(vol, str(tmp_path / "a.nii.gz"), VX, VY, VZ, 0.0)
    assert nib.load(p).get_fdata().max() == pytest.approx(1000.0)


def test_nifti_creates_the_parent_directory(tmp_path):
    p = export.write_nifti(marked_volume(), str(tmp_path / "deep" / "a.nii.gz"),
                           VX, VY, VZ, 0.0)
    assert os.path.exists(p)


# ------------------------------------------------------------------ DICOM

def read_series(paths):
    import pydicom

    ds = [pydicom.dcmread(p) for p in paths]
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    return ds


def test_dicom_writes_one_file_per_plane(tmp_path):
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), HDR,
                               VX, VY, VZ, -394.94)
    assert len(paths) == 6
    ds = read_series(paths)
    assert [d.InstanceNumber for d in ds] == [1, 2, 3, 4, 5, 6]
    assert len({d.SeriesInstanceUID for d in ds}) == 1
    assert len({d.SOPInstanceUID for d in ds}) == 6


def test_dicom_slice_positions_step_by_the_plane_pitch(tmp_path):
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), HDR,
                               VX, VY, VZ, -394.94)
    z = [float(d.ImagePositionPatient[2]) for d in read_series(paths)]
    assert z[0] == pytest.approx(-394.94, abs=1e-3)
    assert np.diff(z) == pytest.approx(VZ, abs=1e-3)


def test_dicom_pixel_values_survive_the_rescale(tmp_path):
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), HDR,
                               VX, VY, VZ, 0.0)
    ds = read_series(paths)
    peak = max(float(d.pixel_array.max()) * float(d.RescaleSlope) for d in ds)
    assert peak == pytest.approx(1000.0, rel=1e-3)


def test_dicom_carries_the_marked_voxel_to_the_right_place(tmp_path):
    """(row, col) in the file, not in STIR's flipped y."""
    vol = marked_volume()
    paths = export.write_dicom(vol, str(tmp_path / "dcm"), HDR, VX, VY, VZ, 0.0)
    ds = read_series(paths)
    d = export.to_dicom_order(vol)
    k, j, i = np.argwhere(d == d.max())[0]
    a = ds[k].pixel_array
    assert np.argwhere(a == a.max())[0].tolist() == [j, i]


def test_dicom_is_tagged_so_a_viewer_can_compute_suv(tmp_path):
    """Units + dose + weight + decay reference; missing one and SUV is refused."""
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), HDR,
                               VX, VY, VZ, 0.0)
    d = read_series(paths)[0]
    assert d.Units == "BQML"
    assert d.DecayCorrection == "START"
    assert float(d.PatientWeight) == 25.0
    rp = d.RadiopharmaceuticalInformationSequence[0]
    assert float(rp.RadionuclideTotalDose) == pytest.approx((185.0 - 3.7) * 1e6)
    assert float(rp.RadionuclideHalfLife) == pytest.approx(6586.2002)
    assert rp.RadiopharmaceuticalStartDateTime.startswith("20260728024500")


def test_the_dicom_decay_interval_matches_the_rdf_header(tmp_path):
    """Acquisition time and injection time have to share one clock.

    The RDF header is UTC throughout and `_dcm_dt` renders the epoch in UTC, so
    the interval a viewer reads out of the series is the real uptake time.  Mix
    a local time into one of the two and every SUV is wrong by exp(lambda*7h).
    """
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), HDR,
                               VX, VY, VZ, 0.0)
    d = read_series(paths)[0]
    acq = dt.datetime.strptime(d.AcquisitionDate + d.AcquisitionTime[:6],
                               "%Y%m%d%H%M%S")
    inj = dt.datetime.strptime(
        d.RadiopharmaceuticalInformationSequence[0]
        .RadiopharmaceuticalStartDateTime[:14], "%Y%m%d%H%M%S")
    true_uptake = HDR["bed_start_time"] - dt.datetime.strptime(
        HDR["radiopharm_start_datetime"][:14],
        "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
    assert (acq - inj).total_seconds() == pytest.approx(true_uptake, abs=1.0)


def test_dicom_frame_of_reference_is_the_exams(tmp_path):
    """This is what makes the PET land on the CT in any viewer."""
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), HDR,
                               VX, VY, VZ, 0.0)
    d = read_series(paths)[0]
    assert d.FrameOfReferenceUID == HDR["sop_instance_uid"]
    assert d.StudyInstanceUID == HDR["study_instance_uid"]


def test_dicom_survives_a_header_without_a_residual_dose(tmp_path):
    hdr = {k: v for k, v in HDR.items() if k != "residual_dose_mbq"}
    paths = export.write_dicom(marked_volume(), str(tmp_path / "dcm"), hdr,
                               VX, VY, VZ, 0.0)
    rp = read_series(paths)[0].RadiopharmaceuticalInformationSequence[0]
    assert float(rp.RadionuclideTotalDose) == pytest.approx(185.0 * 1e6)


def test_dicom_handles_an_all_zero_block(tmp_path):
    """An empty bed must not divide by a zero rescale slope."""
    paths = export.write_dicom(np.zeros((3, 8, 8), np.float32),
                               str(tmp_path / "dcm"), HDR, VX, VY, VZ, 0.0)
    assert read_series(paths)[0].pixel_array.max() == 0


def test_dicom_replaces_nan_and_inf(tmp_path):
    """And the rescale slope is taken from the finite voxels only.

    `np.nanmax` drops NaN but keeps +inf, so one infinite voxel used to make
    `RescaleSlope` the string "inf" -- not a valid DS -- and every pixel divide
    by infinity, zeroing the whole series without an error anywhere.
    """
    vol = marked_volume()
    vol[0, 0, 0] = np.nan
    vol[0, 0, 1] = np.inf
    vol[0, 0, 2] = -np.inf
    paths = export.write_dicom(vol, str(tmp_path / "dcm"), HDR, VX, VY, VZ, 0.0)
    ds = read_series(paths)
    slope = float(ds[0].RescaleSlope)
    assert np.isfinite(slope) and slope > 0
    a = ds[0].pixel_array
    assert (a[0, :3] == 0).all()
    # The real signal is still there, at its real value.
    peak = max(float(d.pixel_array.max()) * float(d.RescaleSlope) for d in ds)
    assert peak == pytest.approx(1000.0, rel=1e-3)


# ----------------------------------------------- CT -> mu -> image -> DICOM

def test_a_ct_feature_comes_back_at_the_same_patient_coordinate(ct_dir, bed24,
                                                                tmp_path):
    """The end-to-end orientation check, in the only units that matter: mm.

    The synthetic CT has its dense insert at +y.  It goes into STIR through
    `mu_image` (which flips y), comes out through `write_dicom` (which flips it
    back), and must land within a voxel of where it started.
    """
    _ad, template = bed24
    ct = attenuation.load(ct_dir)
    span = (attenuation.PLANES_PER_BED - 1) * attenuation.PLANE_MM
    table = float(ct.z[0] + (ct.z[-1] - ct.z[0] - span) / 2)
    mu = attenuation.mu_image(ct, table, template).as_array()

    vz, vy, vx = (float(v) for v in template.voxel_sizes())
    paths = export.write_dicom(mu, str(tmp_path / "dcm"), HDR, vx, vy, vz, table)
    d = read_series(paths)[len(paths) // 2]

    a = d.pixel_array.astype(float) * float(d.RescaleSlope)
    w = np.where(a > 0.75 * a.max(), a, 0.0)          # the insert alone
    rows, cols = np.indices(a.shape)
    row = float((w * rows).sum() / w.sum())
    col = float((w * cols).sum() / w.sum())
    y = float(d.ImagePositionPatient[1]) + row * float(d.PixelSpacing[0])
    x = float(d.ImagePositionPatient[0]) + col * float(d.PixelSpacing[1])

    # synth_ct puts the insert at x = 0, y = +0.15 * 64 * 1.3672 mm.
    assert x == pytest.approx(0.0, abs=vx)
    assert y == pytest.approx(0.15 * 64 * synth_ct.DEFAULT_PIXEL_MM, abs=vy)


# ------------------------------------------------ our time reference vs GE's

def _case_with_beds(tmp_path, starts):
    """A `Case` whose only content is one header sidecar per bed.

    `starts` maps bed number -> seconds after the injection.  Everything else
    comes from `HDR`, so the only thing varying between the beds is the one
    field `scan_start_factor` selects on.
    """
    import json

    inj = dt.datetime.strptime(HDR["radiopharm_start_datetime"][:14],
                               "%Y%m%d%H%M%S").replace(
                                   tzinfo=dt.timezone.utc).timestamp()
    C = get_case("tref", tmp_path).mkdirs()
    for n, dt_s in starts.items():
        hdr = dict(HDR, bed_start_time=inj + dt_s)
        with open(C.decoded / f"bed{n}.json", "w") as f:
            json.dump(hdr, f)
    return C, inj


def test_scan_start_factor_is_the_uptake_decay(tmp_path):
    """One scalar, and it is exp(-lambda*uptake) -- nothing about frame length.

    GE applies the same mean-activity-over-the-frame correction that
    `stitch.decay_factor` does, so that term cancels and must NOT appear here.
    """
    uptake = 58.6 * 60
    C, _ = _case_with_beds(tmp_path, {1: uptake})
    decay, ref, hdr = quant.scan_start_factor(C, [1])

    lam = np.log(2) / HDR["half_life_s"]
    assert ref == 1
    assert decay == pytest.approx(np.exp(-lam * uptake))
    assert hdr["bed_start_time"] == pytest.approx(
        dt.datetime.strptime(HDR["radiopharm_start_datetime"][:14],
                             "%Y%m%d%H%M%S").replace(
                                 tzinfo=dt.timezone.utc).timestamp() + uptake)
    # 1.41-1.64 on the real cases; the factor is large, not a rounding detail.
    assert 1.3 < 1 / decay < 1.8


def test_scan_start_factor_takes_the_earliest_bed_not_the_lowest_numbered(tmp_path):
    """"Series start" is a time, so it is chosen by time.

    Bed order is not guaranteed to follow acquisition order, and a partial
    `--beds` run need not include bed 1 at all.  Picking `beds[0]` would then
    put our reference on a different instant from the `AcquisitionTime` the
    DICOM carries -- a silent scale error of a few per cent.
    """
    C, _ = _case_with_beds(tmp_path, {1: 3600.0, 2: 3000.0, 3: 3300.0})
    decay, ref, hdr = quant.scan_start_factor(C, [1, 2, 3])
    assert ref == 2
    assert decay == pytest.approx(np.exp(-np.log(2) / HDR["half_life_s"] * 3000.0))

    # And a subset that excludes it moves the reference, as it must.
    _, ref2, _ = quant.scan_start_factor(C, [1, 3])
    assert ref2 == 3


def test_moving_to_the_ge_reference_changes_bq_but_never_suv(tmp_path):
    """The invariant that makes the switch safe.

    Bq/mL is quoted at a different instant, so it moves by exactly the decay.
    SUV does not move at all: the volume and the dose are scaled by the same
    factor, which is the whole reason SUV was already right on the injection
    reference.  If this test ever fails, one of the two was scaled and the other
    was not, and every SUV in the export is wrong by that ratio.
    """
    uptake = 70 * 60
    C, _ = _case_with_beds(tmp_path, {1: uptake})
    decay, _, hdr = quant.scan_start_factor(C, [1])
    vol = marked_volume()
    vox = (VZ, VY, VX)
    K = 62_000.0

    inj = quant.report(vol, K, hdr, vox, out=lambda *_: None)
    ge = quant.report(vol * decay, K, hdr, vox,
                      dose=quant.dose_bq(hdr) * decay, out=lambda *_: None)

    assert ge["bqml"].max() == pytest.approx(inj["bqml"].max() * decay, rel=1e-6)
    np.testing.assert_allclose(ge["suv"], inj["suv"], rtol=1e-6)
    # And the fraction of the dose seen in the FOV is unchanged too -- both
    # numerator and denominator moved together.
    assert ge["mbq_in_fov"] / (ge["dose_bq"] / 1e6) == pytest.approx(
        inj["mbq_in_fov"] / (inj["dose_bq"] / 1e6), rel=1e-6)


def test_the_dose_bound_on_k_does_not_depend_on_when_the_scan_happened(tmp_path):
    """K is a property of the scanner, so its bound cannot carry an uptake time."""
    C, _ = _case_with_beds(tmp_path, {1: 70 * 60})
    decay, _, hdr = quant.scan_start_factor(C, [1])
    vol = marked_volume()
    vox = (VZ, VY, VX)

    assert quant.k_from_dose(vol * decay, vox, quant.dose_bq(hdr) * decay) == \
        pytest.approx(quant.k_from_dose(vol, vox, quant.dose_bq(hdr)))


# ------------------------------------------------------- one K for each path

def test_k_export_reads_the_constant_of_the_path_it_is_asked_for(monkeypatch):
    monkeypatch.delenv("D710_K", raising=False)
    monkeypatch.delenv("D710_K_LM", raising=False)
    monkeypatch.setattr(quant, "K_EXPORT", 111.0)
    monkeypatch.setattr(quant, "K_EXPORT_LM", 222.0)
    assert quant.k_export() == 111.0
    assert quant.k_export(lm=True) == 222.0


def test_the_env_var_of_each_path_wins_over_its_constant(monkeypatch):
    monkeypatch.setattr(quant, "K_EXPORT", 111.0)
    monkeypatch.setattr(quant, "K_EXPORT_LM", 222.0)
    monkeypatch.setenv("D710_K", "333")
    monkeypatch.delenv("D710_K_LM", raising=False)
    assert quant.k_export() == 333.0
    # ...and it does NOT leak into the other path.
    assert quant.k_export(lm=True) == 222.0


def test_an_unmeasured_path_returns_none_rather_than_borrowing_the_other(monkeypatch):
    """The two are ~2x apart, so falling back to the other K is worse than the
    WCC guess the caller would otherwise use."""
    monkeypatch.delenv("D710_K", raising=False)
    monkeypatch.delenv("D710_K_LM", raising=False)
    monkeypatch.setattr(quant, "K_EXPORT", 111.0)
    monkeypatch.setattr(quant, "K_EXPORT_LM", None)
    assert quant.k_export(lm=True) is None
