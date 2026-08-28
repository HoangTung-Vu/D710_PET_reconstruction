"""`background/attenuation.py` -- CT series -> mu-map on the bed grid."""

from __future__ import annotations

import numpy as np
import pytest

from utils import attenuation
import synth_ct


# ----------------------------------------------------------------- hu_to_mu

def test_water_and_air_anchors():
    assert attenuation.hu_to_mu(np.array([0.0])) == pytest.approx(0.0096, rel=1e-6)
    assert attenuation.hu_to_mu(np.array([-1000.0])) == pytest.approx(0.0, abs=1e-9)


def test_bilinear_is_continuous_at_the_kink():
    lo = attenuation.hu_to_mu(np.array([-1e-3]))[0]
    hi = attenuation.hu_to_mu(np.array([+1e-3]))[0]
    assert lo == pytest.approx(hi, abs=1e-7)       # float32 output


def test_monotone_and_never_negative():
    hu = np.linspace(-1200, 3000, 4201)
    mu = attenuation.hu_to_mu(hu)
    assert (np.diff(mu) >= 0).all()
    assert (mu >= 0).all()
    assert mu[hu <= -1000].max() == 0.0        # air and below clip to zero


def test_bone_slope_depends_on_kvp():
    """Carney's `b` is per-kVp; a lower tube voltage means a steeper bone arm."""
    hu = np.array([1000.0])
    mu80 = attenuation.hu_to_mu(hu, 80)[0]
    mu120 = attenuation.hu_to_mu(hu, 120)[0]
    mu140 = attenuation.hu_to_mu(hu, 140)[0]
    assert mu80 > mu120 > mu140
    b = attenuation.CARNEY_B[120]
    want = (attenuation.MU_WATER_511
            + 1000 * (attenuation.MU_BONE_511 - attenuation.MU_WATER_511) / (1000 * b))
    assert mu120 == pytest.approx(want, rel=1e-6)


def test_an_unlisted_kvp_falls_back_to_120():
    assert (attenuation.hu_to_mu(np.array([1000.0]), 110)
            == attenuation.hu_to_mu(np.array([1000.0]), 120))


def test_soft_tissue_arm_is_water_scaled():
    hu = np.array([-500.0])
    assert attenuation.hu_to_mu(hu)[0] == pytest.approx(0.0096 * 0.5, rel=1e-6)


# ------------------------------------------------------------ to_radiological

def test_to_radiological_is_its_own_inverse():
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert (attenuation.to_radiological(attenuation.to_radiological(a)) == a).all()


def test_to_radiological_flips_y_only():
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    f = attenuation.to_radiological(a)
    assert (f[:, 0] == a[:, -1]).all()
    assert (f[0, :, 0] == a[0, ::-1, 0]).all()


# ------------------------------------------------------------------- load

def test_load_reads_the_geometry(ct_dir):
    ct = attenuation.load(ct_dir)
    assert ct.hu.shape == (60, 64, 64)
    assert ct.dz == pytest.approx(synth_ct.DEFAULT_DZ, rel=1e-4)
    assert ct.pixel_mm == pytest.approx(synth_ct.DEFAULT_PIXEL_MM, rel=1e-6)
    assert ct.kvp == 120.0
    assert ct.meta["frame_of_reference_uid"] == synth_ct.FRAME_OF_REFERENCE
    assert ct.meta["num_slices"] == 60
    assert ct.z[0] < ct.z[-1]                       # sorted towards the head


def test_load_recovers_the_hu_it_was_given(ct_dir):
    ct = attenuation.load(ct_dir)
    assert ct.hu.min() == pytest.approx(-1000.0, abs=0.5)
    assert ct.hu.max() == pytest.approx(850.0, abs=0.5)


def test_load_sorts_by_z_not_by_file_name(tmp_path):
    """Instance order on disk is not slice order; only `ImagePositionPatient` is."""
    import os

    d = synth_ct.series(tmp_path / "ct", n_slices=6)
    names = sorted(os.listdir(d))
    # Reverse the file names so alphabetical order contradicts z order.
    for i, f in enumerate(names):
        os.rename(os.path.join(d, f), os.path.join(d, "zz%02d.dcm" % (len(names) - i)))
    ct = attenuation.load(d)
    assert (np.diff(ct.z) > 0).all()


def test_load_refuses_a_directory_with_no_ct(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no CT instance"):
        attenuation.load(str(tmp_path / "empty"))


def test_load_refuses_a_tilted_series(tmp_path):
    import glob

    import pydicom

    d = synth_ct.series(tmp_path / "ct", n_slices=6)
    for f in glob.glob(d + "/*"):
        ds = pydicom.dcmread(f)
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "0.9848", "0.1736"]
        ds.save_as(f, enforce_file_format=True)
    with pytest.raises(SystemExit, match="is tilted"):
        attenuation.load(d)


def test_load_refuses_a_partial_export(tmp_path):
    """A missing slice would be bridged by interpolation into a plausible lie."""
    d = synth_ct.series(tmp_path / "ct", n_slices=20, drop=(10,))
    with pytest.raises(SystemExit) as e:
        attenuation.load(d)
    msg = str(e.value)
    assert "incomplete export" in msg
    assert "1 gaps" in msg                       # exactly one gap found
    assert "6.5 mm" in msg                          # and it is the doubled step


def test_load_defaults_the_kvp_when_the_tag_is_empty(tmp_path):
    import glob

    import pydicom

    d = synth_ct.series(tmp_path / "ct", n_slices=6)
    for f in glob.glob(d + "/*"):
        ds = pydicom.dcmread(f)
        ds.KVP = None
        ds.save_as(f, enforce_file_format=True)
    assert attenuation.load(d).kvp == 120.0


# ---------------------------------------------------------------- mu_image

def _bed_start(ct):
    """A table position that puts all 47 planes comfortably inside the CT."""
    span = (attenuation.PLANES_PER_BED - 1) * attenuation.PLANE_MM
    return float(ct.z[0] + (ct.z[-1] - ct.z[0] - span) / 2)


def test_mu_image_is_in_per_cm_on_the_bed_grid(ct_dir, bed24):
    _ad, template = bed24
    ct = attenuation.load(ct_dir)
    mu = attenuation.mu_image(ct, _bed_start(ct), template)
    a = mu.as_array()
    assert a.shape == (attenuation.PLANES_PER_BED, 32, 32)
    # Water at 511 keV is 0.096 /cm.  STIR wants /cm; the PIFA wants /mm, and
    # confusing the two is a factor of ten in every attenuation factor.
    water = a[(a > 0.05) & (a < 0.12)]
    assert water.size, "no water-like voxels: check the /cm conversion"
    assert np.median(water) == pytest.approx(0.096, rel=0.05)
    assert a.max() <= 10 * attenuation.hu_to_mu(np.array([850.0]))[0] + 1e-4
    assert a.min() == 0.0                       # air, not a negative mu


def test_mu_image_is_radiological_so_y_is_flipped(ct_dir, bed24):
    """The bone insert sits at +y in DICOM; STIR must see it at -y."""
    _ad, template = bed24
    ct = attenuation.load(ct_dir)
    mu = attenuation.mu_image(ct, _bed_start(ct), template).as_array()
    row_profile = mu[mu.shape[0] // 2].sum(axis=1)
    assert row_profile[: len(row_profile) // 2].sum() > \
        row_profile[len(row_profile) // 2:].sum()


def test_mu_image_refuses_a_grid_that_is_not_a_bed(ct_dir, bed24, sirf):
    ad, _template = bed24
    ct = attenuation.load(ct_dir)
    wrong = ad.create_uniform_image(1.0, (10, 32, 32))
    with pytest.raises(SystemExit, match="is not"):
        attenuation.mu_image(ct, _bed_start(ct), wrong)


def test_mu_image_refuses_a_bed_far_outside_the_ct(ct_dir, bed24):
    _ad, template = bed24
    ct = attenuation.load(ct_dir)
    with pytest.raises(SystemExit, match="overhang"):
        attenuation.mu_image(ct, float(ct.z[-1]) + 50.0, template)


def test_mu_image_clamps_a_small_overhang_instead_of_filling_air(ct_dir, bed24, capsys):
    """The first and last bed of an exam always stick out by a few mm.

    Filling that with air would under-correct attenuation through tissue that
    is really there, so the outermost CT slice is repeated instead.
    """
    _ad, template = bed24
    ct = attenuation.load(ct_dir)
    inside = _bed_start(ct)
    over = float(ct.z[0]) - 1.0 * attenuation.PLANE_MM      # one plane short
    mu_in = attenuation.mu_image(ct, inside, template).as_array()
    mu_over = attenuation.mu_image(ct, over, template).as_array()
    assert "warning" in capsys.readouterr().out
    # The overhanging plane repeats slice 0 rather than going to air (mu = 0).
    assert mu_over[0].max() > 0.5 * mu_in[0].max()


def test_overhang_tolerance_is_counted_in_pet_planes(ct_dir, bed24, tmp_path):
    """`edge_tol_planes` must mean PET planes whatever the CT pitch is.

    A 1.25 mm CT and a 3.27 mm CT that stick out by the same *distance* have to
    be judged the same; measuring the overhang in CT slices instead would make
    the finer series 2.6x stricter for no physical reason.
    """
    _ad, template = bed24
    coarse = attenuation.load(ct_dir)                         # 3.27 mm pitch
    fine = attenuation.load(synth_ct.series(
        tmp_path / "fine", n_slices=160, dz=1.25, z0=-100.0))
    span = (attenuation.PLANES_PER_BED - 1) * attenuation.PLANE_MM
    out_mm = 1.2 * attenuation.PLANE_MM                       # 1.2 planes over

    for ct in (coarse, fine):
        assert float(ct.z[-1] - ct.z[0]) > span + out_mm, "series too short to test"
        attenuation.mu_image(ct, float(ct.z[0]) - out_mm, template)
        with pytest.raises(SystemExit, match="overhang"):
            attenuation.mu_image(ct, float(ct.z[0]) - 3.0 * attenuation.PLANE_MM,
                                 template)


def test_factors_are_survival_probabilities(ct_dir, bed24):
    """`af` is a probability in (0, 1]; `acf` is its reciprocal, not its twin."""
    ad, template = bed24
    ct = attenuation.load(ct_dir)
    mu = attenuation.mu_image(ct, _bed_start(ct), template)
    af, acf = attenuation.factors(ad, mu)
    a, c = af.as_array(), acf.as_array()
    assert 0 < a.min() and a.max() <= 1.0 + 1e-6
    assert (c >= 1.0 - 1e-6).all()
    nz = a > 0
    assert np.allclose(a[nz] * c[nz], 1.0, rtol=1e-4)
