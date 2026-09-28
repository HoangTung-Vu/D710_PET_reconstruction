"""`simulation/`: inputs, crystal map, TOF convention, the case it writes.

Nothing here runs GATE (a run takes minutes); the GATE-specific pieces that
can be checked without it -- the crystal placement and its GE ids, the TOF
formula, the coincidence branch names -- are.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from simulation import crystals as cr
from simulation import events_io as eio
from simulation import phantom as ph
from utils.geometry import crystal_positions
from utils.scanner import DR_MM, NDET, NRINGS, NSEG0, PLANE_MM, XY

RINGS, NDET_MINI, NTANG = 6, 16, 9


# --- inputs -----------------------------------------------------------------

def _nifti(tmp_path, data_zyx, x0, y0, z0, px, dz, name="v.nii.gz"):
    """Written exactly as `tools/dicom_suv.write_nifti` writes it."""
    import nibabel as nib

    aff = np.array([[-px, 0, 0, -x0], [0, -px, 0, -y0], [0, 0, dz, z0], [0, 0, 0, 1.0]])
    p = tmp_path / name
    nib.save(nib.Nifti1Image(np.transpose(data_zyx, (2, 1, 0)).astype(np.float32), aff), p)
    return p


def test_nifti_comes_back_in_dicom_order(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.random((7, 11, 13)).astype(np.float32)
    v = ph.load_volume(_nifti(tmp_path, a, -40.0, -30.0, -900.0, 2.5, 3.27), "PT")
    assert np.array_equal(v.data, a)
    assert (v.x0, v.y0, v.pixel_mm) == (-40.0, -30.0, 2.5)
    assert np.allclose(v.z, -900.0 + 3.27 * np.arange(7))


def test_a_flipped_nifti_axis_is_undone(tmp_path):
    import nibabel as nib

    rng = np.random.default_rng(1)
    a = rng.random((5, 6, 8)).astype(np.float32)
    ref = ph.load_volume(_nifti(tmp_path, a, -10.0, -12.0, 100.0, 2.0, 3.0), "PT")
    aff = np.array([[2.0, 0, 0, 10.0 - 2.0 * 7], [0, -2.0, 0, 12.0],
                    [0, 0, 3.0, 100.0], [0, 0, 0, 1]])
    p = tmp_path / "flip.nii.gz"
    nib.save(nib.Nifti1Image(np.transpose(a[:, :, ::-1], (2, 1, 0)), aff), p)
    v = ph.load_volume(p, "PT")
    assert np.allclose(v.data, ref.data) and np.isclose(v.x0, ref.x0)


def test_resample_to_bed_is_what_mu_map_always_did():
    """The refactor that split the resampler out of `mu_map` changed nothing."""
    from scipy.ndimage import map_coordinates

    from utils.attenuation import CTAC, hu_to_mu, mu_map, to_radiological

    rng = np.random.default_rng(2)
    hu = rng.uniform(-1000, 1500, (80, 64, 64)).astype(np.float32)
    z = -300.0 + 3.0 * np.arange(80)
    ct = CTAC(hu, z, -120.0, -110.0, 3.9, 120.0, {})
    tp, xy = -250.0, 41
    zc = tp + np.arange(NSEG0) * PLANE_MM
    c = (np.arange(xy) - xy // 2) * DR_MM
    g = np.meshgrid((zc - z[0]) / 3.0, (c - ct.y0) / ct.pixel_mm,
                    (c - ct.x0) / ct.pixel_mm, indexing="ij")
    old = map_coordinates(hu, [x.ravel() for x in g], order=1, mode="constant",
                          cval=-1000.0).reshape(NSEG0, xy, xy)
    old = np.ascontiguousarray(to_radiological(hu_to_mu(old, 120.0) * 10.0), np.float32)
    assert np.array_equal(mu_map(ct, tp, xy, DR_MM), old)


def test_suv_inverts_to_bqml():
    from utils.quant import suv_bw

    t = {"t_scan": 10_000.0, "t_inj": 10_000.0 - 3600.0, "dose_bq": 250e6,
         "weight_kg": 70.0, "half_life_s": 6586.2, "positron_fraction": 0.967}
    bq = np.full((3, 4, 4), 5000.0, np.float32)
    dose_ref = t["dose_bq"] * 2 ** (-3600.0 / t["half_life_s"])
    suv = suv_bw(bq, dose_ref, t["weight_kg"])
    v = ph.Volume(suv, np.arange(3) * 3.0, 0.0, 0.0, 2.0, "test")
    f, info = ph.bqml_scale(v, "suv", t)
    assert np.allclose(suv * f, bq, rtol=1e-6)


def test_relative_map_holds_the_global_activity():
    t = {"t_scan": 1000.0, "t_inj": 0.0, "dose_bq": 1.0, "weight_kg": 1.0,
         "half_life_s": 6586.2, "positron_fraction": 1.0}
    v = ph.Volume(np.random.default_rng(3).random((4, 5, 5)).astype(np.float32),
                  np.arange(4) * 3.0, 0.0, 0.0, 2.0, "test")
    f, info = ph.bqml_scale(v, "relative", t, activity=(100e6, 1000.0))
    assert np.isclose(float(v.data.sum()) * v.voxel_ml * f, 100e6, rtol=1e-6)


def test_the_extended_grid_is_centred_on_the_world_origin():
    first, n = ph.grid_planes(200.0)
    assert first < 0 and n == NSEG0 - 2 * first
    o = ph.world_origin((n, XY, XY))
    centre = o + (np.array([XY, XY, n]) - 1) / 2 * np.array(ph.VOXEL_XYZ)
    assert np.allclose(centre, 0.0, atol=1e-4)


def test_mhd_round_trip(tmp_path):
    a = np.random.default_rng(4).random((3, 5, 7)).astype(np.float32)
    p = ph.write_mhd(tmp_path / "x.mhd", a)
    assert np.array_equal(ph.read_mhd(p), a)


# --- crystals ---------------------------------------------------------------

def test_gate_crystals_map_one_to_one_onto_ge_ids():
    c = cr.gate_crystal_centres()
    ids = cr.ge_ids(c)
    assert c.shape == (NRINGS * NDET, 3)
    assert np.unique(ids).size == ids.size
    phi = np.arctan2(c[:, 1], c[:, 0])
    err = np.angle(np.exp(1j * (phi - cr.ge_azimuth()[ids % NDET])))
    assert np.abs(err).max() < 0.25 * 2 * np.pi / NDET
    lut = crystal_positions()
    assert np.allclose(c[:, 2], lut[ids, 2], atol=1e-4)


def test_the_lookup_returns_the_crystal_of_its_own_centre():
    look = cr.CrystalLookup()
    rng = np.random.default_rng(5)
    i = rng.integers(0, NRINGS * NDET, 2000)
    assert np.array_equal(look(look.centres[i]), i)


def test_blocks_sit_inside_the_ring_without_touching():
    from simulation.gate.geometry import RING_RMAX_MM, RING_RMIN_MM

    tr, rot = cr.block_placements()
    hw = cr.BLOCK_SIZE_MM[1] / 2
    for t, r in zip(tr, rot):
        for sx in (-1, 1):
            for sy in (-1, 1):
                p = np.asarray(t) + np.asarray(r) @ np.array(
                    [sx * cr.BLOCK_SIZE_MM[0] / 2, sy * hw, 0.0])
                assert RING_RMIN_MM < math.hypot(p[0], p[1]) < RING_RMAX_MM
    phi = np.sort(cr.block_azimuth())
    gap = np.diff(np.r_[phi, phi[0] + 2 * np.pi])
    inner_half = math.atan2(hw, cr.BLOCK_CENTRE_R_MM - cr.BLOCK_SIZE_MM[0] / 2)
    assert (gap > 2 * inner_half).all()


# --- TOF --------------------------------------------------------------------

def _point_image(xyz_mm, shape=(65, 65, 15), vox=(4.0, 4.0, 4.0)):
    img = np.zeros(shape, np.float32)
    idx = tuple(int(round(v / d + (n - 1) / 2)) for v, d, n in zip(xyz_mm, vox, shape))
    img[idx] = 1.0
    org = ((-np.asarray(shape) / 2 + 0.5) * np.asarray(vox)).astype(np.float32)
    return img, org, np.asarray(vox, np.float32)


def test_parallelproj_tof_bins_run_towards_the_lor_end():
    import parallelproj

    from simulation import pp

    img, org, vox = _point_image((80.0, 0.0, 0.0))
    xs = np.array([[-400.0, 0, 0]], np.float32)
    xe = np.array([[400.0, 0, 0]], np.float32)
    w, s, o = pp.tof_kernel()
    p = parallelproj.joseph3d_fwd_tof_sino(xs, xe, img, org, vox, w, s, o, 3.0, 55)
    assert int(p[0].argmax()) - 27 == int(round(80.0 / w))


def test_the_tof_sign_is_the_one_the_reconstruction_reads():
    """An event drawn by `pp` projects, through `lm`'s own mapping, onto the value it was drawn from."""
    import parallelproj

    from lm import events as lmev
    from simulation import pp

    lut = crystal_positions()
    a, b = 3 * NDET + 10, 17 * NDET + 300
    near_a = 0.7 * lut[a] + 0.3 * lut[b]
    img, org, vox = _point_image(tuple(near_a), shape=(211, 211, 41), vox=(4.0, 4.0, 4.0))
    w, s, o = pp.tof_kernel()
    sino = parallelproj.joseph3d_fwd_tof_sino(lut[[a]], lut[[b]], img, org, vox,
                                              w, s, o, 3.0, 55)[0]
    j = int(sino.argmax())
    tof_bin = 27 - j
    assert tof_bin > 0, "nearer xtal_a must be a positive GE bin"
    e = eio.events([a], [b], [tof_bin], [0])
    t = lmev.detector_ids(e, 55)[:, 2] - 27
    lm_val = parallelproj.joseph3d_fwd_tof_lm(lut[[a]], lut[[b]], img, org, vox, w, s, o,
                                              3.0, t.astype(np.int16))
    assert np.isclose(float(lm_val[0]), float(sino[j]), rtol=1e-5)


def test_gate_timing_gives_the_same_sign():
    """A pair nearer `xtal_a` reaches it first, so `t_b - t_a > 0` is a positive bin."""
    d_mm = 100.0
    dt_ns = 2 * d_mm / eio.C_MM_NS
    assert eio.tof_bin_from_dt(dt_ns) == round(d_mm / eio.TOF_BIN_MM)
    assert eio.tof_bin_from_dt(-dt_ns) == -round(d_mm / eio.TOF_BIN_MM)


def test_swapping_the_crystals_mirrors_the_tof_bin():
    rng = np.random.default_rng(6)
    xa, xb, t = np.arange(1000), np.arange(1000) + 5000, rng.integers(-27, 28, 1000)
    a2, b2, t2 = eio.swap_randomly(xa, xb, t, rng)
    s = a2 != xa
    assert s.any() and (~s).any()
    assert np.array_equal(b2[s], xa[s]) and np.array_equal(t2[s], -t[s])


def test_coincidence_branch_spellings_are_normalised():
    from simulation.gate.coinc import _normalise

    c = _normalise({"PostPosition1_X": [1.0], "PostPosition_Y2": [2.0], "EventID1": [3]})
    assert set(c) == {"PostPosition_X1", "PostPosition_Y2", "EventID1"}


# --- the model on a miniature scanner ---------------------------------------

@pytest.fixture(scope="module")
def mini(mini_hs):
    from utils.binmap import BinMap

    return BinMap(mini_hs)


def test_ring_pairs_cover_the_lor_table(mini):
    from simulation.pp import RingPairs

    ids, bins = mini.lor_table()
    got_ids, got_bins = [], []
    for _p, a, b, k in RingPairs(mini):
        got_ids.append(np.stack([a, b], 1))
        got_bins.append(k)
    assert np.array_equal(np.concatenate(got_ids), ids)
    assert np.array_equal(np.concatenate(got_bins), bins)


def test_pp_draws_poisson_with_the_stated_mean(mini):
    from simulation import pp

    lut = crystal_positions(nrings=RINGS, ndet=NDET_MINI, r_mm=60.0, pitch_mm=6.54)
    shape = (21, 21, 11)
    vox = (4.0, 4.0, 3.27)
    org = ((-np.asarray(shape) / 2 + 0.5) * np.asarray(vox)).astype(np.float32)
    x = (np.full(shape, 1.0, np.float32), org, np.asarray(vox, np.float32))
    mu = (np.zeros(shape, np.float32), x[1], x[2])
    pairs = pp.RingPairs(mini)
    n_lor = np.ones(mini.n_bin, np.float32)
    tot = []
    for s in range(20):
        _a, _b, _t, lab, exp = pp.simulate(pairs, lut, mu, x, n_lor, 0.05,
                                           np.random.default_rng(s), n_tof=1,
                                           out=lambda *_: None)
        tot.append(len(lab))
    mean = exp[0]
    assert mean > 100
    assert abs(np.mean(tot) - mean) < 4 * math.sqrt(mean / len(tot))


def test_norm_acceptance_only_removes(mini):
    from simulation.gate.coinc import norm_acceptance

    n = np.random.default_rng(8).uniform(0.2, 2.0, mini.n_bin).astype(np.float32)
    p = norm_acceptance(n, mini)
    assert p.min() >= 0 and p.max() <= 1
    r = n.reshape(mini.shape) / n.reshape(mini.shape).mean(axis=1, keepdims=True)
    assert np.allclose(p.reshape(mini.shape)[r <= 1], r[r <= 1], rtol=1e-5)


def test_sampled_times_stay_in_the_frame():
    from simulation.pp import sample_times

    t = sample_times(100_000, 90.0, 6586.2, np.random.default_rng(9))
    assert t.min() >= 0 and t.max() < 90_000
    first, last = (t < 10_000).sum(), (t >= 80_000).sum()
    assert first > last


def test_a_written_bed_reads_back_as_an_ordinary_case(tmp_path, mini, mini_hs):
    """The sinogram on disk is the histogram of the event table on disk."""
    import shutil

    import synth_hs
    from lm.events import histogram
    from utils.paths import Case

    real = Case("real", tmp_path)
    real.mkdirs()
    shutil.copy(mini_hs, real.prompt(1))
    shutil.copy(mini_hs[:-3] + ".s", real.decoded / "mini.s")
    (real.decoded / "bed1.json").write_text(json.dumps(
        {"prompts": 0, "delays": 0, "frame_duration_ms": 90000, "half_life_s": 6586.2}))
    w = real.work_bed(1)
    w.mkdir(parents=True)
    hs = synth_hs.header(RINGS, NDET_MINI, NTANG, data_file="randoms.s",
                         number_format="float", bytes_per_pixel=4)
    (w / "randoms.hs").write_text(hs)
    np.ones(mini.n_bin, "<f4").tofile(w / "normdt.s")
    (w / "normdt.hs").write_text(hs.replace("randoms.s", "normdt.s"))

    ids, _ = mini.lor_table()
    rng = np.random.default_rng(10)
    k = rng.integers(0, len(ids), 5000)
    ev = eio.events(ids[k, 0], ids[k, 1], rng.integers(-27, 28, 5000),
                    rng.integers(0, 90000, 5000))
    dst = Case("sim", tmp_path)
    row = eio.write_bed(real, dst, 1, ev, {"randoms": np.full(mini.shape, 0.1, np.float32),
                                          "scatter": np.full(mini.shape, 0.2, np.float32)},
                        {"delays": 7}, {"label": np.zeros(5000, np.int8)})
    e2 = np.load(dst.decoded / "bed1.lm.npy")
    assert e2.dtype == eio.EVENT_DTYPE and len(e2) == 5000
    assert np.all(np.diff(e2["t_ms"].astype(np.int64)) >= 0)
    h, _ = histogram(e2, mini, 1)
    assert np.array_equal(np.fromfile(dst.decoded / "bed1.s", "<i2"), h.ravel())
    hdr = json.loads((dst.decoded / "bed1.json").read_text())
    assert hdr["prompts"] == row["prompts"] == 5000 and hdr["delays"] == 7
    bg = np.fromfile(dst.work_bed(1) / "background.s", "<f4")
    assert np.allclose(bg, 0.3)
    assert (dst.work_bed(1) / "normdt.hs").exists()


def test_a_crystal_named_by_volume_id_is_the_one_placed_there():
    """`d710_crystal_rep_<c>-0_0_<block>_<c>`: block z-major, crystal as `crystal_offsets`."""
    look = cr.CrystalLookup()
    rng = np.random.default_rng(11)
    blocks, crys = rng.integers(0, 256, 500), rng.integers(0, 54, 500)
    vid = [f"d710_crystal_rep_{c}-0_0_{b}_{c}" for b, c in zip(blocks, crys)]
    gi = cr.volume_id_to_gate_index(vid)
    assert np.array_equal(gi, blocks * 54 + crys)
    centres = cr.gate_crystal_centres()[gi]
    origin = np.zeros_like(centres)
    got = look(np.where(rng.random((500, 1)) < 0.5, centres, origin), vid)
    assert np.array_equal(got, cr.ge_ids(centres))
    assert look.n_fallback > 0


def test_a_bed_plane_on_the_edge_of_the_series_is_not_emptied():
    """Plane 0 exactly on the first slice, 2e-5 mm outside after float32 rounding."""
    from utils.attenuation import resample_to_bed

    vol = np.ones((60, 20, 20), np.float32)
    z = np.float32(-873.98) + 3.27 * np.arange(60)
    kw = dict(x0=-20.0, y0=-20.0, pixel_mm=2.0, table_position_mm=-873.98,
              xy=9, dr_mm=2.0, cval=0.0)
    old = resample_to_bed(vol, z, **kw)
    new = resample_to_bed(vol, z, clamp_edges=True, **kw)
    assert old[0].max() == 0.0, "the case this guards against no longer arises"
    assert np.all(new[:, 4, 4] == 1.0)


def test_export_scales_a_simulated_case_like_a_thinned_one(tmp_path):
    """`simulation.json` carries T_real / T_sim, and `quant.lowdose_k_scale` reads it."""
    from utils.paths import Case
    from utils.quant import lowdose_k_scale

    dst = Case("sim", tmp_path)
    dst.root.mkdir(parents=True)
    eio.manifest(dst, {"method": "gate", "beds": [{"bed": 7, "k_scale": 10.0},
                                                  {"bed": 6, "k_scale": 10.0}]})
    assert lowdose_k_scale(dst) == 10.0
    eio.manifest(dst, {"beds": [{"bed": 5, "k_scale": 30.0}]})
    with pytest.raises(SystemExit):
        lowdose_k_scale(dst)
