"""`lm.geom` -- the crystal <-> bin maps, on the miniature scanner.

The real scanner's map is proven on real data by `test_lm_data.py`; this module
proves the *rules* (inversion, multiplicity, TOF order) without needing any.
"""

from __future__ import annotations

import numpy as np
import pytest

from lm import events as ev
from lm import geom, interfile
from utils import geometry, scanner
from utils.terms import NSEG0

RINGS, NDET, NTANG = 6, 16, 9


@pytest.fixture(scope="module")
def binmap(mini_hs):
    """No `stir` fixture: `lm` parses the header itself, see `lm.interfile`."""
    return geom.BinMap(mini_hs)


def _xtal(binmap, ring, det):
    """The GE crystal id whose STIR detector number is `det`."""
    trans = int(np.argsort(binmap.xtal2det)[det])
    return ring * binmap.ndet + trans


# ---------------------------------------------- the parser vs STIR itself
# `lm` runs where there is no STIR, so it reads the header itself. These two
# tests are the only thing that keeps that parser honest -- they need STIR, and
# skip without it, but the code under test never does.
def test_header_segments_match_stir(mini_hs, mini_info):
    _pd, info = mini_info
    h = interfile.Header(mini_hs)
    got = [(s, lo, hi, n) for s, lo, hi, n in h.segments()]
    want = [(s, info.get_min_ring_difference(s), info.get_max_ring_difference(s),
             info.get_num_axial_poss(s)) for s in geometry.segment_order(info)]
    assert got == want


def test_header_ring_pairs_match_stir(mini_hs, mini_info):
    _pd, info = mini_info
    h = interfile.Header(mini_hs)
    want = geometry.plane_ring_pairs(info, h.n_rings)
    geometry.check_ring_pairs(info, want)
    assert h.ring_pairs() == want


def test_sirf_written_layout_is_refused(tmp_path, mini_hs):
    """A header with view before axial is SIRF's own order -- and unreadable here."""
    p = tmp_path / "attn.hs"
    p.write_text(open(mini_hs).read().replace(
        "matrix axis label [3] := axial coordinate",
        "matrix axis label [3] := view"))
    assert not interfile.Header(p).plane_major
    with pytest.raises(SystemExit, match="SIRF's own segment order"):
        interfile.Header(p).require_plane_major()


# --------------------------------------------------------------- the maps
def test_shape_matches_the_header(binmap, mini_info):
    _pd, info = mini_info
    assert binmap.n_view == info.get_num_views()
    assert binmap.n_tang == info.get_num_tangential_poss()
    assert binmap.n_plane == sum(info.get_num_axial_poss(s)
                                 for s in geometry.segment_order(info))


def test_multiplicity_is_stirs_own(binmap, mini_info):
    _pd, info = mini_info
    assert np.array_equal(binmap.mult,
                          geometry.ring_pair_multiplicity(info).astype(np.int32))
    # span 2: only the odd axial positions of segment 0 gather two ring pairs
    assert set(np.unique(binmap.mult)) <= {1, 2}
    assert binmap.mult[:2 * RINGS - 1][1::2].tolist() == [2] * (RINGS - 1)


def test_flat_inverts_det_pair_map(binmap):
    """Every `(view, tang)` comes back from the crystal pair it is made of."""
    d1, d2 = geometry.det_pair_map(binmap.n_view, binmap.n_tang, NDET)
    inv = np.argsort(binmap.xtal2det)
    for v in range(binmap.n_view):
        a = inv[d1[v]] + 0 * NDET               # ring 0 for both crystals
        b = inv[d2[v]] + 0 * NDET
        got = binmap.flat(a, b)
        plane = binmap.pl[0, 0]                 # ring pair (0, 0) -> its plane
        want = plane * binmap.n_view * binmap.n_tang + v * binmap.n_tang \
            + np.arange(binmap.n_tang)
        assert np.array_equal(got, want)


def test_ring_order_follows_pos1_minus_pos2(binmap, mini_info):
    """`(r1, r2)` and `(r2, r1)` land in the +k and -k segments, not the same one."""
    _pd, info = mini_info
    order = geometry.segment_order(info)
    seg_of = np.concatenate([[s] * info.get_num_axial_poss(s) for s in order])
    p1 = binmap.pl[3, 0]        # ring difference +3
    p2 = binmap.pl[0, 3]
    assert p1 >= 0 and p2 >= 0
    assert seg_of[p1] == -seg_of[p2] != 0


def test_lor_table_covers_every_bin_with_its_multiplicity(binmap):
    ids, bins = binmap.lor_table()
    assert len(ids) == int(binmap.mult.sum()) * binmap.n_view * binmap.n_tang
    n = np.bincount(bins, minlength=binmap.n_bin)
    assert np.array_equal(n, np.repeat(binmap.mult,
                                       binmap.n_view * binmap.n_tang))
    # and each LOR really does map back to the bin the table claims
    assert np.array_equal(binmap.flat(ids[:, 0], ids[:, 1]), bins)


def test_out_of_range_pairs_are_dropped(binmap):
    """Adjacent crystals make a chord no tangential bin holds -> -1, not a wrong bin."""
    a = np.arange(NDET, dtype=np.int32)
    b = (a + 1) % NDET
    assert (binmap.flat(a, b) < 0).all()


# ---------------------------------------------------------------- the LUT
def test_scanner_lut_is_a_cylinder():
    lut = geom.scanner_lut()
    assert lut.shape == (geom.NXTAL, 3)
    r = np.hypot(lut[:, 0], lut[:, 1])
    assert np.allclose(r, geom.R_EFF_MM, atol=1e-3)   # crystal face + DOI, not R_MM
    z = lut[:, 2].reshape(geom.NRINGS, geom.NDET)
    assert np.allclose(np.diff(z[:, 0]), geom.RING_PITCH_MM)
    assert abs(z.mean()) < 1e-4                     # centred on the bed
    # the LUT must reach past the outermost image plane, or PyTomography zeroes it
    assert z.max() >= (NSEG0 - 1) / 2 * geometry.PLANE_MM


def test_scanner_lut_is_in_stirs_frame():
    """Detector 0 at (0, -R), and GE's crystal order reversed into STIR's.

    Both were measured against the sinogram reconstruction of ped bed 1
    (`tools/lm_frame.py`); this pins the result so it cannot drift.
    """
    lut = geom.scanner_lut(offset_deg=0.0)
    d0 = int(np.argsort(geometry.crystal_to_det(geom.NDET))[0])   # GE id of det 0
    assert np.allclose(lut[d0, :2], [0.0, -geom.R_EFF_MM], atol=1e-3)
    # and the raw GE frame is the mirror image, not the same thing
    ge = geom.scanner_lut(offset_deg=0.0, stir_frame=False)
    assert not np.allclose(ge[:, :2], lut[:, :2])


# ---------------------------------------------------------------- the TOF
def test_tof_to_stir_reverses_ge_order():
    t = np.arange(-27, 28)
    idx = geom.tof_to_stir(t)
    assert idx.min() == 0 and idx.max() == 54
    assert np.array_equal(idx, 27 - t)              # GE's last bin is STIR's first
    assert np.array_equal(idx[::-1], np.arange(55))


@pytest.mark.parametrize("n_out", [1, 5, 11, 55])
def test_mashing_commutes_with_the_reversal(n_out):
    """Reverse-then-mash == mash-then-reverse, because the factor divides 55."""
    t = np.arange(-27, 28)
    mash = 55 // n_out
    assert np.array_equal(geom.tof_to_stir(t, n_out),
                          (n_out - 1) - (t + 27) // mash)


def test_tof_bins_must_divide_55():
    with pytest.raises(ValueError):
        geom.tof_to_stir(np.zeros(3, np.int8), 7)


# ---------------------------------------------------------- the histogram
def _events(binmap, n=5000, seed=0):
    """Random *valid* events: pick bins, then read their crystal pair back out."""
    rng = np.random.default_rng(seed)
    ids, bins = binmap.lor_table()
    k = rng.integers(0, len(ids), n)
    e = np.zeros(n, dtype=[("xtal_a", "<u2"), ("xtal_b", "<u2"),
                           ("tof_bin", "i1"), ("t_ms", "<u4")])
    e["xtal_a"], e["xtal_b"] = ids[k, 0], ids[k, 1]
    e["tof_bin"] = rng.integers(-27, 28, n)
    return e, bins[k]


def test_histogram_counts_every_event(binmap):
    e, bins = _events(binmap)
    h, dropped = ev.histogram(e, binmap, n_tof=1)
    assert dropped == 0
    assert h.shape == (1,) + binmap.shape
    assert int(h.sum()) == len(e)
    assert np.array_equal(h.ravel(), np.bincount(bins, minlength=binmap.n_bin))


def test_histogram_splits_the_tof_axis_without_losing_counts(binmap):
    e, _ = _events(binmap)
    h1, _ = ev.histogram(e, binmap, n_tof=1)
    h5, _ = ev.histogram(e, binmap, n_tof=5)
    assert h5.shape[0] == 5
    assert np.array_equal(h5.sum(axis=0, dtype=np.int64), h1[0].astype(np.int64))


def test_tof_index_mirrors_for_the_reversed_crystal_pair(binmap):
    """A bin has one direction, so an event recorded the other way round mirrors.

    Same LOR, same spatial bin, same arrival-time difference -- but the pair
    written `(b, a)` describes the opposite displacement, so its TOF index has to
    come out mirrored. Getting this wrong leaves the counts and every invariant
    untouched and only moves the activity to the wrong half of the LOR.
    """
    e, _ = _events(binmap, n=2000)
    r = e.copy()
    r["xtal_a"], r["xtal_b"] = e["xtal_b"], e["xtal_a"]

    assert np.array_equal(ev.bins(e, binmap), ev.bins(r, binmap))   # same bin
    _, sa = binmap.flat(e["xtal_a"], e["xtal_b"], with_swap=True)
    _, sb = binmap.flat(r["xtal_a"], r["xtal_b"], with_swap=True)
    assert np.array_equal(sa, ~sb)                                  # opposite sense

    n = 11
    assert np.array_equal(ev.tof_index(e, binmap, n),
                          (n - 1) - ev.tof_index(r, binmap, n))


def test_histogram_tof_follows_the_bin_direction(binmap):
    """The mirror really reaches the histogram, and only the TOF axis moves."""
    e, _ = _events(binmap, n=4000)
    r = e.copy()
    r["xtal_a"], r["xtal_b"] = e["xtal_b"], e["xtal_a"]
    h, _ = ev.histogram(e, binmap, 11)
    g, _ = ev.histogram(r, binmap, 11)
    assert np.array_equal(h[::-1], g)
    assert np.array_equal(h.sum(0), g.sum(0))       # invisible without the TOF axis


def test_detector_ids_keep_the_events_own_frame(binmap):
    """`detector_ids` must NOT mirror: PyTomography gets the recorded pair order."""
    e, _ = _events(binmap, n=500)
    r = e.copy()
    r["xtal_a"], r["xtal_b"] = e["xtal_b"], e["xtal_a"]
    assert np.array_equal(ev.detector_ids(e, 11)[:, 2],
                          ev.detector_ids(r, 11)[:, 2])


def test_detector_ids_are_zero_based_and_signed_the_same_way(binmap):
    e, _ = _events(binmap)
    ids = ev.detector_ids(e, 55)
    assert ids.shape == (len(e), 3) and ids.dtype == np.int32
    assert ids[:, 2].min() >= 0 and ids[:, 2].max() <= 54
    assert np.array_equal(ev.detector_ids(e, 55, tof_sign=-1)[:, 2],
                          54 - ids[:, 2])


# ---------------------------------------------- the object's axial support
# `lm.recon.axial_mask` replaces `PETLMSystemMatrix._get_object_initial`'s own
# axial cut. Nothing here needs PyTomography: the rule is arithmetic on the
# real grid, and the bug it fixes was arithmetic too.
def test_every_bed_plane_is_in_the_axial_support():
    """All 47 planes, or a bed comes back with a dead one -- and OSEM is multiplicative.

    Plane 46 was zero in every list-mode bed until 2026-09-05, because
    `_get_object_initial` floors `zmax`, which lands on exactly 46.0 for this
    grid. `osem.stitch` then averaged that hard zero into the seam with the
    weight `norm_BP` gives plane 46 -- the bright line at each bed junction.
    """
    from lm.recon import axial_mask

    m = axial_mask(NSEG0, scanner.PLANE_MM, geom.scanner_lut()[:, 2])
    assert m.shape == (NSEG0,)
    assert m.all(), f"dead image planes: {np.flatnonzero(~m).tolist()}"


def test_the_axial_support_is_symmetric_and_still_guards_a_taller_grid():
    """Round, not floor/ceil -- but a grid past the rings is still cut, at both ends."""
    from lm.recon import axial_mask

    dz, z = 1.0, np.array([-3.0, 3.0])           # rings at +-3, planes 1 mm apart
    assert list(axial_mask(7, dz, z)) == [True] * 7          # exactly the extent
    assert list(axial_mask(9, dz, z)) == [False] + [True] * 7 + [False]
    m = axial_mask(11, dz, z)
    assert list(m) == [False] * 2 + [True] * 7 + [False] * 2
    assert list(m) == list(m[::-1])                          # no end favoured
