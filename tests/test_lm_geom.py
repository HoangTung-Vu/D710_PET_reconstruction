"""`lm.geom`: the crystal and bin maps, on the miniature scanner."""

from __future__ import annotations

import numpy as np
import pytest

from lm import events as ev
from lm import geom
import stir_oracle
from utils import geometry, interfile, scanner
from utils.terms import NSEG0

RINGS, NDET, NTANG = 6, 16, 9


@pytest.fixture(scope="module")
def binmap(mini_hs):
    """A `BinMap` built from the header by `utils.interfile`, without the `stir` fixture."""
    return geom.BinMap(mini_hs)


def _xtal(binmap, ring, det):
    trans = int(np.argsort(binmap.xtal2det)[det])
    return ring * binmap.ndet + trans


def test_header_segments_match_stir(mini_hs, mini_info):
    _pd, info = mini_info
    h = interfile.Header(mini_hs)
    got = [(s, lo, hi, n) for s, lo, hi, n in h.segments()]
    want = [(s, info.get_min_ring_difference(s), info.get_max_ring_difference(s),
             info.get_num_axial_poss(s)) for s in stir_oracle.segment_order(info)]
    assert got == want


def test_header_ring_pairs_match_stir(mini_hs, mini_info):
    _pd, info = mini_info
    h = interfile.Header(mini_hs)
    want = stir_oracle.plane_ring_pairs(info, h.n_rings)
    stir_oracle.check_ring_pairs(info, want)
    assert h.ring_pairs() == want


def test_sirf_written_layout_is_refused(tmp_path, mini_hs):
    p = tmp_path / "attn.hs"
    p.write_text(open(mini_hs).read().replace(
        "matrix axis label [3] := axial coordinate",
        "matrix axis label [3] := view"))
    assert not interfile.Header(p).plane_major
    with pytest.raises(SystemExit, match="SIRF's own segment order"):
        interfile.Header(p).require_plane_major()


def test_shape_matches_the_header(binmap, mini_info):
    _pd, info = mini_info
    assert binmap.n_view == info.get_num_views()
    assert binmap.n_tang == info.get_num_tangential_poss()
    assert binmap.n_plane == sum(info.get_num_axial_poss(s)
                                 for s in stir_oracle.segment_order(info))


def test_multiplicity_is_stirs_own(binmap, mini_info):
    _pd, info = mini_info
    assert np.array_equal(binmap.mult,
                          stir_oracle.ring_pair_multiplicity(info).astype(np.int32))
    assert set(np.unique(binmap.mult)) <= {1, 2}
    assert binmap.mult[:2 * RINGS - 1][1::2].tolist() == [2] * (RINGS - 1)


def test_flat_inverts_det_pair_map(binmap):
    d1, d2 = geometry.det_pair_map(binmap.n_view, binmap.n_tang, NDET)
    inv = np.argsort(binmap.xtal2det)
    for v in range(binmap.n_view):
        a = inv[d1[v]] + 0 * NDET
        b = inv[d2[v]] + 0 * NDET
        got = binmap.flat(a, b)
        plane = binmap.pl[0, 0]
        want = plane * binmap.n_view * binmap.n_tang + v * binmap.n_tang \
            + np.arange(binmap.n_tang)
        assert np.array_equal(got, want)


def test_ring_order_follows_stirs_ring2_minus_ring1(binmap, mini_info):
    """`pl[ring1, ring2]` must land in the segment of `ring2 - ring1`.

    An earlier form asserted only that the two orders fall in opposite-signed
    segments, which holds under BOTH conventions and so constrained nothing.
    The sign decides which ring sits on `det1`; `tests/test_lm_data.py` proves
    the choice against GE's own data.
    """
    _pd, info = mini_info
    order = stir_oracle.segment_order(info)
    seg_of = np.concatenate([[s] * info.get_num_axial_poss(s) for s in order])

    checked = 0
    for r_det1 in range(binmap.nrings):
        for r_det2 in range(binmap.nrings):
            p = int(binmap.pl[r_det1, r_det2])
            if p < 0:
                continue
            s = int(seg_of[p])
            lo = info.get_min_ring_difference(s)
            hi = info.get_max_ring_difference(s)
            assert lo <= r_det2 - r_det1 <= hi, (
                f"pl[{r_det1}, {r_det2}] is plane {p}, segment {s}, whose ring "
                f"difference runs {lo}..{hi} -- but ring2 - ring1 = "
                f"{r_det2 - r_det1}.  The pairing has been reversed; "
                f"see utils/binmap.py.")
            checked += 1
    assert checked == int(binmap.mult.sum())


def test_ring_pairs_read_back_out_of_pl_agree_with_the_header(binmap):
    """`ring_pairs_by_plane` is what everything geometric must go through."""
    r1, r2, planes = binmap.ring_pairs_by_plane()
    assert np.array_equal(binmap.pl[r1, r2], planes)
    assert np.array_equal(np.bincount(planes, minlength=binmap.n_plane),
                          binmap.mult)

    # pl stores (ring on det1, ring on det2), which is ring_pairs reversed
    want = {p: sorted((b, a) for a, b in prs) for p, prs in
            enumerate(binmap.hdr.ring_pairs())}
    got = {p: [] for p in range(binmap.n_plane)}
    for a, b, p in zip(r1, r2, planes):
        got[int(p)].append((int(a), int(b)))
    assert {p: sorted(v) for p, v in got.items()} == want


def test_lor_table_covers_every_bin_with_its_multiplicity(binmap):
    ids, bins = binmap.lor_table()
    assert len(ids) == int(binmap.mult.sum()) * binmap.n_view * binmap.n_tang
    n = np.bincount(bins, minlength=binmap.n_bin)
    assert np.array_equal(n, np.repeat(binmap.mult,
                                       binmap.n_view * binmap.n_tang))
    assert np.array_equal(binmap.flat(ids[:, 0], ids[:, 1]), bins)


def test_out_of_range_pairs_are_dropped(binmap):
    a = np.arange(NDET, dtype=np.int32)
    b = (a + 1) % NDET
    assert (binmap.flat(a, b) < 0).all()


def test_scanner_lut_is_a_cylinder():
    lut = geom.scanner_lut()
    assert lut.shape == (geom.NXTAL, 3)
    r = np.hypot(lut[:, 0], lut[:, 1])
    assert np.allclose(r, geom.R_EFF_MM, atol=1e-3)
    z = lut[:, 2].reshape(geom.NRINGS, geom.NDET)
    assert np.allclose(np.diff(z[:, 0]), geom.RING_PITCH_MM)
    assert abs(z.mean()) < 1e-4
    assert z.max() >= (NSEG0 - 1) / 2 * scanner.PLANE_MM


def test_scanner_lut_is_in_stirs_frame():
    lut = geom.scanner_lut(offset_deg=0.0)
    d0 = int(np.argsort(geometry.crystal_to_det(geom.NDET))[0])
    assert np.allclose(lut[d0, :2], [0.0, -geom.R_EFF_MM], atol=1e-3)
    ge = geom.scanner_lut(offset_deg=0.0, stir_frame=False)
    assert not np.allclose(ge[:, :2], lut[:, :2])


def test_tof_to_stir_reverses_ge_order():
    t = np.arange(-27, 28)
    idx = geom.tof_to_stir(t)
    assert idx.min() == 0 and idx.max() == 54
    assert np.array_equal(idx, 27 - t)
    assert np.array_equal(idx[::-1], np.arange(55))


@pytest.mark.parametrize("n_out", [1, 5, 11, 55])
def test_mashing_commutes_with_the_reversal(n_out):
    t = np.arange(-27, 28)
    mash = 55 // n_out
    assert np.array_equal(geom.tof_to_stir(t, n_out),
                          (n_out - 1) - (t + 27) // mash)


def test_tof_bins_must_divide_55():
    with pytest.raises(ValueError):
        geom.tof_to_stir(np.zeros(3, np.int8), 7)


def _events(binmap, n=5000, seed=0):
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
    e, _ = _events(binmap, n=2000)
    r = e.copy()
    r["xtal_a"], r["xtal_b"] = e["xtal_b"], e["xtal_a"]

    assert np.array_equal(ev.bins(e, binmap), ev.bins(r, binmap))
    _, sa = binmap.flat(e["xtal_a"], e["xtal_b"], with_swap=True)
    _, sb = binmap.flat(r["xtal_a"], r["xtal_b"], with_swap=True)
    assert np.array_equal(sa, ~sb)

    n = 11
    assert np.array_equal(ev.tof_index(e, binmap, n),
                          (n - 1) - ev.tof_index(r, binmap, n))


def test_histogram_tof_follows_the_bin_direction(binmap):
    e, _ = _events(binmap, n=4000)
    r = e.copy()
    r["xtal_a"], r["xtal_b"] = e["xtal_b"], e["xtal_a"]
    h, _ = ev.histogram(e, binmap, 11)
    g, _ = ev.histogram(r, binmap, 11)
    assert np.array_equal(h[::-1], g)
    assert np.array_equal(h.sum(0), g.sum(0))


def test_detector_ids_keep_the_events_own_frame(binmap):
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


def test_every_bed_plane_is_in_the_axial_support():
    from lm.recon import axial_mask

    m = axial_mask(NSEG0, scanner.PLANE_MM, geom.scanner_lut()[:, 2])
    assert m.shape == (NSEG0,)
    assert m.all(), f"dead image planes: {np.flatnonzero(~m).tolist()}"


def test_the_axial_support_is_symmetric_and_still_guards_a_taller_grid():
    from lm.recon import axial_mask

    dz, z = 1.0, np.array([-3.0, 3.0])
    assert list(axial_mask(7, dz, z)) == [True] * 7
    assert list(axial_mask(9, dz, z)) == [False] + [True] * 7 + [False]
    m = axial_mask(11, dz, z)
    assert list(m) == [False] * 2 + [True] * 7 + [False] * 2
    assert list(m) == list(m[::-1])
