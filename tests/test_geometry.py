"""`utils/geometry.py`: the D710 to STIR bin conventions."""

from __future__ import annotations

import numpy as np
import pytest

from utils import geometry
import synth_hs


def test_segment_order_is_stirs_storage_order(mini_info):
    _pd, info = mini_info
    assert geometry.segment_order(info) == [0, 1, -1, 2, -2]


def test_derived_ring_pairs_match_stirs_own_count(mini_info):
    _pd, info = mini_info
    pairs = geometry.plane_ring_pairs(info, 6)
    assert len(pairs) == synth_hs.num_planes(6)
    geometry.check_ring_pairs(info, pairs)


def test_check_ring_pairs_rejects_a_wrong_table(mini_info):
    _pd, info = mini_info
    pairs = geometry.plane_ring_pairs(info, 6)
    pairs[1] = pairs[1][:1]
    with pytest.raises(ValueError, match="plane 1"):
        geometry.check_ring_pairs(info, pairs)


def test_ring_pairs_carry_the_segments_ring_difference(mini_info):
    _pd, info = mini_info
    pairs = geometry.plane_ring_pairs(info, 6)
    p = 0
    for s in geometry.segment_order(info):
        lo, hi = info.get_min_ring_difference(s), info.get_max_ring_difference(s)
        for _a in range(info.get_num_axial_poss(s)):
            for r1, r2 in pairs[p]:
                assert lo <= r1 - r2 <= hi, f"plane {p}: ({r1},{r2}) not in segment {s}"
            p += 1


def test_span_2_doubles_the_odd_planes_of_segment_0(mini_info):
    _pd, info = mini_info
    m = geometry.ring_pair_multiplicity(info)
    n0 = info.get_num_axial_poss(0)
    assert list(m[:n0:2]) == [1.0] * len(m[:n0:2])
    assert list(m[1:n0:2]) == [2.0] * len(m[1:n0:2])
    assert (m[n0:] == 1.0).all()


def test_multiplicity_is_one_value_per_stored_plane(mini_info):
    _pd, info = mini_info
    assert geometry.ring_pair_multiplicity(info).size == synth_hs.num_planes(6)


def test_crystal_to_det_is_a_permutation():
    t = geometry.crystal_to_det(576)
    assert sorted(t.tolist()) == list(range(576))


def test_crystal_to_det_is_the_measured_reflection():
    t = geometry.crystal_to_det(576)
    ge = np.arange(576)
    assert (t == (287 - ge) % 576).all()


def test_crystal_to_det_without_the_reflection_is_a_plain_roll():
    t = geometry.crystal_to_det(576, offset=0, reverse=False)
    assert (t == np.arange(576)).all()


def test_det_pair_map_spans_the_ring_and_is_symmetric():
    num_views, num_tang, num_det = 8, 9, 16
    d1, d2 = geometry.det_pair_map(num_views, num_tang, num_det)
    assert d1.shape == d2.shape == (num_views, num_tang)
    assert d1.min() >= 0 and d1.max() < num_det
    assert d2.min() >= 0 and d2.max() < num_det
    mid = num_tang // 2
    assert ((d2[:, mid] - d1[:, mid]) % num_det == num_views).all()


def test_plane_pitch_is_half_the_ring_spacing(mini_info):
    _pd, info = mini_info
    ring_mm = info.get_scanner().get_ring_spacing()
    assert geometry.PLANE_MM == pytest.approx(ring_mm / 2.0, rel=1e-6)


def test_the_two_names_for_plane_mm_agree():
    from utils import attenuation

    assert attenuation.PLANE_MM == geometry.PLANE_MM


def test_fov_radius_is_the_widest_chord_the_sinogram_holds(stir, mini_hs):
    from utils import scanner

    s = geometry.tangential_s_mm(mini_hs)
    got = scanner.fov_radius_mm(s.size, ndet=16)
    assert got == pytest.approx(float(np.abs(s).max()), rel=1e-7)


def test_fov_radius_of_the_real_scanner():
    from utils import scanner

    assert scanner.fov_radius_mm(381) == pytest.approx(355.8250, abs=1e-3)
    corner = (scanner.XY - 1) / 2 * np.sqrt(2) * scanner.DR_MM
    assert corner > 1.4 * scanner.fov_radius_mm(381)


def test_fov_mask_is_a_centred_disc():
    from utils import scanner

    m = scanner.fov_mask(scanner.XY, 381)
    assert m.shape == (scanner.XY, scanner.XY)
    assert m[scanner.XY // 2, scanner.XY // 2]
    assert not m[0, 0] and not m[-1, -1]
    assert np.array_equal(m, m[::-1]) and np.array_equal(m, m[:, ::-1])
    r = scanner.fov_radius_mm(381) / scanner.DR_MM
    assert m.sum() == pytest.approx(np.pi * r * r, rel=0.01)


def test_tangential_s_mm_is_not_arc_corrected(stir, mini_hs):
    s = geometry.tangential_s_mm(mini_hs)
    assert s.size == 9
    assert s[0] < 0 < s[-1]
    assert s == pytest.approx(-s[::-1], abs=1e-6)
    step = np.diff(s)
    assert step.max() == pytest.approx(step[len(step) // 2], rel=1e-9)
    assert step.min() < step.max()


def test_open_projdata_keeps_the_data_alive(stir, mini_hs):
    pd, info = geometry.open_projdata(mini_hs)
    assert pd is not None
    assert info.get_num_views() == 8
