"""The parallelproj attenuation projector: units, geometry and both backends.

None of this needs SIRF, and none of it needs a case on disk. The cylinder
check is what fixes the unit convention, because it is analytic: parallelproj
integrates in the units of the coordinates it is handed, and those are mm,
while the CT conversion produces 1/cm.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import attn_proj, interfile
from utils.geometry import det_pair_map, detector_xy_mm
from utils.scanner import DR_MM

pytest.importorskip("parallelproj")

MU_PER_CM = 0.096
XY = 320


def image_planes(hdr) -> int:
    """The image has one plane per direct sinogram plane, as a bed does."""
    return 2 * hdr.n_rings - 1


def cylinder(n_plane: int, xy: int, mu_percm: float, radius_mm: float):
    """A cylinder of constant mu on the axis, as `(plane, y, x)` in 1/cm."""
    c = (np.arange(xy) - (xy - 1) / 2.0) * DR_MM
    r = np.hypot(*np.meshgrid(c, c, indexing="ij"))
    return np.repeat(((r <= radius_mm) * mu_percm).astype(np.float32)[None],
                     n_plane, 0)


def tangential_s_mm(hdr) -> np.ndarray:
    """Distance of each tangential bin from the axis, from the detector ring."""
    d1, d2 = det_pair_map(hdr.n_view, hdr.n_tang, hdr.n_det)
    p = detector_xy_mm(hdr.n_det)
    a, b = p[d1[0]], p[d2[0]]                      # view 0, every tangential bin
    d = b - a
    cross = d[:, 0] * (-a[:, 1]) - d[:, 1] * (-a[:, 0])
    return np.abs(cross) / np.linalg.norm(d, axis=1)


def test_image_origin_matches_pytomography():
    """The frame must be the one `PETLMSystemMatrix` uses, or attn and lm disagree."""
    shape, vox = (337, 337, 47), (DR_MM, DR_MM, 3.2699997)
    want = (-np.array(shape) / 2 + 0.5) * np.array(vox)
    assert np.allclose(attn_proj.image_origin(shape, vox), want)


def test_image_for_projector_converts_per_cm_to_per_mm():
    img, _origin, voxel = attn_proj.image_for_projector(
        np.full((4, 6, 6), 0.096, np.float32))
    assert img.shape == (6, 6, 4)                  # (x, y, plane)
    assert np.allclose(img, 0.0096)                # 1/mm
    assert np.allclose(voxel[:2], DR_MM)


def test_image_for_projector_rejects_a_non_square_map():
    with pytest.raises(SystemExit):
        attn_proj.image_for_projector(np.zeros((4, 6, 7), np.float32))


def test_a_uniform_cylinder_gives_the_analytic_chord(mini_hs):
    """`af = exp(-mu * 2 sqrt(R^2 - s^2))`, with mu in 1/mm and the chord in mm."""
    hdr = interfile.Header(mini_hs)
    radius = 300.0
    af = attn_proj.factors(cylinder(image_planes(hdr), XY, MU_PER_CM, radius),
                           mini_hs, out=lambda *_: None)
    assert af.shape == (hdr.n_plane, hdr.n_view, hdr.n_tang)

    s = tangential_s_mm(hdr)
    inside = s < radius - 4 * DR_MM                # away from the partial-volume rim
    assert inside.sum() >= 5, "the test geometry stopped covering the cylinder"

    want = np.exp(-MU_PER_CM / 10.0
                  * 2.0 * np.sqrt(radius ** 2 - s[inside] ** 2))
    got = af[0, 0, inside]                         # segment 0, first plane, view 0
    assert np.allclose(got, want, rtol=0.05), \
        f"max rel {np.abs(got / want - 1).max():.3%}"


def test_a_denser_cylinder_attenuates_more(mini_hs):
    hdr = interfile.Header(mini_hs)
    n = image_planes(hdr)
    a = attn_proj.factors(cylinder(n, XY, MU_PER_CM, 200.0), mini_hs,
                          out=lambda *_: None)
    b = attn_proj.factors(cylinder(n, XY, 2 * MU_PER_CM, 200.0), mini_hs,
                          out=lambda *_: None)
    assert np.allclose(b, a ** 2, rtol=1e-4)       # exp is exp


def test_bins_that_miss_the_object_are_transparent(mini_hs):
    hdr = interfile.Header(mini_hs)
    af = attn_proj.factors(cylinder(image_planes(hdr), 64, MU_PER_CM, 20.0),
                           mini_hs, out=lambda *_: None)
    assert 0.0 < af.min() and af.max() <= 1.0 + 1e-6
    s = tangential_s_mm(hdr)
    assert np.allclose(af[:, :, s > 25.0], 1.0, atol=1e-6)


def test_torch_and_numpy_backends_agree(mini_hs):
    """`--device cuda` is this code path with another torch device."""
    pytest.importorskip("torch")
    hdr = interfile.Header(mini_hs)
    mu = cylinder(image_planes(hdr), XY, MU_PER_CM, 200.0)
    a = attn_proj.factors(mu, mini_hs, device="cpu", out=lambda *_: None)
    b = attn_proj.factors(mu, mini_hs, device="torch:cpu", out=lambda *_: None)
    assert np.allclose(a, b, atol=1e-6)


def test_the_traced_lors_are_the_ones_the_bin_map_names(mini_hs):
    """`attn_proj` must trace the line `BinMap.lor_table()` names, for each bin.

    `lor_table` inverts `flat`, and `flat` is what `d710 lm check` proves
    bit-exact against GE's own sinogram.  If the two ever disagree, `attn.hs`
    is mirrored against every other term in `work/bed<n>/`.
    """
    from utils.binmap import BinMap
    from utils.geometry import crystal_positions

    bm = BinMap(mini_hs)
    lut = crystal_positions(bm.nrings, bm.ndet)
    ids, bins = bm.lor_table()

    xy1, xy2, z_ring = attn_proj._endpoints(bm)
    r1s, r2s, planes = bm.ring_pairs_by_plane()
    vt = bm.n_view * bm.n_tang

    want = {}
    for r1, r2, p in zip(r1s, r2s, planes):
        for k in range(vt):
            want[(int(p) * vt + k, int(r1), int(r2))] = (
                (xy1[k][0], xy1[k][1], z_ring[r1]),
                (xy2[k][0], xy2[k][1], z_ring[r2]))

    # every LOR lor_table names must be one attn_proj traces, endpoint for endpoint
    checked = 0
    for i in range(0, len(ids), max(1, len(ids) // 500)):
        a, b = int(ids[i, 0]), int(ids[i, 1])
        key = (int(bins[i]), a // bm.ndet, b // bm.ndet)
        assert key in want, f"lor_table LOR {i} has no counterpart in attn_proj"
        pa, pb = want[key]
        assert np.allclose(lut[a], pa, atol=1e-3), f"start of LOR {i}"
        assert np.allclose(lut[b], pb, atol=1e-3), f"end of LOR {i}"
        checked += 1
    assert checked > 100


def test_reversing_the_pairing_would_mirror_the_segments(mini_hs):
    """Guards the sensitivity of the test above: the two orders DO differ."""
    from utils.binmap import BinMap

    hdr = interfile.Header(mini_hs)
    mu = cylinder(image_planes(hdr), XY, MU_PER_CM, 200.0)
    mu[:4] *= 2.0                                  # break the axial symmetry

    a = attn_proj.factors(mu, mini_hs, out=lambda *_: None)

    real = BinMap.ring_pairs_by_plane

    def crossed(self):
        r1, r2, p = real(self)
        return r2, r1, p

    BinMap.ring_pairs_by_plane = crossed
    try:
        b = attn_proj.factors(mu, mini_hs, out=lambda *_: None)
    finally:
        BinMap.ring_pairs_by_plane = real
    assert not np.allclose(a, b)

    seg, p0 = {}, 0
    for s, _lo, _hi, n in hdr.segments():
        seg[s], p0 = slice(p0, p0 + n), p0 + n
    assert set(seg) >= {1, -1}
    for s in (1, 2):
        if s in seg and -s in seg:
            assert np.allclose(a[seg[s]], b[seg[-s]], atol=1e-6), f"segment {s}"
