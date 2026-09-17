"""D710 sinogram geometry, read from the STIR header rather than tabulated."""

from __future__ import annotations

import numpy as np

from .scanner import CRYSTAL_OFFSET, CRYSTAL_REVERSE, PLANE_MM


def open_projdata(hs: str):
    """`(proj_data, info)`; proj_data must be kept alive, info is a borrowed pointer."""
    import stir

    pd = stir.ProjData.read_from_file(hs)
    return pd, pd.get_proj_data_info()


def segment_order(info) -> list[int]:
    """The order in which STIR stores segments: 0, +1, -1, +2, -2, and so on."""
    out = [0]
    for k in range(1, info.get_max_segment_num() + 1):
        out += [k, -k]
    return out


def plane_ring_pairs(info, num_rings: int) -> list[list[tuple[int, int]]]:
    """The ring pairs summed into each plane, as `(ring of pos1, ring of pos2)`."""
    out = []
    for s in segment_order(info):
        lo, hi = info.get_min_ring_difference(s), info.get_max_ring_difference(s)
        z0 = min(abs(d) for d in range(lo, hi + 1))
        for a in range(info.get_num_axial_poss(s)):
            z = z0 + a
            out.append([((z - d) // 2 + d, (z - d) // 2) for d in range(lo, hi + 1)
                        if (z - d) % 2 == 0 and 0 <= (z - d) // 2 < num_rings
                        and 0 <= (z - d) // 2 + d < num_rings])
    return out


def check_ring_pairs(info, pairs: list[list[tuple[int, int]]]) -> None:
    """Raise if the derived ring pairs disagree with STIR's own count."""
    p = 0
    for s in segment_order(info):
        for a in range(info.get_num_axial_poss(s)):
            want = info.get_num_ring_pairs_for_segment_axial_pos_num(s, a)
            if len(pairs[p]) != want:
                raise ValueError(
                    f"plane {p} (segment {s}, axial {a}): derived "
                    f"{len(pairs[p])} ring pairs, STIR says {want}")
            p += 1


def ring_pair_multiplicity(info) -> np.ndarray:
    """Ring pairs merged into each plane, along STIR's flattened axial axis."""
    return np.concatenate([
        np.array([info.get_num_ring_pairs_for_segment_axial_pos_num(s, a)
                  for a in range(info.get_num_axial_poss(s))], dtype=np.float32)
        for s in segment_order(info)])


def det_pair_map(num_views: int, num_tang: int, num_det: int):
    """`(view, tangential)` to `(det1, det2)`, as two `(num_views, num_tang)` arrays."""
    v = np.arange(num_views)[:, None]
    t = (np.arange(num_tang) - num_tang // 2)[None, :]
    d1 = (v + np.floor_divide(t, 2)) % num_det
    d2 = (v - (-np.floor_divide(-t, 2)) + num_views) % num_det
    return d1.astype(np.int32), d2.astype(np.int32)


def crystal_to_det(num_det: int, offset: int = CRYSTAL_OFFSET,
                   reverse: bool = CRYSTAL_REVERSE) -> np.ndarray:
    """Lookup table from GE transverse crystal index to STIR detector number."""
    d = np.arange(num_det)
    return np.roll(d[::-1] if reverse else d, offset)


def tangential_s_mm(hs: str) -> np.ndarray:
    """Radial offset `s` of each tangential bin in mm, taken from the header."""
    import stir

    pd = stir.ProjData.read_from_file(hs)
    info = pd.get_proj_data_info()
    lo = info.get_min_tangential_pos_num()
    return np.array([info.get_s(stir.Bin(0, 0, 0, lo + t))
                     for t in range(info.get_num_tangential_poss())])
