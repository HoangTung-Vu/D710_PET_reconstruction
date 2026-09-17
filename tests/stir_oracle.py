"""STIR itself, used as an oracle for the geometry the tree derives by hand.

These functions are the only place left that imports `stir`, and nothing
outside `tests/` calls them. They used to live in `utils/geometry.py`, where
they made the production tree look as though it needed STIR; it does not, and
`d710 attn` was the last command that did.

Every test using them takes the `stir` fixture, which skips when STIR is
absent -- that is, everywhere except a machine with `petct_reconstruction`
activated.
"""

from __future__ import annotations

import numpy as np


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
    """The ring pairs summed into each plane, as `(a, b)` with `a - b` the rd.

    NOT an oracle for the ORDER. This is the same formula `utils/interfile.py`
    uses, driven by STIR's declared ring-difference ranges; comparing the two
    checks the derivation, not the convention. STIR's own ordered-pair APIs
    (`get_det_pos_pair_for_bin`, `get_all_det_pos_pairs_for_bin`,
    `find_cartesian_coordinates_of_detection`) all refuse span-2 data --
    "does not work for data with axial compression" -- so they cannot be used
    here at all.

    The order is settled against GE instead, in `tests/test_lm_data.py`, and
    stated in `utils/binmap.py`.
    """
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


def tangential_s_mm(hs: str) -> np.ndarray:
    """Radial offset `s` of each tangential bin in mm, taken from the header."""
    import stir

    pd = stir.ProjData.read_from_file(hs)
    info = pd.get_proj_data_info()
    lo = info.get_min_tangential_pos_num()
    return np.array([info.get_s(stir.Bin(0, 0, 0, lo + t))
                     for t in range(info.get_num_tangential_poss())])
