"""D710 sinogram geometry — read from the STIR header, never copied into a table.

The three constants here have been checked **bit-exact** against a sinogram the
scanner decoded itself (histogramming NEMA bed 2 list-mode reproduces every bin,
corr 1.000000). They are proven results, not architectural choices, so they are
carried over unchanged:

* `crystal_to_det` — GE numbers crystals opposite to STIR: `stir = (287 - ge) mod 576`
* `plane_ring_pairs` — the sign of the ring difference follows `pos1 - pos2`
* segment 0 of span-2 merges two ring pairs into one odd plane
"""

from __future__ import annotations

import numpy as np

#: Plane spacing = half the ring spacing.
PLANE_MM = 3.2699997

#: GE numbers crystals opposite to STIR, offset by half a ring.
#: Calibrated by histogramming NEMA bed 2 list-mode through all 1152 candidates
#: and scoring against the vendor-decoded sinogram of that same bed
#: (corr 0.990, runner-up 0.981).
CRYSTAL_REVERSE = True
CRYSTAL_OFFSET = 288


def open_projdata(hs: str):
    """``(proj_data, info)`` — proj_data must be kept alive; info is a borrowed pointer."""
    import stir

    pd = stir.ProjData.read_from_file(hs)
    return pd, pd.get_proj_data_info()


def segment_order(info) -> list[int]:
    """The order in which STIR stores segments: 0, +1, -1, +2, -2, ..."""
    out = [0]
    for k in range(1, info.get_max_segment_num() + 1):
        out += [k, -k]
    return out


def plane_ring_pairs(info, num_rings: int) -> list[list[tuple[int, int]]]:
    """The ring pairs summed into each plane, as ``(ring of pos1, ring of pos2)``.

    The plane at axial position ``a`` of a segment covering ring differences
    ``[lo, hi]`` contains every pair whose ring sum is ``z`` and whose difference
    falls in that range. Segment 0 of span-2 covers -1..+1, so odd ``z`` holds
    **two** pairs — the merge that the background term has to sum itself, since
    it does not go through the sensitivity model.

    The pair emitted is ``(r + d, r)``, not ``(r, r + d)``: STIR's signed segment
    follows ``pos1 - pos2``, the opposite of the obvious reading.
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
                    f"plane {p} (segment {s}, axial {a}): suy ra "
                    f"{len(pairs[p])} cặp ring, STIR nói {want}")
            p += 1


def ring_pair_multiplicity(info) -> np.ndarray:
    """Number of ring pairs merged into each plane, along STIR's flattened axial axis.

    ⚠ **DO NOT apply this to the vendor path (`vendor/to_stir.py` + `normdt`).**
    Measured 2026-08-22 on ped bed 4, the odd/even ratio within segment 0:

        prompts 1.985   randoms 1.981   scatter 2.007   background 1.986
        normdt  1.992   norm_only 1.993

    GE's `normdt` **already carries** this multiplicity — it is the sensitivity
    of the whole *bin*, folding in how many ring pairs that bin receives, not the
    bare efficiency of a single LOR. Multiplying by `ring_pair_multiplicity` on
    top **squares** it (4x at odd bins). The chain `y = S(Gx) + b` is
    self-consistent: y, b and S are all 2x at odd bins, the projector fires one
    LOR, so `S·(Gx)` comes out exactly 2x. Checked on the image: the power at the
    Nyquist frequency of the axial profile is down to 0.00-0.49 % (in the
    sinogram it is 68.9 %).

    This function remains here for the **old hand-written** pipeline (since
    deleted), where randoms/scatter/norm were built by hand and did NOT carry the
    multiplicity, so it had to be pushed into `asm` manually.

    **This is geometry, not detector normalisation.** Segment 0 of span-2 merges
    ring differences +1 and -1 into its odd axial positions, so those bins collect
    **two** LORs while STIR's projector fires **one**.

    Ignoring it *when the multiplicative term does not already carry it* settles
    into a period-2 stripe along the axis — sitting exactly at the axial Nyquist
    frequency.
    """
    return np.concatenate([
        np.array([info.get_num_ring_pairs_for_segment_axial_pos_num(s, a)
                  for a in range(info.get_num_axial_poss(s))], dtype=np.float32)
        for s in segment_order(info)])


def det_pair_map(num_views: int, num_tang: int, num_det: int):
    """``(view, tangential) -> (det1, det2)``, two ``(num_views, num_tang)`` arrays."""
    v = np.arange(num_views)[:, None]
    t = (np.arange(num_tang) - num_tang // 2)[None, :]
    d1 = (v + np.floor_divide(t, 2)) % num_det
    d2 = (v - (-np.floor_divide(-t, 2)) + num_views) % num_det
    return d1.astype(np.int32), d2.astype(np.int32)


def crystal_to_det(num_det: int, offset: int = CRYSTAL_OFFSET,
                   reverse: bool = CRYSTAL_REVERSE) -> np.ndarray:
    """Lookup table: GE transverse crystal index -> STIR detector number."""
    d = np.arange(num_det)
    return np.roll(d[::-1] if reverse else d, offset)


def tangential_s_mm(hs: str) -> np.ndarray:
    """Radial offset ``s`` of each tangential bin, in mm, taken from the header.

    Not arc-corrected, so the bins are **not** evenly spaced: on this scanner
    they run -356.7 .. +356.7 mm over 381 bins, 2.261 mm at the centre and
    tightening towards the edges. Anything needing a distance in mm must ask this
    function rather than multiplying by a nominal bin width.
    """
    import stir

    pd = stir.ProjData.read_from_file(hs)
    info = pd.get_proj_data_info()
    lo = info.get_min_tangential_pos_num()
    return np.array([info.get_s(stir.Bin(0, 0, 0, lo + t))
                     for t in range(info.get_num_tangential_poss())])
