"""D710 sinogram geometry: crystal numbering, detector pairs and crystal positions.

Everything here is plain numpy. The STIR cross-checks that used to live in this
module -- they only ever served as an oracle for the tests -- are in
`tests/stir_oracle.py`.
"""

from __future__ import annotations

import numpy as np

from .scanner import (CRYSTAL_OFFSET, CRYSTAL_REVERSE, NDET, NRINGS,
                      RING_PITCH_MM, R_EFF_MM, VIEW_OFFSET_DEG)


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


def ring_z_mm(nrings: int = NRINGS, pitch_mm: float = RING_PITCH_MM) -> np.ndarray:
    """The axial centre of each ring, in mm, about the middle of the scanner."""
    return ((np.arange(nrings) - (nrings - 1) / 2.0) * pitch_mm).astype(np.float32)


def detector_xy_mm(num_det: int = NDET, r_mm: float = R_EFF_MM,
                   offset_deg: float = VIEW_OFFSET_DEG) -> np.ndarray:
    """`(num_det, 2)` transaxial centre of each STIR detector number, in mm."""
    ang = 2.0 * np.pi * np.arange(num_det) / num_det + np.deg2rad(offset_deg)
    return np.stack([r_mm * np.sin(ang), -r_mm * np.cos(ang)], 1).astype(np.float32)


def crystal_positions(nrings: int = NRINGS, ndet: int = NDET, r_mm: float = R_EFF_MM,
                      pitch_mm: float = RING_PITCH_MM,
                      offset_deg: float = VIEW_OFFSET_DEG,
                      stir_frame: bool = True) -> np.ndarray:
    """`(nrings*ndet, 3)` crystal centres in mm, indexed by GE crystal id.

    The GE crystal id is `ring * ndet + transverse`. `stir_frame` puts the
    result in the frame STIR and the image grid share, which is also what
    PyTomography wants as its `scanner_LUT`.
    """
    i = np.arange(nrings * ndet)
    ring, trans = np.divmod(i, ndet)
    d = crystal_to_det(ndet)[trans] if stir_frame else trans
    ang = 2.0 * np.pi * d / ndet + np.deg2rad(offset_deg)
    x, y = (r_mm * np.sin(ang), -r_mm * np.cos(ang)) if stir_frame else \
        (r_mm * np.cos(ang), r_mm * np.sin(ang))
    z = (ring - (nrings - 1) / 2.0) * pitch_mm
    return np.stack([x, y, z], 1).astype(np.float32)
