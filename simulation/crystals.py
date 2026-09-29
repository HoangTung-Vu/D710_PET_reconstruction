from __future__ import annotations

import math

import numpy as np

from utils.geometry import crystal_positions
from utils.scanner import NDET, NRINGS, R_MM, RING_PITCH_MM

CRYSTAL_MM = (25.0, 4.2, 6.3)

BLOCK_CRYSTALS = (9, 6)

N_BLOCKS_T = NDET // BLOCK_CRYSTALS[0]

N_BLOCKS_Z = NRINGS // BLOCK_CRYSTALS[1]

PITCH_T = 2.0 * R_MM * math.tan(math.pi / N_BLOCKS_T) / BLOCK_CRYSTALS[0]

PITCH_Z = RING_PITCH_MM

BLOCK_CLEARANCE_MM = 0.02

BLOCK_SIZE_MM = (CRYSTAL_MM[0], BLOCK_CRYSTALS[0] * PITCH_T - BLOCK_CLEARANCE_MM,
                 BLOCK_CRYSTALS[1] * PITCH_Z)
"""Adjacent blocks meet exactly at their inner corners, so the box is 20 um
narrower than 9 pitches: otherwise Geant4 reports 20-60 nm overlaps there.
The crystals inside (4.2 mm on a 4.4225 mm pitch) are not moved."""

BLOCK_CENTRE_R_MM = R_MM + CRYSTAL_MM[0] / 2.0


def ge_azimuth() -> np.ndarray:
    p = crystal_positions(centre_mm=(0.0, 0.0))[:NDET]
    return np.arctan2(p[:, 1], p[:, 0])


def block_azimuth() -> np.ndarray:
    phi = ge_azimuth().reshape(N_BLOCKS_T, BLOCK_CRYSTALS[0])
    return np.angle(np.exp(1j * phi).mean(axis=1))


def block_z() -> np.ndarray:
    return (np.arange(N_BLOCKS_Z) - (N_BLOCKS_Z - 1) / 2.0) * BLOCK_SIZE_MM[2]


def block_placements():
    tr, rot = [], []
    for z in block_z():
        for phi in block_azimuth():
            c, s = math.cos(phi), math.sin(phi)
            tr.append([BLOCK_CENTRE_R_MM * c, BLOCK_CENTRE_R_MM * s, float(z)])
            rot.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    return tr, rot


def crystal_offsets() -> np.ndarray:
    nt, nz = BLOCK_CRYSTALS
    out = []
    for iy in range(nt):
        for iz in range(nz):
            out.append([0.0, (iy - (nt - 1) / 2.0) * PITCH_T,
                        (iz - (nz - 1) / 2.0) * PITCH_Z])
    return np.asarray(out)


def gate_crystal_centres() -> np.ndarray:
    tr, rot = block_placements()
    off = crystal_offsets()
    return np.concatenate([np.asarray(t) + off @ np.asarray(r).T
                           for t, r in zip(tr, rot)]).astype(np.float64)


def ge_ids(positions) -> np.ndarray:
    p = np.asarray(positions, np.float64)
    phi = np.arctan2(p[:, 1], p[:, 0])
    ge = ge_azimuth()
    d = np.angle(np.exp(1j * (phi[:, None] - ge[None, :])))
    trans = np.abs(d).argmin(axis=1)
    ring = np.rint(p[:, 2] / RING_PITCH_MM + (NRINGS - 1) / 2.0).astype(np.int64)
    if (ring < 0).any() or (ring >= NRINGS).any():
        raise ValueError("a position lies outside the 24 rings")
    return (ring * NDET + trans).astype(np.int64)


def ge_ordered_centres() -> np.ndarray:
    c = gate_crystal_centres()
    ids = ge_ids(c)
    if np.unique(ids).size != ids.size:
        raise RuntimeError("GATE crystals do not map one-to-one onto GE ids")
    out = np.empty_like(c)
    out[ids] = c
    return out


def volume_id_to_gate_index(volume_ids) -> np.ndarray:
    n = BLOCK_CRYSTALS[0] * BLOCK_CRYSTALS[1]
    out = np.empty(len(volume_ids), np.int64)
    for i, s in enumerate(volume_ids):
        s = s.decode() if isinstance(s, bytes) else str(s)
        b, c = s.rsplit("-", 1)[1].split("_")[-2:]
        out[i] = int(b) * n + int(c)
    return out


class CrystalLookup:
    def __init__(self):
        from scipy.spatial import cKDTree

        self.centres = ge_ordered_centres()
        self.tree = cKDTree(self.centres)
        self.gate_to_ge = ge_ids(gate_crystal_centres())

    def __call__(self, xyz, volume_ids=None, tol_mm: float = 1.0) -> np.ndarray:
        d, i = self.tree.query(np.asarray(xyz, np.float64))
        bad = d > tol_mm
        if bad.any():
            if volume_ids is None:
                raise ValueError(f"{int(bad.sum())} positions are more than "
                                 f"{tol_mm} mm from any crystal centre "
                                 f"(max {d.max():.2f}) and no volume ids were given")
            vid = np.asarray(volume_ids, dtype=object)[bad]
            i[bad] = self.gate_to_ge[volume_id_to_gate_index(vid)]
        self.n_fallback = int(bad.sum())
        return i.astype(np.int64)
