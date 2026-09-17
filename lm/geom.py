"""Mapping between crystal identifiers, STIR sinogram bins and PyTomography geometry."""

from __future__ import annotations

import numpy as np

from utils.geometry import crystal_to_det, det_pair_map
from utils.scanner import (C_MM_PS, NDET, NRINGS, NXTAL,
                           N_TOF_RAW, RING_PITCH_MM, R_EFF_MM, R_MM, TIMING_PS,
                           TOF_LSB_PS, TOF_RANGE_MM, VIEW_OFFSET_DEG)

from . import interfile


def scanner_lut(nrings=NRINGS, ndet=NDET, r_mm=R_EFF_MM, pitch_mm=RING_PITCH_MM,
                offset_deg=VIEW_OFFSET_DEG, stir_frame=True):
    """`(nrings*ndet, 3)` crystal centres in mm, as PyTomography's `scanner_LUT`."""
    i = np.arange(nrings * ndet)
    ring, trans = np.divmod(i, ndet)
    d = crystal_to_det(ndet)[trans] if stir_frame else trans
    ang = 2.0 * np.pi * d / ndet + np.deg2rad(offset_deg)
    x, y = (r_mm * np.sin(ang), -r_mm * np.cos(ang)) if stir_frame else \
        (r_mm * np.cos(ang), r_mm * np.sin(ang))
    z = (ring - (nrings - 1) / 2.0) * pitch_mm
    return np.stack([x, y, z], 1).astype(np.float32)


def tof_meta(n_bins=N_TOF_RAW, n_sigmas=3.0):
    """PyTomography's TOF metadata for `n_bins` mashed bins."""
    from pytomography.metadata.PET import PETTOFMeta

    return PETTOFMeta(num_bins=n_bins, tof_range=TOF_RANGE_MM,
                      fwhm=C_MM_PS * TIMING_PS / 2, n_sigmas=n_sigmas)


def tof_to_stir(tof_bin, n_out=N_TOF_RAW, n_raw=N_TOF_RAW):
    """GE's signed bin (-27..+27) to a 0-based STIR timing position, rebinned to `n_out`."""
    if n_raw % n_out:
        raise ValueError(f"{n_out} TOF bins does not divide {n_raw}")
    ge = np.asarray(tof_bin, np.int32) + n_raw // 2
    return (n_raw - 1 - ge) // (n_raw // n_out)


class BinMap:
    """`(xtal_a, xtal_b)` -> flat index into a `(plane, view, tang)` sinogram."""

    def __init__(self, hs, nrings=None, ndet=None):
        self.hdr = interfile.Header(hs)
        nrings = nrings or self.hdr.n_rings
        ndet = ndet or self.hdr.n_det
        self.n_view, self.n_tang = self.hdr.n_view, self.hdr.n_tang
        pairs = self.hdr.ring_pairs()
        self.n_plane = len(pairs)
        self.mult = np.array([len(p) for p in pairs], np.int32)

        d1, d2 = det_pair_map(self.n_view, self.n_tang, ndet)
        k = np.arange(self.n_view * self.n_tang, dtype=np.int32).reshape(d1.shape)
        self.vt = np.full((ndet, ndet), -1, np.int32)
        self.vt[d1, d2] = k

        self.pl = np.full((nrings, nrings), -1, np.int32)
        for p, prs in enumerate(pairs):
            for r1, r2 in prs:
                self.pl[r1, r2] = p

        self.xtal2det = crystal_to_det(ndet).astype(np.int32)
        self.ndet = ndet

    @property
    def n_bin(self) -> int:
        return self.n_plane * self.n_view * self.n_tang

    @property
    def shape(self) -> tuple:
        return (self.n_plane, self.n_view, self.n_tang)

    def flat(self, xa, xb, with_swap: bool = False):
        ra, ta = np.divmod(np.asarray(xa, np.int32), self.ndet)
        rb, tb = np.divmod(np.asarray(xb, np.int32), self.ndet)
        da, db = self.xtal2det[ta], self.xtal2det[tb]

        k = self.vt[da, db]
        swap = k < 0
        k = np.where(swap, self.vt[db, da], k)
        p = self.pl[np.where(swap, rb, ra), np.where(swap, ra, rb)]

        bad = (k < 0) | (p < 0)
        flat = np.where(bad, np.int32(-1),
                        p * (self.n_view * self.n_tang) + k).astype(np.int32)
        return (flat, swap) if with_swap else flat

    def lor_table(self):
        d1, d2 = det_pair_map(self.n_view, self.n_tang, self.ndet)
        det2xtal = np.argsort(self.xtal2det).astype(np.int32)
        t1, t2 = det2xtal[d1].ravel(), det2xtal[d2].ravel()
        k = np.arange(self.n_view * self.n_tang, dtype=np.int32)

        ids, bins = [], []
        for r1, r2 in zip(*np.nonzero(self.pl >= 0)):
            a = (r1 * self.ndet + t1).astype(np.int32)
            b = (r2 * self.ndet + t2).astype(np.int32)
            ids.append(np.stack([a, b], 1))
            bins.append(int(self.pl[r1, r2]) * (self.n_view * self.n_tang) + k)
        return np.concatenate(ids), np.concatenate(bins)
