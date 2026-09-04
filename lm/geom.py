"""Crystal ids <-> STIR sinogram bins, and the geometry PyTomography needs.

Both index conventions are the ones already proven bit-exact by the sinogram
path: the `(287 - ge_trans)` transaxial reflection and the `pos1 - pos2`
ring-difference sign. Nothing new is assumed here -- `BinMap` is built by
*inverting* those same tables, and `lm check` re-proves the result by
histogramming the events back onto `decoded/bed<n>.s`.

**No SIRF and no STIR.** This package runs in the PyTomography environment; the
segment layout comes from the header (`lm.interfile`) rather than from
`ProjDataInfo`. Only the two pure-numpy helpers of `utils.geometry` are used, and
the constants come from `utils.scanner`, which imports nothing.
"""

from __future__ import annotations

import numpy as np

from utils.geometry import crystal_to_det, det_pair_map
from utils.scanner import (C_MM_PS, NDET, NRINGS, NXTAL,  # noqa: F401
                           N_TOF_RAW, RING_PITCH_MM, R_MM, TIMING_PS,
                           TOF_LSB_PS, TOF_RANGE_MM, VIEW_OFFSET_DEG)

from . import interfile


def scanner_lut(nrings=NRINGS, ndet=NDET, r_mm=R_MM, pitch_mm=RING_PITCH_MM,
                offset_deg=VIEW_OFFSET_DEG, stir_frame=True):
    """`(nrings*ndet, 3)` crystal centres in mm -- PyTomography's `scanner_LUT`.

    Indexed by the **GE** crystal id, because that is what the events carry, and
    laid out so the reconstruction lands in STIR's image frame -- otherwise
    stitching, the CT orientation and the DICOM geometry all inherit a frame
    nothing else in the project uses. Two things put it there, and both were
    measured against the sinogram reconstruction of ped bed 1 rather than
    reasoned about:

    * `crystal_to_det` -- GE numbers crystals in the opposite rotational sense,
      the same reflection `gerdf.cli._view_index` undoes for the sinogram path.
      Without it the best match is a transpose, i.e. a mirror image.
    * `(sin, -cos)` rather than `(cos, sin)` -- detector 0 sits at `(0, -R)`, not
      at `(R, 0)`. Left as `(cos, sin)` the image comes out rotated by exactly
      270 deg; with this, the angular profiles of the two reconstructions agree
      at **+0.994 with zero rotation** (`tools/lm_frame.py`).

    `stir_frame=False` gives the raw GE frame, for comparison only.
    """
    i = np.arange(nrings * ndet)
    ring, trans = np.divmod(i, ndet)
    d = crystal_to_det(ndet)[trans] if stir_frame else trans
    ang = 2.0 * np.pi * d / ndet + np.deg2rad(offset_deg)
    x, y = (r_mm * np.sin(ang), -r_mm * np.cos(ang)) if stir_frame else \
        (r_mm * np.cos(ang), r_mm * np.sin(ang))
    z = (ring - (nrings - 1) / 2.0) * pitch_mm
    return np.stack([x, y, z], 1).astype(np.float32)


def tof_meta(n_bins=N_TOF_RAW, n_sigmas=3.0):
    """PyTomography's TOF metadata for `n_bins` mashed bins.

    `tof_range` is the TOTAL range across all bins, so it stays `TOF_RANGE_MM`
    whatever the mashing: 11 bins are 5 LSB wide, they do not cover 11 LSB.
    """
    from pytomography.metadata.PET import PETTOFMeta

    return PETTOFMeta(num_bins=n_bins, tof_range=TOF_RANGE_MM,
                      fwhm=C_MM_PS * TIMING_PS / 2, n_sigmas=n_sigmas)


def tof_to_stir(tof_bin, n_out=N_TOF_RAW, n_raw=N_TOF_RAW):
    """GE's signed bin (-27..+27) -> 0-based STIR timing position, mashed to `n_out`.

    The reversal is `CListRecordGEHDF5::get_tof_bin() = -deltaTime`, the same one
    `gerdf.cli._tof_to_stir` applies to the prompts and `vendor/to_stir.py` to the
    scatter weights. Mashing commutes with it because the factor divides 55.
    """
    if n_raw % n_out:
        raise ValueError(f"{n_out} TOF bins does not divide {n_raw}")
    ge = np.asarray(tof_bin, np.int32) + n_raw // 2
    return (n_raw - 1 - ge) // (n_raw // n_out)


class BinMap:
    """`(xtal_a, xtal_b)` -> flat index into a `(plane, view, tang)` sinogram.

    Built from the bed's own header, so the segment layout is STIR's own rather
    than a table typed out here.
    """

    def __init__(self, hs, nrings=None, ndet=None):
        self.hdr = interfile.Header(hs)
        nrings = nrings or self.hdr.n_rings
        ndet = ndet or self.hdr.n_det
        self.n_view, self.n_tang = self.hdr.n_view, self.hdr.n_tang
        pairs = self.hdr.ring_pairs()
        self.n_plane = len(pairs)
        self.mult = np.array([len(p) for p in pairs], np.int32)

        # (det1, det2) -> view*n_tang + tang, inverted from the forward map so the
        # two can never drift apart. -1 = the LOR misses the tangential range.
        d1, d2 = det_pair_map(self.n_view, self.n_tang, ndet)
        k = np.arange(self.n_view * self.n_tang, dtype=np.int32).reshape(d1.shape)
        self.vt = np.full((ndet, ndet), -1, np.int32)
        self.vt[d1, d2] = k

        # (ring of det1, ring of det2) -> plane
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
        """`(N,)` int32 flat bin index; **-1** where the LOR is outside the sinogram.

        `with_swap` also returns the mask of events whose `(xtal_a, xtal_b)` runs
        against the bin's own `(det1, det2)` direction. A sinogram bin has one
        canonical direction and TOF is a *signed* displacement along it, so those
        events need their TOF index mirrored -- see `lm.events.tof_index`.
        """
        ra, ta = np.divmod(np.asarray(xa, np.int32), self.ndet)
        rb, tb = np.divmod(np.asarray(xb, np.int32), self.ndet)
        da, db = self.xtal2det[ta], self.xtal2det[tb]

        k = self.vt[da, db]
        # Either ordering of the pair can be the one det_pair_map emits; whichever
        # hits decides which ring is pos1, and the ring sign follows pos1 - pos2.
        swap = k < 0
        k = np.where(swap, self.vt[db, da], k)
        p = self.pl[np.where(swap, rb, ra), np.where(swap, ra, rb)]

        bad = (k < 0) | (p < 0)
        flat = np.where(bad, np.int32(-1),
                        p * (self.n_view * self.n_tang) + k).astype(np.int32)
        return (flat, swap) if with_swap else flat

    def lor_table(self):
        """Every valid LOR, as `(ids (M,2) int32, bin (M,) int32)`.

        One row per crystal pair, so segment 0's odd planes contribute **two**
        rows to the same bin -- which is the multiplicity `normdt` already
        carries (see `utils.geometry.ring_pair_multiplicity`), and the reason the
        per-LOR weight is the bin's divided by `mult`.
        """
        d1, d2 = det_pair_map(self.n_view, self.n_tang, self.ndet)
        det2xtal = np.argsort(self.xtal2det).astype(np.int32)   # STIR det -> GE trans
        t1, t2 = det2xtal[d1].ravel(), det2xtal[d2].ravel()
        k = np.arange(self.n_view * self.n_tang, dtype=np.int32)

        ids, bins = [], []
        for r1, r2 in zip(*np.nonzero(self.pl >= 0)):
            a = (r1 * self.ndet + t1).astype(np.int32)
            b = (r2 * self.ndet + t2).astype(np.int32)
            ids.append(np.stack([a, b], 1))
            bins.append(int(self.pl[r1, r2]) * (self.n_view * self.n_tang) + k)
        return np.concatenate(ids), np.concatenate(bins)
