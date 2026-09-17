"""Crystal pairs to sinogram bins, and the one statement of the ring pairing.

`Header.ring_pairs()` returns `(a, b)` with `a - b` the declared ring
difference, and `pl[b, a] = plane` below puts ring `b` on `det1` of
`det_pair_map`. The first index of `pl` is always the ring on `det1`, which is
what `ring_pairs_by_plane()` and `lor_table()` rely on.

That is STIR's rule, `ring2 - ring1 = rd`, and since 2026-09-18 the decoder
declares the ring differences with the same sign
(`custom_tool/gerdf/interfile.py::ordered_segments`). Reversing the pairing
mirrors the segment axis, so anything that turns a plane back into a geometric
line must come through here rather than through `Header.ring_pairs()` -- that
way `d710 lm check`, which proves this map bit-exact against GE's own data,
proves it for every consumer at once.

**A header written before 2026-09-18 says the opposite and carries no marker;
decode the case again.** `d710 lm check` fails outright on one.

Measurements, the earlier false lead, and what the fix does not change:
`.claude/D710_AUDITS.md`, "Ring-pairing audit".
"""

from __future__ import annotations

import numpy as np

from . import interfile
from .geometry import crystal_to_det, det_pair_map


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
                self.pl[r2, r1] = p

        self.xtal2det = crystal_to_det(ndet).astype(np.int32)
        self.ndet = ndet
        self.nrings = nrings

    @property
    def n_bin(self) -> int:
        return self.n_plane * self.n_view * self.n_tang

    @property
    def shape(self) -> tuple:
        return (self.n_plane, self.n_view, self.n_tang)

    def ring_pairs_by_plane(self):
        """`(r1, r2, plane)`, one row per ring pair, read back out of `pl`.

        This is the module docstring as an array: `r1` belongs with `det1` and
        `r2` with `det2`. Anything that turns a plane back into a geometric
        line should come through here rather than through
        `Header.ring_pairs()`, so that there is one statement of the pairing
        and `d710 lm check` proves it for every consumer at once.
        """
        r1, r2 = np.nonzero(self.pl >= 0)
        return (r1.astype(np.int32), r2.astype(np.int32),
                self.pl[r1, r2].astype(np.int32))

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
