"""The decoded event table, and the arrays every consumer wants from it.

`d710 decode --listmode --format npy` writes `decoded/bed<n>.lm.npy` of
`gerdf.listmode.EVENT_DTYPE`: `xtal_a`, `xtal_b` (`ring*576 + trans`), `tof_bin`
(-27..+27, GE order) and `t_ms`.

**Two TOF frames, and they are not the same.** TOF is a signed displacement
along a directed LOR, so the index depends on which way the LOR is pointing:

* into a **sinogram bin**, whose direction is the bin's own `(det1, det2)` --
  events recorded the other way round have to be mirrored (`tof_index`);
* into **PyTomography**, which is handed `(xtal_a, xtal_b)` as they are, so the
  event's own direction is the frame and one global sign covers every event
  (`detector_ids`, measured by `d710 lm tofcheck`).

Mixing them up is invisible: counts, file size and every invariant are identical
either way. Only the image differs.
"""

from __future__ import annotations

import numpy as np

from . import geom

FIELDS = ("xtal_a", "xtal_b", "tof_bin", "t_ms")


def load(path, mmap: bool = True):
    e = np.load(str(path), mmap_mode="r" if mmap else None)
    missing = [f for f in FIELDS[:3] if f not in (e.dtype.names or ())]
    if missing:
        raise SystemExit(f"error: {path} is not an event table (no {missing})")
    return e


def tof_index(e, binmap, n_tof: int, swap=None):
    """`(N,)` 0-based TOF index in the **sinogram bin's** frame.

    Measured, not assumed: histogramming ped bed 1 this way reproduces
    `pedtof/decoded/bed1.s` and `pedtof5/decoded/bed1.s` bit for bit, and no
    other combination of the two signs comes close (`d710 lm check`).
    """
    if swap is None:
        _, swap = binmap.flat(np.asarray(e["xtal_a"]), np.asarray(e["xtal_b"]),
                              with_swap=True)
    t = geom.tof_to_stir(np.asarray(e["tof_bin"]), n_tof)
    return np.where(swap, t, (n_tof - 1) - t).astype(np.int32)


def detector_ids(e, n_tof: int = geom.N_TOF_RAW, tof_sign: int = 1):
    """`(N, 3)` int32 `[xtal_a, xtal_b, tof_idx]` in the **event's own** frame.

    No per-event mirroring here: the crystal pair is passed through in the order
    it was recorded, so the LOR direction is the event's own and one global
    `tof_sign` covers the lot.
    """
    t = geom.tof_to_stir(np.asarray(e["tof_bin"]), n_tof)
    if tof_sign < 0:
        t = (n_tof - 1) - t
    return np.stack([np.asarray(e["xtal_a"], np.int32),
                     np.asarray(e["xtal_b"], np.int32),
                     t.astype(np.int32)], axis=1)


def bins(e, binmap, with_swap: bool = False):
    """`(N,)` flat non-TOF bin index, -1 outside the sinogram."""
    return binmap.flat(np.asarray(e["xtal_a"]), np.asarray(e["xtal_b"]),
                       with_swap=with_swap)


def histogram(e, binmap, n_tof: int = 1, dtype="<i2"):
    """Events -> `(n_tof, plane, view, tang)`, the layout `decoded/bed<n>.s` uses.

    Returns `(array, n_dropped)`.
    """
    b, swap = bins(e, binmap, with_swap=True)
    ok = b >= 0
    out = np.empty((n_tof,) + binmap.shape, dtype)
    # One TOF bin at a time: a single bincount over 55 x 60.7 M would be 26 GB.
    t = (tof_index(e, binmap, n_tof, swap) if n_tof > 1
         else np.zeros(len(b), np.int32))
    for k in range(n_tof):
        sel = ok if n_tof == 1 else (ok & (t == k))
        h = np.bincount(b[sel], minlength=binmap.n_bin)
        if h.max() > np.iinfo(dtype).max:
            raise SystemExit(f"error: {h.max()} counts in one bin overflows {dtype}")
        out[k] = h.reshape(binmap.shape)
    return out, int((~ok).sum())
