"""Per-event sensitivity weights and additive term, from the bed's own sinograms."""

from __future__ import annotations

import numpy as np

from utils import terms as sino

NEEDED = ("normdt", "attn", "randoms", "scatter")

PER_BIN = ("normdt", "norm_only", "randoms", "scatter", "background")


def read(case, bed: int, name: str, binmap, per_lor: bool = True):
    """One term as a flat `(plane*view*tang,)` float32 array."""
    from utils import interfile

    p = case.work_bed(bed) / f"{name}.hs"
    if not p.exists():
        how = (f"run: d710 attn --case {case.name}" if name == "attn" else
               f"run: d710 tostir --case {case.name} --bed {bed}")
        raise SystemExit(f"error: no {p}\n  {how}")
    h = interfile.Header(p)
    h.require_plane_major()
    if h.n_tof != 1:
        raise SystemExit(f"error: {p} has a TOF axis; every term here is non-TOF")
    a = np.fromfile(h.data_file(), "<f4")
    if a.size != binmap.n_bin:
        raise SystemExit(f"error: {p} holds {a.size:,} bins, the bin map has "
                         f"{binmap.n_bin:,}")
    a = a.reshape(binmap.shape)
    if per_lor and name in PER_BIN:
        a = a / binmap.mult[:, None, None]
    return a.reshape(-1)


def lor_sensitivity(case, bed: int, binmap):
    """Per-LOR weight, flat over the non-TOF bins: norm times dead time times attenuation."""
    return read(case, bed, "normdt", binmap) * read(case, bed, "attn", binmap)


def scatter_tof_weights(case, bed: int, binmap, n_tof: int, e=None):
    """`(w, note)`: the scatter's TOF shape, summing to 1 over TOF."""
    got = (sino.vendor_tof_weights(case, bed, n_tof, binmap.n_view, binmap.n_tang)
           or sino.measured_tof_weights(case, bed, n_tof))
    if got:
        return got
    if e is None:
        raise SystemExit(
            f"error: bed {bed} has no scatter_tof.npy and no events to measure a "
            f"profile from.\n  re-estimate in TOF mode: d710 estimate "
            f"--case {case.name} --bed {bed} --tof")

    from . import events as ev

    b, swap = ev.bins(e, binmap, with_swap=True)
    ok = b >= 0
    t = ev.tof_index(e, binmap, n_tof, swap)[ok].astype(np.int64)
    u = (b[ok] % binmap.n_tang).astype(np.int64)
    P = np.bincount(t * binmap.n_tang + u,
                    minlength=n_tof * binmap.n_tang).reshape(n_tof, binmap.n_tang)

    def tang(name):
        a = read(case, bed, name, binmap, per_lor=False)
        return a.reshape(binmap.shape).sum((0, 1))[None, None, None, :]

    return sino.scatter_tof_profile({"prompts": P[:, None, None, :].astype(np.float64),
                                     "randoms": tang("randoms"),
                                     "scatter": tang("scatter")}, n_tof)


def event_terms(case, bed: int, e, binmap, n_tof: int, tof_scatter=None):
    """`(keep, weights, additive)`, the last two given only for the events kept."""
    from . import events as ev

    b, swap = ev.bins(e, binmap, with_swap=True)
    keep = b >= 0
    b = b[keep].astype(np.int64)

    w = lor_sensitivity(case, bed, binmap)[b]
    rnd = read(case, bed, "randoms", binmap)[b] / n_tof
    sct = read(case, bed, "scatter", binmap)[b]

    if n_tof > 1:
        note = "supplied"
        if tof_scatter is None:
            tof_scatter, note = scatter_tof_weights(case, bed, binmap, n_tof, e)
        print(f"  TOF scatter: {note}")
        wt = np.asarray(tof_scatter, np.float32)
        t = ev.tof_index(e, binmap, n_tof, swap)[keep].astype(np.int64)
        if wt.ndim == 1:
            sct = sct * wt[t]
        else:
            sct = sct * wt.reshape(n_tof, -1)[t, b % (binmap.n_view * binmap.n_tang)]

    add = (rnd + sct) / np.maximum(w, 1e-12)
    return keep, w.astype(np.float32), add.astype(np.float32)


def sensitivity(case, bed: int, binmap):
    """`(ids (M, 2) int32, weights (M,) float32)` over every valid LOR."""
    ids, b = binmap.lor_table()
    return ids, lor_sensitivity(case, bed, binmap)[b.astype(np.int64)]
