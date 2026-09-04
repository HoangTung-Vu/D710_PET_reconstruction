"""Per-event weights and additive term, looked up from the bed's own sinograms.

The list-mode model PyTomography implements is, per event,

    y_i = (H x)_i + a_i          with the sensitivity image  H~^T w

so the weight `w` does **not** divide the forward projection: the background has
to arrive already divided by it. That is the `additive_term / weights` in
`PyTomo/t_GE_HDF5.ipynb`, and it makes this the same model as `y = S(Gx) + b` on
the sinogram side.

Everything on disk is per **bin**, and a bin at an odd axial position of segment
0 collects two ring pairs. `normdt`, `randoms` and `scatter` all carry that
factor of two (`utils.geometry.ring_pair_multiplicity`); an event is one LOR, so
those three are divided by it here -- the exact inverse of the mistake that
docstring warns about. `attn` does **not** carry it: it is a survival
probability of one LOR, not a count or a bin sensitivity.
"""

from __future__ import annotations

import numpy as np

from utils import terms as sino

#: What has to be on disk before a bed can be reconstructed in list mode.
NEEDED = ("normdt", "attn", "randoms", "scatter")

#: Terms that are per-bin sums over ring pairs, so an event gets `1 / mult` of them.
PER_BIN = ("normdt", "norm_only", "randoms", "scatter", "background")


def read(case, bed: int, name: str, binmap, per_lor: bool = True):
    """One term as a flat `(plane*view*tang,)` float32 array, read with numpy.

    Every file in `work/bed<n>/` is a header cloned from the decoded prompts with
    the array in `(1, plane, view, tang)` order, so `np.fromfile` is enough and
    no SIRF is needed -- see `lm.interfile.Header.require_plane_major`.
    """
    from . import interfile

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
    """`w` per LOR, flat over the non-TOF bins: norm x dead time x attenuation."""
    return read(case, bed, "normdt", binmap) * read(case, bed, "attn", binmap)


def scatter_tof_weights(case, bed: int, binmap, n_tof: int, e=None):
    """`(w, note)`: the scatter's TOF shape, summing to 1 over TOF.

    GE's own (`scatter_tof.npy`) when the bed was estimated with `reconMethod 3`;
    otherwise measured from this bed's own tail ring, exactly the way
    `utils.terms.scatter_tof_profile` does it for the sinogram path -- fed the
    events' own `(tof, tangential)` histogram in place of a TOF sinogram.
    """
    got = sino.vendor_tof_weights(case, bed, n_tof, binmap.n_view, binmap.n_tang)
    if got:
        return got
    if e is None:
        raise SystemExit(
            f"error: bed {bed} has no scatter_tof.npy and no events to measure a "
            f"profile from.\n  re-estimate in TOF mode: d710 estimate "
            f"--case {case.name} --bed {bed} --tof")

    from . import events as ev
    from . import geom

    b = ev.bins(e, binmap)
    ok = b >= 0
    t = geom.tof_to_stir(np.asarray(e["tof_bin"]), n_tof)[ok].astype(np.int64)
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
    """`(keep, weights, additive)`; the last two only for the events kept."""
    from . import events as ev
    from . import geom

    b = ev.bins(e, binmap)
    keep = b >= 0
    b = b[keep].astype(np.int64)

    w = lor_sensitivity(case, bed, binmap)[b]
    rnd = read(case, bed, "randoms", binmap)[b] / n_tof   # randoms are flat in TOF
    sct = read(case, bed, "scatter", binmap)[b]

    if n_tof > 1:
        note = "supplied"
        if tof_scatter is None:
            tof_scatter, note = scatter_tof_weights(case, bed, binmap, n_tof, e)
        print(f"  TOF scatter: {note}")
        wt = np.asarray(tof_scatter, np.float32)
        t = geom.tof_to_stir(np.asarray(e["tof_bin"])[keep], n_tof).astype(np.int64)
        if wt.ndim == 1:
            sct = sct * wt[t]
        else:
            sct = sct * wt.reshape(n_tof, -1)[t, b % (binmap.n_view * binmap.n_tang)]

    add = (rnd + sct) / np.maximum(w, 1e-12)
    return keep, w.astype(np.float32), add.astype(np.float32)


def sensitivity(case, bed: int, binmap):
    """`(ids (M,2) int32, weights (M,) float32)` over every valid LOR."""
    ids, b = binmap.lor_table()
    return ids, lor_sensitivity(case, bed, binmap)[b.astype(np.int64)]
