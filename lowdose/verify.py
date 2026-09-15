"""The checks a thinned case has to pass before anything is reconstructed from it.

All of them aggregate **per plane**. The raw sinogram runs at ~0.06 count/bin, so
`p < r` is true at ~82 % of bins from Poisson noise alone and a per-bin assertion
says nothing.
"""

from __future__ import annotations

import numpy as np


def plane_sums(path, n_plane: int, n_vt: int, dtype="<f4"):
    """`(n_plane,)` float64 sums of a `(tof, plane, view, tang)` file."""
    a = np.fromfile(path, dtype)
    if a.size % (n_plane * n_vt):
        raise SystemExit(f"error: {path} holds {a.size:,} samples, not a multiple "
                         f"of {n_plane} planes x {n_vt} bins")
    return a.reshape(-1, n_plane, n_vt).sum(axis=(0, 2), dtype=np.float64)


def plane_tang_sums(path, n_plane: int, n_tang: int, n_vt: int, dtype="<f4"):
    """`(n_plane, n_tang)` float64 sums, collapsing TOF and **view**.

    The resolution `rho` is estimated at in low-dose mode, and therefore the
    resolution the binomial expectation has to be computed at: `q` is constant
    within a `(plane, u)` cell but not within a plane.
    """
    a = np.fromfile(path, dtype)
    if a.size % (n_plane * n_vt) or n_vt % n_tang:
        raise SystemExit(f"error: {path} holds {a.size:,} samples, which is not "
                         f"{n_plane} planes x {n_vt // n_tang} views x {n_tang}")
    return a.reshape(-1, n_plane, n_vt // n_tang, n_tang).sum(axis=(0, 2),
                                                              dtype=np.float64)


def lm_matches_sinogram(src, beds, out=print) -> list:
    """Does the event table reproduce the decoded sinogram? Returns the bad beds.

    Run **before** any thinning, because the default `derived` sinogram is the
    histogram of the thinned events: if the two disagree at full count, the
    thinned sinogram is a thinning of the wrong parent and the sinogram path
    silently reconstructs something the list-mode path never saw.

    `decode_in.sh` enforces this at decode time from the `.lm.json` sidecar, but
    a thinned case has no sidecar, so it is measured here instead. Measured on
    fdg26081008: exact on every bed, with no event falling outside the sinogram.
    """
    bad = []
    out(f"{'bed':>4} {'sum bed<n>.s':>16} {'events':>16} {'diff':>12}")
    for n in beds:
        y = int(np.fromfile(src.decoded / f"bed{n}.s", "<i2").sum(dtype=np.int64))
        p = src.decoded / f"bed{n}.lm.npy"
        if not p.exists():
            out(f"{n:>4} {y:>16,} {'(no event table)':>16}")
            bad.append(n)
            continue
        e = len(np.load(str(p), mmap_mode="r"))
        if e != y:
            bad.append(n)
        out(f"{n:>4} {y:>16,} {e:>16,} {e - y:>+12,}")
    return bad


def binomial(src, dst, beds, q, n_plane: int, n_vt: int, out=print) -> int:
    """Per plane, `sum(y')` must sit within 3 sd of `Binomial(sum(y), q)`.

    `q` is the keep probability, per bed (`{bed: ...}`) or shared:

    * a **scalar** `f` -- low-count thinning;
    * `(n_plane,)` -- low-dose with `rho` per plane;
    * `(n_plane, n_tang)` -- low-dose with `rho` per `(plane, u)`, the default;
    * a **`(mu, var)` pair** of `(n_plane,)` **absolute counts** -- the exact mean
      and variance of the draw, summed per event by `thin.expectation`. Used as
      given, with no `y` weighting, and it is what a TOF-resolved draw needs:
      modelling that expectation instead biased it, see `thin.expectation`.

    For the probability forms the expectation must be computed at whatever
    resolution `q` has, because `sum(q_b y_b) != f sum(y_b)` the moment `q` varies
    inside the plane. Passing a scalar `f` for a low-dose run fails 553/553 planes
    by construction.

    Returns the number of planes outside the band; ~0.3 % is expected by chance.
    """
    bad = 0
    out(f"{'bed':>4} {'sum y':>14} {'sum y_thin':>14} {'expected':>14} "
        f"{'planes > 3 sd':>16}")
    for n in beds:
        p = q[n] if isinstance(q, dict) else q
        if isinstance(p, tuple):
            # Already absolute: mean and variance of the draw itself.
            mu, var = (np.asarray(x, np.float64) for x in p)
            y = plane_sums(src.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
            t = plane_sums(dst.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
            sd = np.sqrt(np.maximum(var, 1e-12))
            k = int((np.abs(t - mu) > 3 * sd).sum())
            bad += k
            out(f"{n:>4} {y.sum():>14,.0f} {t.sum():>14,.0f} {mu.sum():>14,.0f} "
                f"{k:>11d}/{len(y)}")
            continue
        p = np.asarray(p, np.float64)
        mu_w, var_w = p, p * (1 - p)
        if mu_w.ndim == 2:
            n_tang = mu_w.shape[1]
            y = plane_tang_sums(src.decoded / f"bed{n}.s", n_plane, n_tang, n_vt, "<i2")
            t = plane_tang_sums(dst.decoded / f"bed{n}.s", n_plane, n_tang, n_vt, "<i2")
            mu, var = (mu_w * y).sum(1), (var_w * y).sum(1)
            y, t = y.sum(1), t.sum(1)
        else:
            y = plane_sums(src.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
            t = plane_sums(dst.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
            mu, var = mu_w * y, var_w * y
        sd = np.sqrt(np.maximum(var, 1e-12))
        k = int((np.abs(t - mu) > 3 * sd).sum())
        bad += k
        out(f"{n:>4} {y.sum():>14,.0f} {t.sum():>14,.0f} {mu.sum():>14,.0f} "
            f"{k:>11d}/{len(y)}")
    return bad


def invariants(case, beds, n_plane: int, n_vt: int, out=print) -> int:
    """`sum(p) >= sum(r)` and `sum(s) <= sum(p - r)`, per plane. Both must be 0."""
    bad = 0
    out(f"{'bed':>4} {'planes p<r':>14} {'planes s>p-r':>16}")
    for n in beds:
        p = plane_sums(case.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
        r = plane_sums(case.work_bed(n) / "randoms.s", n_plane, n_vt)
        s = plane_sums(case.work_bed(n) / "scatter.s", n_plane, n_vt)
        a, b = int((p < r).sum()), int((s > p - r).sum())
        bad += a + b
        out(f"{n:>4} {a:>9d}/{len(p)} {b:>11d}/{len(p)}")
    return bad
