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


def binomial(src, dst, beds, q, n_plane: int, n_vt: int, out=print) -> int:
    """Per plane, `sum(y')` must sit within 3 sd of `Binomial(sum(y), q)`.

    `q` is the keep probability: a scalar `f` for uniform thinning, or one value
    per plane per bed (`{bed: array}`) for the randoms-aware mode, where it is
    `f(1-rho) + f^2 rho` and varies down the axis. Passing `f` there instead
    would fail every plane by construction.

    Returns the number of planes outside the band; ~0.3 % is expected by chance.
    """
    bad = 0
    out(f"{'bed':>4} {'sum y':>14} {'sum y_thin':>14} {'expected':>14} "
        f"{'planes > 3 sd':>16}")
    for n in beds:
        y = plane_sums(src.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
        t = plane_sums(dst.decoded / f"bed{n}.s", n_plane, n_vt, "<i2")
        p = np.asarray(q[n] if isinstance(q, dict) else q, np.float64)
        mu, sd = p * y, np.sqrt(np.maximum(y * p * (1 - p), 1e-12))
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
