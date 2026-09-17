"""Checks a thinned case must pass before anything is reconstructed from it."""

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
    """`(n_plane, n_tang)` float64 sums, collapsing TOF and view."""
    a = np.fromfile(path, dtype)
    if a.size % (n_plane * n_vt) or n_vt % n_tang:
        raise SystemExit(f"error: {path} holds {a.size:,} samples, which is not "
                         f"{n_plane} planes x {n_vt // n_tang} views x {n_tang}")
    return a.reshape(-1, n_plane, n_vt // n_tang, n_tang).sum(axis=(0, 2),
                                                              dtype=np.float64)


def lm_matches_sinogram(src, beds, out=print) -> list:
    """Check that the event table reproduces the decoded sinogram."""
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
    """Per plane, require `sum(y')` within three standard deviations of `Binomial(sum(y), q)`."""
    bad = 0
    out(f"{'bed':>4} {'sum y':>14} {'sum y_thin':>14} {'expected':>14} "
        f"{'planes > 3 sd':>16}")
    for n in beds:
        p = q[n] if isinstance(q, dict) else q
        if isinstance(p, tuple):
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
    """Require `sum(p) >= sum(r)` and `sum(s) <= sum(p - r)` per plane."""
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
