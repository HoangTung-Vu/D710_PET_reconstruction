"""Event-level and histogram-level decimation."""

from __future__ import annotations

import numpy as np

MODES = ("low-count", "low-dose")

ALIASES = {"uniform": "low-count", "randoms": "low-dose"}


def canonical(mode: str) -> str:
    """Either spelling of a mode, resolved to the `MODES` member."""
    m = ALIASES.get(mode, mode)
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES} (or {tuple(ALIASES)}), "
                         f"got {mode!r}")
    return m


def randoms_power(mode: str) -> int:
    """`1` for low-count, `2` for low-dose; the only place this is decided."""
    return 2 if canonical(mode) == "low-dose" else 1


def keep(e, f: float, mode: str = "low-count", rng=None, bins=None, rho=None,
         tof=None):
    """Boolean keep-mask of length `len(e)`."""
    if f > 1 or f <= 0:
        raise ValueError(f"dose fraction must be in (0, 1], got {f}")
    mode = canonical(mode)
    n = len(e)
    if f == 1.0:
        return np.ones(n, bool)
    rng = np.random.default_rng() if rng is None else rng
    if mode == "low-count":
        return rng.random(n) < f
    if bins is None or rho is None:
        raise ValueError("mode 'low-dose' needs bins and rho")
    q, ok = event_q(f, bins, rho, tof)
    out = np.zeros(n, bool)
    out[ok] = rng.random(len(q)) < q
    return out


def event_q(f: float, bins, rho, tof=None):
    """`(q, ok)`: the keep probability of each event inside the sinogram."""
    ok = bins >= 0
    b = bins[ok]
    r = np.asarray(rho)[b]
    if tof is not None:
        t_idx, factor = tof
        factor = np.asarray(factor, np.float64)
        u = b % factor.shape[0]
        r = np.clip(r * factor[u, np.asarray(t_idx)[ok]], 0.0, 1.0)
    return f * (1.0 - r) + f * f * r, ok


def expectation(q, plane_of_event, n_plane: int):
    """`(mu, var)` per plane: the exact mean and variance of the thinned count."""
    q = np.asarray(q, np.float64)
    p = np.asarray(plane_of_event, np.int64)
    mu = np.bincount(p, weights=q, minlength=n_plane)
    var = np.bincount(p, weights=q * (1.0 - q), minlength=n_plane)
    return mu, var


def binomial_sinogram(y, q, rng=None):
    """Thin a histogram directly, as `y'_b ~ Binomial(y_b, q_b)`."""
    y = np.asarray(y)
    q = np.asarray(q, np.float64)
    n_plane = y.shape[1]
    if q.shape[0] != n_plane:
        raise ValueError(f"{q.shape[0]} keep probabilities for {n_plane} planes")
    if q.ndim == 2:
        n_tang = q.shape[1]
        if y.shape[2] % n_tang:
            raise ValueError(f"{y.shape[2]} view*tang bins is not a multiple of "
                             f"the {n_tang} tangential bins of q")
        n_view = y.shape[2] // n_tang
    elif q.ndim != 1:
        raise ValueError(f"q must be 1-D or 2-D, got {q.ndim}-D")
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty_like(y)
    for p in range(n_plane):
        yp = y[:, p].astype(np.int64)
        if q.ndim == 1:
            out[:, p] = yp if q[p] >= 1.0 else rng.binomial(yp, q[p])
        else:
            qp = np.broadcast_to(q[p], (n_view, n_tang)).reshape(-1)
            out[:, p] = np.where(qp >= 1.0, yp, rng.binomial(yp, np.minimum(qp, 1.0)))
    return out


RHO_AXES = ("tangential", "plane")


def rho_bins(prompts, randoms, binmap, axis: str = "tangential"):
    """`(n_bin,)` float32 `rho = randoms/prompts`, aggregated at `axis`."""
    if axis not in RHO_AXES:
        raise ValueError(f"axis must be one of {RHO_AXES}, got {axis!r}")
    p = np.asarray(prompts, np.float64)
    r = np.asarray(randoms, np.float64)
    want = 1 if axis == "plane" else 2
    if p.ndim != want or r.shape != p.shape:
        raise ValueError(f"axis {axis!r} needs {want}-D prompts and randoms of "
                         f"the same shape, got {p.shape} and {r.shape}")
    rho = np.clip(np.divide(r, p, out=np.zeros_like(p), where=p > 0),
                  0.0, 1.0).astype(np.float32)
    if axis == "plane":
        return np.repeat(rho, binmap.n_view * binmap.n_tang)

    out = np.empty((binmap.n_plane, binmap.n_view, binmap.n_tang), np.float32)
    out[:] = rho[:, None, :]
    return out.reshape(-1)


def rho_per_plane(prompts_per_plane, randoms_per_plane, binmap):
    """`rho_bins(..., axis="plane")`."""
    return rho_bins(prompts_per_plane, randoms_per_plane, binmap, "plane")


def tof_rho_factor(phi, n_tof: int):
    """`(n_tang, n_tof)` factor turning a non-TOF `rho` into a TOF-resolved one."""
    phi = np.asarray(phi, np.float64)
    if phi.shape[1] != n_tof:
        raise ValueError(f"phi has {phi.shape[1]} TOF bins, expected {n_tof}")
    tot = phi.sum(axis=1, keepdims=True)
    phi = np.divide(phi, tot, out=np.full_like(phi, 1.0 / n_tof), where=tot > 0)
    return np.divide(1.0 / n_tof, phi, out=np.ones_like(phi), where=phi > 0)


WINDOWS = ("uniform", "time")


def decay_scales(half_life_s: float, frame_ms: float, f: float):
    """`(linear, quadratic)`: the effect of a time window on the terms."""
    lam = np.log(2.0) / float(half_life_s)
    T = float(frame_ms) / 1000.0
    tau = f * T
    lin = -np.expm1(-lam * tau) / -np.expm1(-lam * T)
    quad = -np.expm1(-2.0 * lam * tau) / -np.expm1(-2.0 * lam * T)
    return float(lin), float(quad)


def time_window(t_ms, f: float, frame_ms: float):
    """Keep the first fraction `f` of the frame, that is, a shorter acquisition."""
    if f > 1 or f <= 0:
        raise ValueError(f"dose fraction must be in (0, 1], got {f}")
    t = np.asarray(t_ms)
    if f == 1.0:
        return np.ones(len(t), bool)
    return t < t.min() + f * float(frame_ms)


def split(n: int, k: int, rng=None):
    """`(n,)` labels in `0..k-1`, giving `k` disjoint and mutually independent replicates."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, k, n, dtype=np.int8)
