"""Event-level decimation. Everything here is a keep-mask over the event table.

Thinning a Poisson variable gives a Poisson variable, so histogramming after
thinning and binomially thinning the histogram are the same distribution -- which
is why one decimator serves both the sinogram and the list-mode path, and why
`binomial_sinogram` is a legitimate substitute for `keep` on a bed that has no
event table at all.

Two modes, named after what they SIMULATE rather than after the draw they make:

    low-count   a shorter acquisition.  Everything linear in `f`.
    low-dose    less activity.  Trues and scatter linear in `f`, randoms in `f^2`.
"""

from __future__ import annotations

import numpy as np

#: The two things this package can simulate. `low-count` is the default and the
#: primary result -- see README.md.
MODES = ("low-count", "low-dose")

#: What the modes were called when they were named after their mechanism. Kept
#: because they are written into the `mode` field of every `lowdose.json` already
#: on disk, and typed into scripts.
ALIASES = {"uniform": "low-count", "randoms": "low-dose"}


def canonical(mode: str) -> str:
    """Either spelling of a mode -> the `MODES` member. Raises on anything else."""
    m = ALIASES.get(mode, mode)
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES} (or {tuple(ALIASES)}), "
                         f"got {mode!r}")
    return m


def randoms_power(mode: str) -> int:
    """`1` for low-count, `2` for low-dose -- **the only place this is decided**.

    Randoms are coincidences between two unrelated singles, so their rate goes as
    the square of the activity while trues and scatter go linearly. A shorter
    scan of the same patient scales everything alike; a smaller injection does
    not. Three call sites used to carry their own `2 if mode == "randoms" else 1`
    and nothing stopped them drifting apart.
    """
    return 2 if canonical(mode) == "low-dose" else 1


def keep(e, f: float, mode: str = "low-count", rng=None, bins=None, rho=None,
         tof=None):
    """Boolean keep-mask of length `len(e)`.

    `low-count` -- probability `f` per event. Exactly a reduced-acquisition-time
    simulation, and conservative as a low-dose one: trues and scatter scale
    right, the randoms fraction stays artificially high.

    `low-dose` -- probability `q_b = f(1-rho_b) + f^2 rho_b`, so trues and scatter
    go as `f` and randoms as `f^2`, the way activity really works. Needs `bins`
    (flat bin index per event) and `rho` (randoms fraction per bin). `q_b <= f`
    always, so it is a valid probability for every `f`.

    `tof` -- optional `(tof_index_per_event, factor)`, where `factor` comes from
    `tof_rho_factor` and the index is each event's TOF bin **in the sinogram
    bin's frame** (`lm.events.tof_index`). It makes `rho` follow the TOF axis as
    well: randoms are flat in TOF while trues are peaked, so without it every TOF
    bin of a LOR shares one `q` and the deep TOF tails keep up to 7x too many
    counts (measured, fdg26081008 bed 1). See `tof_rho_factor`.

    `rho` is gathered per event before `q` is formed rather than after: `q` over
    all 60.7 M bins is 243 MB, `q` over the events is a quarter of that.
    """
    if f > 1 or f <= 0:
        raise ValueError(f"dose fraction must be in (0, 1], got {f}")
    mode = canonical(mode)
    n = len(e)
    if f == 1.0:
        return np.ones(n, bool)          # verification 1: f=1 is the identity
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
    """`(q, ok)` -- the keep probability of each event **inside** the sinogram.

    Separate from `keep` because the exact mean and variance of the draw are sums
    over these numbers, and nothing else gives them: see `expectation`.

    `rho` is per bin and gathered here rather than turned into a per-bin `q` first
    -- `q` over all 60.7 M bins is 243 MB, `q` over the events a quarter of that.
    """
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
    """`(mu, var)` per plane: the **exact** mean and variance of the thinned count.

    A thinned plane total is a sum of independent Bernoulli draws, one per event,
    so `mu = sum q_i` and `var = sum q_i (1 - q_i)` over the events in that plane.
    Exact, with no model in it at all.
    """
    q = np.asarray(q, np.float64)
    p = np.asarray(plane_of_event, np.int64)
    mu = np.bincount(p, weights=q, minlength=n_plane)
    var = np.bincount(p, weights=q * (1.0 - q), minlength=n_plane)
    return mu, var


def binomial_sinogram(y, q, rng=None):
    """Thin a histogram directly: `y'_b ~ Binomial(y_b, q_b)`.
    `y` is `(tof, plane, view*tang)` as it sits on disk. `q` is `(n_plane,)` or
    `(n_plane, n_tang)`, matching the two resolutions `rho_bins` estimates at; it
    broadcasts over views either way. One plane at a time: a single `binomial`
    over 60.7 M bins allocates 486 MB of int64 for no reason.
    """
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


#: Resolutions `rho_bins` will aggregate at. Never per bin -- raw bins hold ~0.06
#: counts, so `rho > 1` happens constantly from Poisson noise alone.
RHO_AXES = ("tangential", "plane")


def rho_bins(prompts, randoms, binmap, axis: str = "tangential"):
    """`(n_bin,)` float32 `rho = randoms/prompts`, aggregated at `axis`.

    `plane` -- the original, coarser choice. Reachable for a bed so thin that a
    `(plane, u)` cell has too few counts to divide; it is wrong in the tails by
    construction, so prefer `tangential` unless the counts say otherwise.

    `prompts` and `randoms` are `(n_plane,)` for `plane` and `(n_plane, n_tang)`
    for `tangential`.
    """
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
    """`rho_bins(..., axis="plane")`. Kept for callers that named the old shape."""
    return rho_bins(prompts_per_plane, randoms_per_plane, binmap, "plane")


def tof_rho_factor(phi, n_tof: int):
    """`(n_tang, n_tof)` factor turning a non-TOF `rho` into a TOF-resolved one.

    **Why a factor and not a fresh estimate.** `rho` in a `(bin, TOF)` cell would
    need the prompts there, and that is ~0.001 counts a cell -- hopeless. But
    randoms are **flat in TOF** (measured: CoV 0.0404 against a Poisson floor of
    0.0393, `tools/tof_profile.py`), so the whole TOF dependence of `rho` comes
    from how the *other* components are distributed:

        rho(b, t) = (R_b / n_tof) / y(b, t)

    and if the TOF shape at radial offset `u` is the same for every plane and view,
    `y(b, t) = y_b * phi(u, t)`, giving

        rho(b, t) = rho_nonTOF(b) * (1 / n_tof) / phi(u, t)

    -- the non-TOF `rho`, which already carries the axial structure, times a factor
    that depends only on `(u, t)`. `phi` is measured at `n_tang * n_tof` cells

    `phi` is `(n_tang, n_tof)` and need not be normalised; it is normalised here.
    A flat `phi` returns all ones, i.e. the non-TOF behaviour, exactly.
    """
    phi = np.asarray(phi, np.float64)
    if phi.shape[1] != n_tof:
        raise ValueError(f"phi has {phi.shape[1]} TOF bins, expected {n_tof}")
    tot = phi.sum(axis=1, keepdims=True)
    phi = np.divide(phi, tot, out=np.full_like(phi, 1.0 / n_tof), where=tot > 0)
    # Where a (u, t) cell saw nothing, the factor would be infinite. Those cells
    # hold no events either, so the value never gets used -- but it must not be a
    # nan, because it is multiplied into an array that is then clipped.
    return np.divide(1.0 / n_tof, phi, out=np.ones_like(phi), where=phi > 0)


WINDOWS = ("uniform", "time")


def decay_scales(half_life_s: float, frame_ms: float, f: float):
    """`(linear, quadratic)` -- what a time window really does to the terms.

    Truncating the frame to `tau = f*T` is NOT `x f` for anything, because the
    activity decays *during* the frame, so the retained window holds a higher mean
    activity than the frame average:

        trues, scatter   rate ~ A    ->  integral A dt    ->  (1-e^-mu.tau)/(1-e^-mu.T)
        randoms          rate ~ A^2  ->  integral A^2 dt  ->  (1-e^-2mu.tau)/(1-e^-2mu.T)

    **Randoms therefore carry a different exponent from trues even in low-count
    mode** -- not because of dose, but because a random is a coincidence of TWO
    coincidence and its rate falls twice as fast as the activity.

    F-18, 90 s frame, 9 s window: 0.100427 and 0.100855 against a nominal 0.1, so
    +0.43 % and +0.85 %. Checked against the data: the predicted prompts factor
    `(1-rho).lin + rho.quad` = 0.10061 at rho = 0.435, measured 0.1004-0.1007 on
    fdg26081008 beds 1/4/7.

    It scales with frame/half-life and stops being negligible fast: the randoms
    factor is +0.9 % out for F-18, +4.7 % for C-11 and **+52 % for O-15**.

    A *uniform* thinning has none of this -- it keeps a fraction `f` of every
    event whatever time it arrived, so every integral scales by exactly `f`.
    """
    lam = np.log(2.0) / float(half_life_s)
    T = float(frame_ms) / 1000.0
    tau = f * T
    lin = -np.expm1(-lam * tau) / -np.expm1(-lam * T)
    quad = -np.expm1(-2.0 * lam * tau) / -np.expm1(-2.0 * lam * T)
    return float(lin), float(quad)


def time_window(t_ms, f: float, frame_ms: float):
    """Keep the first `f` of the frame: a shorter acquisition, literally.

    The window starts where the frame does, so it is "the scanner was stopped
    early". That is the interpretation that makes `frame_duration_ms` simply
    shorter, which is all `osem.stitch.decay_factor` needs to stay correct. A
    window centred on the frame instead would reproduce a uniform thinning.

    Deterministic: no RNG, and the same data twice gives the same answer.
    """
    if f > 1 or f <= 0:
        raise ValueError(f"dose fraction must be in (0, 1], got {f}")
    t = np.asarray(t_ms)
    if f == 1.0:
        return np.ones(len(t), bool)
    return t < t.min() + f * float(frame_ms)


def split(n: int, k: int, rng=None):
    """`(n,)` labels in `0..k-1`: `k` disjoint, mutually independent replicates.

    Independent multinomial assignment of a Poisson total gives exactly
    independent Poisson subsets -- unlike repeated thinning, whose realisations
    overlap. `k=2` is the Noise2Noise pair.
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, k, n, dtype=np.int8)
