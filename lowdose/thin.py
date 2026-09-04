"""Event-level decimation. Everything here is a keep-mask over the event table.

Thinning a Poisson variable gives a Poisson variable, so histogramming after
thinning and binomially thinning the histogram are the same distribution -- which
is why one decimator serves both the sinogram and the list-mode path.
"""

from __future__ import annotations

import numpy as np

MODES = ("uniform", "randoms")


def keep(e, f: float, mode: str = "uniform", rng=None, bins=None, rho=None):
    """Boolean keep-mask of length `len(e)`.

    `uniform` -- probability `f` per event. Exactly a reduced-acquisition-time
    simulation, and conservative as a low-dose one: trues and scatter scale
    right, the randoms fraction stays artificially high.

    `randoms` -- probability `q_b = f(1-rho_b) + f^2 rho_b`, so trues and scatter
    go as `f` and randoms as `f^2`, the way activity really works. Needs `bins`
    (flat bin index per event) and `rho` (randoms fraction per bin). `q_b <= f`
    always, so it is a valid probability for every `f`.
    """
    if f > 1 or f <= 0:
        raise ValueError(f"dose fraction must be in (0, 1], got {f}")
    n = len(e)
    if f == 1.0:
        return np.ones(n, bool)          # verification 1: f=1 is the identity
    rng = np.random.default_rng() if rng is None else rng
    if mode == "uniform":
        return rng.random(n) < f
    if mode != "randoms":
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if bins is None or rho is None:
        raise ValueError("mode 'randoms' needs bins and rho")
    q = f * (1.0 - rho) + f * f * rho
    out = np.zeros(n, bool)
    ok = bins >= 0
    out[ok] = rng.random(int(ok.sum())) < q[bins[ok]]
    return out


def rho_per_plane(prompts_per_plane, randoms_per_plane, binmap):
    """`rho_b` = randoms/prompts, estimated per PLANE and broadcast back to bins.

    Not per bin: raw bins hold ~0.06 counts, so `rho > 1` happens constantly from
    Poisson noise alone. A plane is the coarsest aggregate that still follows the
    real axial structure of the randoms.
    """
    p = np.asarray(prompts_per_plane, np.float64)
    r = np.asarray(randoms_per_plane, np.float64)
    rho = np.clip(np.divide(r, p, out=np.zeros_like(p), where=p > 0), 0.0, 1.0)
    return np.repeat(rho.astype(np.float32), binmap.n_view * binmap.n_tang)


def split(n: int, k: int, rng=None):
    """`(n,)` labels in `0..k-1`: `k` disjoint, mutually independent replicates.

    Independent multinomial assignment of a Poisson total gives exactly
    independent Poisson subsets -- unlike repeated thinning, whose realisations
    overlap. `k=2` is the Noise2Noise pair.
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, k, n, dtype=np.int8)
