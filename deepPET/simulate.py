"""One SUV + mu slice to one precorrected sinogram: DeepPET's input, PETSTEP-style.

    Px    = fwd(blur_psf(SUV))          line integrals, SUV.mm
    AF    = exp(-fwd(mu))               attenuation factor per LOR
    T     = s * n * AF * Px             trues mean (n: optional crystal efficiency)
    R     = flat,                rf * C  randoms mean      (paper mode)
    S     = blur_tang(n*AF*Px),  sf * C  scatter mean      (paper mode)
    y     ~ Poisson(T + R + S)
    x_in  = (y - R - S) / (s * n * AF)  E[x_in] = Px, exactly

`C` is the total prompts of the slice, drawn log-uniform in `count_range`, and
`s` follows from it: `s = (1 - rf - sf) C / sum(n AF Px)`. The randoms and
scatter means are subtracted exactly, as DeepPET did (paper eq. 3); negative
bins are kept, clipping them would bias the input. `rf` and `sf` are drawn from
the ranges measured on the real D710 beds of fdg26081008 (randoms 0.24-0.54,
scatter 0.17-0.18 of the prompts).

Modes: `paper` as above; `attn` drops R and S; `pure` also drops AF, i.e. only
the forward model and Poisson noise.

Everything outside the 350 mm bore is zeroed first, in the activity and in mu:
a LOR through the FOV also crosses the image corners, and activity there would
reach the sinogram while the (masked) target ignores it.

    python -m deepPET.simulate --study 0001_20200702 --slice 120 [--grid 128] [--mode paper]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from utils.scanner import PSF_MM

from .scanner2d import Scanner2D, fov_mask

MODES = ("paper", "attn", "pure")

COUNT_RANGE = (1e5, 1e7)

RF_RANGE = (0.24, 0.54)

SF_RANGE = (0.15, 0.20)

SCATTER_FWHM_MM = 200.0

TANG_MM = 2.255

FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))

AUG_SHIFT_PX_128 = 25

AUG_ROT_DEG = 10.0


def downsample(a: np.ndarray, grid: int) -> np.ndarray:
    """A `(..., 256, 256)` array on `grid`: itself, or its 2 x 2 mean for 128."""
    n = a.shape[-1]
    if grid == n:
        return a
    f = n // grid
    if f * grid != n:
        raise ValueError(f"cannot take grid {grid} from {n}")
    return a.reshape(*a.shape[:-2], grid, f, grid, f).mean(axis=(-3, -1))


def augment(suv, mu, rng, grid: int):
    """DeepPET's augmentation, one transform applied to both images.

    With p = 1/3 the slice is shifted by up to +-25 pixels (of the 128 grid) and
    rotated by up to +-10 degrees; one in three of those is also flipped left to
    right (p = 1/9 overall), as in the paper's 3-of-9 and 1-of-3 realisations.
    """
    from scipy.ndimage import affine_transform

    if rng.random() >= 1.0 / 3.0:
        return suv, mu
    th = math.radians(rng.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
    sh = rng.uniform(-AUG_SHIFT_PX_128, AUG_SHIFT_PX_128, 2) * grid / 128.0
    flip = rng.random() < 1.0 / 3.0
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    m = rot @ np.diag([1.0, -1.0 if flip else 1.0])
    c = (np.array([grid, grid]) - 1) / 2.0
    off = c + sh - m @ c
    return tuple(affine_transform(a, m, offset=off, order=1, mode="constant", cval=0.0)
                 for a in (suv, mu))


def simulate(suv, mu, scanner: Scanner2D, rng, mode: str = "paper", counts=None,
             count_range=COUNT_RANGE, psf_mm: float = PSF_MM, eff=None,
             noiseless: bool = False, return_raw: bool = False):
    """`(x_in (288, 371), target (grid, grid), info)` for one slice.

    `suv`, `mu`: `(grid, grid)` on `scanner`'s grid. `eff`: an optional
    `(288, 371)` crystal-pair efficiency (a real bed's normdt in 2D).
    `noiseless` returns the mean instead of a Poisson draw. `return_raw` adds
    `y`, `gamma` and `mult` (so that `mean = mult * Px + gamma`) to `info`, which
    is what OSEM needs.
    """
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    mask = fov_mask(scanner.grid)
    target = np.where(mask, np.asarray(suv, np.float32), 0.0).astype(np.float32)
    img = target
    if psf_mm > 0:
        img = gaussian_filter(target, psf_mm / FWHM / scanner.voxel_mm, mode="constant")
    px = np.maximum(scanner.fwd(img), 0.0)
    if mode == "pure":
        af = np.ones_like(px)
    else:
        mu_m = np.where(mask, np.asarray(mu, np.float32), 0.0)
        af = np.exp(-np.maximum(scanner.fwd(mu_m), 0.0)).astype(np.float32)
    n = np.ones_like(px) if eff is None else np.asarray(eff, np.float32)
    shape = n * af * px

    if counts is None:
        lo, hi = count_range
        counts = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    rf = rng.uniform(*RF_RANGE) if mode == "paper" else 0.0
    sf = rng.uniform(*SF_RANGE) if mode == "paper" else 0.0
    total = float(shape.sum(dtype=np.float64))
    info = {"counts": float(counts), "rf": rf, "sf": sf, "empty": total <= 0}
    if total <= 0:
        z = np.zeros_like(px)
        if return_raw:
            info.update(y=z, gamma=z, mult=z)
        return z, target, info

    s = (1.0 - rf - sf) * counts / total
    mult = (s * n * af).astype(np.float32)
    gamma = np.zeros_like(px)
    if mode == "paper":
        gamma += np.float32(rf * counts / px.size)
        sc = gaussian_filter1d(shape, SCATTER_FWHM_MM / FWHM / TANG_MM, axis=1,
                               mode="constant")
        gamma += (sf * counts / float(sc.sum(dtype=np.float64)) * sc).astype(np.float32)
    mean = mult * px + gamma
    y = mean.astype(np.float32) if noiseless else rng.poisson(mean).astype(np.float32)
    x_in = ((y - gamma) / mult).astype(np.float32)
    info.update(s=s, trues=float(s * total))
    if return_raw:
        info.update(y=y, gamma=gamma, mult=mult, px=px)
    return x_in, target, info


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .dataset import SliceStore, default_data

    ap = argparse.ArgumentParser(description="draw one slice at three count levels")
    ap.add_argument("--data", default=None, help="default: $D710_OUT/deeppet/data")
    ap.add_argument("--study", required=True)
    ap.add_argument("--slice", type=int, default=None, help="default: the middle slice")
    ap.add_argument("--grid", type=int, default=128, choices=(128, 256))
    ap.add_argument("--mode", default="paper", choices=MODES)
    ap.add_argument("--counts", type=float, nargs="+", default=[1e5, 1e6, 1e7])
    ap.add_argument("--png", default=None)
    a = ap.parse_args(argv)

    store = SliceStore(a.data or default_data())
    k = store.studies.index(a.study) if a.study in store.studies else None
    if k is None:
        raise SystemExit(f"error: no study {a.study} in {store.root}")
    i = a.slice if a.slice is not None else store.n[k] // 2
    suv, mu = store.get(k, i, a.grid)
    sc = Scanner2D(a.grid)
    rng = np.random.default_rng(0)

    x0, target, _ = simulate(suv, mu, sc, rng, a.mode, counts=a.counts[0], noiseless=True)
    print(f"{a.study} slice {i}/{store.n[k]}, grid {a.grid}, mode {a.mode}")
    rows = []
    for c in a.counts:
        reps = [simulate(suv, mu, sc, rng, a.mode, counts=c)[0] for _ in range(20)]
        m = np.mean(reps, 0)
        print(f"  {c:.0e} prompts: mean(x_in)/Px over 20 draws = "
              f"{m.sum() / x0.sum():.4f}, per-bin CV at the Px peak "
              f"{np.std(reps, 0).flat[x0.argmax()] / x0.max():.3f}")
        rows.append((c, reps[0]))

    fig, ax = plt.subplots(1, 2 + len(rows), figsize=(4 * (2 + len(rows)), 4.4))
    ax[0].imshow(target, cmap="gray_r", vmax=np.percentile(target, 99.8))
    ax[0].set_title("target SUV")
    ax[1].imshow(x0, aspect="auto", cmap="gray_r")
    ax[1].set_title("P x (noiseless)")
    for j, (c, x) in enumerate(rows):
        ax[2 + j].imshow(x, aspect="auto", cmap="gray_r", vmin=0, vmax=x0.max())
        ax[2 + j].set_title(f"x_in, {c:.0e} prompts")
    for x in ax:
        x.axis("off")
    png = a.png or f"deeppet_sim_{a.study}_{i}_{a.mode}_{a.grid}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    print(f"  wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
