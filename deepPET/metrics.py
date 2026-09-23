"""Image metrics inside the FOV mask; no scikit-image, which the env does not carry.

rRMSE is the paper's: sqrt(MSE) / mean(ground truth). SSIM is Wang et al. 2004
with an 11-tap Gaussian window of sigma 1.5 and K1, K2 = 0.01, 0.03, the
standard constants, over the data range of the truth.
"""

from __future__ import annotations

import numpy as np


def rrmse(x, gt, mask) -> float:
    m = mask.astype(bool)
    g = gt[m].astype(np.float64)
    return float(np.sqrt(np.mean((x[m] - g) ** 2)) / max(g.mean(), 1e-12))


def psnr(x, gt, mask) -> float:
    m = mask.astype(bool)
    g = gt[m].astype(np.float64)
    mse = np.mean((x[m] - g) ** 2)
    return float(10 * np.log10(max(g.max(), 1e-12) ** 2 / max(mse, 1e-20)))


def ssim(x, gt, mask, sigma: float = 1.5) -> float:
    from scipy.ndimage import gaussian_filter

    x, g = x.astype(np.float64), gt.astype(np.float64)
    L = max(float(g[mask].max() - g[mask].min()), 1e-12)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    f = lambda a: gaussian_filter(a, sigma, truncate=3.5)
    mx, mg = f(x), f(g)
    vx, vg, cxg = f(x * x) - mx * mx, f(g * g) - mg * mg, f(x * g) - mx * mg
    s = ((2 * mx * mg + c1) * (2 * cxg + c2)) / ((mx * mx + mg * mg + c1) * (vx + vg + c2))
    return float(s[mask.astype(bool)].mean())


def corr(x, gt, mask) -> float:
    m = mask.astype(bool)
    return float(np.corrcoef(x[m], gt[m])[0, 1])
