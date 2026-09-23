"""2D OSEM on the raw counts, the conventional baseline DeepPET was compared with.

The model is the one the data were drawn from, `ybar = mult * P x + gamma`, with
`mult = s * n * AF` and `gamma` the randoms + scatter mean -- corrections inside
the model, not subtracted, as a clinical OSEM does. The paper ran 5 iterations
x 16 subsets and, "typically for the GE D710/690", a 6.4 mm Gaussian post-filter
(transaxial only, the data being 2D). Subsets are interleaved views.
"""

from __future__ import annotations

import numpy as np

from utils.scanner import POST_FILTER_FWHM_MM

from .scanner2d import N_VIEW, Scanner2D, fov_mask

N_ITER, N_SUBSETS = 5, 16

FWHM = 2.3548200450309493


def osem(y, mult, gamma, scanner: Scanner2D, n_iter: int = N_ITER,
         n_subsets: int = N_SUBSETS, post_fwhm_mm: float = POST_FILTER_FWHM_MM):
    """`(grid, grid)` image in the units `mult` converts from (SUV for our data)."""
    from scipy.ndimage import gaussian_filter

    y, mult, gamma = (np.asarray(a, np.float32) for a in (y, mult, gamma))
    mask = fov_mask(scanner.grid)
    x = mask.astype(np.float32)
    subsets = [np.arange(k, N_VIEW, n_subsets) for k in range(n_subsets)]
    sens = [scanner.back(mult[v], v) for v in subsets]
    for _ in range(n_iter):
        for v, s in zip(subsets, sens):
            ybar = mult[v] * scanner.fwd(x, v) + gamma[v]
            ratio = np.where(ybar > 0, y[v] / np.maximum(ybar, 1e-12), 0.0)
            x = np.where(s > 0, x * scanner.back(mult[v] * ratio, v) / np.maximum(s, 1e-12), 0.0)
            x = np.where(mask, x, 0.0).astype(np.float32)
    if post_fwhm_mm > 0:
        x = gaussian_filter(x, post_fwhm_mm / FWHM / scanner.voxel_mm, mode="constant")
    return np.where(mask, x, 0.0).astype(np.float32)
