"""2D OSEM on the raw counts, the conventional baseline DeepPET was compared with.

The model is the one the data were drawn from, `ybar = mult * P x + gamma`, with
`mult = s * n * AF` and `gamma` the randoms + scatter mean -- corrections inside
the model, not subtracted, as a clinical OSEM does. The paper ran 5 iterations
x 16 subsets and, "typically for the GE D710/690", a 6.4 mm Gaussian post-filter
(transaxial only, the data being 2D). Subsets are interleaved views.

`osem_ge` is GE's clinical protocol instead, for a stack of axial slices: 2
iterations x 24 subsets (`utils.scanner`, from the DICOM tags), then GE's
post-filter on the stack -- 6.4 mm transaxial and the axial [1, R, 1], R = 4
(`osem.stitch.post_filter`, the one the reconstruction pipeline uses). On real
bed 4 of fdg26081901 that brought the noise of slice 12 from 0.161 (5 x 16,
transaxial filter only) to 0.111, against 0.122 for GE's own image.
"""

from __future__ import annotations

import numpy as np

from utils.scanner import (N_ITERATIONS, N_SUBSETS as GE_N_SUBSETS, PLANE_MM,
                           POST_FILTER_FWHM_MM, POST_FILTER_Z_RATIO)

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


def osem_ge(y, mult, gamma, scanner: Scanner2D, slices=None, n_iter: int = N_ITERATIONS,
            n_subsets: int = GE_N_SUBSETS, fwhm_mm: float = POST_FILTER_FWHM_MM,
            z_ratio: float = POST_FILTER_Z_RATIO, recon=None):
    """GE's protocol on `(n, 288, 371)` stacks of axial slices: `(len(slices), grid, grid)`.

    Each slice is reconstructed with `n_iter` x `n_subsets` and no filter, then
    the stack gets GE's post-filter: `fwhm_mm` transaxial and the axial
    `[1, z_ratio, 1]`, edges replicated. `slices`: only these (their neighbours
    are reconstructed too, for the axial filter); all of them when None.
    `recon`: an already reconstructed, unfiltered `{slice: image}` to reuse.
    """
    from osem.stitch import post_filter

    n = len(y)
    want = list(range(n)) if slices is None else [int(q) for q in slices]
    need = sorted({j for q in want for j in (q - 1, q, q + 1) if 0 <= j < n})
    rec = dict(recon or {})
    for j in need:
        if j not in rec:
            rec[j] = osem(y[j], mult[j], gamma[j], scanner, n_iter, n_subsets, post_fwhm_mm=0.0)
    vox = (PLANE_MM, scanner.voxel_mm, scanner.voxel_mm)
    mask = fov_mask(scanner.grid)
    out = []
    for q in want:
        trio = np.stack([rec[max(q - 1, 0)], rec[q], rec[min(q + 1, n - 1)]])
        out.append(post_filter(trio, vox, fwhm_mm, z_ratio, verbose=False)[1] * mask)
    return np.stack(out).astype(np.float32)
