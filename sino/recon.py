from __future__ import annotations

import time

import numpy as np

from utils import scanner
from utils.scanner import N_ITERATIONS, N_SUBSETS, NSEG0, PSF_FWHM_MM, XY

from . import terms


def initial(xy: int, n_tang: int, n_plane: int = NSEG0):
    import torch

    m = np.repeat(scanner.fov_mask(xy, n_tang)[:, :, None], n_plane, axis=2)
    return torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))


def reconstruct(case, bed: int, xy: int = XY, n_sub: int = N_SUBSETS,
                n_it: int = N_ITERATIONS, psf=PSF_FWHM_MM,
                max_rays: int | None = None):
    import pytomography
    import torch
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood

    from .projector import MAX_RAYS, SinogramSystemMatrix

    binmap, y, sens, add, dead = terms.load(case, bed)
    print(f"  {float(y.sum(dtype=np.float64)):,.0f} prompts in {binmap.n_bin:,} "
          f"bins, {dead:,} with zero sensitivity; grid {xy}x{xy}x{NSEG0}")

    dev = pytomography.device
    sm = SinogramSystemMatrix(binmap, sensitivity=torch.from_numpy(sens), xy=xy,
                              psf=psf, max_rays=max_rays or MAX_RAYS)
    del sens
    ll = PoissonLogLikelihood(sm, torch.from_numpy(y).to(dev),
                              additive_term=torch.from_numpy(add).to(dev))
    del y, add

    t0 = time.time()
    ll._set_n_subsets(n_sub)
    print(f"  sensitivity image: {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    x = OSEM(ll, object_initial=initial(xy, binmap.n_tang))(
        n_iters=n_it, n_subsets=n_sub)
    print(f"  OSEM {n_it}x{n_sub}: {time.time() - t0:.0f} s", flush=True)

    sens_img = (torch.stack(ll.norm_BPs).sum(0) if n_sub > 1 else ll.norm_BP)
    mask = scanner.fov_mask(xy, binmap.n_tang)
    img = x.cpu().numpy().transpose(2, 1, 0) * mask
    sens_img = sens_img.cpu().numpy().transpose(2, 1, 0) * mask
    return (np.ascontiguousarray(img, np.float32),
            np.ascontiguousarray(sens_img, np.float32))
