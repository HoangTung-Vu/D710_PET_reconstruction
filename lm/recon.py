"""LM-OSEM / BSREM for one bed, through PyTomography + parallelproj.

Output is `(47, xy, xy)` count/voxel -- the same array shape and the same
`work/bed<n>/osem.npz` layout `osem/recon.py` writes, so `osem.stitch` and
`utils.export` take it unchanged.
"""

from __future__ import annotations

import time

import numpy as np

from utils import scanner
from utils.scanner import (DR_MM, N_ITERATIONS, N_SUBSETS,  # noqa: F401
                           NSEG0, PLANE_MM, PSF_MM, XY)

from . import events as ev
from . import geom, terms


def _initial(sm, n_tang):
    """PyTomography's initial estimate, further masked to the transaxial FOV.

    `_get_object_initial` masks only the AXIAL extent — see `scanner.fov_mask`
    for what the unmasked transaxial corners do to the result.
    """
    import torch

    x = sm._get_object_initial()
    m = scanner.fov_mask(sm.object_meta.shape[0], n_tang)
    return x * torch.from_numpy(m).to(x.device)[:, :, None]


def object_meta(xy: int = XY, n_plane: int = NSEG0):
    from pytomography.metadata import ObjectMeta

    return ObjectMeta(dr=(DR_MM, DR_MM, PLANE_MM), shape=(xy, xy, n_plane))


def system_matrix(case, bed, e, binmap, n_tof, xy=XY, psf=PSF_MM, tof_sign=1,
                  n_splits=8, tof_scatter=None):
    """`(system matrix, additive term, events kept)`."""
    import torch
    from pytomography.metadata.PET import PETLMProjMeta
    from pytomography.projectors.PET import PETLMSystemMatrix
    from pytomography.transforms.shared import GaussianFilter

    keep, w, add = terms.event_terms(case, bed, e, binmap, n_tof, tof_scatter)
    ids = ev.detector_ids(e, n_tof, tof_sign)[keep]
    if n_tof == 1:
        ids = ids[:, :2]
    sens_ids, sens_w = terms.sensitivity(case, bed, binmap)
    print(f"  {len(ids):,} events kept, {int((~keep).sum()):,} outside the "
          f"sinogram; {len(sens_ids):,} sensitivity LORs")

    proj_meta = PETLMProjMeta(
        torch.from_numpy(ids), info=None,
        scanner_LUT=torch.from_numpy(geom.scanner_lut()),
        tof_meta=geom.tof_meta(n_tof) if n_tof > 1 else None,
        weights=torch.from_numpy(w),
        detector_ids_sensitivity=torch.from_numpy(sens_ids),
        weights_sensitivity=torch.from_numpy(sens_w))

    sm = PETLMSystemMatrix(object_meta(xy), proj_meta,
                           obj2obj_transforms=[GaussianFilter(psf)] if psf else [],
                           N_splits=n_splits)
    return sm, torch.from_numpy(add), int(keep.sum())


def reconstruct(case, bed: int, npy, n_tof: int = geom.N_TOF_RAW, xy: int = XY,
                n_sub: int = N_SUBSETS, n_it: int = N_ITERATIONS, psf: float = PSF_MM,
                tof_sign: int = 1, n_splits: int = 8, beta: float = 0.0,
                tof_scatter=None):
    """`(image (47, xy, xy), sensitivity (47, xy, xy))`, both float32."""
    import torch
    from pytomography.algorithms import BSREM, OSEM
    from pytomography.likelihoods import PoissonLogLikelihood

    binmap = geom.BinMap(case.prompt(bed))
    e = ev.load(npy)
    print(f"  {len(e):,} events, {n_tof} TOF bins, grid {xy}x{xy}x{NSEG0}")

    t0 = time.time()
    sm, add, _ = system_matrix(case, bed, e, binmap, n_tof, xy, psf, tof_sign,
                               n_splits, tof_scatter)
    print(f"  sensitivity image: {time.time() - t0:.0f} s", flush=True)

    ll = PoissonLogLikelihood(sm, additive_term=add)
    x0 = _initial(sm, binmap.n_tang)
    if beta > 0:
        from pytomography.priors import RelativeDifferencePrior

        algo = BSREM(ll, prior=RelativeDifferencePrior(beta=beta),
                     object_initial=x0)
    else:
        algo = OSEM(ll, object_initial=x0)

    t0 = time.time()
    x = algo(n_iters=n_it, n_subsets=n_sub)
    print(f"  {'BSREM' if beta > 0 else 'OSEM'} {n_it}x{n_sub}: "
          f"{time.time() - t0:.0f} s", flush=True)

    # PyTomography is (x, y, z); STIR and everything downstream is (z, y, x).
    img = np.ascontiguousarray(x.cpu().numpy().transpose(2, 1, 0), np.float32)
    # PETLMSystemMatrix parks 1e7 in the voxels its sensitivity never reaches, to
    # keep its own division finite. As a bed-stitching weight that is exactly
    # backwards, so it goes back to zero here.
    sens = sm.norm_BP.cpu().numpy().transpose(2, 1, 0)
    sens = np.where(sens >= 1e7 - 1, 0.0, sens)
    # Same mask on the stitching weight: outside the FOV it is noise, not a
    # small sensitivity, and stitch() would weight two beds by it.
    sens = np.ascontiguousarray(
        sens * scanner.fov_mask(xy, binmap.n_tang), np.float32)
    return img, sens
