"""Attenuation factors by line integral through the mu-map, with parallelproj.

This replaces SIRF's `AcquisitionSensitivityModel.compute_attenuation_factors`,
which was the last thing in the reconstruction path that needed STIR. The factor
stored for a bin is the survival probability

    af = mean over the bin's ring pairs of  exp( -int mu dl )

with `mu` in 1/mm and the path in mm, on the image grid and in the world frame
PyTomography reconstructs in -- `image_origin` is copied from
`PETLMSystemMatrix`, so `d710 attn` and `d710 lm` cannot drift apart.

Two things are easy to get wrong. **Units**: parallelproj integrates in the
units of the coordinates it is given, which are mm, while the CT conversion
produces 1/cm; `tests/test_attn_proj.py` pins this on a cylinder whose chord is
known in closed form. **Which detector carries which ring**: this module does
not decide that and must not -- it reads `BinMap.ring_pairs_by_plane()`, and
`utils/binmap.py` is where the pairing is stated.

Agreement with SIRF, the ring-order false lead it used to encode, and what the
correction moved: `.claude/D710_AUDITS.md`, "Ring-pairing audit".
"""

from __future__ import annotations

import numpy as np

from .binmap import BinMap
from .geometry import det_pair_map, detector_xy_mm, ring_z_mm
from .scanner import DR_MM, PLANE_MM


def image_origin(shape, voxel_mm) -> np.ndarray:
    """World coordinates of voxel `[0, 0, 0]`, in PyTomography's convention."""
    return ((-np.asarray(shape, np.float64) / 2 + 0.5)
            * np.asarray(voxel_mm, np.float64)).astype(np.float32)


def image_for_projector(mu_percm, dr_mm: float = DR_MM,
                        plane_mm: float = PLANE_MM):
    """`(img, origin, voxel)` for parallelproj, from a `(plane, y, x)` map in 1/cm."""
    mu = np.asarray(mu_percm)
    if mu.ndim != 3 or mu.shape[1] != mu.shape[2]:
        raise SystemExit(f"error: mu-map {mu.shape} is not (plane, xy, xy)")
    img = np.ascontiguousarray(mu.transpose(2, 1, 0) / 10.0, np.float32)
    voxel = np.array([dr_mm, dr_mm, plane_mm], np.float32)
    return img, image_origin(img.shape, voxel), voxel


def _endpoints(binmap: BinMap):
    """`(xy1, xy2, z)`: the transaxial ends of every bin of one sinogram, and ring z.

    `xy1` belongs with `det1`, and so with the ring `BinMap` calls `r1`.
    """
    d1, d2 = det_pair_map(binmap.n_view, binmap.n_tang, binmap.ndet)
    xy = detector_xy_mm(binmap.ndet)
    return xy[d1.ravel()], xy[d2.ravel()], ring_z_mm(binmap.nrings)


def factors(mu_percm, hs, dr_mm: float = DR_MM, plane_mm: float = PLANE_MM,
            device: str = "auto", out=print) -> np.ndarray:
    """`(n_plane, n_view, n_tang)` attenuation factors for the sinogram `hs`.

    `mu_percm` is `(n_plane_image, xy, xy)` in 1/cm, in the image's own
    `(plane, y, x)` order -- exactly what `utils.attenuation.mu_map` returns.
    """
    binmap = BinMap(hs)
    img, origin, voxel = image_for_projector(mu_percm, dr_mm, plane_mm)
    xy1, xy2, z_ring = _endpoints(binmap)
    n_lor = xy1.shape[0]

    r1s, r2s, planes = binmap.ring_pairs_by_plane()
    fwd, to_host, xs, xe, im, og, vx = _bind(device, img, origin, voxel,
                                             xy1, xy2, out)

    acc = np.zeros((binmap.n_plane, n_lor), np.float32)
    mult = np.zeros(binmap.n_plane, np.int32)
    for r1, r2, p in zip(r1s, r2s, planes):
        xs[:, 2] = float(z_ring[r1])
        xe[:, 2] = float(z_ring[r2])
        acc[p] += to_host(fwd(xs, xe, im, og, vx))
        mult[p] += 1
    if not np.array_equal(mult, binmap.mult):
        raise SystemExit(f"error: {hs} -- the ring pairs read back out of the "
                         f"bin map do not match its own multiplicity")

    af = (acc / mult[:, None]).reshape(binmap.shape)
    out(f"  {int(mult.sum())} ring pairs x {n_lor:,} LORs over "
        f"{binmap.n_plane} planes = {int(mult.sum()) * n_lor:,} rays, "
        f"af mean {af.mean():.4f}  min {af.min():.4f}")
    return af


def _bind(device: str, img, origin, voxel, xy1, xy2, out):
    """Pick numpy or torch for the projection; the LOR ends keep their z free.

    `auto` is `cuda` where torch sees a GPU and `cpu` otherwise. `cpu` and
    `numpy` both take the numpy route, which has no torch in it at all;
    anything else names a torch device, and `torch:cpu` exercises that route
    on a machine with no GPU, which is how it is tested.
    """
    if device == "auto":
        device = "cuda" if _cuda_ready() else "cpu"

    if device in ("cpu", "numpy"):
        xs = np.empty((xy1.shape[0], 3), np.float32)
        xe = np.empty((xy2.shape[0], 3), np.float32)
        xs[:, :2], xe[:, :2] = xy1, xy2
        return _fwd_exp_numpy, (lambda a: a), xs, xe, img, origin, voxel

    import torch

    dev = torch.device(device[6:] if device.startswith("torch:") else device)
    out(f"  projector on {dev}")

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), device=dev)

    xs = torch.empty((xy1.shape[0], 3), dtype=torch.float32, device=dev)
    xe = torch.empty((xy2.shape[0], 3), dtype=torch.float32, device=dev)
    xs[:, :2], xe[:, :2] = t(xy1), t(xy2)
    return (_fwd_exp_torch, (lambda a: a.cpu().numpy()), xs, xe,
            t(img), t(origin), t(voxel))


def _fwd_exp_numpy(xs, xe, img, origin, voxel):
    import parallelproj

    return np.exp(-parallelproj.joseph3d_fwd(xs, xe, img, origin, voxel))


def _fwd_exp_torch(xs, xe, img, origin, voxel):
    import parallelproj
    import torch

    return torch.exp(-parallelproj.joseph3d_fwd(xs, xe, img, origin, voxel))


def _cuda_ready() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
