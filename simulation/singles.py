from __future__ import annotations

import time

import numpy as np

from utils.attenuation import hu_to_mu
from utils.scanner import NDET, NRINGS

from . import phantom as ph
from .gpu import is_torch

DOWN_ZYX = (3, 4, 4)
BLOCK_T, BLOCK_Z = 9, 6
MODULE_T, N_BLOCKS_Z = 2, 4
CRYSTAL_STEP = 3
CHUNK = 1000
CHUNK_GPU_MAX = 8192
POINT_BYTES_PER_CRYSTAL = 160
KEEP_FRACTION = 0.999


def _block(a, f, how):
    offs = [(s % k) // 2 for s, k in zip(a.shape, f)]
    n = [s // k for s, k in zip(a.shape, f)]
    c = a[offs[0]:offs[0] + n[0] * f[0], offs[1]:offs[1] + n[1] * f[1],
          offs[2]:offs[2] + n[2] * f[2]]
    r = c.reshape(n[0], f[0], n[1], f[1], n[2], f[2])
    return (r.sum((1, 3, 5)) if how == "sum" else r.mean((1, 3, 5))), offs


def coarse_phantom(pdir, meta, down=DOWN_ZYX):
    act = ph.read_mhd(pdir / "act_bqml.mhd")
    hu = ph.read_mhd(pdir / "ct_hu.mhd")
    k = meta["decay_scan_to_bed"] * meta["timing"]["positron_fraction"] * ph.VOXEL_ML
    A, offs = _block(act * np.float32(k), down, "sum")
    M, _ = _block(hu_to_mu(hu, meta.get("ct_kvp", 120.0)).astype(np.float32), down, "mean")
    fine = np.asarray(ph.VOXEL_XYZ, np.float64)
    f_xyz = np.asarray(down[::-1], np.float64)
    o_xyz = np.asarray(offs[::-1], np.float64)
    origin = ph.world_origin(act.shape).astype(np.float64) + (o_xyz + (f_xyz - 1) / 2) * fine
    vox = fine * f_xyz
    mu = (np.ascontiguousarray(M.transpose(2, 1, 0), np.float32),
          origin.astype(np.float32), vox.astype(np.float32))
    return A.astype(np.float64), mu


def geometric_singles(A, mu, lut, crystal_step: int = CRYSTAL_STEP, chunk: int | None = None,
                      keep: float = KEEP_FRACTION, shield=None, device=None,
                      out=print) -> np.ndarray:
    import parallelproj
    import torch

    from .gpu import batch_for
    from .gpu import device as pick_device

    dev = pick_device(device)
    crystals = np.arange(0, NDET, crystal_step)
    det = (np.arange(NRINGS)[:, None] * NDET + crystals[None, :]).ravel()
    rdet = lut[det].astype(np.float64)
    rxy = rdet[:, :2] / np.linalg.norm(rdet[:, :2], axis=1, keepdims=True)
    flat = A.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    n = int(np.searchsorted(csum, keep * csum[-1])) + 1
    zyx = np.column_stack(np.unravel_index(order[:n], A.shape))
    act = flat[order[:n]]
    pts = mu[1][None, :].astype(np.float64) + zyx[:, ::-1] * mu[2][None, :]
    D = det.size
    chunk = int(chunk or batch_for(dev, POINT_BYTES_PER_CRYSTAL * D, CHUNK, CHUNK_GPU_MAX))
    G = torch.zeros(D, dtype=torch.float64, device=dev)
    rdet_t = torch.from_numpy(rdet).to(dev)
    rxy_t = torch.from_numpy(rxy).to(dev)
    act_t = torch.from_numpy(np.ascontiguousarray(act)).to(dev)
    xe_one = np.ascontiguousarray(rdet, np.float32)
    t0 = time.time()
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        B = len(p)
        xs = np.repeat(p.astype(np.float32), D, axis=0)
        T = parallelproj.joseph3d_fwd(xs, np.tile(xe_one, (B, 1)), *mu).reshape(B, D)
        pt = torch.from_numpy(p).to(dev)
        d = rdet_t[None, :, :] - pt[:, None, :]
        r2 = (d ** 2).sum(-1)
        cos = torch.clamp((d[..., :2] * rxy_t[None]).sum(-1) / torch.sqrt(r2), min=0.0)
        g = act_t[s:s + B, None] * cos / r2 * torch.exp(-torch.from_numpy(T).to(dev))
        if shield is not None:
            g = g * aperture(pt, rdet_t, *shield)
        G += g.sum(0)
    out(f"    singles: {len(pts):,} voxels x {D} crystals in {time.time() - t0:.0f} s "
        f"({dev}, {chunk} per chunk)")
    g = G.cpu().numpy().reshape(NRINGS, len(crystals))
    full = np.empty((NRINGS, NDET), np.float64)
    t = np.arange(NDET)
    i0 = t // crystal_step
    w = (t % crystal_step) / crystal_step
    i1 = (i0 + 1) % len(crystals)
    full[:] = (1 - w)[None, :] * g[:, i0] + w[None, :] * g[:, i1]
    return full.reshape(-1)


def aperture(p, rdet, z_shield: float, r_open: float):
    if is_torch(p):
        import torch as xp
    else:
        xp = np
    zv = p[:, None, 2]
    zs = xp.sign(zv) * z_shield
    outside = xp.abs(zv) > z_shield
    t = xp.where(outside, (zs - zv) / xp.where(outside, rdet[None, :, 2] - zv, 1.0), 0.0)
    xy = p[:, None, :2] + t[..., None] * (rdet[None, :, :2] - p[:, None, :2])
    return (~outside) | (xp.hypot(xy[..., 0], xy[..., 1]) <= r_open)


def rate_at_bed_start(counts, frame_s: float, half_life_s: float) -> np.ndarray:
    return np.asarray(counts, np.float64) / ph.frame_integral_s(frame_s, half_life_s)


def stored_to_crystal() -> np.ndarray:
    i = np.arange(NRINGS * NDET)
    block, w = np.divmod(i, BLOCK_T * BLOCK_Z)
    module, m = np.divmod(block, MODULE_T * N_BLOCKS_Z)
    a, t = np.divmod(m, MODULE_T)
    row, col = np.divmod(w, BLOCK_T)
    return (a * BLOCK_Z + row) * NDET + (module * MODULE_T + t) * BLOCK_T + col


def measured(case, bed: int) -> np.ndarray:
    p = case.decoded / f"bed{bed}.singles.npy"
    if not p.exists():
        raise SystemExit(f"error: no {p}\n  run: d710 decode --case {case.name}")
    s = np.load(p).astype(np.float64)
    out = np.empty_like(s)
    out[stored_to_crystal()] = s
    return out


def model(G, eff, c_s: float, c_0: float) -> np.ndarray:
    return np.asarray(eff, np.float64) * (c_s * np.asarray(G, np.float64) + c_0)
