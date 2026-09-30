from __future__ import annotations

import time

import numpy as np

from utils.scanner import NDET, NRINGS

E_LOW_KEV = 425.0
E_RES = 0.12
IMAGE_STEP = 4
MU_CUTOFF = 0.004
RING_STEP = 4
CRYSTAL_STEP = 4
BATCH = 8
BATCH_GPU_MAX = 256
PAIR_BYTES = 96
PROGRESS_POINTS = 1600


def sample_rings(step: int = RING_STEP) -> np.ndarray:
    return np.array(sorted(set(range(0, NRINGS, step)) | {NRINGS - 1}), np.int64)


def sample_crystals(step: int = CRYSTAL_STEP) -> np.ndarray:
    if NDET % step:
        raise ValueError(f"crystal step {step} does not divide {NDET}")
    return np.arange(0, NDET, step, dtype=np.int64)


def sample_detectors(rings, crystals) -> np.ndarray:
    return (np.asarray(rings)[:, None] * NDET + np.asarray(crystals)[None, :]).ravel()


def scatter_points(mu_img, origin, voxel, step: int = IMAGE_STEP,
                   cutoff: float = MU_CUTOFF, rng=None):
    idx = np.argwhere(mu_img[::step, ::step, ::step] > cutoff) * step
    mu_at = mu_img[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
    jit = 0.0 if rng is None else rng.random(idx.shape) - 0.5
    pos = (np.asarray(origin)[None, :] + (idx + jit) * np.asarray(voxel)[None, :])
    weight = float(step ** 3 * np.prod(voxel))
    return pos.astype(np.float32), mu_at, weight


def physics():
    from pytomography.utils.sss import (detector_efficiency,
                                        diff_compton_cross_section,
                                        photon_energy_after_compton_scatter_511kev,
                                        total_compton_cross_section)

    return (total_compton_cross_section, diff_compton_cross_section,
            photon_energy_after_compton_scatter_511kev, detector_efficiency)


def pair_kernel(pos, mu_at, E, T, rdet, I, J, e_low: float = E_LOW_KEV,
                e_res: float = E_RES, open_=None):
    import torch

    dev = rdet.device
    total_cs, diff_cs, e_after, det_eff = physics()
    e511 = torch.tensor(511.0, device=dev)
    s511 = total_cs(e511)
    S = torch.as_tensor(pos, device=dev)
    rS = rdet[None, :, :] - S[:, None, :]
    rn = torch.linalg.norm(rS, dim=2)
    u = rS / rn[..., None]
    rxy = rdet[:, :2] / torch.linalg.norm(rdet[:, :2], dim=1, keepdim=True)
    cinc = (u[..., :2] * rxy[None]).sum(-1)
    cos_th = -(u[:, I] * u[:, J]).sum(-1)
    e_new = e_after(cos_th)
    eff = det_eff(e_new, e_res, e_low)
    k = total_cs(e_new) / s511 - 1.0
    E, T = torch.as_tensor(E, device=dev), torch.as_tensor(T, device=dev)
    emis = (E[:, I] * torch.exp(-T[:, J] * k) + E[:, J] * torch.exp(-T[:, I] * k))
    val = (emis * torch.exp(-(T[:, I] + T[:, J]))
           / (rn[:, I] ** 2 * rn[:, J] ** 2)
           * eff * cinc[:, I] * cinc[:, J] * diff_cs(cos_th, e511) / s511
           * torch.as_tensor(mu_at, device=dev)[:, None])
    if open_ is not None:
        o = torch.as_tensor(open_, dtype=val.dtype, device=dev)
        val = val * o[:, I] * o[:, J]
    return val.sum(0, dtype=torch.float64)


def simulate_sparse(x, mu, lut, rings=None, crystals=None, image_step: int = IMAGE_STEP,
                    cutoff: float = MU_CUTOFF, e_low: float = E_LOW_KEV,
                    e_res: float = E_RES, seed: int = 0, batch: int | None = None,
                    threads: int | None = None, shield=None, device=None, out=print):
    import parallelproj
    import torch

    from .gpu import batch_for
    from .gpu import device as pick_device

    dev = pick_device(device)
    if threads:
        torch.set_num_threads(threads)
    rings = sample_rings() if rings is None else np.asarray(rings)
    crystals = sample_crystals() if crystals is None else np.asarray(crystals)
    det = sample_detectors(rings, crystals)
    D = det.size
    I, J = np.triu_indices(D, k=1)
    I, J = torch.from_numpy(I).to(dev), torch.from_numpy(J).to(dev)
    rdet = torch.from_numpy(np.ascontiguousarray(lut[det], np.float32)).to(dev)
    pos, mu_at, weight = scatter_points(mu[0], mu[1], mu[2], image_step, cutoff,
                                        None if seed is None else np.random.default_rng(seed))
    batch = int(batch or batch_for(dev, PAIR_BYTES * I.numel(), BATCH, BATCH_GPU_MAX))
    every = max(1, PROGRESS_POINTS // batch)
    acc = torch.zeros(I.numel(), dtype=torch.float64, device=dev)
    xe_one = np.ascontiguousarray(lut[det], np.float32)
    t0 = time.time()
    for s in range(0, len(pos), batch):
        p = pos[s:s + batch]
        B = len(p)
        xs = np.repeat(p, D, axis=0)
        xe = np.tile(xe_one, (B, 1))
        E = parallelproj.joseph3d_fwd(xs, xe, *x).reshape(B, D)
        T = parallelproj.joseph3d_fwd(xs, xe, *mu).reshape(B, D)
        open_ = None
        if shield is not None:
            from .singles import aperture

            open_ = aperture(p.astype(np.float64), xe_one.astype(np.float64), *shield)
        acc += pair_kernel(p, mu_at[s:s + batch], E, T, rdet, I, J, e_low, e_res, open_)
        if (s // batch) % every == 0:
            out(f"    sss: {s + B}/{len(pos)} scatter points  {time.time() - t0:.0f} s")
    R, C = len(rings), len(crystals)
    f = np.zeros((D, D), np.float64)
    v = acc.cpu().numpy() * weight
    i, j = I.cpu().numpy(), J.cpu().numpy()
    f[i, j] = v
    f[j, i] = v
    info = {"scatter_points": int(len(pos)), "image_step": int(image_step),
            "cutoff": float(cutoff), "rings": rings.tolist(),
            "crystal_step": int(crystals[1] - crystals[0]) if C > 1 else NDET,
            "e_low_kev": float(e_low), "e_res": float(e_res),
            "shield": None if shield is None else [float(v) for v in shield],
            "seconds": round(time.time() - t0, 1), "device": str(dev), "batch": batch}
    return f.reshape(R, C, R, C).astype(np.float32), info


def ring_weights(r, rings):
    rings = np.asarray(rings)
    i0 = np.clip(np.searchsorted(rings, r, side="right") - 1, 0, len(rings) - 2)
    w = (r - rings[i0]) / (rings[i0 + 1] - rings[i0])
    return int(i0), float(w)


def crystal_weights(t, step: int, n_nodes: int):
    t = np.asarray(t, np.int64)
    i0 = t // step
    w = (t % step) / float(step)
    return i0 % n_nodes, (i0 + 1) % n_nodes, w.astype(np.float32)


def interpolate_ring_pair(f, rings, crystal_step: int, r1: int, r2: int, t1, t2):
    C = f.shape[1]
    ia, wa = ring_weights(r1, rings)
    ib, wb = ring_weights(r2, rings)
    a0, a1, u = crystal_weights(t1, crystal_step, C)
    b0, b1, v = crystal_weights(t2, crystal_step, C)
    out = np.zeros(len(a0), np.float32)
    for ra, wra in ((ia, 1.0 - wa), (ia + 1, wa)):
        for rb, wrb in ((ib, 1.0 - wb), (ib + 1, wb)):
            w = wra * wrb
            if w == 0.0:
                continue
            g = f[ra, :, rb, :]
            out += np.float32(w) * ((1 - u) * (1 - v) * g[a0, b0] + (1 - u) * v * g[a0, b1]
                                    + u * (1 - v) * g[a1, b0] + u * v * g[a1, b1])
    return out


def to_bins(f, info, pairs) -> np.ndarray:
    b = pairs.binmap
    acc = np.zeros(b.n_bin, np.float64)
    nd = b.ndet
    for p, a, c, bins in pairs:
        r1, t1 = np.divmod(a, nd)
        r2, t2 = np.divmod(c, nd)
        acc[bins] += interpolate_ring_pair(f, info["rings"], info["crystal_step"],
                                           int(r1[0]), int(r2[0]), t1, t2)
    return acc.astype(np.float32)
