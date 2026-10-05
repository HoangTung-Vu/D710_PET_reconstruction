from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytomography
import torch

from lm.recon import psf_fwhm
from sino import terms
from sino.projector import MAX_RAYS, SinogramSystemMatrix
from utils import scanner
from utils.scanner import GEOMETRY, NSEG0, PSF_FWHM_MM, XY

DELTA = pytomography.delta


class Bed:
    def __init__(self, binmap, y, sens, add, n_sub: int, psf=PSF_FWHM_MM,
                 xy: int = XY, meta=None, mask=None, s=None, device=None,
                 max_rays: int | None = None):
        dev = torch.device(device or pytomography.device)
        self.device = dev
        self.sm = SinogramSystemMatrix(binmap, xy=xy, psf=psf, meta=meta,
                                       max_rays=max_rays or MAX_RAYS, device=dev)
        self.views = self.sm.set_n_subsets(n_sub)
        self.n_sub = len(self.views)
        self.shape = self.sm.shape
        self.y = torch.as_tensor(y, dtype=torch.float32).to(dev)
        self.add = torch.as_tensor(add, dtype=torch.float32).to(dev)
        if mask is None:
            mask = scanner.fov_mask(self.shape[0], binmap.n_tang)
        mask = np.asarray(mask, np.float32)
        if mask.ndim == 2:
            mask = np.repeat(mask[:, :, None], self.shape[2], axis=2)
        self.mask = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
        if s is None:
            S = torch.as_tensor(sens, dtype=torch.float32).to(dev)
            s = torch.stack([self.sm.backward(S[:, v].contiguous(), m)
                             for m, v in enumerate(self.views)])
            del S
        self.s = torch.as_tensor(s, dtype=torch.float32).to(dev)

    def initial(self):
        return self.mask.clone()

    def sens_full(self):
        return self.s.sum(0) * self.mask

    @torch.no_grad()
    def em(self, x, m: int):
        v = self.views[m]
        s = self.s[m]
        ratio = self.y[:, v] / (self.sm.forward(x, m) + self.add[:, v] + DELTA)
        g = self.sm.backward(ratio, m) - s
        x = x + x * (1 / (s + DELTA)) * g
        x[x <= 0] = 0
        return x

    @classmethod
    def from_arrays(cls, a: dict, n_sub: int, psf=PSF_FWHM_MM, device=None,
                    max_rays: int | None = None, xy: int = XY):
        return cls(a["binmap"], a["y"], a["sens"], a["add"], n_sub, psf=psf,
                   xy=xy, s=a.get("s"), device=device, max_rays=max_rays)


def sens_key(C, n: int, n_sub: int, psf, xy: int = XY) -> str:
    parts = []
    for name in ("normdt", "attn"):
        for p in (C.work_bed(n) / f"{name}.hs", C.work_bed(n) / f"{name}.s"):
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    parts += [f"n_sub={n_sub}", f"psf={psf_fwhm(psf)!r}", f"xy={xy}",
              f"geometry={GEOMETRY}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def sens_path(cache, C, n: int, n_sub: int, psf, xy: int = XY) -> Path:
    tag = f"{C.root.parent.name}__{C.name}"
    key = sens_key(C, n, n_sub, psf, xy)
    return Path(cache) / tag / f"bed{n}_s{n_sub}_{key}.npy"


def label(C, n: int) -> np.ndarray:
    xt = np.load(C.raw_sim / f"bed{n}_x_true.npy")
    return np.ascontiguousarray(xt.transpose(2, 1, 0), dtype=np.float32)


def ready(C, n: int, need_label: bool = False) -> bool:
    for hs in terms.paths(C, n):
        if not (hs.exists() and hs.with_suffix(".s").exists()):
            return False
    return not need_label or (C.raw_sim / f"bed{n}_x_true.npy").exists()


def load_arrays(C, n: int, n_sub: int, psf, cache=None,
                with_label: bool = False, xy: int = XY) -> dict:
    binmap, y, sens, add, dead = terms.load(C, n)
    a = {"binmap": binmap, "y": y, "sens": sens, "add": add, "dead": dead,
         "case": C, "bed": n}
    if cache:
        p = sens_path(cache, C, n, n_sub, psf, xy)
        a["s_path"] = p
        if p.exists():
            a["s"] = np.load(p)
    if with_label:
        a["label"] = label(C, n)
        p = C.work_bed(n) / "sino.npz"
        if p.exists():
            a["osem"] = np.ascontiguousarray(np.load(p)["img"].transpose(2, 1, 0))
    return a


def build(a: dict, n_sub: int, psf, device=None, max_rays=None,
          xy: int = XY) -> Bed:
    bed = Bed.from_arrays(a, n_sub, psf=psf, device=device, max_rays=max_rays,
                          xy=xy)
    p = a.get("s_path")
    if p is not None and a.get("s") is None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp.npy")
        np.save(tmp, bed.s.cpu().numpy())
        tmp.replace(p)
    return bed
