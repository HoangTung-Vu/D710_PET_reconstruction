from __future__ import annotations

import numpy as np

from lm.recon import psf_fwhm
from utils.scanner import NSEG0, PSF_FWHM_MM, XY

NAME = "sens_lm.npz"

STAMPED = ("normdt.s", "attn.s")

SENTINEL = 1e7


def _psf(psf) -> list[float]:
    return psf_fwhm(psf) or [0.0, 0.0, 0.0]


def path(case, bed: int, xy: int = XY, psf=PSF_FWHM_MM,
         n_plane: int = NSEG0):
    f = _psf(psf)
    if (int(xy), f, int(n_plane)) == (XY, _psf(PSF_FWHM_MM), NSEG0):
        return case.work_bed(bed) / NAME
    return case.work_bed(bed) / (
        f"sens_lm_xy{int(xy)}_psf{f[0]:g}-{f[1]:g}-{f[2]:g}_z{int(n_plane)}.npz")


def stamp(case, bed: int) -> str:
    out = []
    for n in STAMPED:
        p = case.work_bed(bed) / n
        if not p.exists():
            how = ("d710 attn --case " + case.name if n.startswith("attn")
                   else f"d710 tostir --case {case.name} --bed {bed}")
            raise SystemExit(f"error: no {p}\n  run: {how}")
        st = p.stat()
        out.append(f"{n}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(out)


def load(case, bed: int, xy: int = XY, psf=PSF_FWHM_MM,
         n_plane: int = NSEG0):
    p = path(case, bed, xy, psf, n_plane)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "norm_BP" not in z.files:
        return None
    if (str(z["stamp"]) != stamp(case, bed) or int(z["xy"]) != int(xy)
            or tuple(np.atleast_1d(z["psf"]).tolist()) != tuple(_psf(psf))
            or int(z["n_plane"]) != int(n_plane)):
        return None
    return np.ascontiguousarray(z["norm_BP"], np.float32)


def save(case, bed: int, norm_bp, xy: int = XY, psf=PSF_FWHM_MM,
         n_plane: int = NSEG0):
    p = path(case, bed, xy, psf, n_plane)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, norm_BP=np.ascontiguousarray(norm_bp, np.float32),
                        stamp=stamp(case, bed), xy=int(xy),
                        psf=np.array(_psf(psf), np.float64),
                        n_plane=int(n_plane), bed=int(bed))
    return p


def build(case, bed: int, binmap, xy: int = XY, psf=PSF_FWHM_MM,
          n_plane: int = NSEG0, n_splits: int = 8):
    from lm import recon, terms

    sens_ids, sens_w = terms.sensitivity(case, bed, binmap)
    sm = recon.build_sm(np.zeros((1, 2), np.int32), 1, xy=xy, n_plane=n_plane,
                        psf=psf, n_splits=n_splits, sens_ids=sens_ids,
                        sens_w=sens_w)
    return np.ascontiguousarray(sm.norm_BP.cpu().numpy(), np.float32)


def get(case, bed: int, binmap=None, xy: int = XY, psf=PSF_FWHM_MM,
        n_plane: int = NSEG0, n_splits: int = 8, rebuild: bool = False):
    a = None if rebuild else load(case, bed, xy, psf, n_plane)
    if a is not None:
        return a
    if binmap is None:
        from lm import geom

        binmap = geom.BinMap(case.prompt(bed))
    a = build(case, bed, binmap, xy, psf, n_plane, n_splits)
    save(case, bed, a, xy, psf, n_plane)
    return a


def image(norm_bp, n_tang: int | None = None):
    a = np.asarray(norm_bp, np.float32)
    a = np.where(a >= SENTINEL - 1, np.float32(0.0), a)
    if n_tang:
        from utils import scanner

        a = a * scanner.fov_mask(a.shape[0], n_tang)[:, :, None]
    return np.ascontiguousarray(a, np.float32)


def kappa(n_events: int, s) -> float:
    total = float(np.asarray(s, np.float64).sum())
    if total <= 0:
        raise SystemExit("error: the sensitivity image sums to zero")
    return float(n_events) / total
