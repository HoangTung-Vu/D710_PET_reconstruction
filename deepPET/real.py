"""Run a trained DeepPET on a real D710 bed.

    python -m deepPET.real --case fdg26081008 --bed 1 --name g128

The real sinogram is 3D (553 planes of span 2, non-TOF). Every bin is
precorrected with the pipeline's own terms, then single-slice rebinned into the
47 direct planes, as a ratio of sums:

    x[k] = sum_{planes -> k} (prompts - background)  /  sum_{planes -> k} (normdt * attn)

`background` is randoms + scatter (`work/bed<n>/`, from `vendor/`); normdt is a
sensitivity, span-2 multiplicity included, so it divides. Plane `q` of the 3D
sinogram lands on direct plane `r1 + r2` of its ring pairs
(`BinMap.ring_pairs_by_plane`). Bins with `normdt * attn < 1e-3` are left out.

`x` is in counts per unit of line integral; the network wants SUV.mm, so one
scale `s_real` is fitted per bed against the projection of GE's own SUV image
of the same bed (`export/<case>_ge_suvbw.nii.gz`). That makes GE's image the
answer key for the *scale* only -- it is stated in the output, and a real
calibration (K, frame time, decay, SUV factor) would replace it.

Tripwire: the correlation of `x`, smoothed by 2 bins, with P(GE image) is
printed for the image as is and flipped in x and in y. The unflipped one must be
the highest and above 0.95, or the 2D geometry and the image frame disagree.
Unsmoothed, the per-bin correlation is capped by the noise, not the geometry:
on fdg26081008 bed 6 (6e5 prompts per plane, 71 % background) it is 0.757 as
is, 0.727 x-flipped, 0.641 y-flipped; smoothed, 0.988 / 0.952 / 0.840.

Writes `real_<case>_bed<n>.npz` and `.png` into the run directory.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from utils.scanner import NSEG0, PLANE_MM, PSF_MM

from . import metrics as M
from . import nifti
from .scanner2d import TANG, N_TANG, N_VIEW, Scanner2D, fov_mask
from .simulate import FWHM

DEN_MIN = 1e-3

TRIP_SIGMA_BINS = 2.0

TRIP_MIN = 0.95


def direct_plane_of(binmap) -> np.ndarray:
    """For each of the 3D sinogram's planes, the direct plane `r1 + r2` it rebins to."""
    r1, r2, p = binmap.ring_pairs_by_plane()
    k = np.full(binmap.n_plane, -1, np.int64)
    for a, b, q in zip(r1, r2, p):
        if k[q] >= 0 and k[q] != a + b:
            raise SystemExit(f"error: plane {q} mixes ring pairs of r1+r2 {k[q]} and {a + b}")
        k[q] = a + b
    if (k < 0).any() or k.max() >= NSEG0:
        raise SystemExit("error: the ring pairing does not fill 0..46")
    return k


def ssrb(case, bed: int):
    """`(x, den, y, gamma)`, each `(47, 288, N_TANG)`, from the 3D sinogram and its terms."""
    from utils.binmap import BinMap

    hs = case.prompt(bed)
    bm = BinMap(hs)
    bm.hdr.require_plane_major()
    shape = bm.shape
    if shape[1:] != (N_VIEW, 381):
        raise SystemExit(f"error: {hs} is {shape}, expected (*, 288, 381)")
    k = direct_plane_of(bm)
    prom = np.memmap(hs.with_suffix(".s"), "<i2", "r", shape=shape)
    w = case.work_bed(bed)
    t = {n: np.memmap(w / f"{n}.s", "<f4", "r", shape=shape)
         for n in ("background", "normdt", "attn")}
    out = {n: np.zeros((NSEG0, N_VIEW, N_TANG), np.float64) for n in ("num", "den", "y", "g")}
    for q in range(shape[0]):
        d = t["normdt"][q][:, TANG].astype(np.float64) * t["attn"][q][:, TANG]
        ok = d >= DEN_MIN
        y = np.where(ok, prom[q][:, TANG], 0.0)
        g = np.where(ok, t["background"][q][:, TANG], 0.0)
        out["num"][k[q]] += y - g
        out["den"][k[q]] += np.where(ok, d, 0.0)
        out["y"][k[q]] += y
        out["g"][k[q]] += g
    x = np.where(out["den"] > 0, out["num"] / np.maximum(out["den"], 1e-30), 0.0)
    return (x.astype(np.float32), out["den"].astype(np.float32),
            out["y"].astype(np.float32), out["g"].astype(np.float32))


def ge_planes(case, bed: int, grid: int, path=None) -> np.ndarray:
    """GE's SUV image on the bed's 47 planes and the DeepPET grid, `(47, grid, grid)`."""
    p = path or case.export / f"{case.name}_ge_suvbw.nii.gz"
    if not p.exists():
        raise SystemExit(f"error: no {p}; pass --ge <SUV NIfTI of GE's reconstruction>")
    vol = nifti.load(p)
    tp = float(case.header(bed)["table_position_mm"])
    sl = nifti.at_z(vol, tp + np.arange(NSEG0) * PLANE_MM)
    g = nifti.to_grid(sl, vol.x0, vol.y0, vol.pixel_mm, grid, 0.0)
    return np.clip(g, 0.0, None) * fov_mask(grid)


def _osem_job(args):
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .osem2d import osem

    y, mult, gamma, grid = args
    return osem(y, mult, gamma, Scanner2D(grid))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--bed", type=int, required=True)
    ap.add_argument("--name", default=None, help="trained run; omit for the tripwire only")
    ap.add_argument("--ckpt", default="best", choices=("best", "last"))
    ap.add_argument("--grid", type=int, default=None, help="default: the run's grid, else 128")
    ap.add_argument("--ge", default=None)
    ap.add_argument("--out", default=None, help="runs root; default $D710_OUT/deeppet/runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-osem", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    from scipy.ndimage import gaussian_filter

    from utils.paths import case as get_case

    from .train import pick_device, run_dir

    C = get_case(a.case)
    ck = None
    if a.name:
        import torch

        rd = run_dir(a.name, a.out)
        ck = torch.load(rd / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    else:
        from utils.paths import out_root

        rd = out_root() / "deeppet" / "real"
    rd.mkdir(parents=True, exist_ok=True)
    grid = a.grid or (ck["args"]["grid"] if ck else 128)
    sc = Scanner2D(grid)
    mask = fov_mask(grid)

    t0 = time.time()
    x, den, y, gam = ssrb(C, a.bed)
    ge = ge_planes(C, a.bed, grid, a.ge)
    blur = lambda im: gaussian_filter(im, PSF_MM / FWHM / sc.voxel_mm, mode="constant")
    pge = np.stack([sc.fwd(blur(g)) for g in ge])
    valid = den > 0
    s_real = float(x[valid].sum(dtype=np.float64) / pge[valid].sum(dtype=np.float64))
    print(f"{a.case} bed {a.bed}: {y.sum():.4g} prompts rebinned into {NSEG0} planes "
          f"({y.sum() / NSEG0:.3g} per plane), background {gam.sum() / y.sum():.3f} of them; "
          f"{time.time() - t0:.0f} s")
    print(f"  count scale s_real = {s_real:.6g} counts per SUV.mm, fitted to GE's image")
    trip = {}
    xs = gaussian_filter(x, (0, TRIP_SIGMA_BINS, TRIP_SIGMA_BINS))
    for name, f in (("as is", lambda g: g), ("flip x", lambda g: g[:, ::-1]),
                    ("flip y", lambda g: g[::-1, :])):
        p = pge if name == "as is" else np.stack([sc.fwd(blur(f(g))) for g in ge])
        trip[name] = float(np.corrcoef(xs[valid], p[valid])[0, 1])
    per_bin = float(np.corrcoef(x[valid], pge[valid])[0, 1])
    print("  tripwire, corr(real sinogram smoothed, P(GE)): " +
          ", ".join(f"{k} {v:.4f}" for k, v in trip.items()) +
          f"  (per bin, unsmoothed: {per_bin:.4f}, noise-limited)")
    if trip["as is"] < TRIP_MIN or trip["as is"] < max(trip.values()):
        print(f"  WARNING: the unflipped image is not the best match above {TRIP_MIN} -- the "
              "2D geometry or the image frame is off; do not trust the images below")

    x_in = (x / s_real).astype(np.float32)
    res = {"x_in": x_in, "ge": ge.astype(np.float32), "s_real": s_real,
           "tripwire": json.dumps(trip)}
    if ck is not None:
        import torch

        from .model import DeepPET

        dev = pick_device(a.device)
        model = DeepPET(grid, ck["args"]["bn_momentum"]).to(dev).eval()
        model.load_state_dict(ck["model"])
        with torch.no_grad():
            pred = model(torch.from_numpy(x_in)[:, None].to(dev)).float().cpu().numpy()[:, 0]
        res["deeppet"] = pred * mask
    if not a.no_osem:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        jobs = [(y[k], den[k] * np.float32(s_real), gam[k], grid) for k in range(NSEG0)]
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
            res["osem"] = np.stack(list(ex.map(_osem_job, jobs)))

    body = [k for k in range(NSEG0) if ge[k].sum() > 0]
    for name in ("osem", "deeppet"):
        if name in res:
            im = res[name]
            print(f"  {name:8s} vs GE: corr {np.mean([M.corr(im[k], ge[k], mask) for k in body]):.4f}, "
                  f"rRMSE {np.mean([M.rrmse(im[k], ge[k], mask) for k in body]):.4f}, "
                  f"total {im.sum() / ge.sum():.4f} of GE's")
    stem = rd / f"real_{a.case}_bed{a.bed}"
    np.savez_compressed(f"{stem}.npz", **res)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("x_in", "sinogram (SSRB)"), ("ge", "GE"), ("osem", "OSEM 2D"), ("deeppet", "DeepPET")]
    cols = [c for c in cols if c[0] in res]
    planes = (12, 23, 35)
    fig, ax = plt.subplots(len(planes), len(cols), figsize=(3.6 * len(cols), 3.4 * len(planes)),
                           squeeze=False)
    for r, k in enumerate(planes):
        vmax = float(np.percentile(ge[k][mask], 99.9)) or 1.0
        for c, (key, title) in enumerate(cols):
            kw = {"aspect": "auto", "cmap": "gray_r"} if key == "x_in" else \
                {"cmap": "hot", "vmin": 0, "vmax": vmax}
            ax[r, c].imshow(res[key][k], **kw)
            ax[r, c].set_title(f"{title}, plane {k}", fontsize=9)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=90)
    print(f"  wrote {stem}.npz / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
