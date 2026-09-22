"""Images reconstructed from simulated data against real ones, on the SUV scale.

    python -m simulation.compare_images --case fdg26081008 --sim gate_s1 \
        --lowcount fdg26081008_lowcount_time

All images are resampled, by their affines, onto the grid of our full-dose
reconstruction. The comparison is restricted to the planes the simulation
covers, less `EDGE_PLANES` at each end of each simulated bed, where the
scanner's sensitivity falls to 7-30 % of the centre and every reconstruction
is noise.

  full   our full-dose list-mode reconstruction of the real exam (reference)
  ge     GE's clinical reconstruction -- the activity the simulation was fed
  low    the real exam thinned to the same time (`d710 lowdose --window time`)
  sim    the simulated exam, reconstructed by `d710 lm recon` with its own terms

Per image, inside the body (full-dose SUV > 0.3): median SUV and its ratio to
full dose, voxelwise Pearson r and NRMSE against full dose and against GE,
p99 SUV, and noise -- the standard deviation of the image minus a 2-voxel
Gaussian smoothing of itself, over its mean, in soft tissue (0.5 < GE SUV <
1.5). A simulation that behaves like the real low-count scan should match
`low` in noise and `full` in median SUV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from utils.paths import Case
from utils.paths import case as get_case

EDGE_PLANES = 3

BODY_SUV = 0.3

SOFT_TISSUE = (0.5, 1.5)


def resample_onto(src, ref) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    m = np.linalg.inv(src.affine) @ ref.affine
    g = np.indices(ref.shape).reshape(3, -1)
    idx = m[:3, :3] @ g + m[:3, 3:4]
    return map_coordinates(np.asarray(src.get_fdata(), np.float32), idx, order=1,
                           mode="constant", cval=0.0).reshape(ref.shape)


def sim_planes(sim_case: Case, ref_z0: float, n_ref: int, plane_mm: float) -> np.ndarray:
    """`(n_ref,)` bool: reference planes inside a simulated bed, away from its edges."""
    keep = np.zeros(n_ref, bool)
    for b in sim_case.decoded_beds():
        tp = float(sim_case.header(b)["table_position_mm"])
        i0 = int(round((tp - ref_z0) / plane_mm))
        keep[max(i0 + EDGE_PLANES, 0):min(i0 + 47 - EDGE_PLANES, n_ref)] = True
    return keep


def frame_s(case: Case, bed: int) -> float:
    """Frame length of `bed`: its sidecar, or the manifest a derived case keeps."""
    p = case.decoded / f"bed{bed}.json"
    if p.exists():
        return case.header(bed)["frame_duration_ms"] / 1000.0
    for name in ("lowdose.json", "simulation.json"):
        m = case.root / name
        if m.exists():
            for r in json.loads(m.read_text()).get("beds", []):
                if int(r["bed"]) == bed and r.get("frame_duration_ms"):
                    return float(r["frame_duration_ms"]) / 1000.0
    return float("nan")


def noise(img, soft) -> float:
    from scipy.ndimage import gaussian_filter

    hf = img - gaussian_filter(img, 2.0)
    return float(np.std(hf[soft]) / max(np.mean(img[soft]), 1e-9))


def metrics(x, full, ge, body, soft) -> dict:
    a, f, g = x[body], full[body], ge[body]
    return {"median_suv": float(np.median(a)),
            "median_ratio_to_full": float(np.median(a) / max(np.median(f), 1e-9)),
            "mean_ratio_to_full": float(a.mean() / max(f.mean(), 1e-9)),
            "r_vs_full": float(np.corrcoef(a, f)[0, 1]),
            "r_vs_ge": float(np.corrcoef(a, g)[0, 1]),
            "nrmse_vs_full": float(np.sqrt(np.mean((a - f) ** 2)) / max(f.mean(), 1e-9)),
            "p99_suv": float(np.percentile(a, 99)),
            "noise": noise(x, soft)}


def figure(images: dict, keep, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(images)
    zs = np.nonzero(keep)[0]
    ref = images["full"]
    yc = int(np.argmax(ref[:, :, zs].sum(axis=(0, 2))))
    zc = int(zs[np.argmax(ref[:, :, zs].sum(axis=(0, 1)))])
    fig, ax = plt.subplots(2, len(names), figsize=(4 * len(names), 8))
    vmax = float(np.percentile(ref[:, :, zs][ref[:, :, zs] > BODY_SUV], 99.5))
    for j, n in enumerate(names):
        cor = images[n][:, yc, zs.min():zs.max() + 1].T
        ax[0, j].imshow(cor, origin="lower", cmap="gray_r", vmin=0, vmax=vmax,
                        aspect=3.27 / 2.1306)
        ax[0, j].set_title(f"{n}: coronal y={yc}")
        ax[1, j].imshow(images[n][:, :, zc].T, cmap="gray_r", vmin=0, vmax=vmax)
        ax[1, j].set_title(f"{n}: axial plane {zc}")
        for a in ax[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"{title}   (SUV 0..{vmax:.1f}, same scale in every panel)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    import nibabel as nib

    ap = argparse.ArgumentParser(prog="simulation.compare_images", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--sim", default="gate_s1", help="label: <case>_sim_<label>")
    ap.add_argument("--lowcount", required=True, help="the real case thinned to the same time")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    C = get_case(a.case, a.out)
    S = Case(f"{C.name}_sim_{a.sim}", C.root.parent)
    L = Case(a.lowcount, C.root.parent)
    paths = {"ge": C.export / f"{C.name}_ge_suvbw.nii.gz",
             "full": C.export / f"{C.name}_lm_suvbw.nii.gz",
             "low": L.export / f"{L.name}_lm_suvbw.nii.gz",
             "sim": S.export / f"{S.name}_lm_suvbw.nii.gz"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("error: missing\n  " + "\n  ".join(missing))

    ref = nib.load(str(paths["full"]))
    imgs = {k: (np.asarray(ref.get_fdata(), np.float32) if k == "full"
                else resample_onto(nib.load(str(p)), ref)) for k, p in paths.items()}
    plane_mm = float(abs(ref.affine[2, 2]))
    keep = sim_planes(S, float(ref.affine[2, 3]), ref.shape[2], plane_mm)
    if not keep.any():
        raise SystemExit(f"error: {S.name} covers no plane of the reference")
    body = (imgs["full"] > BODY_SUV) & keep[None, None, :]
    soft = body & (imgs["ge"] > SOFT_TISSUE[0]) & (imgs["ge"] < SOFT_TISSUE[1])
    rows = {k: metrics(v, imgs["full"], imgs["ge"], body, soft) for k, v in imgs.items()}

    b0 = S.decoded_beds()[0]
    frames = {"full": frame_s(C, b0), "low": frame_s(L, b0), "sim": frame_s(S, b0)}
    out_dir = C.root.parent / f"{C.name}_sim" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"images_{a.sim}_vs_{L.name}"
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"case": C.name, "sim": S.name, "lowcount": L.name,
         "beds": S.decoded_beds(), "planes": int(keep.sum()),
         "body_voxels": int(body.sum()), "frame_s": frames, "images": rows},
        indent=2))
    figure(imgs, keep, out_dir / f"{stem}.png",
           f"{C.name} beds {S.decoded_beds()}: GE / full dose / real low-count / simulated")

    print(f"beds {S.decoded_beds()}, {int(keep.sum())} planes, {int(body.sum()):,} body voxels")
    print(f"{'image':<6}{'frame s':>8}{'med SUV':>9}{'/full':>7}{'r full':>8}"
          f"{'r GE':>7}{'NRMSE':>7}{'p99':>7}{'noise':>7}")
    for k, m in rows.items():
        print(f"{k:<6}{frames.get(k, float('nan')):>8.1f}{m['median_suv']:>9.3f}"
              f"{m['median_ratio_to_full']:>7.3f}{m['r_vs_full']:>8.3f}{m['r_vs_ge']:>7.3f}"
              f"{m['nrmse_vs_full']:>7.3f}{m['p99_suv']:>7.2f}{m['noise']:>7.3f}")
    print(f"-> {out_dir / stem}.json / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
