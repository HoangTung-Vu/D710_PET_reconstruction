from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from utils.paths import case as get_case
from utils.scanner import DR_MM, GEOMETRY, NSEG0, PLANE_MM, POST_FILTER_Z_RATIO, XY

from .scanner2d import FOV_MM, SCANNER_GEOMETRY, TRAIN_GEOMETRY, Scanner2D, fov_mask

ENGINE = "DeepPET (Haggstrom et al. 2019), 2D encoder-decoder on SSRB sinograms"


def parser():
    ap = argparse.ArgumentParser(prog="deepPET recon")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", help="output root; defaults to $D710_OUT")
    ap.add_argument("--model", required=True, help="checkpoint (.pt) from deepPET.train")
    ap.add_argument("--beds", type=int, nargs="+")
    ap.add_argument("--scale", choices=("phys", "ge"), default="phys")
    ap.add_argument("--ge", help="GE SUV NIfTI; default <case>_ge_suvbw.nii.gz in export/")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--xy", type=int, default=XY)
    ap.add_argument("--ct")
    ap.add_argument("--device", default="auto")
    return ap


def file_sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def rigid_resample(img, n_out: int, v_out: float, src=TRAIN_GEOMETRY, dst=SCANNER_GEOMETRY):
    from scipy.ndimage import map_coordinates

    img = np.asarray(img, np.float32)
    n_in = img.shape[-1]
    v_in = FOV_MM / n_in
    d = np.deg2rad(dst["offset_deg"] - src["offset_deg"])
    c_old = np.asarray(src["centre_mm"], np.float64)
    c_new = np.asarray(dst["centre_mm"], np.float64)
    a = (np.arange(n_out) - (n_out - 1) / 2.0) * v_out
    qy, qx = np.meshgrid(a, a, indexing="ij")
    dx, dy = qx - c_new[0], qy - c_new[1]
    px = np.cos(d) * dx + np.sin(d) * dy + c_old[0]
    py = -np.sin(d) * dx + np.cos(d) * dy + c_old[1]
    rc = [py / v_in + (n_in - 1) / 2.0, px / v_in + (n_in - 1) / 2.0]
    one = img.ndim == 2
    stack = img[None] if one else img
    out = np.stack([map_coordinates(p, rc, order=1, mode="constant", cval=0.0) for p in stack])
    return out[0] if one else out.astype(np.float32)


def suv_per_bqml(hdr) -> float:
    from utils.quant import dose_bq

    return dose_bq(hdr) / (float(hdr["patient_weight_kg"]) * 1000.0)


def bed_decay(hdr) -> float:
    from osem.stitch import decay_factor, injection_epoch

    return float(decay_factor(hdr, injection_epoch(hdr)))


def s_phys(hdr, calib: dict) -> float:
    if "ssrb_counts_per_bqml_mm_s" not in calib:
        raise SystemExit("error: deepPET/calib.json has no ssrb_counts_per_bqml_mm_s; run "
                         "python -m deepPET.calibrate --ssrb-only")
    t = float(hdr["frame_duration_ms"]) / 1000.0
    return suv_per_bqml(hdr) * float(calib["ssrb_counts_per_bqml_mm_s"]) * t / bed_decay(hdr)


def counts_per_suv(case, hdr, K: float) -> float:
    from utils.quant import lowdose_k_scale

    return suv_per_bqml(hdr) / (bed_decay(hdr) * K * lowdose_k_scale(case))


def ge_path(case, explicit=None):
    if explicit:
        return Path(explicit)
    for d in (case.export, case.root.parent / "export"):
        for p in sorted(d.glob("*_ge_suvbw.nii.gz")):
            return p
    return None


def mean_corr(a, b, mask) -> float:
    from . import metrics as M

    body = [k for k in range(len(b)) if b[k][mask].sum() > 0]
    return float(np.mean([M.corr(a[k], b[k], mask) for k in body])) if body else float("nan")


def load_model(path, device: str):
    import torch

    from .model import DeepPET
    from .train import pick_device

    dev = pick_device(device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    grid = int(a.get("grid", 128))
    net = DeepPET(grid, a.get("bn_momentum", 0.2)).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net, dev, grid, ck


def infer(net, dev, x_in, batch: int) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(x_in), batch):
            t = torch.from_numpy(np.ascontiguousarray(x_in[i:i + batch], np.float32))[:, None]
            out.append(net(t.to(dev)).float().cpu().numpy()[:, 0])
    return np.concatenate(out)


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    from osem import stitch
    from utils import quant
    from utils import terms as sidecar

    from .real import ge_planes, ssrb
    from .simulate import CALIB

    C = get_case(args.case, args.out)
    need = ("normdt", "background", "attn")
    beds = args.beds or [n for n in C.decoded_beds()
                         if C.prompt(n).with_suffix(".s").exists()
                         and all((C.work_bed(n) / f"{t}.s").exists() for t in need)]
    if not beds:
        raise SystemExit(f"error: no bed of {C.root} has prompts, normdt, background and attn")
    K = quant.k_export(sino=True)
    if K is None:
        raise SystemExit("error: no sinogram K (quant.K_EXPORT_SINO / $D710_K_SINO)")

    net, dev, grid, ck = load_model(args.model, args.device)
    tag = f"{Path(args.model).name}:{file_sha(args.model)}"
    sc = Scanner2D(grid)
    mask = fov_mask(grid)
    gp = ge_path(C, args.ge)
    print(f"case {C.root}: beds {beds}, grid {grid}, scale {args.scale}, {tag}, epoch "
          f"{ck.get('epoch')}, on {dev}")

    img, sens, info = {}, {}, {}
    for n in beds:
        t0 = time.time()
        hdr = C.header(n)
        x, den, _, _ = ssrb(C, n)
        valid = den > 0
        ge = s_ge = None
        if gp is not None and gp.exists():
            ge = ge_planes(C, n, grid, gp)
            pge = np.stack([sc.fwd(g) for g in ge])
            s_ge = float(x[valid].sum(dtype=np.float64) / max(pge[valid].sum(dtype=np.float64), 1e-30))
        sp = s_phys(hdr, CALIB)
        if args.scale == "ge" and s_ge is None:
            raise SystemExit(f"error: --scale ge needs GE's image; none found for {C.name}")
        s = s_ge if args.scale == "ge" else sp
        pred = infer(net, dev, (x / s).astype(np.float32), args.batch) * mask
        fixed = rigid_resample(pred, grid, FOV_MM / grid) * mask
        r = {"s_phys": sp, "s_ge": s_ge if s_ge is not None else float("nan"), "s_used": s,
             "corr_ge_raw": float("nan"), "corr_ge_fixed": float("nan")}
        if ge is not None:
            r["corr_ge_raw"] = mean_corr(pred, ge, mask)
            r["corr_ge_fixed"] = mean_corr(fixed, ge, mask)
        cps = counts_per_suv(C, hdr, K)
        img[n] = (rigid_resample(pred, args.xy, DR_MM) * np.float32(cps)).astype(np.float32)
        w = den.sum(axis=(1, 2), dtype=np.float64)
        sens[n] = np.broadcast_to(w[:, None, None], img[n].shape)
        el = time.time() - t0
        info[n] = r
        np.savez_compressed(C.work_bed(n) / "deeppet.npz", img=img[n], key=tag, bed=n,
                            seconds=el, counts_per_suv=cps, axial_weight=w, **r)
        print(f"  bed {n}: s_phys {sp:.4f}  s_ge {r['s_ge']:.4f}  (phys/ge {sp / r['s_ge']:.3f})  "
              f"corr GE raw {r['corr_ge_raw']:.4f} fixed {r['corr_ge_fixed']:.4f}  {el:.0f} s",
              flush=True)

    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    stitch.overlap_report(C, beds, img, factors)
    try:
        ct = args.ct or sidecar.ct_dir(C, beds[0])
    except (SystemExit, OSError, KeyError):
        ct = ""
    per = {k: np.array([info[n][k] for n in beds]) for k in next(iter(info.values()))}
    np.savez_compressed(C.recon_deeppet, vol=vol, z0=z0,
                        vox=np.array([PLANE_MM, DR_MM, DR_MM]), beds=np.array(beds),
                        decay=np.array([factors[n] for n in beds]),
                        n_subsets=0, n_iterations=0, ct=ct, n_tof=1, tangential_lors=0,
                        psf_fwhm_mm=np.zeros(3), post_filter_fwhm_mm=0.0,
                        post_filter_z_ratio=POST_FILTER_Z_RATIO, engine=ENGINE,
                        geometry=GEOMETRY, model=tag, K=K, scale=args.scale,
                        train_geometry=json.dumps(TRAIN_GEOMETRY), n_plane=NSEG0, **per)
    print(f"\nwrote {C.recon_deeppet}  ({vol.shape})")
    print(f"next:  d710 export --out {C.root.parent} --case {C.name} --deeppet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
