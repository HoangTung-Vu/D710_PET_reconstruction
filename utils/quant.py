from __future__ import annotations

import os

import numpy as np

from .scanner import K_EXPORT, K_EXPORT_LM, K_EXPORT_SINO, WCC_UNIT_SCALE


def k_source(lm: bool = False, sino: bool = False) -> tuple[str, str]:
    if lm and sino:
        raise ValueError("a reconstruction is list-mode or sinogram, not both")
    if lm:
        return "D710_K_LM", "K_EXPORT_LM"
    if sino:
        return "D710_K_SINO", "K_EXPORT_SINO"
    return "D710_K", "K_EXPORT"


def k_export(lm: bool = False, sino: bool = False):
    var, name = k_source(lm, sino)
    const = {"K_EXPORT": K_EXPORT, "K_EXPORT_LM": K_EXPORT_LM,
             "K_EXPORT_SINO": K_EXPORT_SINO}[name]
    raw = os.environ.get(var)
    if raw:
        return float(raw)
    return None if const is None else float(const)


def lowdose_k_scale(case) -> float:
    import json

    for name in ("lowdose.json", "simulation.json"):
        p = case.root / name
        if not p.exists():
            continue
        with open(p) as f:
            k = json.load(f).get("k_scale")
        if k is None:
            raise SystemExit(f"error: {p} has no single k_scale -- its beds were "
                             "simulated for different lengths of time")
        return float(k)
    return 1.0


def dose_bq(hdr) -> float:
    return (hdr["dose_mbq"] - hdr.get("residual_dose_mbq", 0.0)) * 1e6


def voxel_ml(vox) -> float:
    return float(vox[0] * vox[1] * vox[2]) / 1000.0


def scan_start_factor(case, beds) -> tuple[float, int, dict]:
    from osem.stitch import injection_epoch

    hdrs = {n: case.header(n) for n in beds}
    ref = min(beds, key=lambda n: hdrs[n]["bed_start_time"])
    hdr = hdrs[ref]
    lam = np.log(2) / hdr["half_life_s"]
    dt_s = hdr["bed_start_time"] - injection_epoch(hdr)
    return float(np.exp(-lam * dt_s)), int(ref), hdr


def wcc_activity_factor(case, bed: int, verbose: bool = True):
    from . import container, terms

    try:
        est = terms.meta(case, bed).get("estimate", {})
    except (OSError, ValueError):
        est = {}

    if est.get("wcc_activity_factor"):
        if verbose:
            print("WCC declared by the exam: %s" % est.get("wcc_name", "?"))
        return float(est["wcc_activity_factor"])

    uid = (est.get("rdf_header") or {}).get("wcc_cal_uid")
    if not uid:
        if verbose:
            print("no wcc_cal_uid in the sidecar of bed %d" % bed)
        return None

    got = container.cal_tags(uid, "3dwcc",
                            [("name", 0x00191006), ("factor", 0x0019100B)])
    if not got or not got.get("factor"):
        if verbose:
            print("could not read %s.3dwcc inside the container" % uid)
        return None
    if verbose:
        print("WCC declared by the exam: %s" % got.get("name"))
    return float(got["factor"])


def k_from_wcc(factor):
    return None if factor is None else float(factor) * WCC_UNIT_SCALE


def k_from_dose(vol, vox, dose: float) -> float:
    total = float(np.asarray(vol).sum(dtype=np.float64))
    return dose / (total * voxel_ml(vox))


def body_mask(vol, frac: float = 0.02, pct: float = 99.9):
    v = np.asarray(vol)
    return v > frac * np.percentile(v, pct)


def suv_bw(bqml, dose: float, weight_kg: float):
    return np.asarray(bqml) / (dose / (weight_kg * 1000.0))


def bsa_m2(weight_kg: float, height_m: float) -> float:
    return 0.007184 * weight_kg ** 0.425 * (height_m * 100) ** 0.725


def suv_bsa(bqml, dose: float, weight_kg: float, height_m: float):
    return np.asarray(bqml) * (bsa_m2(weight_kg, height_m) * 1e4) / dose


def suv_table(bqml, mask, hdr, out=print) -> dict:
    dose = dose_bq(hdr)
    w = hdr["patient_weight_kg"]
    h = hdr.get("patient_height_m") or 0.0

    got = {"SUVbw": suv_bw(bqml, dose, w)}
    if h > 0:
        got["SUVbsa"] = suv_bsa(bqml, dose, w, h)

    out(f"{w} kg   {h} m   actual dose {dose / 1e6:.1f} MBq")
    out(f"SUVbw denominator = {dose / (w * 1000):,.1f} Bq/mL"
        + (f"   BSA = {bsa_m2(w, h):.3f} m²" if h > 0 else ""))
    out("")
    for name, s in got.items():
        v = s[mask]
        out(f"{name:7s} median {np.median(v):6.3f}   "
            f"p90 {np.percentile(v, 90):6.2f}   p99 {np.percentile(v, 99):6.2f}   "
            f"max {s.max():8.1f}")
    out("\n[muscle/fat ~0.5-1 | liver ~1.5-2.5 | paediatric brain high | bladder >20]")
    return got


def report(vol, K: float, hdr, vox, dose: float | None = None, out=print) -> dict:
    dose = dose_bq(hdr) if dose is None else float(dose)
    vml = voxel_ml(vox)
    bqml = np.asarray(vol) * K
    suv = suv_bw(bqml, dose, hdr["patient_weight_kg"])
    mask = body_mask(vol)
    in_fov = float(bqml.sum(dtype=np.float64)) * vml

    out(f">>> K in use = {K:,.2f} (Bq/mL)/(count/voxel)")
    out(f"total inside the FOV {in_fov / 1e6:6.1f} MBq = {100 * in_fov / dose:.1f} %"
        f" of the {dose / 1e6:.1f} MBq present at the reference time"
        "     [<100 % is correct: the scan does not cover the whole body]")
    out(f"Bq/mL  max {bqml.max():>12,.0f}   body median {np.median(bqml[mask]):>10,.0f}")
    out(f"SUVbw  max {suv.max():>12.1f}   body median {np.median(suv[mask]):>10.3f}"
        "     [soft tissue ~0.5-1; brain/bladder much higher]")

    return {"bqml": bqml, "suv": suv, "mask": mask, "K": float(K),
            "dose_bq": dose, "voxel_ml": vml, "mbq_in_fov": in_fov / 1e6}
