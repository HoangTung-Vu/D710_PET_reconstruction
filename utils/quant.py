"""Conversion from counts per voxel to Bq/mL and SUV."""

from __future__ import annotations

import os

import numpy as np

from .scanner import K_EXPORT, K_EXPORT_LM, WCC_UNIT_SCALE


def k_export(lm: bool = False):
    """Export's own `K`: the environment variable, then the `scanner.py` constant, then `None`."""
    var, const = (("D710_K_LM", K_EXPORT_LM) if lm else ("D710_K", K_EXPORT))
    raw = os.environ.get(var)
    if raw:
        return float(raw)
    return None if const is None else float(const)


def lowdose_k_scale(case) -> float:
    """`1/f` for a case built by `d710 lowdose` or `d710 simulate`, otherwise 1.

    A simulated case records `T_real / T_sim` in `simulation.json`; it is
    `null` when its beds were simulated for different lengths of time, which
    one K cannot serve.
    """
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
    """Dose that entered the patient: injected minus syringe residual, in Bq."""
    return (hdr["dose_mbq"] - hdr.get("residual_dose_mbq", 0.0)) * 1e6


def voxel_ml(vox) -> float:
    """`(z, y, x)` in mm to a volume in mL."""
    return float(vox[0] * vox[1] * vox[2]) / 1000.0


def scan_start_factor(case, beds) -> tuple[float, int, dict]:
    """`(exp(-lambda*dt), reference bed, its header)`: our time reference to GE's."""
    from osem.stitch import injection_epoch

    hdrs = {n: case.header(n) for n in beds}
    ref = min(beds, key=lambda n: hdrs[n]["bed_start_time"])
    hdr = hdrs[ref]
    lam = np.log(2) / hdr["half_life_s"]
    dt_s = hdr["bed_start_time"] - injection_epoch(hdr)
    return float(np.exp(-lam * dt_s)), int(ref), hdr


def wcc_activity_factor(case, bed: int, verbose: bool = True):
    """This scanner's own `hrActivityFactor`, or `None`."""
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
    """`hrActivityFactor` to `K`, via the assumed unit convention."""
    return None if factor is None else float(factor) * WCC_UNIT_SCALE


def k_from_dose(vol, vox, dose: float) -> float:
    """An upper bound on `K`, assuming the whole dose is inside the FOV."""
    total = float(np.asarray(vol).sum(dtype=np.float64))
    return dose / (total * voxel_ml(vox))


def body_mask(vol, frac: float = 0.02, pct: float = 99.9):
    """Body mask obtained by percentile threshold rather than an absolute value."""
    v = np.asarray(vol)
    return v > frac * np.percentile(v, pct)


def suv_bw(bqml, dose: float, weight_kg: float):
    """Body-weight SUV."""
    return np.asarray(bqml) / (dose / (weight_kg * 1000.0))


def bsa_m2(weight_kg: float, height_m: float) -> float:
    """Du Bois body-surface area: 0.007184 * W(kg)^0.425 * H(cm)^0.725."""
    return 0.007184 * weight_kg ** 0.425 * (height_m * 100) ** 0.725


def suv_bsa(bqml, dose: float, weight_kg: float, height_m: float):
    """Body-surface-area SUV."""
    return np.asarray(bqml) * (bsa_m2(weight_kg, height_m) * 1e4) / dose


def suv_table(bqml, mask, hdr, out=print) -> dict:
    """SUVbw, and SUVbsa where height is known: median, p90, p99 and maximum inside the body."""
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
    """Apply `K`, print the numbers it depends on, and return `{bqml, suv, ...}`."""
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
