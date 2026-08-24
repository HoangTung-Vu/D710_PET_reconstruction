"""count/voxel -> Bq/mL -> SUV. All the quantification, and the constant `K`.

The image coming out of `osem/` is **count/voxel decay-corrected back to the
injection time**. Converting to Bq/mL takes exactly one scalar:

    Bq/mL = K · x_(count/voxel)

⚠ **`K` is only valid for THE EXACT correction chain that measured it and for
ONE voxel size.** The projector accumulates along the voxel step rather than by
volume, so a constant measured at 2.1306 mm and reused at 1.3672 mm reads 1.56×
high.

Two reference points, neither of which is the final answer:

* **The exam's own WCC.** The exam names its WCC file in the header
  (`wcc_cal_uid`) and that file really exists in
  `/usr/PET/systemConfig/cal/<uid>.3dwcc`, tag `(0019,100B)` =
  `hrActivityFactor`. The `1e4` multiplier is a **GUESS** about GE's unit
  convention, not yet derived.
* **A dose-based upper bound.** The image is decay-corrected to injection time,
  so Σ(activity) ≤ the injected dose; forcing equality gives an upper bound,
  since in reality some of the dose lies outside the FOV.

Both are linear in SUV, so every SUV number below is wrong by exactly the factor
`K` is wrong by.
"""

from __future__ import annotations

import numpy as np

#: ASSUMED unit convention between `hrActivityFactor` and (Bq/mL)/(count/voxel).
#: Not yet derived; it puts 68 % of the dose inside the FOV in the paediatric
#: case, consistent with a 77 cm scan of a 118 cm child (legs not covered).
#: Only a NEMA measurement will settle it.
WCC_UNIT_SCALE = 1e4


def dose_bq(hdr) -> float:
    """Dose that actually entered the patient: injected minus syringe residual, Bq."""
    return (hdr["dose_mbq"] - hdr.get("residual_dose_mbq", 0.0)) * 1e6


def voxel_ml(vox) -> float:
    """`(z, y, x)` mm -> mL."""
    return float(vox[0] * vox[1] * vox[2]) / 1000.0


def wcc_activity_factor(case, bed: int, verbose: bool = True):
    """This scanner's own `hrActivityFactor`, or `None`.

    Read from the `estimate.json` sidecar first — `d710 estimate` writes it while
    it still has the container at hand. Only when the sidecar lacks it (a bed
    built by an older version) is the container queried again.
    """
    from . import container, terms

    try:
        est = terms.meta(case, bed).get("estimate", {})
    except (OSError, ValueError):
        est = {}

    if est.get("wcc_activity_factor"):
        if verbose:
            print("WCC exam khai báo: %s" % est.get("wcc_name", "?"))
        return float(est["wcc_activity_factor"])

    uid = (est.get("rdf_header") or {}).get("wcc_cal_uid")
    if not uid:
        if verbose:
            print("không có wcc_cal_uid trong sidecar của bed %d" % bed)
        return None

    got = container.cal_tags(uid, "3dwcc",
                            [("name", 0x00191006), ("factor", 0x0019100B)])
    if not got or not got.get("factor"):
        if verbose:
            print("không đọc được %s.3dwcc trong container" % uid)
        return None
    if verbose:
        print("WCC exam khai báo: %s" % got.get("name"))
    return float(got["factor"])


def k_from_wcc(factor):
    """`hrActivityFactor` -> `K`, via the ASSUMED unit convention.

    `None` in gives `None` out rather than a `TypeError`: callers usually write
    `k_from_wcc(wcc_activity_factor(...))` and the inner call **may** fail to
    find anything (a case without a current sidecar, or a header that does not
    name a `wcc_cal_uid`). The fallback is then the dose-based reference point,
    and that is the caller's decision.
    """
    return None if factor is None else float(factor) * WCC_UNIT_SCALE


def k_from_dose(vol, vox, dose: float) -> float:
    """An **upper bound** on `K`: assumes 100 % of the dose is inside the FOV.

    In reality it is < 100 % (the scan does not cover the whole patient), so the
    true `K` is smaller than this.
    """
    total = float(np.asarray(vol).sum(dtype=np.float64))
    return dose / (total * voxel_ml(vox))


def body_mask(vol, frac: float = 0.02, pct: float = 99.9):
    """Rough body mask: threshold by percentile, not by absolute value.

    An absolute threshold is meaningless here because the scale is uncalibrated
    (and shifts with `K`).
    """
    v = np.asarray(vol)
    return v > frac * np.percentile(v, pct)


def suv_bw(bqml, dose: float, weight_kg: float):
    """Body-weight SUV. Assumes tissue at 1 g/mL."""
    return np.asarray(bqml) / (dose / (weight_kg * 1000.0))


def bsa_m2(weight_kg: float, height_m: float) -> float:
    """Du Bois: BSA(m²) = 0.007184 · W(kg)^0.425 · H(cm)^0.725."""
    return 0.007184 * weight_kg ** 0.425 * (height_m * 100) ** 0.725


def suv_bsa(bqml, dose: float, weight_kg: float, height_m: float):
    """Body-surface-area SUV. Less biased than body weight for paediatric cases.

    No SUVlbm: the Janmahasatian formula needs sex, which the RDF header lacks.
    """
    return np.asarray(bqml) * (bsa_m2(weight_kg, height_m) * 1e4) / dose


def suv_table(bqml, mask, hdr, out=print) -> dict:
    """SUVbw and (if height is known) SUVbsa: median, p90, p99, max inside the body.

    ⚠ SUV is **linear in `K`**, and `K` is currently a guess — every number below
    is wrong by exactly the factor `K` is wrong by. Switching to the dose-based
    reference multiplies SUV by ~1.46× straight away. Not yet usable for clinical
    conclusions.
    """
    dose = dose_bq(hdr)
    w = hdr["patient_weight_kg"]
    h = hdr.get("patient_height_m") or 0.0

    got = {"SUVbw": suv_bw(bqml, dose, w)}
    if h > 0:
        # Paediatric case: weight normalisation is more biased than in adults.
        got["SUVbsa"] = suv_bsa(bqml, dose, w, h)
    # No SUVlbm: the Janmahasatian formula needs sex, which the RDF header lacks.

    out(f"{w} kg   {h} m   liều thực {dose / 1e6:.1f} MBq")
    out(f"mẫu số SUVbw = {dose / (w * 1000):,.1f} Bq/mL"
        + (f"   BSA = {bsa_m2(w, h):.3f} m²" if h > 0 else ""))
    out("")
    for name, s in got.items():
        v = s[mask]
        out(f"{name:7s} trung vị {np.median(v):6.3f}   "
            f"p90 {np.percentile(v, 90):6.2f}   p99 {np.percentile(v, 99):6.2f}   "
            f"max {s.max():8.1f}")
    out("\n[cơ/mỡ ~0,5–1 | gan ~1,5–2,5 | não trẻ em cao | bàng quang >20]")
    return got


def report(vol, K: float, hdr, vox, out=print) -> dict:
    """Apply `K`, print the numbers to check before trusting it, return `{bqml, suv, ...}`."""
    dose = dose_bq(hdr)
    vml = voxel_ml(vox)
    bqml = np.asarray(vol) * K
    suv = suv_bw(bqml, dose, hdr["patient_weight_kg"])
    mask = body_mask(vol)
    in_fov = float(bqml.sum(dtype=np.float64)) * vml

    out(f">>> K đang dùng = {K:,.2f} (Bq/mL)/(count/voxel)")
    out(f"tổng trong FOV {in_fov / 1e6:6.1f} MBq = {100 * in_fov / dose:.1f} % liều"
        "     [<100 % là đúng: scan không phủ hết người]")
    out(f"Bq/mL  max {bqml.max():>12,.0f}   trung vị thân {np.median(bqml[mask]):>10,.0f}")
    out(f"SUVbw  max {suv.max():>12.1f}   trung vị thân {np.median(suv[mask]):>10.3f}"
        "     [mô mềm ~0,5–1; não/bàng quang cao hơn nhiều]")

    return {"bqml": bqml, "suv": suv, "mask": mask, "K": float(K),
            "dose_bq": dose, "voxel_ml": vml, "mbq_in_fov": in_fov / 1e6}
