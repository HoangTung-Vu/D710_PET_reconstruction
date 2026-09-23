"""Measure SUV -> counts on real D710 beds, for the simulation's absolute scale.

    python -m deepPET.calibrate [--cases fdg26081901] [--out deepPET/calib.json]

The simulation needs one number, `s` counts per SUV.mm, in its own 2D units:
a slice's trues are `s * AF * P(SUV)`, per LOR of one direct plane. It is the
product of two factors, both measured here:

  A  Bq/mL per SUV    = net dose x decay(injection -> scan start) / weight.
                        GE's SUV is decay-corrected to the scan start and uses
                        the net dose (`quant.dose_bq`, `quant.scan_start_factor`).
  B  counts per Bq/mL.mm = sum_k trues_k / sum_k sum_bins AF0_k * P(blur(C_k))
                        per bed, where trues_k are the real prompts minus
                        background rebinned to direct plane k (as `real.ssrb`),
                        AF0_k is segment 0's attenuation for plane k, and C_k is
                        GE's image of the bed in Bq/mL (its SUV times A). B
                        therefore carries everything the 2D model leaves out:
                        the oblique segments SSRB adds, crystal efficiency, dead
                        time, the frame length and the decay to the bed.

`s = mean(A) x mean(B)`. The randoms and scatter fractions of the prompts
(inside the 371 tangential bins) are measured too; the simulation draws them
from their range.

The H108 data carry no dose or weight, so their SUV is scaled by this one mean
A: an assumption about the population, written into `calib.json` with the
cases it came from. The default case is fdg26081901, the adult (77 y, 40 kg);
fdg26081008 is a child and is left out.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from utils.scanner import NSEG0, PSF_MM

from .scanner2d import TANG, Scanner2D

CALIB_JSON = Path(__file__).with_name("calib.json")

DEFAULT_CASES = ("fdg26081901",)

GRID = 256


def bqml_per_suv(case) -> dict:
    """`A` and what it is made of, from the case's own RDF headers."""
    from utils.quant import dose_bq, scan_start_factor

    beds = case.decoded_beds()
    f, ref, hdr = scan_start_factor(case, beds)
    w = float(hdr["patient_weight_kg"])
    d = dose_bq(hdr)
    age = int(str(hdr["radiopharm_start_datetime"])[:4]) - int(str(hdr["patient_birth_date"])[:4])
    return {"bqml_per_suv": d * f / (w * 1000.0), "net_dose_mbq": d / 1e6,
            "decay_to_scan_start": f, "weight_kg": w, "age_y": age,
            "frame_s": float(hdr["frame_duration_ms"]) / 1000.0}


def bed_factor(case, bed: int, a: float, sc: Scanner2D) -> dict:
    """`B` of one bed, with the randoms and scatter fractions of its prompts."""
    from scipy.ndimage import gaussian_filter

    from utils.binmap import BinMap

    from .real import direct_plane_of, ge_planes, ssrb
    from .simulate import FWHM

    bm = BinMap(case.prompt(bed))
    k = direct_plane_of(bm)
    if not np.array_equal(k[:NSEG0], np.arange(NSEG0)):
        raise SystemExit(f"error: {case.name} bed {bed}: segment 0 is not planes 0..46 in order")
    _, _, y, g = ssrb(case, bed)
    w = case.work_bed(bed)
    attn = np.memmap(w / "attn.s", "<f4", "r", shape=bm.shape)
    r = float(np.memmap(w / "randoms.s", "<f4", "r", shape=bm.shape)[:, :, TANG].sum(dtype=np.float64))
    s = float(np.memmap(w / "scatter.s", "<f4", "r", shape=bm.shape)[:, :, TANG].sum(dtype=np.float64))
    c = ge_planes(case, bed, sc.grid) * np.float32(a)
    sig = PSF_MM / FWHM / sc.voxel_mm
    trues = (y - g).sum(axis=(1, 2), dtype=np.float64)
    q = np.array([float((attn[p][:, TANG] * sc.fwd(gaussian_filter(c[p], sig, mode="constant")))
                        .sum(dtype=np.float64)) for p in range(NSEG0)])
    body = q > 0.05 * q.max()
    per_plane = trues[body] / q[body]
    prompts = float(y.sum(dtype=np.float64))
    return {"bed": bed, "B": float(trues.sum() / q.sum()),
            "B_plane_min": float(per_plane.min()), "B_plane_max": float(per_plane.max()),
            "prompts": prompts, "prompts_per_plane": prompts / NSEG0,
            "randoms_fraction": r / prompts, "scatter_fraction": s / prompts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    ap.add_argument("--out", default=str(CALIB_JSON))
    a = ap.parse_args(argv)

    from utils.paths import case as get_case

    sc = Scanner2D(GRID)
    rows, cases = [], []
    for name in a.cases:
        C = get_case(name)
        info = {"case": name, **bqml_per_suv(C)}
        print(f"{name}: age {info['age_y']} y, {info['weight_kg']:.0f} kg, net dose "
              f"{info['net_dose_mbq']:.1f} MBq, decay to scan start "
              f"{info['decay_to_scan_start']:.4f} -> A = {info['bqml_per_suv']:.1f} Bq/mL per SUV")
        beds = C.beds(("background", "normdt", "attn", "randoms", "scatter"))
        for b in beds:
            r = {"case": name, **bed_factor(C, b, info["bqml_per_suv"], sc)}
            r["s"] = r["B"] * info["bqml_per_suv"]
            rows.append(r)
            print(f"  bed {b}: B = {r['B']:.5g} counts per Bq/mL.mm (planes "
                  f"{r['B_plane_min']:.3g}..{r['B_plane_max']:.3g}), s = {r['s']:.4f} "
                  f"counts per SUV.mm; {r['prompts_per_plane']:.3g} prompts/plane, "
                  f"randoms {r['randoms_fraction']:.3f}, scatter {r['scatter_fraction']:.3f}")
        info["beds"] = beds
        cases.append(info)
    if not rows:
        raise SystemExit("error: no bed with prompts + background, normdt, attn, randoms, scatter")

    A = float(np.mean([c["bqml_per_suv"] for c in cases]))
    B = float(np.mean([r["B"] for r in rows]))
    rf = [r["randoms_fraction"] for r in rows]
    sf = [r["scatter_fraction"] for r in rows]
    out = {"bqml_per_suv": A, "counts_per_bqml_mm": B, "counts_per_suv_mm": A * B,
           "counts_per_suv_mm_bed_range": [min(r["s"] for r in rows), max(r["s"] for r in rows)],
           "randoms_fraction_range": [min(rf), max(rf)],
           "scatter_fraction_range": [min(sf), max(sf)],
           "frame_s": cases[0]["frame_s"], "grid_mm": sc.voxel_mm,
           "made": dt.date.today().isoformat(), "by": "python -m deepPET.calibrate",
           "cases": cases, "beds": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nA = {A:.1f} Bq/mL per SUV, B = {B:.5g} counts per Bq/mL.mm "
          f"-> s = {A * B:.4f} counts per SUV.mm (beds {out['counts_per_suv_mm_bed_range'][0]:.4f}"
          f"..{out['counts_per_suv_mm_bed_range'][1]:.4f}); randoms {min(rf):.3f}..{max(rf):.3f}, "
          f"scatter {min(sf):.3f}..{max(sf):.3f}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
