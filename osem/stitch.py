"""Decay-correct, then stitch the beds along the axis.

### Decay correction must happen BEFORE stitching

The six beds are acquired 91 s apart, with bed 6 up to 458 s later than bed 1.
Stitching directly glues six different time points into one volume → a spurious
axial gradient. Everything is referred back to the **injection time**, matching
DICOM's `DecayCorrection = START`:

    f = exp(−λ·Δt) · (1 − exp(−λ·T)) / (λ·T)
        exp(−λ·Δt)          decay from injection to the start of the bed
        (1−exp(−λT))/(λT)   the mean activity DURING frame T, not the instantaneous one

Dropping the second factor (using the instantaneous activity at the bed start)
is off by ~0.5 % here and considerably more for long frames — and it is off **by
a different amount per bed**, so it survives as an axial gradient instead of
dissolving into the constant `K`.

### The overlap is where BOTH beds are weakest — hence the weighting

The table step of 124.26 mm is **exactly 38 planes**, and a bed is 47 planes
long → a 9-plane overlap. But those 9 planes are planes 38–46 of the lower bed
and planes 0–8 of the upper one: **the two weakest ends meet**. The weight is
`SENS[n]`, STIR's own sensitivity image — the denominator OSEM divides by on
each iteration, so it already includes norm, dead time, attenuation and how the
projector really samples LORs, and it is a **3D** image, i.e. a PER-VOXEL weight.
For Poisson ML, `Var(x̂) ≈ x / sens`, so the inverse-variance weight is exactly
`w ∝ sens`.

The plane indices can be rounded because the table step really is an integer
number of planes; a half-plane offset would pile two beds into the same cell with
a 1.6 mm axial error. `tests/test_notebook_contract.py` pins exactly that down.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from utils.geometry import PLANE_MM


def injection_epoch(hdr) -> float:
    """Injection time, UTC epoch. The RDF header records UTC (DICOM records local time)."""
    t = dt.datetime.strptime(hdr["radiopharm_start_datetime"][:14], "%Y%m%d%H%M%S")
    return t.replace(tzinfo=dt.timezone.utc).timestamp()


def decay_factor(hdr, t_inj: float) -> float:
    """The MULTIPLICATIVE factor taking a bed's image back to the activity at injection time."""
    lam = np.log(2) / hdr["half_life_s"]
    dt_s = hdr["bed_start_time"] - t_inj
    T = hdr["frame_duration_ms"] / 1000.0
    return 1.0 / (np.exp(-lam * dt_s) * (1 - np.exp(-lam * T)) / (lam * T))


def plane_index(case, beds):
    """`(idx, z0, nz)` — which planes of the shared volume each bed maps into.

    `z = table_position + i·PLANE_MM`, **the very formula `attenuation.mu_image`
    uses to cut the CT for that bed**, so the two geometries agree by
    construction rather than by coincidence.
    """
    nz_bed = int(round(2 * 24 - 1))          # 47 planes per bed
    zs = {n: case.header(n)["table_position_mm"] + np.arange(nz_bed) * PLANE_MM
          for n in beds}
    z0 = min(z[0] for z in zs.values())
    idx = {n: np.rint((zs[n] - z0) / PLANE_MM).astype(int) for n in beds}
    return idx, float(z0), int(max(i[-1] for i in idx.values()) + 1)


def stitch(case, beds, img: dict, sens: dict, verbose: bool = True):
    """Stitch the beds into one whole-body volume. Returns `(vol, z0, factors)`.

    `vol` is count/voxel **referred back to the injection time**.
    """
    t_inj = injection_epoch(case.header(beds[0]))
    factors = {n: decay_factor(case.header(n), t_inj) for n in beds}

    if verbose:
        up = (case.header(beds[0])["bed_start_time"] - t_inj) / 60
        inj = dt.datetime.fromtimestamp(t_inj, dt.timezone.utc)
        print(f"tiêm {inj:%Y-%m-%d %H:%M:%S} UTC   "
              f"uptake tới bed {beds[0]}: {up:.1f} phút")
        for n in beds:
            print(f"  bed {n}: × {factors[n]:.4f}")

    idx, z0, nz = plane_index(case, beds)
    shape = (nz,) + img[beds[0]].shape[1:]
    num = np.zeros(shape, dtype=np.float64)
    den = np.zeros_like(num)
    for n in beds:
        w = sens[n].astype(np.float64)
        num[idx[n]] += img[n] * factors[n] * w
        den[idx[n]] += w

    ok = den > 0
    vol = np.zeros_like(num)
    vol[ok] = num[ok] / den[ok]
    vol = vol.astype(np.float32)

    if verbose:
        print(f"\ntoàn thân: {vol.shape}  "
              f"z {z0:.1f} .. {z0 + (nz - 1) * PLANE_MM:.1f} mm")
    return vol, z0, factors


def overlap_report(case, beds, img: dict, factors: dict, out=print):
    """CHECK the seams with NUMBERS, do not just look at them.

    The "correlation" column compares two **independent** reconstructions of *the
    same stretch of body*. Get the axial direction wrong, or misplace a bed, and
    it collapses immediately — while the stitched image still looks like a body.
    """
    idx, _z0, _nz = plane_index(case, beds)
    out(f"\n{'cặp bed':>10} {'plane chồng':>12} {'tương quan':>11} {'tỉ lệ biên độ':>14}")
    rows = []
    for a, b in zip(beds, beds[1:]):
        common = np.intersect1d(idx[a], idx[b])
        if not common.size:
            out(f"{f'{a}-{b}':>10} {0:>12}   (không chồng)")
            continue
        va = img[a][np.searchsorted(idx[a], common)].ravel() * factors[a]
        vb = img[b][np.searchsorted(idx[b], common)].ravel() * factors[b]
        corr = float(np.corrcoef(va, vb)[0, 1])
        ratio = float(vb.sum() / max(va.sum(), 1e-9))
        out(f"{f'{a}-{b}':>10} {common.size:>12} {corr:>11.4f} {ratio:>14.3f}")
        rows.append({"pair": (a, b), "planes": int(common.size),
                     "corr": corr, "ratio": ratio})
    return rows
