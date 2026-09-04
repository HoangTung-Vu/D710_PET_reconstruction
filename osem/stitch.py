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

### The post-filter is not cosmetic — it is the missing half of the algorithm

Unregularised OSEM has no noise control at all: no prior, no inter-iteration
filter. GE's does not either — `(0009,10B6) ir_loop_filter = 0` and
`(0009,10BC) ir_regularize = 0`. What GE has and this did not is a **post**
filter, and without it the two images are not comparable: measured against GE's
own VPFXS of the same exam, the liver CoV was 36.5 % against GE's 18.3 %, while
the *means* agreed to 5.7 %. Noise, not bias.

`post_filter` runs on the STITCHED volume, not per bed. Filtering a bed on its
own convolves its 47-plane ends against zeros, which deepens exactly the axial
roll-off the sensitivity weighting above exists to avoid.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from utils.scanner import (PLANE_MM, POST_FILTER_FWHM_MM,  # noqa: F401
                           POST_FILTER_Z_RATIO)


def post_filter(vol, vox, fwhm_mm: float = POST_FILTER_FWHM_MM,
                z_ratio: float = POST_FILTER_Z_RATIO, verbose: bool = True):
    """GE's two-part post-filter on the stitched volume. Returns a new array.

    `vox` is `(z, y, x)` in mm — `sirf.STIR` image `voxel_sizes()` order, which
    is what `osem/__main__.py` already holds. Two different filters, because GE
    uses two:

    * **transaxial** — Gaussian, `fwhm_mm` FWHM, applied in-plane only.
    * **axial** — the three-tap `[1, z_ratio, 1]`.

    `fwhm_mm = 0` skips the transaxial half, `z_ratio = 0` the axial one, so a
    run can be compared against an unfiltered one without editing anything.

    **Edge handling differs by axis, on purpose.** Transaxially the filter runs
    against zeros: outside the reconstructed FOV there genuinely is no activity,
    so letting the rim decay is correct. Axially it replicates the end plane
    instead — the volume does not stop because the patient does, it stops
    because the beds ran out, and convolving that against zeros would darken the
    end planes for a reason that is an artefact of where the scan ended.
    """
    from scipy.ndimage import gaussian_filter

    out = np.asarray(vol, dtype=np.float32)

    if fwhm_mm and fwhm_mm > 0:
        # FWHM -> sigma, then mm -> voxels, per axis: the grid is not isotropic.
        sig = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        out = gaussian_filter(out, sigma=(0.0, sig / vox[1], sig / vox[2]),
                              mode="constant", cval=0.0)

    if z_ratio and z_ratio > 0:
        k = np.array([1.0, z_ratio, 1.0], dtype=np.float32)
        k /= k.sum()
        p = np.pad(out, ((1, 1), (0, 0), (0, 0)), mode="edge")
        out = k[0] * p[:-2] + k[1] * p[1:-1] + k[2] * p[2:]

    if verbose:
        sig = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0))) if fwhm_mm else 0.0
        print(f"post-filter: transaxial FWHM {fwhm_mm:g} mm "
              f"(sigma {sig / vox[2]:.2f} voxel), axial [1,{z_ratio:g},1]")

    return out.astype(np.float32)


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
        print(f"injection {inj:%Y-%m-%d %H:%M:%S} UTC   "
              f"uptake up to bed {beds[0]}: {up:.1f} min")
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
        print(f"\nwhole body: {vol.shape}  "
              f"z {z0:.1f} .. {z0 + (nz - 1) * PLANE_MM:.1f} mm")
    return vol, z0, factors


def overlap_report(case, beds, img: dict, factors: dict, out=print):
    """CHECK the seams with NUMBERS, do not just look at them.

    The "correlation" column compares two **independent** reconstructions of *the
    same stretch of body*. Get the axial direction wrong, or misplace a bed, and
    it collapses immediately — while the stitched image still looks like a body.
    """
    idx, _z0, _nz = plane_index(case, beds)
    out(f"\n{'bed pair':>10} {'overlap pl.':>12} {'correlation':>11} {'amplitude ratio':>14}")
    rows = []
    for a, b in zip(beds, beds[1:]):
        common = np.intersect1d(idx[a], idx[b])
        if not common.size:
            out(f"{f'{a}-{b}':>10} {0:>12}   (no overlap)")
            continue
        va = img[a][np.searchsorted(idx[a], common)].ravel() * factors[a]
        vb = img[b][np.searchsorted(idx[b], common)].ravel() * factors[b]
        corr = float(np.corrcoef(va, vb)[0, 1])
        ratio = float(vb.sum() / max(va.sum(), 1e-9))
        out(f"{f'{a}-{b}':>10} {common.size:>12} {corr:>11.4f} {ratio:>14.3f}")
        rows.append({"pair": (a, b), "planes": int(common.size),
                     "corr": corr, "ratio": ratio})
    return rows
