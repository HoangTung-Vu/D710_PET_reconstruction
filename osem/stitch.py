"""Decay correction and axial stitching of the reconstructed beds."""

from __future__ import annotations

import datetime as dt

import numpy as np

from utils.scanner import (PLANE_MM, POST_FILTER_FWHM_MM,
                           POST_FILTER_Z_RATIO)


def post_filter(vol, vox, fwhm_mm: float = POST_FILTER_FWHM_MM,
                z_ratio: float = POST_FILTER_Z_RATIO, verbose: bool = True):
    """GE's two-part post-filter, applied to the stitched volume."""
    from scipy.ndimage import gaussian_filter

    out = np.asarray(vol, dtype=np.float32)

    if fwhm_mm and fwhm_mm > 0:
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
    """Injection time as a UTC epoch."""
    t = dt.datetime.strptime(hdr["radiopharm_start_datetime"][:14], "%Y%m%d%H%M%S")
    return t.replace(tzinfo=dt.timezone.utc).timestamp()


def decay_factor(hdr, t_inj: float) -> float:
    """The multiplicative factor taking a bed's image back to the activity at injection time."""
    lam = np.log(2) / hdr["half_life_s"]
    dt_s = hdr["bed_start_time"] - t_inj
    T = hdr["frame_duration_ms"] / 1000.0
    return 1.0 / (np.exp(-lam * dt_s) * (1 - np.exp(-lam * T)) / (lam * T))


def plane_index(case, beds):
    """`(idx, z0, nz)`: which planes of the shared volume each bed maps into."""
    nz_bed = int(round(2 * 24 - 1))
    zs = {n: case.header(n)["table_position_mm"] + np.arange(nz_bed) * PLANE_MM
          for n in beds}
    z0 = min(z[0] for z in zs.values())
    idx = {n: np.rint((zs[n] - z0) / PLANE_MM).astype(int) for n in beds}
    return idx, float(z0), int(max(i[-1] for i in idx.values()) + 1)


def stitch(case, beds, img: dict, sens: dict, verbose: bool = True):
    """Stitch the beds into one whole-body volume."""
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
    """Report the bed seams numerically."""
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
