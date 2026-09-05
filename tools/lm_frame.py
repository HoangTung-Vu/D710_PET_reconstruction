#!/usr/bin/env python3
"""Is the list-mode image in the same frame as the sinogram one?

    python3 tools/lm_frame.py --case ped --bed 1

Reads `work/bed<n>/{osem,lm}.npz` and reports, for the list-mode image and its
transpose, the rotation that best lines its angular profile up with the sinogram
image's. A correct `lm.geom.scanner_lut` gives **0 deg on the untransposed
image**; a transpose winning means the crystal order is mirrored, and a non-zero
rotation means the detector-0 position is wrong.

⚠ **This compares the two paths to EACH OTHER, never to reality**, and only their
rotation — so it is blind to a shared angle error (10.04 deg, found 2026-09-04)
and to a scale error (list-mode is 2.3 % small). Use the vendor's own PT series
or the CT for the absolute frame. See GEOMETRY_AUDIT.md.

Angular profiles rather than voxel correlation: they are insensitive to the
resolution and noise difference between the two reconstructions, which is what
buries the signal in a plain voxel-wise comparison.
"""

from __future__ import annotations

import argparse

import numpy as np

from utils.paths import case as get_case


def angular_profile(vol, n_bins: int = 360):
    """Intensity against angle about the image centre, radius-weighted."""
    s = vol.sum(0)
    n = s.shape[0]
    y, x = np.mgrid[0:n, 0:n] - (n - 1) / 2
    r = np.hypot(x, y)
    m = (r > 8) & (r < n / 2 - 2)
    idx = ((np.degrees(np.arctan2(y, x)) % 360)[m] / 360 * n_bins).astype(int)
    return np.bincount(idx % n_bins, s[m] * r[m], minlength=n_bins)


def best_rotation(p, q):
    """`(degrees, correlation)` -- how far `q` must turn to match `p`."""
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    p, q = z(p), z(q)
    c = np.array([p @ np.roll(q, k) for k in range(len(q))]) / len(q)
    return int(c.argmax()) * 360 // len(q), float(c.max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out")
    ap.add_argument("--bed", type=int, default=1)
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    w = C.work_bed(args.bed)
    for p in (w / "osem.npz", w / "lm.npz"):
        if not p.exists():
            raise SystemExit(f"error: no {p}\n  run `d710 osem --beds {args.bed}` "
                             f"and `d710 lm recon --beds {args.bed}` first")
    a = np.load(w / "osem.npz")["img"]
    b = np.load(w / "lm.npz")["img"]
    print(f"sinogram {a.shape} sum {a.sum():,.1f}   "
          f"list-mode {b.shape} sum {b.sum():,.1f}")

    pa = angular_profile(a)
    for name, v in (("as reconstructed", b), ("transposed", b.transpose(0, 2, 1))):
        deg, c = best_rotation(pa, angular_profile(v))
        print(f"  {name:17s} rotation {deg:>4d} deg   profile corr {c:+.4f}"
              + ("   <- want this, at 0 deg" if name.startswith("as") else ""))

    m = a > np.percentile(a, 99.0)
    print(f"  voxel corr on the top 1 % of the sinogram image: "
          f"{float(np.corrcoef(a[m], b[m])[0, 1]):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
