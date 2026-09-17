#!/usr/bin/env python3
"""Build a PIFA, GE's mu-map container, from a real CT series."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from utils import attenuation

from make_pifa import HEADER_BYTES, pack_header


def resample_to_pifa(ct, table_location_mm, matrix, dfov_mm, planes, plane_mm):
    """HU volume to mu in mm^-1 on the PIFA grid, `[z][y][x]` with x fastest."""
    from scipy.ndimage import map_coordinates

    pixel_mm = dfov_mm / matrix
    zc = table_location_mm + np.arange(planes) * plane_mm
    gz = (zc - ct.z[0]) / ct.dz
    lo, hi = gz.min(), gz.max()
    covered = -0.5 <= lo and hi <= len(ct.z) - 0.5
    print("  axial: planes %d x %.6f mm from table %.2f -> z %.1f..%.1f mm"
          % (planes, plane_mm, table_location_mm, zc[0], zc[-1]))
    print("  CT covers z %.1f..%.1f mm (%d slices, step %.4f)"
          % (ct.z[0], ct.z[-1], len(ct.z), ct.dz))
    if not covered:
        print("  !! the bed is NOT fully inside the CT; edges fill with air "
              "(HU -1000)", file=sys.stderr)

    c = (np.arange(matrix) - matrix // 2) * pixel_mm
    g = np.meshgrid(gz,
                    (c - ct.y0) / ct.pixel_mm,
                    (c - ct.x0) / ct.pixel_mm, indexing="ij")
    hu = map_coordinates(ct.hu, [a.ravel() for a in g], order=1,
                         mode="constant", cval=-1000.0).reshape(planes, matrix, matrix)
    return attenuation.hu_to_mu(hu, ct.kvp)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ct_dir")
    ap.add_argument("out")
    ap.add_argument("--table-location", type=float, required=True,
                    help="bed table position in mm (NEMA bed2 = -125.17)")
    ap.add_argument("--matrix", type=int, default=128)
    ap.add_argument("--dfov", type=float, default=700.0)
    ap.add_argument("--planes", type=int, default=attenuation.PLANES_PER_BED)
    ap.add_argument("--plane-mm", type=float, default=3.264583,
                    help="PET plane pitch; 3.264583 is what pet_recon reports "
                         "as rawDataTheta[0].spacing_v")
    ap.add_argument("--frame-of-reference", default=None,
                    help="defaults to the CT series' own FrameOfReferenceUID, "
                         "which is what ValidateCTAC strcmp's against; a "
                         "literal placeholder makes it fail with "
                         "'frame_of_reference EX: ... CTAC: ...'")
    ap.add_argument("--copy-spare-from", metavar="PIFA",
                    help="lift spareFields[64] from a known-good PIFA")
    ap.add_argument("--patient-entry", type=int, default=0)
    ap.add_argument("--patient-position", type=int, default=0)
    args = ap.parse_args()

    ct = attenuation.load(args.ct_dir)
    print(ct.describe())
    forf = args.frame_of_reference or ct.meta["frame_of_reference_uid"]
    if not forf:
        raise SystemExit("error: the CT has no FrameOfReferenceUID and none "
                         "was given; ValidateCTAC will reject the PIFA")
    print("  frame_of_reference: %s" % forf)

    mu = resample_to_pifa(ct, args.table_location, args.matrix, args.dfov,
                          args.planes, args.plane_mm)
    nz = mu[mu > 0]
    print("  mu (mm^-1): max %.6f  mean(>0) %.6f  nonzero %d/%d"
          % (mu.max(), nz.mean() if nz.size else 0.0, nz.size, mu.size))
    print("  mu (cm^-1): max %.4f   [water at 511 keV ~0.096 cm^-1;"
          " GE's selftest PIFA peaks at 0.0934]" % (mu.max() * 10))
    if not (0.005 < mu.max() < 0.05):
        print("  !! max mu is outside the plausible mm^-1 range -- check units",
              file=sys.stderr)

    spare = None
    if args.copy_spare_from:
        with open(args.copy_spare_from, "rb") as f:
            spare = f.read(HEADER_BYTES)[96:160]

    header = pack_header(args.matrix, args.matrix, args.planes, args.dfov,
                         args.table_location, forf,
                         patient_entry=args.patient_entry,
                         patient_position=args.patient_position, spare=spare)
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(mu, dtype="<f4").tobytes())
    print("wrote %s: %dx%dx%d, %d bytes"
          % (args.out, args.matrix, args.matrix, args.planes,
             HEADER_BYTES + mu.size * 4))
    print("verify:  python3 make_pifa.py --inspect %s" % args.out)


if __name__ == "__main__":
    main()
