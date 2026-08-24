#!/usr/bin/env python3
"""Build a PIFA (GE's mu-map container) from a real CT series.

No PIFA exists for any of our exams -- the only four on disk are GE's selftest
files -- so a real case has to start from the CT.  This does that:

    CT DICOM series  ->  HU  ->  mu(511 keV) in mm^-1  ->  PIFA grid  ->  .pifa

`estimate.py` runs this INSIDE d710:full, which already has numpy/scipy/pydicom
-- so the host needs neither.  By hand, from D710/:

    docker run --rm -e PYTHONPATH=/d710 -v "$PWD:/d710:ro" -v <ct>:/ct:ro \\
        -v <outdir>:/out d710:full \\
        python3 /d710/vendor/ct_to_pifa.py /ct /out/mu.pifa \\
        --table-location -125.17

Then point the job at it (job.gdb line 12,
IgJobReq.inputTransmissionFileName[0]) and run vendor/extract.gdb.

--------------------------------------------------------------------------
The PIFA grid is not a guess.  pet_recon printed its own numbers while
forward-projecting the selftest PIFA:

    pPifaHeader->xMatrix: 128   yMatrix: 128   zMatrix: 47
    sx: 5.468750                       <- 700 / 128 = ctacDfov / xMatrix
    m_pParamStruct->rawDataTheta[0].spacing_v: 3.264583   <- PET plane pitch

so: 128 x 128 over a 700 mm transaxial FOV centred on the scanner axis, and 47
planes at the PET plane pitch starting at the bed's table position.  That is
the same sampling utils/attenuation.py:mu_image() already builds for SIRF;
this writes it on the PIFA grid instead, and keeps mu in **mm^-1** (mu_image
multiplies by 10 for STIR's cm^-1 -- the PIFA does NOT want that).

--------------------------------------------------------------------------
frame_of_reference must equal the exam's -- ValidateCTAC does a strcmp on it
and rejects the PIFA otherwise ("frame_of_reference EX: PIFA CTAC: 1.2.840...").
It defaults to the CT series' own FrameOfReferenceUID, which is that value.

--------------------------------------------------------------------------
Orientation: DICOM patient (LPS) is what GE wants, and this is MEASURED, not
assumed.  mu_image() in utils/attenuation.py applies to_radiological() (a
y-flip) because that is STIR's convention; the PIFA must NOT have it.

The discriminator is GE's own table masking.  CScatterFully3dModel::
MaskTableMuImage calls sharcApTableMaskModified, which models the patient table
as a fixed circle (r = 464.05 mm, offset 45.974) positioned from
-ctacXaxisTranslation / -ctacYaxisTranslation, then rotates the image by
sysGeometry.vqc_ZaxisRoll.  If y is flipped the mask removes tissue instead of
the table, the table survives in the mu-map as dense material, and the SSS
estimate inflates.  Measured on NEMA bed 2:

    scatter fraction S/(T+S)      as written   y mirrored   x mirrored
    NEMA phantom bed 2              32.98 %      49.28 %      33.77 %
    patient chest/abdomen bed       37.28 %          --       76.48 %

y is settled by NEMA: 49.28 % is not a physical scatter fraction, so the table
had been left in the mu-map.  x needs an asymmetric subject -- the NEMA phantom
is nearly symmetric about x, hence the useless 33.77 vs 32.98 -- so it was
repeated on a patient chest/abdomen bed (11082026 SINO0004, mu-map 22.1 %
left-right asymmetric), where 76.48 % is impossible.

**Both axes are measured, so there are no flip options: plain DICOM LPS is
written and that is that.**  If a future scanner or export turns out to differ,
re-run the measurement above rather than adding a flag back on a hunch.

The grid itself is not a guess either -- pet_recon printed its own numbers:
zMatrix views of xMatrix*yMatrix float32 in CRawDataMem::m_pAttn (47 views of
65536 B = 128x128x4), i.e. mu[z][y][x] with x fastest, which is forced by the
view size rather than chosen.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# `D710/` is on PYTHONPATH (the container gets it from `d710`, the host from the
# CLI), so this is an ordinary package import -- no sys.path surgery.
from utils import attenuation

# `make_pifa` is a sibling script in this directory, which is sys.path[0] when
# this file is run as a script.
from make_pifa import HEADER_BYTES, pack_header


def resample_to_pifa(ct, table_location_mm, matrix, dfov_mm, planes, plane_mm):
    """HU volume -> mu(mm^-1) on the PIFA grid, [z][y][x] with x fastest."""
    from scipy.ndimage import map_coordinates

    pixel_mm = dfov_mm / matrix                      # 700/128 = 5.46875
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
    return attenuation.hu_to_mu(hu, ct.kvp)          # mm^-1 -- no x10


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
