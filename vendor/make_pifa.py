#!/usr/bin/env python3
"""Write a PIFA file from your own mu-map, so GE's scatter model uses it.

The PIFA is the mu-map container `pet_recon` reads for model-based (SSS)
scatter.  `sharcCmpOpenDataFiles` parses it into the global
`transmissionCTACHeader`, and `CScatterFully3dModel` builds its mu image from
there -- which is why the scatter model runs even when the CTAC attenuation
path is dead.  The job points at it through ONE field:

    IgJobReq.inputTransmissionFileName[0]

so switching to your own mu-map is a one-line edit in job.gdb.

Layout, field names and offsets straight out of the DWARF
(`ptype /o transmissionCTACHeader`):

      0  f32 versionID                 32  s8  frame_of_reference[64]
      4  n32 cmpJobID                  96  s8  spareFields[64]
      8  n16 xMatrix                  160  n32 offsetToStartOfImage = 164
     10  n16 yMatrix                  --- 164 bytes, then the volume:
     12  n16 zMatrix                      xMatrix*yMatrix*zMatrix float32
     16  f32 ctacDfov  (mm)               mu in mm^-1, x fastest, z slowest
     20  n32 patientEntry
     24  n32 patientPosition
     28  f32 tableLocation (mm)

Units matter: mu is in **mm^-1**, not cm^-1.  GE's own selftest file peaks at
0.009336 mm^-1, i.e. 0.0934 cm^-1, water at 511 keV.  If your mu-map is in
cm^-1 -- STIR's convention -- divide by 10 (or pass --units cm-1).

Usage:
    ./make_pifa.py mu.raw out.pifa --matrix 128 128 47 --dfov 700 \\
                   --table-location -47.92 --units cm-1
    ./make_pifa.py mu.npy out.pifa --dfov 700         # shape read from the .npy
    ./make_pifa.py --inspect selftest_kh3d_pifa.dat

ValidateCTAC only compares two int fields against the job, the table location
against apCfg.exCtacFrameTableLocTolerance (1.975), and the
frame_of_reference string -- it checks neither the matrix size nor the voxel
data.  Since the job is ours too, matching --frame-of-reference on both sides
is enough; it need not be a real DICOM UID.
"""
import argparse
import os
import struct
import sys

HEADER_BYTES = 164


def pack_header(x, y, z, dfov, table_location, frame_of_reference,
                version=1.0, job_id=1, patient_entry=0, patient_position=0,
                spare=None):
    h = bytearray(HEADER_BYTES)
    if spare:
        # GE's own files carry non-zero bytes in spareFields[64] (96..159).
        # The DWARF calls the field "spareFields", so pet_recon's parser has no
        # name for it and cannot be reading it -- but that is an argument, not
        # a measurement, so --copy-spare-from exists to sidestep the question.
        h[96:160] = spare[:64].ljust(64, b"\0")
    struct.pack_into("<f", h, 0, version)
    struct.pack_into("<I", h, 4, job_id)
    struct.pack_into("<HHH", h, 8, x, y, z)
    struct.pack_into("<f", h, 16, dfov)
    struct.pack_into("<I", h, 20, patient_entry)
    struct.pack_into("<I", h, 24, patient_position)
    struct.pack_into("<f", h, 28, table_location)
    for_bytes = frame_of_reference.encode()[:63]
    h[32:32 + len(for_bytes)] = for_bytes
    struct.pack_into("<I", h, 160, HEADER_BYTES)
    return bytes(h)


def inspect(path):
    with open(path, "rb") as f:
        h = f.read(HEADER_BYTES)
    ver, job = struct.unpack_from("<fI", h, 0)
    x, y, z = struct.unpack_from("<HHH", h, 8)
    dfov, = struct.unpack_from("<f", h, 16)
    entry, pos = struct.unpack_from("<II", h, 20)
    table, = struct.unpack_from("<f", h, 28)
    forf = h[32:96].split(b"\0")[0].decode(errors="replace")
    off, = struct.unpack_from("<I", h, 160)
    size = os.path.getsize(path)
    print("%s" % path)
    print("  version %.1f  cmpJobID %d" % (ver, job))
    print("  matrix %dx%dx%d   dfov %.1f mm   tableLocation %.2f" % (x, y, z, dfov, table))
    print("  patientEntry %d  patientPosition %d" % (entry, pos))
    print("  frame_of_reference %s" % forf)
    print("  offsetToStartOfImage %d, file %d, payload %d (%.3f B/voxel)"
          % (off, size, size - off, (size - off) / float(x * y * z or 1)))

    import array
    with open(path, "rb") as f:
        f.seek(off)
        a = array.array("f")
        a.frombytes(f.read(x * y * z * 4))
    nz = [v for v in a if v]
    if nz:
        print("  mu (mm^-1): min %.6f  max %.6f  mean(nonzero) %.6f  nonzero %d/%d"
              % (min(a), max(a), sum(nz) / len(nz), len(nz), len(a)))
        print("  mu (cm^-1): max %.4f   [water at 511 keV is ~0.096 cm^-1]"
              % (max(a) * 10))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mu", nargs="?", help="mu-map: .npy, or raw float32")
    ap.add_argument("out", nargs="?", help="PIFA file to write")
    ap.add_argument("--inspect", metavar="PIFA", help="decode an existing PIFA and exit")
    ap.add_argument("--matrix", nargs=3, type=int, metavar=("X", "Y", "Z"),
                    help="required for raw input; read from the array for .npy")
    ap.add_argument("--dfov", type=float, default=700.0, help="ctacDfov in mm (default 700)")
    ap.add_argument("--table-location", type=float, default=0.0)
    ap.add_argument("--frame-of-reference", default="PIFA",
                    help="must match the value ValidateCTAC compares against")
    ap.add_argument("--units", choices=["mm-1", "cm-1"], default="mm-1",
                    help="units of the input; cm-1 is divided by 10")
    ap.add_argument("--job-id", type=int, default=1)
    ap.add_argument("--copy-spare-from", metavar="PIFA",
                    help="lift spareFields[64] (offsets 96..159) from an existing "
                         "PIFA.  The DWARF calls the field spare, so pet_recon "
                         "should ignore it, but that is unproven -- use this to "
                         "be byte-faithful to a known-good file.")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return
    if not (args.mu and args.out):
        ap.error("need MU and OUT (or --inspect)")

    if args.mu.endswith(".npy"):
        try:
            import numpy as np
        except ImportError:
            sys.exit(".npy input needs numpy; convert to raw float32 instead")
        vol = np.load(args.mu).astype("<f4")
        if vol.ndim != 3:
            sys.exit("expected a 3-D mu-map, got shape %s" % (vol.shape,))
        z, y, x = vol.shape          # numpy is z-slowest, matching the file order
        if args.matrix and tuple(args.matrix) != (x, y, z):
            sys.exit("--matrix %s contradicts the array's %s"
                     % (tuple(args.matrix), (x, y, z)))
        if args.units == "cm-1":
            vol = vol / 10.0
        payload = vol.astype("<f4").tobytes()
    else:
        if not args.matrix:
            ap.error("--matrix X Y Z is required for raw input")
        x, y, z = args.matrix
        with open(args.mu, "rb") as f:
            payload = f.read()
        want = x * y * z * 4
        if len(payload) != want:
            sys.exit("raw input is %d bytes, expected %d for %dx%dx%d float32"
                     % (len(payload), want, x, y, z))
        if args.units == "cm-1":
            import array
            a = array.array("f")
            a.frombytes(payload)
            for i in range(len(a)):
                a[i] /= 10.0
            payload = a.tobytes()

    spare = None
    if args.copy_spare_from:
        with open(args.copy_spare_from, "rb") as f:
            spare = f.read(HEADER_BYTES)[96:160]

    header = pack_header(x, y, z, args.dfov, args.table_location,
                         args.frame_of_reference, job_id=args.job_id, spare=spare)
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(payload)
    print("wrote %s: %dx%dx%d, %d bytes" % (args.out, x, y, z, HEADER_BYTES + len(payload)))
    print("now point the job at it:")
    print('  job.gdb line 12 -> _s("IgJobReq.inputTransmissionFileName[0]", "%s")'
          % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
