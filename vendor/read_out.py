#!/usr/bin/env python3
"""Read what extract.gdb wrote.

    python3 read_out.py out/normdt.f32              # summary
    python3 read_out.py out/normdt.f32 --view 0     # one view's stats
    python3 read_out.py out/normdt.f32 --npy        # -> out/normdt.npy

Every dump is a flat little-endian array plus a .json sidecar carrying the
shape and the provenance, so nothing here has to hard-code geometry.  numpy is
used when available and the stdlib `array` module otherwise -- the container
that produces these files has no numpy, and neither does every environment
that reads them.
"""
import argparse
import json
import os
import sys

DTYPE = {"f4": ("f", 4, "<f4"), "i4": ("i", 4, "<i4"), "u2": ("H", 2, "<u2")}


def sidecar(path):
    with open(path + ".json") as f:
        return json.load(f)


def shape_of(meta):
    """(view, v, u) when the sidecar describes a sinogram, else None."""
    if all(k in meta for k in ("number_phi", "number_v_theta", "number_u")):
        return (meta["number_phi"], meta["number_v_theta"], meta["number_u"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--view", type=int, help="summarise a single view")
    ap.add_argument("--npy", action="store_true", help="also write a .npy next to it")
    args = ap.parse_args()

    meta = sidecar(args.path)
    code, width, np_dtype = DTYPE[meta.get("dtype", "f4")]
    shape = shape_of(meta)
    size = os.path.getsize(args.path)

    print("%s" % args.path)
    print("  %-14s %s" % ("produced by", meta.get("produced_by", "-")))
    print("  %-14s %s" % ("axes", meta.get("axes", meta.get("layout", "-"))))
    print("  %-14s %d bytes, %d elements of %s"
          % ("size", size, size // width, meta.get("dtype", "f4")))
    if meta.get("wcc_applied") is False:
        print("  %-14s no WCC applied" % "scale")
    if meta.get("note"):
        print("  %-14s %s" % ("note", meta["note"]))

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is None:
        import array
        a = array.array(code)
        with open(args.path, "rb") as f:
            a.frombytes(f.read())
        nz = [x for x in a if x]
        print("  %-14s min %g  max %g  mean %g  nonzero %d/%d"
              % ("stats", min(nz) if nz else 0, max(nz) if nz else 0,
                 sum(a) / len(a) if a else 0, len(nz), len(a)))
        if args.npy:
            sys.exit("--npy needs numpy")
        return

    a = np.fromfile(args.path, dtype=np_dtype)
    if shape and a.size == shape[0] * shape[1] * shape[2]:
        a = a.reshape(shape)
        print("  %-14s %s" % ("shape", a.shape))
    b = a.astype("f8")
    nz = b[b != 0]
    print("  %-14s min %g  max %g  mean %g  median %g"
          % ("stats", b.min(), b.max(), b.mean(), np.median(b)))
    print("  %-14s %d/%d (%.2f%%)"
          % ("nonzero", nz.size, b.size, 100.0 * nz.size / b.size))

    if args.view is not None and a.ndim == 3:
        v = b[args.view]
        print("  view %-9d min %g  max %g  mean %g"
              % (args.view, v.min(), v.max(), v.mean()))

    if args.npy:
        out = os.path.splitext(args.path)[0] + ".npy"
        np.save(out, a)
        print("  wrote %s" % out)


if __name__ == "__main__":
    main()
