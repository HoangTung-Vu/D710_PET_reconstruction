#!/usr/bin/env python3
"""Vendor ``.f32`` sinograms -> STIR Interfile.  The step the notebook lacked.

Normally driven by ``d710 tostir --case ped --bed 4``, which is exactly:

    ./to_stir.py --vendor   $D710_OUT/ped/vendor/bed4 \\
                 --template $D710_OUT/ped/decoded/bed4.hs \\
                 --out      $D710_OUT/ped/work/bed4

``estimate.py`` leaves GE's correction sinograms as flat little-endian arrays
in the vendor's own bin order.  STIR wants a different one, and the whole
difference is two index conventions:

    GE    (view, plane, u)             288 x 553 x 381
    STIR  (tof, axial, view, tang)   1 x 553 x 288 x 381

* the **plane axis is the identity**.  GE stores segments in the order
  ``0, +1, -1, +2, -2, ...`` holding ``47 - 4|k|`` axial positions each --
  which is exactly STIR's own storage order, so plane ``p`` lands in axial
  slot ``p`` untouched.  (This is why no interpolation happens anywhere here:
  the two michelograms are the same michelogram.)
* the **view axis is reversed**: ``stir_view = 287 - ge_view``.  GE numbers
  views in the opposite rotational sense; writing them straight through
  mirrors the reconstruction transaxially -- see ``gerdf.cli._view_index``,
  which settled the direction against this scanner's own CT of the NEMA
  phantom.

Neither is assumed.  ``prompts.u16`` (what the vendor's kernel loaded) and the
template's ``.s`` (what ``custom_tool`` decoded) are the same acquisition
reached by two independent paths, so **every run re-proves the mapping
bit-exact on real data** before writing anything.  A mismatch writes nothing.

The header is the template's own, with three keys changed (data file name,
number format, bytes per pixel).  Copying it rather than generating one means
the geometry STIR reads for a correction term is, by construction, the same
geometry it reads for the prompts.

Outputs (float32, all in the raw detected-count domain, no WCC anywhere):

    randoms.hs/.s     randoms
    scatter.hs/.s     model-based (SSS) scatter
    background.hs/.s  randoms + scatter      -> `b` in y = S(Gx) + b
    normdt.hs/.s      normalisation x dead time -> `S` (before attenuation)
    norm_only.hs/.s   normalisation alone; dead time = normdt / norm_only

``normdt`` is a **sensitivity**, not a correction factor: dividing the data by
it is what corrects.  Two independent measurements fix that direction:

* ``normdt/norm_only`` is a livetime fraction (< 1) that **falls as the singles
  rate rises** -- NEMA bed 2 at 1.88 Mcps gives 0.9874, pediatric bed 4 at
  11.76 Mcps gives 0.9339.  A *correction* factor would have to rise with rate;
  a sensitivity has to fall, and it does.
* multiplying makes the data worse: the high-frequency ripple of ``(p-r-s)``
  along the detector axis grows on both exams when ``normdt`` is multiplied in
  (ped 20.2 -> 25.3 %, NEMA 24.7 -> 28.5 %) and does not when it is divided out.

SIRF consumes it directly: ``AcquisitionSensitivityModel`` built from an
``AcquisitionData`` treats it as bin efficiencies and *multiplies* the forward
projection by it.

Dead time depends on the singles rate, so neither of those two livetimes is
"the" livetime and neither should be compared against the static 1.10811 that
``PARAMS.md`` derives from ``sysGeometry`` -- see ``README.md`` §3b.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

GE_SHAPE = (288, 553, 381)          # view x plane x tangential, as GE stores it

#: vendor file -> (dtype, output stem).  Order is the order they get written.
TERMS = [("randoms.f32", "<f4", "randoms"),
         ("scatter.f32", "<f4", "scatter"),
         ("normdt.f32", "<f4", "normdt"),
         ("norm_only.f32", "<f4", "norm_only")]


def ge_to_stir(a: np.ndarray) -> np.ndarray:
    """``(view, plane, u)`` -> ``(tof=1, axial, view, tang)``.

    Transpose the plane axis to the front, then reverse the view axis.  Both
    steps are validated by :func:`verify` against independently decoded data.
    """
    return np.ascontiguousarray(a.transpose(1, 0, 2)[:, ::-1, :])[None]


def read_ge(path: str, dtype: str) -> np.ndarray:
    a = np.fromfile(path, dtype=dtype)
    want = int(np.prod(GE_SHAPE))
    if a.size != want:
        raise SystemExit(
            f"error: {path} holds {a.size:,} elements, expected {want:,} "
            f"({GE_SHAPE[0]} x {GE_SHAPE[1]} x {GE_SHAPE[2]}).  This converter "
            "only knows the D710 sinogram geometry.")
    return a.reshape(GE_SHAPE)


def template_data_file(hs: str) -> str:
    """The ``.s`` the template header points at."""
    with open(hs) as f:
        for line in f:
            m = re.match(r"\s*name of data file\s*:=\s*(.+?)\s*$", line, re.I)
            if m:
                return os.path.join(os.path.dirname(os.path.abspath(hs)), m.group(1))
    raise SystemExit(f"error: {hs} has no 'name of data file' key")


def verify(vendor: str, template: str) -> dict:
    """Re-prove the bin mapping on this exam's own data.

    ``prompts.u16`` is what GE's kernel loaded; the template ``.s`` is what
    custom_tool decoded from the same SINO* by a completely separate path.
    They must agree bit for bit once the mapping is applied.
    """
    pu = os.path.join(vendor, "prompts.u16")
    ts = template_data_file(template)
    if not os.path.exists(pu):
        raise SystemExit(f"error: no {pu} -- cannot verify the bin mapping, "
                         "so nothing is written.  Re-run estimate.py.")
    if not os.path.exists(ts):
        raise SystemExit(f"error: the template names {ts}, which does not exist")

    ge = read_ge(pu, "<u2")
    stir = ge_to_stir(ge)
    ref = np.fromfile(ts, dtype="<i2")
    if ref.size != stir.size:
        raise SystemExit(
            f"error: {ts} holds {ref.size:,} samples but the vendor array has "
            f"{stir.size:,} -- template and vendor run are not the same bed")
    ref = ref.reshape(stir.shape)
    exact = bool(np.array_equal(stir.astype("<i2"), ref))
    total = int(ge.sum(dtype=np.int64))
    if not exact:
        bad = int((stir.astype("<i2") != ref).sum())
        raise SystemExit(
            f"error: the bin mapping does not reproduce {ts}: {bad:,} of "
            f"{ref.size:,} bins differ.\n"
            "  Vendor total %d, decoded total %d.\n"
            "  Refusing to write correction sinograms whose bin order is "
            "unproven." % (total, int(ref.sum(dtype=np.int64))))
    return {"bit_exact_vs_decoded": True, "prompts": total}


def write_term(arr_ge: np.ndarray, stem: str, out: str, template: str) -> dict:
    """Write one ``.hs``/``.s`` pair, header cloned from the template."""
    data_name = stem + ".s"
    with open(template) as f:
        hdr = f.read()
    hdr = re.sub(r"(?im)^(\s*name of data file\s*:=).*$",
                 r"\1 " + data_name, hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number format\s*:=).*$", r"\1 float", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number of bytes per pixel\s*:=).*$", r"\1 4", hdr)
    with open(os.path.join(out, stem + ".hs"), "w") as f:
        f.write(hdr)

    a = ge_to_stir(arr_ge).astype("<f4", copy=False)
    a.tofile(os.path.join(out, data_name))
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sum": float(a.sum(dtype=np.float64)),
            "nonzero": int(np.count_nonzero(a))}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", required=True, help="an estimate.py output directory")
    ap.add_argument("--template", required=True,
                    help="the decoded bed .hs for the SAME bed "
                         "($D710_OUT/<case>/decoded/bed<n>.hs)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    vendor, out = os.path.abspath(args.vendor), os.path.abspath(args.out)
    template = os.path.abspath(args.template)
    for p in (vendor, template):
        if not os.path.exists(p):
            raise SystemExit(f"error: no such path: {p}")
    os.makedirs(out, exist_ok=True)

    print("== verifying the bin mapping against independently decoded data")
    checks = verify(vendor, template)
    print("   prompts.u16 -> STIR order reproduces %s bit-exact  (%s prompts)"
          % (os.path.basename(template_data_file(template)), f"{checks['prompts']:,}"))

    stats: dict[str, dict] = {}
    print("== converting")
    randoms = scatter = None
    for name, dtype, stem in TERMS:
        src = os.path.join(vendor, name)
        if not os.path.exists(src):
            print("   %-14s missing -- skipped" % name, file=sys.stderr)
            continue
        a = read_ge(src, dtype)
        if stem == "randoms":
            randoms = a
        elif stem == "scatter":
            scatter = a
        stats[stem] = write_term(a, stem, out, template)
        print("   %-14s -> %s.hs  sum %s"
              % (name, stem, f"{stats[stem]['sum']:,.0f}"))

    # b = randoms + scatter.  Summed here rather than in the notebook so the
    # term that actually enters set_background_term exists as one file.
    if randoms is not None and scatter is not None:
        stats["background"] = write_term(randoms + scatter, "background", out, template)
        print("   %-14s -> background.hs  sum %s"
              % ("randoms+scatter", f"{stats['background']['sum']:,.0f}"))

    meta = {"vendor": vendor, "template": template,
            "mapping": "stir[0, plane, 287 - ge_view, u] = ge[ge_view, plane, u]",
            "verified": checks,
            "sensitivity_term": "normdt (a sensitivity: divide data by it to correct)",
            "background_term": "background = randoms + scatter",
            "wcc_applied": False,
            "stats": stats}
    src_est = os.path.join(vendor, "estimate.json")
    if os.path.exists(src_est):
        with open(src_est) as f:
            meta["estimate"] = json.load(f)
    with open(os.path.join(out, "to_stir.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("\nwrote into %s:" % out)
    for stem in sorted(stats):
        print("   %s.hs / %s.s" % (stem, stem))

    # Exit non-zero on a partial set, the way estimate.py does.  Without this a
    # missing scatter.f32 leaves no background.hs, the notebook silently drops
    # the bed, and run_exam.sh still reports it as finished.
    missing = [stem for _n, _d, stem in TERMS if stem not in stats]
    if missing or "background" not in stats:
        print("\n!! incomplete: no %s -- rerun estimate.py for this bed"
              % ", ".join(missing + (["background"] if "background" not in stats
                                     else [])), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
