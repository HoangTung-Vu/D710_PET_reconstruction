#!/usr/bin/env python3
"""Vendor `.f32` sinograms to STIR Interfile."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

GE_SHAPE = (288, 553, 381)

TERMS = [("randoms.f32", "<f4", "randoms"),
         ("scatter.f32", "<f4", "scatter"),
         ("normdt.f32", "<f4", "normdt"),
         ("norm_only.f32", "<f4", "norm_only")]

TOF_AXIAL_SHAPE = (4, 4)

TOF_AXIS = "stir"


def ge_to_stir(a: np.ndarray) -> np.ndarray:
    """`(view, plane, u)` to `(tof=1, axial, view, tang)`."""
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
    """The `.s` file the template header points at."""
    with open(hs) as f:
        for line in f:
            m = re.match(r"\s*name of data file\s*:=\s*(.+?)\s*$", line, re.I)
            if m:
                return os.path.join(os.path.dirname(os.path.abspath(hs)), m.group(1))
    raise SystemExit(f"error: {hs} has no 'name of data file' key")


def template_tof_bins(hs: str) -> int:
    """Timing positions the template declares; 1 for a non-TOF header."""
    with open(hs) as f:
        for line in f:
            m = re.match(r"\s*!?\s*matrix size\s*\[5\]\s*:=\s*(\d+)\s*$",
                         line, re.I)
            if m:
                return int(m.group(1))
    return 1


def strip_tof(hdr: str) -> str:
    """Turn a 5-D TOF projdata header into the 4-D one these terms need."""
    hdr = re.sub(r"(?im)^\s*!?\s*matrix axis label\s*\[5\]\s*:=.*\n", "", hdr)
    hdr = re.sub(r"(?im)^\s*!?\s*matrix size\s*\[5\]\s*:=.*\n", "", hdr)
    hdr = re.sub(r"(?im)^\s*TOF mashing factor\s*:=.*\n", "", hdr)
    hdr = re.sub(r"(?im)^\s*;\s*TOF axis\s*:=.*\n", "", hdr)
    return re.sub(r"(?im)^(\s*number of dimensions\s*:=).*$", r"\1 4", hdr)


def verify(vendor: str, template: str) -> dict:
    """Re-prove the bin mapping on this exam's own data."""
    pu = os.path.join(vendor, "prompts.u16")
    ts = template_data_file(template)
    if not os.path.exists(pu):
        raise SystemExit(f"error: no {pu} -- cannot verify the bin mapping, "
                         "so nothing is written.  Re-run estimate.py.")
    if not os.path.exists(ts):
        raise SystemExit(f"error: the template names {ts}, which does not exist")

    ge = read_ge(pu, "<u2")
    stir = ge_to_stir(ge).astype(np.int64)
    n_tof = template_tof_bins(template)
    ref = np.fromfile(ts, dtype="<i2")
    if ref.size != stir.size * n_tof:
        raise SystemExit(
            f"error: {ts} holds {ref.size:,} samples but the vendor array has "
            f"{stir.size:,} x {n_tof} TOF bins -- template and vendor run are "
            "not the same bed")
    ref = ref.reshape((n_tof,) + stir.shape[1:]).sum(axis=0, dtype=np.int64)
    ref = ref.reshape(stir.shape)
    exact = bool(np.array_equal(stir, ref))
    total = int(ge.sum(dtype=np.int64))
    if not exact:
        bad = int((stir != ref).sum())
        raise SystemExit(
            f"error: the bin mapping does not reproduce {ts}: {bad:,} of "
            f"{ref.size:,} bins differ.\n"
            "  Vendor total %d, decoded total %d.\n"
            "  Refusing to write correction sinograms whose bin order is "
            "unproven." % (total, int(ref.sum(dtype=np.int64))))
    return {"bit_exact_vs_decoded": True, "prompts": total,
            "template_tof_bins": n_tof}


def write_term(arr_ge: np.ndarray, stem: str, out: str, template: str) -> dict:
    """Write one `.hs`/`.s` pair, with the header cloned from the template."""
    data_name = stem + ".s"
    with open(template) as f:
        hdr = f.read()
    hdr = re.sub(r"(?im)^(\s*name of data file\s*:=).*$",
                 r"\1 " + data_name, hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number format\s*:=).*$", r"\1 float", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number of bytes per pixel\s*:=).*$", r"\1 4", hdr)
    hdr = strip_tof(hdr)
    a = ge_to_stir(arr_ge).astype("<f4", copy=False)
    a.tofile(os.path.join(out, data_name))
    with open(os.path.join(out, stem + ".hs"), "w") as f:
        f.write(hdr)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sum": float(a.sum(dtype=np.float64)),
            "nonzero": int(np.count_nonzero(a))}


def convert_scatter_tof(vendor: str, out: str, scatter_ge=None) -> dict | None:
    """`scatter_tof.f32` to `scatter_tof.npy`, or `None` outside a TOF run."""
    src = os.path.join(vendor, "scatter_tof.f32")
    meta = src + ".json"
    if not os.path.exists(src):
        return None
    if not os.path.exists(meta):
        raise SystemExit(f"error: {src} has no sidecar {meta}; without it the "
                         "ds_nu / numTOF_bins of this run are unknown and the "
                         "buffer cannot be reshaped")
    with open(meta) as f:
        m = json.load(f)
    nview = int(m.get("number_phi") or GE_SHAPE[0])
    ntof = int(m.get("numTOF_bins") or 0)
    dsnu = int(m.get("ds_nu") or 0)
    if not (ntof and dsnu):
        raise SystemExit(f"error: {meta} does not give numTOF_bins and ds_nu")

    a = np.fromfile(src, dtype="<f4")
    want = nview * ntof * dsnu * TOF_AXIAL_SHAPE[0] * TOF_AXIAL_SHAPE[1]
    if a.size != want:
        raise SystemExit(
            f"error: {src} holds {a.size:,} floats, expected {want:,} "
            f"({nview} view x {ntof} tof x {dsnu} ds_nu x "
            f"{TOF_AXIAL_SHAPE[0]} x {TOF_AXIAL_SHAPE[1]}).")
    a = a.reshape((nview, ntof, dsnu) + TOF_AXIAL_SHAPE).sum(axis=(3, 4))

    per_tof = a.sum(axis=(0, 2), dtype=np.float64)[::-1]
    total = float(per_tof.sum())
    stats = {"shape": [ntof, nview, dsnu],
             "axes": "tof(STIR order) x view(STIR order) x ds_nu",
             "tof_axis": TOF_AXIS,
             "num_tof_bins": ntof, "ds_nu": dsnu,
             "sum": total,
             "empty_tof_bins": int((per_tof == 0).sum()),
             "peak_tof_bin": int(per_tof.argmax()),
             "peak_over_mean": (float(per_tof.max() / per_tof.mean())
                                if total > 0 else None)}

    if scatter_ge is not None:
        x = a.sum(axis=1)
        edges = np.linspace(0, scatter_ge.shape[2], dsnu + 1).astype(int)
        y = np.add.reduceat(scatter_ge.sum(axis=1), edges[:-1], axis=1)

        def corr(u, v):
            return float(np.corrcoef(u.ravel(), v.ravel())[0, 1])

        c, c_tang = corr(x, y), corr(x, y[:, ::-1])
        stats["corr_vs_scatter"] = c
        stats["corr_vs_scatter_tangential_flipped"] = c_tang
        if c < 0.95 or c_tang > c:
            raise SystemExit(
                "error: scatter_tof.f32 summed over TOF correlates %.4f with "
                "scatter.f32 over\n"
                "  (view, tangential), against %.4f with the tangential axis "
                "flipped.  They come\n"
                "  from the same model and must agree, so the reshape or the "
                "axis order in this\n"
                "  file is wrong for this run.  Nothing written."
                % (c, c_tang))

    w = np.ascontiguousarray(a.transpose(1, 0, 2)[::-1, ::-1, :]).astype("<f4")
    np.save(os.path.join(out, "scatter_tof.npy"), w)
    return stats


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

    if randoms is not None and scatter is not None:
        stats["background"] = write_term(randoms + scatter, "background", out, template)
        print("   %-14s -> background.hs  sum %s"
              % ("randoms+scatter", f"{stats['background']['sum']:,.0f}"))

    tof = convert_scatter_tof(vendor, out, scatter_ge=scatter)
    if tof:
        print("   %-14s -> scatter_tof.npy  %s, peak TOF bin %d (%sx mean), "
              "corr vs scatter %.4f"
              % ("scatter_tof.f32", tuple(tof["shape"]), tof["peak_tof_bin"],
                 "%.2f" % tof["peak_over_mean"] if tof["peak_over_mean"]
                 else "n/a ", tof.get("corr_vs_scatter", float("nan"))))
        if tof["empty_tof_bins"]:
            print("   !! %d TOF bins are empty" % tof["empty_tof_bins"],
                  file=sys.stderr)
    else:
        print("   %-14s absent -- non-TOF estimate; OSEM will fall back to a "
              "measured TOF profile" % "scatter_tof.f32")

    meta = {"vendor": vendor, "template": template,
            "mapping": "stir[0, plane, 287 - ge_view, u] = ge[ge_view, plane, u]",
            "verified": checks,
            "scatter_tof": tof,
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

    missing = [stem for _n, _d, stem in TERMS if stem not in stats]
    if missing or "background" not in stats:
        print("\n!! incomplete: no %s -- rerun estimate.py for this bed"
              % ", ".join(missing + (["background"] if "background" not in stats
                                     else [])), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
