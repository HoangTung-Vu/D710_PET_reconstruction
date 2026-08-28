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
    scatter_tof.npy   TOF distribution of the scatter (TOF estimates only)

``scatter_tof.npy`` is deliberately **not** an Interfile.  It is GE's own coarse
grid -- ``(55 TOF, 288 view, 43 downsampled tangential)``, 2.7 MB -- and it is a
*shape*, not a sinogram: the amplitude lives in ``scatter.hs``.  Written out at
full resolution it would be 13.4 GB per bed, which is exactly why GE does not
store it either; ``utils.terms.expand_to_tof`` upsamples it per bed at load
time, the same place the background gets assembled.  The view axis is reversed
into STIR order here so nothing downstream has to know about GE's numbering, and
the two length-4 axes of the vendor buffer are summed away -- measured on ped
bed 1, keeping them moves the TOF centroid by a median 0.45 bins (40 ps, 6 mm).

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

#: The two trailing axes of ``scatter_tof.f32``: GE's downsampled axial
#: sampling, 4 x 4.  They are summed away -- see the module docstring.
TOF_AXIAL_SHAPE = (4, 4)


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


def template_tof_bins(hs: str) -> int:
    """Timing positions the template declares; 1 for a non-TOF header.

    Every correction term written here is non-TOF -- norm, dead time,
    attenuation and randoms genuinely do not depend on arrival time, and the
    scatter's time axis travels separately in ``scatter_tof.npy``.  So a TOF
    template has to be handled in two places rather than followed blindly: the
    prompts check below collapses it, and ``strip_tof`` takes the TOF keys back
    out of the cloned header.
    """
    with open(hs) as f:
        for line in f:
            m = re.match(r"\s*!?\s*matrix size\s*\[5\]\s*:=\s*(\d+)\s*$",
                         line, re.I)
            if m:
                return int(m.group(1))
    return 1


def strip_tof(hdr: str) -> str:
    """Turn a 5-D TOF projdata header into the 4-D one these terms need.

    The scanner block keeps its TOF keys -- they describe the hardware, and STIR
    is happy to read non-TOF data from a TOF-capable scanner.  What has to go is
    the *data* description: axis 5, its size, and the mashing factor.
    """
    hdr = re.sub(r"(?im)^\s*!?\s*matrix axis label\s*\[5\]\s*:=.*\n", "", hdr)
    hdr = re.sub(r"(?im)^\s*!?\s*matrix size\s*\[5\]\s*:=.*\n", "", hdr)
    hdr = re.sub(r"(?im)^\s*TOF mashing factor\s*:=.*\n", "", hdr)
    return re.sub(r"(?im)^(\s*number of dimensions\s*:=).*$", r"\1 4", hdr)


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
    stir = ge_to_stir(ge).astype(np.int64)
    n_tof = template_tof_bins(template)
    ref = np.fromfile(ts, dtype="<i2")
    if ref.size != stir.size * n_tof:
        raise SystemExit(
            f"error: {ts} holds {ref.size:,} samples but the vendor array has "
            f"{stir.size:,} x {n_tof} TOF bins -- template and vendor run are "
            "not the same bed")
    # A TOF template is the SAME acquisition with the time axis kept, and
    # mashing preserves counts exactly, so summing it back down is not an
    # approximation -- the comparison stays bit-exact.  int64 because 11 int16
    # bins of a busy LOR do not fit an int16 once added.
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
    """Write one ``.hs``/``.s`` pair, header cloned from the template."""
    data_name = stem + ".s"
    with open(template) as f:
        hdr = f.read()
    hdr = re.sub(r"(?im)^(\s*name of data file\s*:=).*$",
                 r"\1 " + data_name, hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number format\s*:=).*$", r"\1 float", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number of bytes per pixel\s*:=).*$", r"\1 4", hdr)
    # Every term written here is non-TOF, whatever the prompts are.
    hdr = strip_tof(hdr)
    with open(os.path.join(out, stem + ".hs"), "w") as f:
        f.write(hdr)

    a = ge_to_stir(arr_ge).astype("<f4", copy=False)
    a.tofile(os.path.join(out, data_name))
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sum": float(a.sum(dtype=np.float64)),
            "nonzero": int(np.count_nonzero(a))}


def convert_scatter_tof(vendor: str, out: str, scatter_ge=None) -> dict | None:
    """``scatter_tof.f32`` -> ``scatter_tof.npy``, or ``None`` if not a TOF run.

    The vendor buffer is one CViewBuffer of ``number_phi`` views, each holding
    ``[tof][ds_nu][4][4]`` floats in C order.  That layout is not a guess: the
    last thing ``CScatterFully3dModel::PhiUpsampleTofScatter`` does is

        permute_41253(buf, 4, ds_nu, number_phi, 4, numTOF_bins, m_pScatterTOF)

    a column-major 5-D permutation, so the output runs (slowest first)
    ``number_phi, numTOF_bins, ds_nu, 4, 4``.  The sidecar ``estimate.py`` wrote
    carries ``ds_nu`` and ``numTOF_bins``, so the shape is read rather than
    assumed and a scanner with different downsampling still works.

    Returns the stats dict, and writes ``(tof, view, ds_nu)`` float32 in **STIR
    view order** -- the same ``287 - ge_view`` reversal every other term gets.

    Given ``scatter_ge`` (the full ``scatter.f32``, still in GE order) the two
    are checked against each other: summed over TOF, the compact buffer is the
    same distribution as the full one, so their ``(view, tangential)`` maps must
    correlate. That is what catches the mistake this conversion is actually
    exposed to -- reversing the view axis of one term and not the other, which
    is invisible in every per-term statistic and ruins the reconstruction.
    """
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

    # A TOF bin with no scatter anywhere would make the per-LOR normalisation in
    # terms.py divide by zero, and an all-zero buffer means the run was not
    # really TOF -- report both rather than let either pass silently.
    per_tof = a.sum(axis=(0, 2), dtype=np.float64)
    total = float(per_tof.sum())
    stats = {"shape": [ntof, nview, dsnu],
             "axes": "tof x view(STIR order) x ds_nu",
             "num_tof_bins": ntof, "ds_nu": dsnu,
             "sum": total,
             "empty_tof_bins": int((per_tof == 0).sum()),
             "peak_tof_bin": int(per_tof.argmax()),
             # None rather than a NaN from 0/0: an all-zero buffer means the
             # estimate did not really run in TOF mode, and `terms` says so with
             # a usable message. A NaN here would just look like a shape bug.
             "peak_over_mean": (float(per_tof.max() / per_tof.mean())
                                if total > 0 else None)}

    if scatter_ge is not None:
        # Both maps in GE view order, before the reversal below.
        #   x: (view, ds_nu) from the compact buffer, summed over TOF
        #   y: the same map from the full scatter, its tangential axis binned
        #      down to ds_nu so neither side has to be interpolated
        x = a.sum(axis=1)
        edges = np.linspace(0, scatter_ge.shape[2], dsnu + 1).astype(int)
        y = np.add.reduceat(scatter_ge.sum(axis=1), edges[:-1], axis=1)

        def corr(u, v):
            return float(np.corrcoef(u.ravel(), v.ravel())[0, 1])

        c, c_tang = corr(x, y), corr(x, y[:, ::-1])
        stats["corr_vs_scatter"] = c
        stats["corr_vs_scatter_tangential_flipped"] = c_tang
        # What this catches: a wrong reshape, and a tangential axis running the
        # wrong way -- measured on ped bed 1, 0.9949 the right way round against
        # 0.8592 flipped.
        #
        # What it does NOT catch, said plainly: anything that only permutes the
        # VIEW axis, including its direction.  Scatter is nearly symmetric in
        # view, so the reversed orientation still scores 0.9895 here and no
        # threshold separates them.  The view reversal is
        # `ge_to_stir`'s, the same one `verify` above proves bit-exact against
        # independently decoded prompts, and it is applied to this buffer by the
        # single transpose below -- so it is right for the same reason the other
        # terms are, not because this check says so.
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

    # (view, tof, ds_nu) -> (tof, view, ds_nu), views reversed into STIR order.
    # Written last, so a failed check leaves no file for the next step to trust.
    w = np.ascontiguousarray(a.transpose(1, 0, 2)[:, ::-1, :]).astype("<f4")
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

    # b = randoms + scatter.  Summed here rather than in the notebook so the
    # term that actually enters set_background_term exists as one file.
    if randoms is not None and scatter is not None:
        stats["background"] = write_term(randoms + scatter, "background", out, template)
        print("   %-14s -> background.hs  sum %s"
              % ("randoms+scatter", f"{stats['background']['sum']:,.0f}"))

    # `scatter` is still in GE order here, which is what the cross-check wants.
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

    # Exit non-zero on a partial set, the way estimate.py does.  Without this a
    # missing scatter.f32 leaves no background.hs, the notebook silently drops
    # the bed, and `d710 exam` still reports it as finished.
    missing = [stem for _n, _d, stem in TERMS if stem not in stats]
    if missing or "background" not in stats:
        print("\n!! incomplete: no %s -- rerun estimate.py for this bed"
              % ", ".join(missing + (["background"] if "background" not in stats
                                     else [])), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
