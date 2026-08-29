#!/usr/bin/env python3
"""Which way round is our TOF axis? The one TOF check nothing else performs.

    conda activate petct_reconstruction
    PYTHONPATH=<D710> python3 tools/tof_direction.py \\
        --nontof-case ped --tof-case pedtof5 --bed 1

STIR numbers timing positions **signed and centred**, and `ProjDataInfo::get_k`
turns a timing position into a signed displacement along the LOR — measured on
this scanner, `timing_pos -2 -> -294.31 mm` and `+2 -> +294.31 mm`. So the TOF
axis decides *which half of the LOR* an event came from. The decoder writes GE's
bins straight through in file order, and nothing in the RDF header states which
end GE starts from.

**Why no other check catches this.** Prompts and scatter travel down the same
axis, so they stay consistent with each other whichever way it points. Counts
are conserved, `Sum(p) >= Sum(r)` holds, no TOF bin goes negative, and every
test in `tests/test_pipeline_data.py` passes. Only the image is wrong — and it
is wrong in the specific way that makes TOF worse than no TOF at all.

**The method.** Reconstruct the bed WITHOUT TOF, so the reference image cannot
be biased by the very axis under test. Forward project it through STIR's own TOF
model, and see which orientation of the measured data it matches:

    corr( G_tof . x_nonTOF ,  y_tof )   vs   corr( G_tof . x_nonTOF , flip(y_tof) )

STIR supplies both the geometry and the TOF binning, so agreement means the
whole chain agrees. One forward projection, not one reconstruction.

Two statistics, because the raw one is weak on its own: the sinogram runs at
~0.06 counts/bin, so per-bin correlation is dominated by Poisson noise. The
sharp statistic is the **TOF centroid per (view, tangential)** — a signed
number that is precisely what the axis direction controls, and summing over the
axial axis first removes most of the noise. The analysis is also repeated on the
highest-count LORs alone, where trues dominate scatter, so that the scatter's
own TOF distribution cannot be what is driving the answer.

Result on ped bed 1, 2026-08-27 — the axis was **mirrored**:

    top 10 % of LORs   corr(sinogram)  corr(centroid)  mean |d centroid|
    as written             +0.734          -0.977          0.541 bins
    flipped                +0.983          +0.977          0.055 bins

Fixed 2026-08-29 by reversing the axis where it is written, in
`gerdf.cli._tof_to_stir` (prompts) and `vendor/to_stir.py` (the scatter
weights). On data decoded since, the two columns are swapped and "AS WRITTEN"
must win — that is what this tool is for now: a regression check, run on any
case before trusting a TOF image, and after any change to the axis convention.
Data decoded BEFORE that date is refused outright by `utils.terms.load`, which
is why this script cannot be pointed at it any more.
"""
from __future__ import annotations

import argparse

import numpy as np

from osem import recon
from utils import attn, sirf_env, terms
from utils.paths import case as get_case


def corr(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def centroid(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """TOF centroid per (view, tangential), over one SHARED mask.

    The mask has to be shared between the two arrays being compared, or the two
    centroid maps cover different LORs and cannot be correlated at all.
    """
    w = np.clip(v, 0, None)[:, mask]
    tot = w.sum(axis=0)
    c = np.arange(v.shape[0])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (w * c).sum(axis=0) / np.where(tot > 0, tot, 1),
                        np.nan)


def project(nontof_case, tof_case, bed: int, xy: int):
    """`(pred, meas, n_tof)` reduced over the axial axis -> `(tof, view, tang)`.

    The axial axis carries no information about the TOF direction, and dropping
    it turns two 1.2 GB arrays into two 2 MB ones.
    """
    sirf_env.setup(nontof_case)

    print("== reconstructing bed %d WITHOUT TOF (the reference object)" % bed,
          flush=True)
    y0, x0 = recon.image_grid(nontof_case, bed, xy=xy)
    if int(y0.dimensions()[0]) != 1:
        raise SystemExit(
            "error: case %r has TOF prompts, so it cannot be the reference.\n"
            "  --nontof-case must name a case decoded with --no-tof."
            % nontof_case.name)
    at = attn.Attenuation(nontof_case, terms.ct_dir(nontof_case, bed), x0, y0,
                          verbose=False)
    af = at.af(bed)
    img, _ = recon.reconstruct(nontof_case, bed, af, x0, n_sub=12, n_it=1,
                               xy=xy, projector="ray")
    x = x0.get_uniform_copy(0)
    x.fill(img)
    del y0, x0

    print("== TOF model on the same bed", flush=True)
    sirf_env.setup(tof_case)
    objs, A = terms.load(tof_case, bed, af=af, lean=True)
    n_tof = int(objs["prompts"].dimensions()[0])
    if n_tof < 2:
        raise SystemExit(
            "error: case %r has no TOF axis; --tof-case must name a case "
            "decoded with --tof." % tof_case.name)
    S = objs["prompts"].get_uniform_copy(0)
    S.fill(A["sensitivity"])
    del A
    am = recon.acquisition_model(objs, S, x, projector="parallelproj")

    print("== forward projecting (%d TOF bins)" % n_tof, flush=True)
    bg = objs["background"].as_array()
    # forward() returns S.(Gx) + b; subtracting b leaves the model's TRUES, which
    # is what the measured prompts minus background is.
    pred = am.forward(x).as_array() - bg
    meas = objs["prompts"].as_array().astype(np.float32) - bg
    return (pred.sum(axis=1, dtype=np.float64),
            meas.sum(axis=1, dtype=np.float64), n_tof)


def report(P, M, n_tof, out=print) -> bool:
    """Print both orientations at several count thresholds. True if flipped wins."""
    tot = np.clip(M, 0, None).sum(axis=0)
    live = tot > 0

    out("\n%-34s %14s %14s" % ("LOR selection", "as written", "flipped"))
    for q, label in ((0, "all"), (50, "top 50%"), (80, "top 20%"),
                     (90, "top 10%"), (99, "top 1%")):
        m = live if q == 0 else (tot >= np.percentile(tot[live], q))
        out("%-34s %+14.4f %+14.4f"
            % ("%s by counts (n=%d)" % (label, m.sum()),
               corr(P[:, m], M[:, m]), corr(P[:, m], M[::-1][:, m])))

    # The sharp statistic, on the LORs where trues dominate scatter.
    m = tot >= np.percentile(tot[live], 90)
    cp, cm = centroid(P, m), centroid(M, m)
    ok = np.isfinite(cp) & np.isfinite(cm)
    d_same = np.abs(cp[ok] - cm[ok]).mean()
    d_flip = np.abs(cp[ok] - ((n_tof - 1) - cm[ok])).mean()
    out("\nTOF centroid, top 10%% of LORs (%d):" % ok.sum())
    out("   corr(model, data)        %+.4f" % corr(cp[ok], cm[ok]))
    out("   mean |d centroid|        as written %.3f bins, flipped %.3f bins"
        % (d_same, d_flip))

    flipped_wins = d_flip < d_same
    out("")
    if flipped_wins:
        out("=> FLIPPED wins: the TOF axis is MIRRORED -- a REGRESSION.")
        out("   Both reversals must be in place, and they are one change:")
        out("     1. custom_tool/gerdf/cli.py:_tof_to_stir -- the prompts;")
        out("        it lives in d710:full, so check the IMAGE, not just the")
        out("        host file (docker build -t d710:full -f D710/Dockerfile .)")
        out("     2. vendor/to_stir.py:convert_scatter_tof -- the scatter")
        out("        weights, which must stay aligned with the prompts")
        out("   See TOF_SCATTER_REVERSE.md section 7.")
    else:
        out("=> AS WRITTEN wins: the TOF axis is correct.")
    out("   margin in centroid: %.3f bins (a small margin means inconclusive)"
        % abs(d_flip - d_same))
    return flipped_wins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nontof-case", required=True,
                    help="a case decoded WITHOUT TOF -- the reference image")
    ap.add_argument("--tof-case", required=True,
                    help="the SAME bed of the same exam, decoded WITH TOF")
    ap.add_argument("--bed", type=int, default=1)
    ap.add_argument("--out", help="output root; defaults to $D710_OUT")
    ap.add_argument("--xy", type=int, default=128,
                    help="transverse voxels (default %(default)d -- small on "
                         "purpose, this is a direction test, not an image)")
    ap.add_argument("--save", help="write the reduced (tof, view, tang) arrays "
                                   "here as .npz, to re-analyse without "
                                   "projecting again")
    args = ap.parse_args(argv)

    P, M, n_tof = project(get_case(args.nontof_case, args.out),
                          get_case(args.tof_case, args.out),
                          args.bed, args.xy)
    if args.save:
        np.savez_compressed(args.save, pred=P, meas=M, n_tof=n_tof)
        print("   saved %s" % args.save)
    report(P, M, n_tof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
