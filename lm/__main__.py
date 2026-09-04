"""`d710 lm` — list-mode reconstruction, and the two checks it must pass first.

    d710 lm check    --case ped --bed 1
    d710 lm tofcheck --case ped --bed 1
    d710 lm recon    --case ped [--beds 1 2 3] [--tof-bins 55]

`check` histograms the events back through the bin map and demands the result be
**bit-identical** to `decoded/bed<n>.s`, which the vendor decoder produced by a
completely separate path. `tofcheck` measures which way parallelproj wants the
TOF axis — the one thing about this path that no invariant can catch.
"""

from __future__ import annotations

import argparse

import numpy as np

from utils.paths import case as get_case
from utils.scanner import DR_MM, PLANE_MM, XY

from . import events as ev
from . import geom, interfile, recon


def _events_path(C, bed: int):
    p = C.decoded / f"bed{bed}.lm.npy"
    if not p.exists():
        raise SystemExit(
            f"error: no {p}\n"
            f"  decode the list mode first:\n"
            f"    d710 decode --raw <SINO dir> --lists <petLists dir> "
            f"--case {C.name} --listmode --format npy")
    return p


def _template_counts(C, bed: int):
    """`(array, n_tof)` of the decoded prompts, in file order."""
    h = interfile.Header(C.prompt(bed))
    return np.fromfile(h.data_file(), "<i2"), h.n_tof


def cmd_check(C, args) -> int:
    """Histogram the events back and demand `decoded/bed<n>.s` bit for bit.

    The TOF mashing comes from the prompts header, never from a flag: the file
    being compared against is the only thing that can decide it.
    """
    binmap = geom.BinMap(C.prompt(args.bed))
    e = ev.load(_events_path(C, args.bed))
    ref, n_tof = _template_counts(C, args.bed)

    h, dropped = ev.histogram(e, binmap, n_tof)
    got, want = int(h.sum(dtype=np.int64)), int(ref.sum(dtype=np.int64))
    print(f"events {len(e):,}   binned {got:,}   dropped {dropped:,}")
    print(f"decoded bed{args.bed}.s: {want:,} counts, {n_tof} TOF bins")

    if ref.size != h.size:
        print(f"!! the decoded sinogram has {ref.size:,} samples, the histogram "
              f"{h.size:,} — the header and the data file disagree")
        return 1
    bad = int((ref.reshape(h.shape) != h).sum())
    print(f"bit-exact vs decoded: {'YES' if not bad else f'NO ({bad:,} bins differ)'}")
    return 0 if not bad else 1


def cmd_tofcheck(C, args) -> int:
    """Which TOF sign parallelproj wants, measured against a non-TOF image.

    Same idea as `tools/tof_direction.py`: reconstruct without TOF, so the
    reference cannot be biased by the axis under test, then score both signs by
    the Poisson log-likelihood of the events. The right sign puts the activity on
    the half of the LOR the events say it is on.
    """
    import torch
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood

    from . import terms

    binmap = geom.BinMap(C.prompt(args.bed))
    e = ev.load(_events_path(C, args.bed))
    if args.n_events and len(e) > args.n_events:
        e = np.asarray(e[:: len(e) // args.n_events])[: args.n_events]
    n_tof = args.tof_bins

    sm, add_tof, _ = recon.system_matrix(C, args.bed, e, binmap, n_tof, args.xy,
                                         args.psf, 1, args.n_splits)
    keep, w, add1 = terms.event_terms(C, args.bed, e, binmap, 1)

    sm.TOF = False
    x0 = OSEM(PoissonLogLikelihood(sm, additive_term=torch.from_numpy(add1)))(
        n_iters=args.iters, n_subsets=args.subsets)
    sm.TOF = True

    # Events far from the centre of the TOF axis are where the sign actually
    # bites; the middle bin is the same either way, so it only dilutes.
    far = np.abs(np.asarray(e["tof_bin"])[keep]) >= args.tof_far

    t = sm.proj_meta.detector_ids[:, 2].clone()
    ll = {}
    for sign, ids in ((+1, t), (-1, (n_tof - 1) - t)):
        sm.proj_meta.detector_ids[:, 2] = ids
        p = torch.log(torch.clamp(sm.forward(x0).cpu() + add_tof, min=1e-12))
        ll[sign] = p.numpy()
    sm.proj_meta.detector_ids[:, 2] = t

    d = ll[+1] - ll[-1]
    print(f"\nlog(H_tof x0 + a), {len(d):,} events "
          f"({int(far.sum()):,} with |tof_bin| >= {args.tof_far}):")
    for sign in (+1, -1):
        print(f"  tof_sign {sign:+d} : all {ll[sign].mean():+.6f}   "
              f"far {ll[sign][far].mean():+.6f}")
    for name, sel in (("all", slice(None)), ("far", far)):
        m, s = float(d[sel].mean()), float(d[sel].std() / np.sqrt(d[sel].size))
        print(f"  difference ({name}): {m:+.6f} +- {s:.6f}   "
              f"-> tof_sign {'+1' if m > 0 else '-1'} by {abs(m) / max(s, 1e-12):.0f} sd")
    print("\nUse it as:  d710 lm recon --tof-sign <n>")
    return 0


def _bed_key(C, n: int, args, tof_scatter) -> str:
    """Fingerprint of everything that changes bed `n`. A resume that reuses a bed
    made with different settings is worse than no cache: nothing would flag it."""
    import hashlib

    parts = []
    for p in (C.prompt(n), C.decoded / f"bed{n}.lm.npy",
              C.work_bed(n) / "attn.hs"):
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    parts += [f"{k}={getattr(args, k)!r}" for k in
              ("tof_bins", "xy", "subsets", "iters", "psf", "tof_sign",
               "n_splits", "beta")]
    if tof_scatter is not None:
        parts.append(f"tof_scatter={float(np.sum(tof_scatter)):.12e}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def cmd_recon(C, args) -> int:
    from osem import stitch
    from utils import terms

    beds = args.beds or [n for n in C.beds()
                         if (C.decoded / f"bed{n}.lm.npy").exists()]
    if not beds:
        raise SystemExit(f"error: no bed of {C.name!r} has both a list-mode "
                         f"bed<n>.lm.npy and the correction terms")
    # Attenuation is the one term SIRF has to build, and SIRF lives in the other
    # runtime. It is required here, never built here.
    missing = [n for n in beds if not (C.work_bed(n) / "attn.hs").exists()]
    if missing:
        raise SystemExit(
            f"error: bed {missing} has no work/bed<n>/attn.hs.\n"
            f"  build it in the SIRF runtime first:\n"
            f"    ./d710_isolate_stir.sh attn --case {C.name}")
    # Before anything runs, so an explicit --beds with no event table fails now
    # and with the message that says how to make one.
    npy = {n: _events_path(C, n) for n in beds}
    ct_dir = args.ct or terms.ct_dir(C, beds[0])
    print(f"case {C.name!r}: {len(beds)} beds  ->  {beds}\n")

    tof_scatter = np.load(args.tof_scatter) if args.tof_scatter else None
    img, sens = {}, {}
    for n in beds:
        p, key = C.work_bed(n) / "lm.npz", _bed_key(C, n, args, tof_scatter)
        if args.resume and p.exists():
            z = np.load(p, allow_pickle=False)
            # An npz from before the key existed has no settings to match, so it
            # is a mismatch -- not a KeyError.
            if "key" in z.files and str(z["key"]) == key:
                img[n], sens[n] = z["img"], z["sens"]
                print(f"\n=== bed {n}: reused {p.name}")
                continue
            print(f"\n=== bed {n}: {p.name} was made with different settings "
                  f"-- reconstructing again")
        else:
            print(f"\n=== bed {n}")
        img[n], sens[n] = recon.reconstruct(
            C, n, npy[n], n_tof=args.tof_bins, xy=args.xy,
            n_sub=args.subsets, n_it=args.iters, psf=args.psf,
            tof_sign=args.tof_sign, n_splits=args.n_splits, beta=args.beta,
            tof_scatter=tof_scatter)
        np.savez_compressed(p, img=img[n], sens=sens[n], key=key, bed=n,
                            n_tof=args.tof_bins, beta=args.beta)

    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    stitch.overlap_report(C, beds, img, factors)
    vox = [PLANE_MM, DR_MM, DR_MM]      # (z, y, x) mm
    vol = stitch.post_filter(vol, vox, args.post_filter, args.z_ratio)

    out = C.root / "recon_lm.npz"
    np.savez_compressed(out, vol=vol, z0=z0, vox=np.array(vox),
                        beds=np.array(beds),
                        decay=np.array([factors[n] for n in beds]),
                        n_subsets=args.subsets, n_iterations=args.iters,
                        ct=ct_dir, n_tof=args.tof_bins, beta=args.beta,
                        tangential_lors=0,
                        post_filter_fwhm_mm=args.post_filter,
                        post_filter_z_ratio=args.z_ratio,
                        tof_scatter=(args.tof_scatter or "per-bed"))
    print(f"\nwrote {out}  ({vol.shape}, count/voxel referred to the injection time)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "tofcheck", "recon"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--out")
    ap.add_argument("--bed", type=int, default=1)
    ap.add_argument("--beds", type=int, nargs="+")
    ap.add_argument("--ct")
    ap.add_argument("--tof-bins", type=int, default=geom.N_TOF_RAW,
                    help="TOF bins to reconstruct with (must divide 55); "
                         "1 disables TOF")
    ap.add_argument("--tof-sign", type=int, choices=(1, -1), default=1,
                    help="recon: direction of the TOF axis in the EVENT's frame; "
                         "measure it with `tofcheck`.  `check` derives its own "
                         "per-event direction from the bin, so it ignores this")
    ap.add_argument("--tof-far", type=int, default=8, metavar="N",
                    help="tofcheck: score events with |tof_bin| >= N separately")
    ap.add_argument("--tof-scatter", metavar="PROF.npy")
    ap.add_argument("--xy", type=int, default=XY,
                    help=f"transaxial matrix size at {DR_MM} mm voxels "
                         f"(default %(default)d -> the same grid `d710 osem` "
                         f"builds, in either SIRF runtime)")
    ap.add_argument("--iters", type=int, default=recon.N_ITERATIONS)
    ap.add_argument("--subsets", type=int, default=recon.N_SUBSETS)
    ap.add_argument("--psf", type=float, default=recon.PSF_MM)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="> 0 switches OSEM for BSREM with a relative-difference prior")
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="recon: reuse any bed in work/bed<n>/lm.npz whose "
                         "settings match this run exactly; a bed that does not "
                         "match is reconstructed again, never reused silently")
    ap.add_argument("--n-events", type=int, default=4_000_000,
                    help="tofcheck: events to score (0 = all)")
    from osem import stitch

    ap.add_argument("--post-filter", type=float, default=stitch.POST_FILTER_FWHM_MM)
    ap.add_argument("--z-ratio", type=float, default=stitch.POST_FILTER_Z_RATIO)
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    if args.tof_bins != 1 and geom.N_TOF_RAW % args.tof_bins:
        raise SystemExit(f"error: --tof-bins {args.tof_bins} does not divide "
                         f"{geom.N_TOF_RAW}")
    return {"check": cmd_check, "tofcheck": cmd_tofcheck,
            "recon": cmd_recon}[args.cmd](C, args)


if __name__ == "__main__":
    raise SystemExit(main())
