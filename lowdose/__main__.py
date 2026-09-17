"""Command-line entry point for `d710 lowdose`."""

from __future__ import annotations

import argparse

import numpy as np

from lm import events as ev
from lm import geom
from utils.paths import Case
from utils.paths import case as get_case

from . import thin, verify, write


def _source_prompts(C, n, binmap, axis, b=None):
    nvt = binmap.n_view * binmap.n_tang
    if b is None:
        return (verify.plane_sums(C.decoded / f"bed{n}.s", binmap.n_plane, nvt, "<i2")
                if axis == "plane" else
                verify.plane_tang_sums(C.decoded / f"bed{n}.s", binmap.n_plane,
                                       binmap.n_tang, nvt, "<i2"))
    ok = b[b >= 0]
    if axis == "plane":
        return np.bincount(ok // nvt, minlength=binmap.n_plane)
    return np.bincount((ok // nvt) * binmap.n_tang + ok % binmap.n_tang,
                       minlength=binmap.n_plane * binmap.n_tang
                       ).reshape(binmap.n_plane, binmap.n_tang)


def _tof_profile(b, t_idx, binmap, n_tof):
    ok = b >= 0
    u = (b[ok] % binmap.n_tang).astype(np.int64)
    t = np.asarray(t_idx)[ok].astype(np.int64)
    return np.bincount(u * n_tof + t,
                       minlength=binmap.n_tang * n_tof
                       ).reshape(binmap.n_tang, n_tof).astype(np.float64)


def _rho(C, n, binmap, axis, b=None):
    nvt = binmap.n_view * binmap.n_tang
    p = _source_prompts(C, n, binmap, axis, b)
    rp = C.work_bed(n) / "randoms.s"
    r = (verify.plane_sums(rp, binmap.n_plane, nvt) if axis == "plane" else
         verify.plane_tang_sums(rp, binmap.n_plane, binmap.n_tang, nvt))
    rho = thin.rho_bins(p, r, binmap, axis)
    small = (rho[::nvt] if axis == "plane" else
             rho.reshape(binmap.n_plane, binmap.n_view, binmap.n_tang)[:, 0, :])
    return small, rho


def build(C, dst_name, beds, f, mode, seed, label=None, part=None,
          sinogram="derived", rho_axis="tangential", tof_rho="model",
          window="uniform"):
    """Build the thinned case; returns `(case, binmap, q)`."""
    mode = thin.canonical(mode)
    D = Case(dst_name, C.root.parent)
    _refuse_to_clobber(D, C.name, f, mode, label, sinogram,
                       rho_axis if mode == "low-dose" else None,
                       tof_rho if mode == "low-dose" else None, window)
    D.root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, q = [], {}
    for n in beds:
        binmap = geom.BinMap(C.prompt(n))
        q[n] = np.full(binmap.n_plane, f)
        e = mask = b = swap = None
        if sinogram != "binomial":
            e = ev.load(C.decoded / f"bed{n}.lm.npy", mmap=False)
            b, swap = ev.bins(e, binmap, with_swap=True)
        if window == "time":
            hdr = C.header(n)
            mask = thin.time_window(np.asarray(e["t_ms"]), f,
                                    hdr["frame_duration_ms"])
            lin, quad = thin.decay_scales(hdr["half_life_s"],
                                          hdr["frame_duration_ms"], f)
            small, _ = _rho(C, n, binmap, rho_axis, b)
            q[n] = lin * (1 - small) + quad * small
        elif part is not None:
            k, of = part
            mask = thin.split(len(e), of, np.random.default_rng(seed)) == k
        elif mode == "low-dose":
            small, rho = _rho(C, n, binmap, rho_axis, b)
            q[n] = f * (1 - small) + f * f * small
            tof = None
            if e is not None and tof_rho == "model" and rho_axis == "tangential":
                n_tof = geom.N_TOF_RAW
                t_idx = ev.tof_index(e, binmap, n_tof, swap)
                phi = _tof_profile(b, t_idx, binmap, n_tof)
                tof = (t_idx, thin.tof_rho_factor(phi, n_tof))
            if e is not None:
                mask = thin.keep(e, f, mode, rng, bins=b, rho=rho, tof=tof)
                qe, okm = thin.event_q(f, b, rho, tof)
                q[n] = thin.expectation(
                    qe, b[okm] // (binmap.n_view * binmap.n_tang), binmap.n_plane)
        elif e is not None:
            mask = thin.keep(e, f, mode, rng)
        rows.append(write.bed(C, D, n, e, mask, binmap, f, mode,
                              sinogram=sinogram, q=q[n], rng=rng, window=window))
        r = rows[-1]
        if r["events"] is None:
            print(f"  bed {n}: {r['prompts']:,} counts binned "
                  f"(sinogram thinned directly, no event table)")
        else:
            print(f"  bed {n}: {len(e):,} -> {r['events']:,} events "
                  f"({r['events'] / max(len(e), 1):.4f}), {r['prompts']:,} binned, "
                  f"{r['dropped']:,} outside the sinogram")
        if r["scatter_tof_profile"]:
            print(f"          scatter TOF profile carried: {r['scatter_tof_profile']}")
        del e
    write.manifest(D, C.name, f, mode, seed, rows, replicate=label,
                   sinogram=sinogram, rho_axis=rho_axis, tof_rho=tof_rho,
                   window=window)
    write.readme(D, C.name, f, mode, seed, sinogram, rho_axis, tof_rho, window)
    return D, binmap, q


def _refuse_to_clobber(D, src, f, mode, label, sinogram=None, rho_axis=None,
                       tof_rho=None, window=None) -> None:
    import json

    p = D.root / "lowdose.json"
    if not p.exists():
        return
    old = json.loads(p.read_text())
    try:
        old_mode = thin.canonical(old["mode"]) if old.get("mode") else None
    except ValueError:
        old_mode = old.get("mode")
    now = {"source_case": src, "dose_fraction": f, "mode": thin.canonical(mode),
           "replicate": label, "sinogram": sinogram, "rho_axis": rho_axis,
           "tof_rho": tof_rho, "window": window}
    was = {**old, "mode": old_mode}
    diff = {k: (was.get(k), v) for k, v in now.items() if was.get(k) != v}
    if diff:
        raise SystemExit(
            f"error: {D.root} already holds a low-dose case built differently:\n"
            + "".join(f"    {k}: {a!r} -> {b!r}\n" for k, (a, b) in diff.items())
            + "  pass --dst <name> to write elsewhere, or delete it first.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lowdose", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dst", help="destination case name (default <case>_drf<D>)")
    ap.add_argument("--drf", type=float, default=1.0,
                    help="dose reduction factor; f = 1/DRF")
    ap.add_argument("--mode", choices=thin.MODES + tuple(thin.ALIASES),
                    default="low-count", metavar="{low-count,low-dose}",
                    help="low-count: everything x f (a shorter scan). "
                         "low-dose: randoms x f^2 (less activity).")
    ap.add_argument("--sinogram", choices=write.SINOGRAM, default="derived",
                    help="derived: histogram the thinned events, so both paths "
                         "see the same data. binomial: thin the source sinogram "
                         "directly -- no event table, so `d710 lm` cannot run.")
    ap.add_argument("--window", choices=thin.WINDOWS, default="uniform",
                    help="uniform: keep a fraction f of every event, whatever time "
                         "it arrived -- a short scan at the frame's MID time. "
                         "time: keep the first f of the frame -- a short scan at "
                         "its START, with decay-weighted term factors")
    ap.add_argument("--tof-rho", choices=("model", "off"), default="model",
                    help="low-dose only: model rho's TOF dependence from the "
                         "measured TOF profile (randoms are flat in TOF, trues are "
                         "not). off reproduces the pre-2026-09-12 behaviour, which "
                         "keeps up to 7x too many counts in the deep TOF tails")
    ap.add_argument("--rho", choices=thin.RHO_AXES, default="tangential",
                    help="low-dose only: resolution the randoms fraction is "
                         "estimated at. tangential (per plane per tangential bin) "
                         "is the axis rho really varies along; plane is the old, "
                         "coarser choice and keeps ~6x too many counts in the tails")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beds", type=int, nargs="+")
    ap.add_argument("--replicates", action="store_true",
                    help="write DRF disjoint, mutually independent realisations "
                         "instead of one thinned copy")
    ap.add_argument("--split", type=int, metavar="K",
                    help="shorthand for --drf K --replicates")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)

    if args.split:
        args.drf, args.replicates = float(args.split), True
    if args.drf < 1:
        raise SystemExit("error: --drf must be >= 1")
    mode = thin.canonical(args.mode)
    if args.replicates and mode != "low-count":
        raise SystemExit("error: --replicates partitions the stream, which is "
                         "uniform by construction; drop --mode low-dose")
    if args.replicates and args.sinogram != "derived":
        raise SystemExit("error: --replicates splits the event stream, so it "
                         "needs --sinogram derived")
    if args.window == "time":
        if mode != "low-count":
            raise SystemExit(
                "error: --window time simulates a shorter SCAN, which is a "
                "low-count experiment.\n  Combining it with --mode low-dose would "
                "reduce the activity and the duration at once, and the term "
                "factors for that\n  are not the ones `thin.decay_scales` "
                "computes. Use --mode low-count.")
        if args.sinogram != "derived":
            raise SystemExit("error: --window time selects events by timestamp, "
                             "so it needs the event table: --sinogram derived")
        if args.replicates:
            raise SystemExit(
                "error: --replicates with --window time would give replicates that "
                "are different TIME windows -- \n  different decay, different "
                "motion, not exchangeable. Use the default --window uniform.")
    f = 1.0 / args.drf

    C = get_case(args.case, args.out)
    lm = args.sinogram == "derived"
    beds = args.beds or [n for n in C.beds()
                         if not lm or (C.decoded / f"bed{n}.lm.npy").exists()]
    if not beds:
        raise SystemExit(
            f"error: no bed of {C.name!r} has both a list-mode bed<n>.lm.npy and the "
            f"correction terms.\n  d710 decode --raw <SINO> --lists <petLists> "
            f"--case {C.name} --listmode --format npy\n"
            f"  or thin the sinogram alone: --sinogram binomial")

    if lm and not args.no_check:
        print("=== does the event table reproduce the decoded sinogram?")
        bad = verify.lm_matches_sinogram(C, beds)
        if bad:
            raise SystemExit(
                f"\nerror: beds {bad} disagree, so histogramming the thinned events "
                f"would\n  thin a different sinogram from the one on disk. Re-decode "
                f"the case, or\n  thin the sinogram directly with --sinogram binomial "
                f"(no event table).")

    base = args.dst or (f"{C.name}_drf{args.drf:g}"
                        + ("" if mode == "low-count" else "_lowdose"))
    jobs = ([(f"{base}_r{k}", (k, int(args.drf)), k) for k in range(int(args.drf))]
            if args.replicates else [(base, None, None)])

    for name, part, label in jobs:
        print(f"\n=== {C.name} -> {name}   f = {f:g} (DRF {args.drf:g}), "
              f"mode {mode}, window {args.window}, sinogram {args.sinogram}"
              + (f", rho per {args.rho}, TOF rho {args.tof_rho}"
                 if mode == "low-dose" else "")
              + (f", replicate {label}" if part else ""))
        if args.window == "time":
            h = C.header(beds[0])
            lin, quad = thin.decay_scales(h["half_life_s"],
                                          h["frame_duration_ms"], f)
            print(f"    frame {h['frame_duration_ms'] / 1000:g} s -> "
                  f"{h['frame_duration_ms'] * f / 1000:g} s;  decay-weighted "
                  f"scales: trues/scatter x {lin:.6f}, randoms x {quad:.6f} "
                  f"(nominal f = {f:g})")
        D, binmap, q = build(C, name, beds, f, mode, args.seed, label, part,
                             sinogram=args.sinogram, rho_axis=args.rho,
                             tof_rho=args.tof_rho, window=args.window)
        if not args.no_check:
            nvt = binmap.n_view * binmap.n_tang
            print()
            k = verify.binomial(C, D, beds, q, binmap.n_plane, nvt)
            print(f"\nplanes outside the 3 sd binomial band: {k} "
                  f"({'expected ~0.3 %' if k else 'none'})")
            print()
            bad = verify.invariants(D, beds, binmap.n_plane, nvt)
            print(f"\ninvariant violations: {bad}"
                  + ("" if not bad else "   <- do not reconstruct this"))
        print(f"\nwrote {D.root}\n  raw:   {D.raw_sim}"
              f"\n  next:  d710 osem --case {name}"
              f"   (export applies K x {1 / f:g} on its own)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
