"""`d710 lowdose` — build a lower-dose copy of a decoded exam.

    d710 lowdose --case ped --drf 10                  -> ped_drf10
    d710 lowdose --case ped --drf 10 --mode randoms   -> randoms go as f^2
    d710 lowdose --case ped --split 2                 -> ped_r0, ped_r1 (Noise2Noise)
    d710 lowdose --case ped --drf 4 --replicates      -> 4 disjoint realisations

Needs the list-mode `decoded/bed<n>.lm.npy` of the source case, and its `work/bed<n>`
terms. The result is an ordinary case: `d710 osem`, `d710 lm` and `d710 export`
take it unchanged.
"""

from __future__ import annotations

import argparse

import numpy as np

from lm import events as ev
from lm import geom
from utils.paths import Case
from utils.paths import case as get_case

from . import thin, verify, write


def _rho(C, n, e, binmap):
    """`(rho per plane, rho per bin, flat bin index)` -- see `thin.rho_per_plane`."""
    b = ev.bins(e, binmap)
    nvt = binmap.n_view * binmap.n_tang
    p = np.bincount((b[b >= 0] // nvt), minlength=binmap.n_plane)
    r = verify.plane_sums(C.work_bed(n) / "randoms.s", binmap.n_plane, nvt)
    rho = thin.rho_per_plane(p, r, binmap)
    return rho[:: nvt], rho, b


def build(C, dst_name, beds, f, mode, seed, label=None, part=None):
    """Returns `(case, binmap, q)` -- `q[bed]` is the per-plane keep probability."""
    D = Case(dst_name, C.root.parent)
    _refuse_to_clobber(D, C.name, f, mode, label)
    D.root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, q = [], {}
    for n in beds:
        binmap = geom.BinMap(C.prompt(n))
        e = ev.load(C.decoded / f"bed{n}.lm.npy", mmap=False)
        q[n] = np.full(binmap.n_plane, f)
        if part is not None:
            k, of = part
            mask = thin.split(len(e), of, np.random.default_rng(seed)) == k
        elif mode == "randoms":
            rho_plane, rho, b = _rho(C, n, e, binmap)
            q[n] = f * (1 - rho_plane) + f * f * rho_plane
            mask = thin.keep(e, f, "randoms", rng, bins=b, rho=rho)
        else:
            mask = thin.keep(e, f, "uniform", rng)
        rows.append(write.bed(C, D, n, e, mask, binmap, f,
                              randoms_power=2 if mode == "randoms" else 1))
        r = rows[-1]
        print(f"  bed {n}: {len(e):,} -> {r['events']:,} events "
              f"({r['events'] / max(len(e), 1):.4f}), {r['prompts']:,} binned, "
              f"{r['dropped']:,} outside the sinogram")
        del e
    write.manifest(D, C.name, f, mode, seed, rows, replicate=label)
    return D, binmap, q


def _refuse_to_clobber(D, src, f, mode, label) -> None:
    """A case built with different settings must not be silently replaced.

    The destination name carries the mode, so this only fires on a real
    collision -- but when it does, the overwritten case would look perfectly
    valid and no measurement would ever flag it.
    """
    import json

    p = D.root / "lowdose.json"
    if not p.exists():
        return
    old = json.loads(p.read_text())
    now = {"source_case": src, "dose_fraction": f, "mode": mode,
           "replicate": label}
    diff = {k: (old.get(k), v) for k, v in now.items() if old.get(k) != v}
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
    ap.add_argument("--mode", choices=thin.MODES, default="uniform")
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
    if args.replicates and args.mode != "uniform":
        raise SystemExit("error: --replicates partitions the stream, which is "
                         "uniform by construction; drop --mode randoms")
    f = 1.0 / args.drf

    C = get_case(args.case, args.out)
    beds = args.beds or [n for n in C.beds()
                         if (C.decoded / f"bed{n}.lm.npy").exists()]
    if not beds:
        raise SystemExit(
            f"error: no bed of {C.name!r} has both a list-mode bed<n>.lm.npy and the "
            f"correction terms.\n  d710 decode --raw <SINO> --lists <petLists> "
            f"--case {C.name} --listmode --format npy")

    # The mode goes in the name: a randoms-aware run and a uniform one at the
    # same DRF are different datasets and must not land on top of each other.
    base = args.dst or (f"{C.name}_drf{args.drf:g}"
                        + ("" if args.mode == "uniform" else f"_{args.mode}"))
    jobs = ([(f"{base}_r{k}", (k, int(args.drf)), k) for k in range(int(args.drf))]
            if args.replicates else [(base, None, None)])

    for name, part, label in jobs:
        print(f"\n=== {C.name} -> {name}   f = {f:g} (DRF {args.drf:g}), "
              f"mode {args.mode}" + (f", replicate {label}" if part else ""))
        D, binmap, q = build(C, name, beds, f, args.mode, args.seed, label, part)
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
        print(f"\nwrote {D.root}\n  next:  d710 osem --case {name}"
              f"   (export applies K x {1 / f:g} on its own)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
