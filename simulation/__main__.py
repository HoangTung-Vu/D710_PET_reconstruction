"""Command-line entry point for `d710 simulate`.

    d710 simulate phantom --case C [--beds 1] [--ct X] [--pet X]
                          [--pet-units suv|bqml|relative] [--activity MBq@UTC]
    d710 simulate gate    --case C --beds 1 [--seconds T] [--chunk-seconds 10]
    d710 simulate pp      --case C --beds 1 [--gate-seed 1]
    d710 simulate compare --case C --beds 1 [--sims gate_s1 pp_s1]

`phantom` runs first: both methods read the activity and CT it puts on the bed
grid. `pp` takes its randoms, scatter and absolute scale from the `gate` run
of the same bed, so `gate` runs before it. See simulation/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.paths import Case
from utils.paths import case as get_case

from . import events_io as eio
from . import phantom as ph


def _default_input(C: Case, kind: str) -> Path:
    root = C.root.parent
    cands = {"ct": [C.export / f"{C.name}_ct.nii.gz",
                    root / "Pipeline reproduced" / C.name / f"{C.name}_ct.nii.gz"],
             "pet": [C.export / f"{C.name}_ge_suvbw.nii.gz",
                     root / "Pipeline reproduced" / C.name / f"{C.name}_ge_suvbw.nii.gz"]}[kind]
    for p in cands:
        if p.exists():
            return p
    raise SystemExit(f"error: no default --{kind} for {C.name}; looked at\n  "
                     + "\n  ".join(map(str, cands)) + f"\n  pass --{kind} <NIfTI or DICOM dir>")


def _beds(C: Case, beds) -> list[int]:
    have = C.decoded_beds()
    if not beds:
        return have
    missing = [b for b in beds if b not in have]
    if missing:
        raise SystemExit(f"error: {C.name} has no decoded bed {missing}; it has {have}")
    return beds


def cmd_phantom(C, a) -> int:
    ct = Path(a.ct) if a.ct else _default_input(C, "ct")
    pet = Path(a.pet) if a.pet else _default_input(C, "pet")
    units = a.pet_units or ("suv" if "suv" in pet.name.lower() else "bqml")
    act = ph.parse_activity(a.activity) if a.activity else None
    for bed in _beds(C, a.beds):
        print(f"phantom bed {bed}: CT {ct}\n  PET {pet} ({units})")
        ph.build(C, bed, ct, pet, units, ph.directory(eio.sim_root(C), bed),
                 activity=act, margin_mm=a.margin_mm, kvp=a.kvp)
    return 0


def cmd_gate(C, a) -> int:
    from .gate import driver
    from .gate.geometry import ShieldSpec

    dst = eio.sim_case(C, "gate", a.seed)
    shield = ShieldSpec(enabled=a.shield)
    for bed in _beds(C, a.beds):
        if not a.convert_only:
            driver.run_bed(C, dst, bed, a.seconds, a.chunk_seconds, a.singles_seconds,
                           a.threads, a.seed, shield, a.positron, a.ct_step)
        if not a.no_convert:
            driver.convert_bed(C, dst, bed, a.seconds, a.seed)
    print(f"-> {dst.root}")
    return 0


def cmd_pp(C, a) -> int:
    from . import pp

    dst = eio.sim_case(C, "pp", a.seed)
    gate = None if a.no_gate else eio.sim_case(C, "gate", a.gate_seed)
    for bed in _beds(C, a.beds):
        pp.run_bed(C, dst, bed, gate, a.seconds, a.seed, a.tof_bins, a.kappa)
    print(f"-> {dst.root}")
    return 0


def cmd_compare(C, a) -> int:
    from . import compare

    labels = a.sims or [p.name[len(C.name) + 5:] for p in sorted(C.root.parent.glob(
        f"{C.name}_sim_*_s*")) if (p / "decoded").is_dir()]
    if not labels:
        raise SystemExit(f"error: no simulated case of {C.name} yet")
    sims = {lab: Case(f"{C.name}_sim_{lab}", C.root.parent) for lab in labels}
    out = eio.sim_root(C) / "compare"
    for bed in _beds(C, a.beds):
        have = {k: S for k, S in sims.items() if bed in S.decoded_beds()}
        if not have:
            print(f"  bed {bed}: no simulation has it")
            continue
        print(f"compare bed {bed}: real vs {', '.join(have)}")
        compare.compare_bed(C, have, bed, out)
    print(f"-> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d710 simulate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--case", required=True, help="the real case the simulation copies")
        p.add_argument("--beds", type=int, nargs="*", default=None)
        p.add_argument("--out", default=None, help="output root (default $D710_OUT)")
        return p

    p = common(sub.add_parser("phantom", help="activity and CT on each bed's grid"))
    p.add_argument("--ct", help="CT: NIfTI (preferred) or DICOM folder")
    p.add_argument("--pet", help="PET: NIfTI (preferred) or DICOM folder")
    p.add_argument("--pet-units", choices=ph.UNITS, default=None,
                   help="default: suv if the file name says so, else bqml")
    p.add_argument("--activity", help="MBq@YYYYmmddHHMMSS (UTC), for --pet-units relative")
    p.add_argument("--margin-mm", type=float, default=ph.DEFAULT_MARGIN_MM,
                   help="activity beyond each end of the bed that GATE also sees")
    p.add_argument("--kvp", type=float, default=None, help="CT kVp (NIfTI has none; 120)")

    p = common(sub.add_parser("gate", help="GATE Monte Carlo, in resumable chunks"))
    p.add_argument("--seconds", type=float, default=None,
                   help="simulated time (default: the real frame -- ~42 h per bed on a 16-thread laptop)")
    p.add_argument("--chunk-seconds", type=float, default=1.0)
    p.add_argument("--singles-seconds", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shield", action="store_true",
                   help="add the assumed lead end shields (dimensions unknown; see gate/geometry.py)")
    p.add_argument("--positron", action="store_true",
                   help="e+ on the F-18 spectrum instead of back-to-back 511 keV pairs")
    p.add_argument("--ct-step", type=int, nargs=2, default=(3, 3),
                   metavar=("Z", "XY"), help="CT voxels averaged for the geometry")
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--no-convert", action="store_true",
                   help="run the missing chunks only; convert later with --convert-only")

    p = common(sub.add_parser("pp", help="parallelproj analytic model"))
    p.add_argument("--seconds", type=float, default=None, help="default: the real frame")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gate-seed", type=int, default=1,
                   help="the GATE run that supplies randoms, scatter and scale")
    p.add_argument("--no-gate", action="store_true", help="trues only; needs --kappa")
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--tof-bins", type=int, default=55, choices=(1, 5, 11, 55))

    p = common(sub.add_parser("compare", help="simulated against real raw data"))
    p.add_argument("--sims", nargs="*", help="labels such as gate_s1 pp_s1 (default: all)")

    a = ap.parse_args(argv)
    C = get_case(a.case, a.out)
    if not C.decoded.is_dir():
        raise SystemExit(f"error: {C.root} is not a decoded case")
    return {"phantom": cmd_phantom, "gate": cmd_gate, "pp": cmd_pp,
            "compare": cmd_compare}[a.cmd](C, a)


if __name__ == "__main__":
    sys.exit(main())
