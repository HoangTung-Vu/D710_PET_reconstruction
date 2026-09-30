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


def resolve(name: str, out=None) -> Case:
    C = get_case(name, out)
    if not C.decoded.is_dir() and (C.root / eio.VIRTUAL / "decoded").is_dir():
        return Case(eio.VIRTUAL, C.root)
    return C


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

    labels = a.sims or [p.name[len("sim_"):] for p in sorted(eio.owner(C).glob("sim_*_s*"))
                        if (p / "decoded").is_dir()]
    if not labels:
        raise SystemExit(f"error: no simulated case of {C.name} yet")
    sims = {lab: eio.sim_named(C, lab) for lab in labels}
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


def cmd_calibrate(C, a) -> int:
    from . import calibrate

    calibrate.run(C, a.beds, a.train, a.sss_step, a.sss_crystal_step, a.sss_ring_step,
                  a.sss_margin, not a.no_shield)
    return 0


def cmd_analytic(C, a) -> int:
    from . import analytic

    calib = analytic.load_calib(a.calib) if a.calib else analytic.load_calib()
    dst = eio.sim_case(C, analytic.METHOD, a.seed)
    for bed in _beds(C, a.beds):
        print(f"analytic bed {bed}")
        analytic.run_bed(C, dst, bed, a.seconds, a.seed, calib, a.singles)
    print(f"-> {dst.root}")
    return 0


def cmd_check_terms(C, a) -> int:
    from . import check_terms

    check_terms.run(C, a.sim, a.beds)
    return 0


def cmd_virtual(a) -> int:
    from utils.paths import out_root

    from . import virtual

    T = get_case(a.template, a.out)
    exam = virtual.exam_from_manifest(Path(a.exam), a.utc_offset_h) if a.exam else None
    V = virtual.build(T, eio.VIRTUAL, Path(a.ct), Path(a.pet), a.pet_units,
                      out_root(a.out) / a.name, a.template_bed, a.margin_mm, a.kvp,
                      exam=exam)
    print(f"-> {V.root}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d710 simulate",
                                 description="raw D710 data simulated from a case's CT and PET",
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

    p = sub.add_parser("virtual", help="a header-only case around a PET/CT NIfTI, from a template")
    p.add_argument("--template", required=True, help="real case whose scanner terms are borrowed")
    p.add_argument("--case", dest="name", required=True, help="the new case")
    p.add_argument("--ct", required=True, help="CT in HU: NIfTI or DICOM folder")
    p.add_argument("--pet", required=True, help="PET: NIfTI or DICOM folder")
    p.add_argument("--pet-units", required=True, choices=ph.UNITS)
    p.add_argument("--template-bed", type=int, default=4)
    p.add_argument("--margin-mm", type=float, default=ph.DEFAULT_MARGIN_MM)
    p.add_argument("--kvp", type=float, default=None)
    p.add_argument("--exam", default=None,
                   help="the patient's dose, weight and times from a lympho2 manifest.json "
                        "(default: the template's)")
    p.add_argument("--utc-offset-h", type=float, default=7.0,
                   help="hours the manifest's local times are ahead of UTC")
    p.add_argument("--out", default=None, help="output root (default $D710_OUT)")

    p = common(sub.add_parser("calibrate", help="fit the analytic model to a real case"))
    p.add_argument("--train", type=int, nargs="+", default=[2, 3, 5, 6],
                   help="beds the constants are fitted on; the rest are held out")
    p.add_argument("--sss-step", type=int, default=4, help="scatter points every N voxels")
    p.add_argument("--sss-crystal-step", type=int, default=4)
    p.add_argument("--sss-ring-step", type=int, default=4)
    p.add_argument("--sss-margin", type=int, default=24,
                   help="planes of activity and mu beyond each end of the bed for SSS")
    p.add_argument("--no-shield", action="store_true",
                   help="no end-shield aperture for out-of-FOV photons")

    p = common(sub.add_parser("analytic", help="analytic model, calibrated; no GATE"))
    p.add_argument("--seconds", type=float, default=None, help="default: the real frame")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--singles", choices=("model", "measured"), default="model",
                   help="randoms from modelled singles, or from the case's own")
    p.add_argument("--calib", default=None, help="default: simulation/analytic_calib.json")

    p = common(sub.add_parser("check-terms",
                              help="simulated randoms and scatter against the real case's"))
    p.add_argument("--sim", default="an_s1", help="label: <case>/sim_<label>")

    p = common(sub.add_parser("compare", help="simulated against real raw data"))
    p.add_argument("--sims", nargs="*", help="labels such as gate_s1 pp_s1 (default: all)")

    a = ap.parse_args(argv)
    if a.cmd == "virtual":
        return cmd_virtual(a)
    C = resolve(a.case, a.out)
    if not C.decoded.is_dir():
        raise SystemExit(f"error: {C.root} is not a decoded case")
    return {"phantom": cmd_phantom, "gate": cmd_gate, "pp": cmd_pp,
            "compare": cmd_compare, "calibrate": cmd_calibrate,
            "analytic": cmd_analytic, "check-terms": cmd_check_terms}[a.cmd](C, a)


if __name__ == "__main__":
    sys.exit(main())
