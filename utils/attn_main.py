"""Command-line entry point for `d710 attn`."""

from __future__ import annotations

import argparse

from . import attn, scanner, terms
from .paths import case as get_case


def _ct_from_sidecar(case, bed: int) -> str:
    """The CT `d710 estimate` used, or an error naming the two ways out.

    `attn` itself needs nothing from `estimate`; only this default does, which
    is why it is worth saying so rather than reporting a missing file.
    """
    if not (case.work_bed(bed) / "to_stir.json").exists():
        raise SystemExit(
            f"error: bed {bed} has no work/bed{bed}/to_stir.json, so there is "
            f"no CT recorded for it.\n"
            f"  name the series:      d710 attn --case {case.name} --ct <CT dir>\n"
            f"  or run the terms first:  d710 exam --case {case.name} ...")
    return terms.ct_dir(case, bed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="attn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ct", help="CT series; defaults to the first bed's sidecar")
    ap.add_argument("--beds", type=int, nargs="+")
    ap.add_argument("--xy", type=int, default=scanner.XY,
                    help="transaxial matrix size; must match the reconstruction")
    ap.add_argument("--device", default="auto", metavar="D",
                    help="where parallelproj runs: auto (default: cuda when "
                         "torch sees a GPU, else cpu), cpu (numpy, no torch), "
                         "or a torch device such as cuda or cuda:1")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even where attn.hs is already there")
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    beds = args.beds or C.decoded_beds()
    if not beds:
        raise SystemExit(f"error: case {args.case!r} has no decoded bed")

    if args.force:
        for n in beds:
            for p in (C.work_bed(n) / "attn.hs", C.work_bed(n) / "attn.s"):
                p.unlink(missing_ok=True)

    ct = args.ct or _ct_from_sidecar(C, beds[0])
    at = attn.Attenuation(C, ct, xy=args.xy, device=args.device)
    print(at.describe())
    at.all(beds)
    print(f"\nwrote work/bed<n>/attn.hs for {beds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
