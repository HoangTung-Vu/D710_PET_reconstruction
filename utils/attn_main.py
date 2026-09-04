"""`d710 attn` — build `work/bed<n>/attn.hs` for a case. SIRF runtime only.

Attenuation is the one term of the model SIRF has to compute, and it is the only
reason the list-mode path would otherwise need SIRF. Splitting it out means:

    SIRF runtime (sirf-local image)   decode -> estimate -> tostir -> attn -> osem
    PyTomography env                  lm, lowdose      (numpy + torch, no SIRF)

`d710 osem` still builds it on the fly when it is missing, so this command is for
preparing a case that only the list-mode path will read.
"""

from __future__ import annotations

import argparse

from . import attn, scanner, sirf_env, terms
from .paths import case as get_case


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="attn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ct", help="CT series; defaults to the first bed's sidecar")
    ap.add_argument("--beds", type=int, nargs="+")
    ap.add_argument("--xy", type=int, default=scanner.XY,
                    help="transaxial matrix size; must match `d710 osem`")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even where attn.hs is already there")
    args = ap.parse_args(argv)

    import sirf.STIR as pet

    C = get_case(args.case, args.out)
    beds = args.beds or C.beds()
    if not beds:
        raise SystemExit(f"error: case {args.case!r} has no bed with vendor terms")
    sirf_env.setup(C)

    if args.force:
        for n in beds:
            for p in (C.work_bed(n) / "attn.hs", C.work_bed(n) / "attn.s"):
                p.unlink(missing_ok=True)

    y0 = pet.AcquisitionData(str(C.prompt(beds[0])))
    x0 = scanner.sirf_grid(y0, args.xy)
    at = attn.Attenuation(C, args.ct or terms.ct_dir(C, beds[0]), x0, y0)
    print(at.describe())
    at.all(beds)
    print(f"\nwrote work/bed<n>/attn.hs for {beds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
