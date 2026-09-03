#!/usr/bin/env python3
"""Where do the hours in a TOF bed go? Ray tracing vs parallelproj, head to head.

    conda activate petct_reconstruction
    python3 tools/projector_bench.py --case ped2 --bed 1

⚠ Do NOT set `PYTHONPATH=<D710>` for this: the env's activate hook puts SIRF on
PYTHONPATH, and overwriting it makes `import sirf.STIR` fail. The script adds its
own parent to `sys.path` instead.

Answers one question the timings alone cannot: is parallelproj *slow*, or is it
being *asked to do too much*? It times a full projection and a one-subset
projection for each projector. If the two are equal, that projector is ignoring
the subset.

Measured 2026-08-30 on ped2 bed 1 (non-TOF, 60.7 M bins, 16 threads, warm):

    projector              fwd FULL   fwd 1of24   back FULL   back 1of24
    parallelproj              10.7       10.7         5.7        0.3
    ray lors=5 cache=off      10.1        0.7         7.3        0.3
    ray lors=5 cache=on        8.1        0.5         4.2        0.2
    ray lors=1 cache=on        5.8        0.4         3.2        0.1

Read it this way: on a FULL sinogram the two are the same speed -- parallelproj
is even the faster of the two backwards. The entire gap in an OSEM subiteration
(0.7 s vs 11.0 s) is that `ForwardProjectorByBinParallelproj::set_input()`
projects every LOR and `actual_forward_project()` then just memcpy's viewgrams
out of the result, once per subset (`distributable.cxx:409`). Hence a TOF bed's
cost is ∝ n_subsets × n_iters and OSEM subsets buy nothing.

The matrix cache is worth only 1.25x here, and `--lors` 5→1 only 1.9x -- both
much less than `osem/recon.py`'s docstrings claim.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def bench(pet, name, make, y, x0, x1, n_sub, out):
    """One projector: set_up, full forward twice, subset forward, backwards."""
    am = make()

    def t(f, *a):
        s = time.time()
        f(*a)
        return time.time() - s

    t_setup = t(am.set_up, y, x0)
    t_f1 = t(am.forward, x1)               # cold: builds any cache
    t_f2 = t(am.forward, x1)               # warm: the number that matters
    t_fs = t(am.forward, x1, 0, n_sub)     # one subset -- equal to full = ignored
    t_b1 = t(am.backward, y)
    t_bs = t(am.backward, y, 0, n_sub)
    sub = t_fs + t_bs
    out(f"{name:<26} set_up {t_setup:7.1f}  fwd {t_f1:7.1f}/{t_f2:7.1f} "
        f"(cold/warm)  fwd 1of{n_sub} {t_fs:7.1f}  back {t_b1:7.1f}  "
        f"back 1of{n_sub} {t_bs:7.1f}   [OSEM subiter ~{sub:6.1f}s]")
    del am


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--bed", type=int, default=1)
    ap.add_argument("--out", help="output root; defaults to $D710_OUT")
    ap.add_argument("--subsets", type=int, default=24)
    ap.add_argument("--skip-ray", action="store_true",
                    help="TOF only: the ray-tracing matrix cache reaches 30 GB "
                         "on a 5-bin bed, so do not let it run unattended")
    args = ap.parse_args(argv)

    import sirf.STIR as pet

    from osem import recon
    from utils import attn, sirf_env, terms
    from utils.paths import case as get_case

    C = get_case(args.case, args.out)
    sirf_env.setup(C)
    y0, x0 = recon.image_grid(C, args.bed)
    at = attn.Attenuation(C, terms.ct_dir(C, args.bed), x0, y0)
    af = at.all([args.bed])
    objs, _ = terms.load(C, args.bed, af=af[args.bed], lean=True)
    y = objs["prompts"]
    x1 = x0.get_uniform_copy(1.0)

    d = tuple(int(v) for v in y.dimensions())
    print(f"\n{args.case} bed {args.bed}: sinogram {d} = {int(np.prod(d)):,} bins"
          f"   image {x0.as_array().shape}\n", flush=True)

    def ray(lors, cache):
        def make():
            am = pet.AcquisitionModelUsingRayTracingMatrix()
            am.set_num_tangential_LORs(lors)
            am.get_matrix().enable_cache(cache)
            return am
        return make

    def out(s):
        print(s, flush=True)

    bench(pet, "parallelproj", pet.AcquisitionModelUsingParallelproj,
          y, x0, x1, args.subsets, out)
    if not args.skip_ray:
        for lors, cache in ((5, True), (5, False), (1, True)):
            bench(pet, f"ray lors={lors} cache={'on' if cache else 'off'}",
                  ray(lors, cache), y, x0, x1, args.subsets, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
