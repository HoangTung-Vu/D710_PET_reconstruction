from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np

from utils.paths import case as get_case
from utils.scanner import (DR_MM, GEOMETRY, N_ITERATIONS, N_SUBSETS,
                           PLANE_MM, POST_FILTER_FWHM_MM, POST_FILTER_Z_RATIO,
                           PSF_FWHM_MM, XY)

from . import terms

ENGINE = "PyTomography OSEM, parallelproj joseph3d, sinogram non-TOF"


def bed_key(C, n: int, args) -> str:
    from lm.recon import psf_fwhm

    parts = []
    for hs in terms.paths(C, n):
        for p in (hs, hs.with_suffix(".s")):
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    parts += [f"xy={args.xy!r}", f"subsets={args.subsets!r}",
              f"iters={args.iters!r}", f"psf={psf_fwhm(args.psf)!r}",
              f"engine={ENGINE}", f"geometry={GEOMETRY}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sino")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", help="output root; defaults to $D710_OUT")
    ap.add_argument("--beds", type=int, nargs="+",
                    help="default: every bed with prompts, normdt, background "
                         "and attn")
    ap.add_argument("--ct", help="recorded in recon_sino.npz; defaults to the "
                                 "CT the first bed was estimated with")
    ap.add_argument("--xy", type=int, default=XY,
                    help=f"transaxial matrix size at {DR_MM} mm voxels "
                         f"(default %(default)d)")
    ap.add_argument("--iters", type=int, default=N_ITERATIONS)
    ap.add_argument("--subsets", type=int, default=N_SUBSETS,
                    help="view subsets; should divide the 288 views "
                         "(default %(default)d)")
    ap.add_argument("--psf", type=float, nargs="+", default=list(PSF_FWHM_MM),
                    metavar="MM",
                    help="XY [Z] mm FWHM; one value = isotropic; 0 disables "
                         "(default %(default)s, as `d710 lm recon`)")
    ap.add_argument("--post-filter", type=float, default=POST_FILTER_FWHM_MM,
                    metavar="MM")
    ap.add_argument("--z-ratio", type=float, default=POST_FILTER_Z_RATIO,
                    metavar="R")
    ap.add_argument("--max-rays", type=int, default=None, metavar="N",
                    help="rays per parallelproj call; lower it if memory is short")
    ap.add_argument("--resume", action="store_true",
                    help="reuse any bed in work/bed<n>/sino.npz whose inputs and "
                         "settings match this run exactly")
    args = ap.parse_args(argv)

    from lm.recon import psf_fwhm
    from osem import stitch
    from utils import terms as sidecar

    from . import recon

    C = get_case(args.case, args.out)
    beds = args.beds or [n for n in C.decoded_beds()
                         if all(p.exists() for p in terms.paths(C, n))]
    if not beds:
        raise SystemExit(
            f"error: no bed of {C.name!r} has prompts, normdt, background and "
            f"attn.\n  run: d710 exam --raw <...> --ct <...> --case {C.name}\n"
            f"       d710 attn --case {C.name}")
    ct_dir = args.ct or sidecar.ct_dir(C, beds[0])
    print(f"case {C.name!r}: {len(beds)} beds  ->  {beds}")

    psf = np.array(psf_fwhm(args.psf) or [0.0, 0.0, 0.0])
    img, sens = {}, {}
    for n in beds:
        p, key = C.work_bed(n) / "sino.npz", bed_key(C, n, args)
        if args.resume and p.exists():
            z = np.load(p, allow_pickle=False)
            if "key" in z.files and str(z["key"]) == key:
                img[n], sens[n] = z["img"], z["sens"]
                print(f"\n=== bed {n}: reused {p.name}")
                continue
            print(f"\n=== bed {n}: {p.name} was made with different settings "
                  f"-- reconstructing again")
        else:
            print(f"\n=== bed {n}")
        t0 = time.time()
        img[n], sens[n] = recon.reconstruct(C, n, xy=args.xy, n_sub=args.subsets,
                                            n_it=args.iters, psf=args.psf,
                                            max_rays=args.max_rays)
        el = time.time() - t0
        np.savez_compressed(p, img=img[n], sens=sens[n], key=key, bed=n,
                            seconds=el, psf_fwhm_mm=psf)
        print(f"  bed {n}: {el:.0f} s  -> {p.name}")

    print()
    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    stitch.overlap_report(C, beds, img, factors)
    vox = [PLANE_MM, DR_MM, DR_MM]
    vol = stitch.post_filter(vol, vox, args.post_filter, args.z_ratio)

    C.root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.recon_sino, vol=vol, z0=z0, vox=np.array(vox),
                        beds=np.array(beds),
                        decay=np.array([factors[n] for n in beds]),
                        n_subsets=args.subsets, n_iterations=args.iters,
                        ct=ct_dir, n_tof=1, tangential_lors=0, psf_fwhm_mm=psf,
                        post_filter_fwhm_mm=args.post_filter,
                        post_filter_z_ratio=args.z_ratio, engine=ENGINE,
                        geometry=GEOMETRY)
    print(f"\nwrote {C.recon_sino}  ({vol.shape}, count/voxel referred to the "
          f"injection time)")
    print(f"next:  d710 export --case {C.name} --sino")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
