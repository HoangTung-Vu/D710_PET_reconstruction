"""Command-line entry point for `d710 osem`."""

from __future__ import annotations

import argparse

import numpy as np

from utils import attn, sirf_env, terms
from utils.paths import case as get_case

from . import recon, stitch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="osem", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", help="output root; defaults to $D710_OUT")
    ap.add_argument("--ct", help="CT series; defaults to the first bed's sidecar")
    ap.add_argument("--beds", type=int, nargs="+",
                    help="default: every bed that finished all three steps")
    ap.add_argument("--iters", type=int, default=recon.N_ITERATIONS)
    ap.add_argument("--subsets", type=int, default=recon.N_SUBSETS)
    ap.add_argument("--xy", type=int, default=recon.XY,
                    help=f"transaxial matrix size at {recon.scanner.DR_MM} mm "
                         f"voxels (default %(default)d).  The two SIRF builds "
                         f"disagree about what `xy` means -- the host pins the "
                         f"voxel, the sirf-local image pins the FOV -- so "
                         f"anything but the default is rescaled to keep the "
                         f"voxel, and with it K")
    ap.add_argument("--projector", choices=("auto", "ray", "parallelproj"),
                    default="auto",
                    help="G in y = S(Gx)+b.  auto = ray-tracing when non-TOF, "
                         "parallelproj when TOF -- because the ray-tracing "
                         "matrix cache explodes along the TOF axis (measured: "
                         "30 GB for ONE 5-bin bed)")
    ap.add_argument("--lors", type=int, default=recon.TANGENTIAL_LORS,
                    metavar="N",
                    help="rays per tangential bin (default %(default)d).  This "
                         "is the biggest COST knob: 1 is ~5x faster but models "
                         "the transaxial direction more coarsely -- use it to "
                         "test the pipeline, not to produce results")
    ap.add_argument("--post-filter", type=float, default=stitch.POST_FILTER_FWHM_MM,
                    metavar="MM",
                    help="transaxial post-reconstruction filter FWHM, mm "
                         "(default %(default)g = GE's own setting; 0 disables)")
    ap.add_argument("--z-ratio", type=float, default=stitch.POST_FILTER_Z_RATIO,
                    metavar="R",
                    help="three-tap axial filter [1,R,1] (default %(default)g; 0 disables)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse any bed already in work/bed<n>/osem.npz whose "
                         "settings match this run exactly (prompts file, grid, "
                         "attenuation, subsets, iterations, projector, LORs).  "
                         "A bed that does not match is reconstructed again -- "
                         "nothing is ever reused silently")
    ap.add_argument("--tof-scatter", metavar="PROF.npy",
                    help="OVERRIDE how the scatter is spread over TOF with a "
                         "single saved profile (tools/tof_profile.py --save). "
                         "Left off, each bed uses GE's own per-(view, u) "
                         "distribution from work/bed<n>/scatter_tof.npy, and "
                         "falls back to measuring its own tail ring only if "
                         "that file is absent")
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    beds = args.beds or C.beds()
    if not beds:
        raise SystemExit(
            "error: case %r has no bed that finished steps 2-3 yet.\n"
            "  run: d710 exam --raw <...> --ct <...> --case %s"
            % (args.case, args.case))

    missing = [n for n in beds if not (C.work_bed(n) / "normdt.hs").exists()]
    if missing:
        raise SystemExit("error: bed %s has no terms yet in %s"
                         % (missing, C.work))

    sirf_env.setup(C)
    freed = sirf_env.clear_scratch(C)
    if freed:
        print(f"scratch: removed {freed} tmp_* files from a previous run")
    print(f"case {C.name!r}: {len(beds)} beds  ->  {beds}\n")

    y0, x0 = recon.image_grid(C, beds[0], xy=args.xy)
    vox = [float(v) for v in x0.voxel_sizes()]
    n_tof = int(y0.dimensions()[0])
    print(f"image {x0.as_array().shape}  "
          f"voxel {vox[2]:.4f} × {vox[1]:.4f} × {vox[0]:.4f} mm")
    if n_tof > 1:
        src = args.tof_scatter or "per bed: GE's own, else measured in place"
        print(f"TOF: {n_tof} bins, mash {y0.get_tof_mash_factor()}, "
              f"scatter {src}")
    print()

    ct_dir = args.ct or terms.ct_dir(C, beds[0])
    at = attn.Attenuation(C, ct_dir, x0, y0)
    print(at.describe())
    print("\nattenuation per bed (water at 511 keV ≈ 0.096 1/cm):")
    af = at.all(beds)

    tof_scatter = np.load(args.tof_scatter) if args.tof_scatter else None

    print()
    img, sens = recon.reconstruct_all(C, beds, af, x0, resume=args.resume,
                                      n_sub=args.subsets, n_it=args.iters,
                                      tof_scatter=tof_scatter,
                                      tangential_lors=args.lors,
                                      projector=args.projector)

    print()
    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    stitch.overlap_report(C, beds, img, factors)

    vol = stitch.post_filter(vol, vox, args.post_filter, args.z_ratio)

    C.root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.recon, vol=vol, z0=z0, vox=np.array(vox),
        beds=np.array(beds), decay=np.array([factors[n] for n in beds]),
        n_subsets=args.subsets, n_iterations=args.iters, ct=ct_dir,
        tangential_lors=args.lors,
        post_filter_fwhm_mm=args.post_filter, post_filter_z_ratio=args.z_ratio,
        n_tof=n_tof,
        tof_scatter=((args.tof_scatter or "per-bed")
                     if n_tof > 1 else "n/a"))
    print(f"\nwrote {C.recon}  ({vol.shape}, count/voxel referred to the injection time)")
    print(f"next:  d710 export --case {C.name} --format nifti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
