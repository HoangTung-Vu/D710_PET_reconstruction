"""`d710 osem` — reconstruct every bed of a case and stitch them, no notebook needed.

    python3 -m osem --case ped [--beds 1 2 3] [--iters 3] [--subsets 12]

Needs the project environment (`conda activate petct_reconstruction`): SIRF is
not in the `d710:full` image.

Writes `<case>/recon.npz` — count/voxel referred back to the injection time, plus
enough geometry for `d710 export` to convert to Bq/mL without rerunning anything.
"""

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
                    help="NUMBER OF transaxial VOXELS -> sets the FOV, not the resolution")
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
    # SIRF leaves a tmp_*.hs/.s pair in scratch for every get_uniform_copy and
    # never collects them. Non-TOF that is 231 MB a pair; at mash 5 it is 2.7 GB,
    # and two TOF runs of one bed left 16 GB behind. Here is the one moment it is
    # safe to clear -- before any AcquisitionData is alive.
    freed = sirf_env.clear_scratch(C)
    if freed:
        print(f"scratch: removed {freed} tmp_* files from a previous run")
    print(f"case {C.name!r}: {len(beds)} beds  ->  {beds}\n")

    y0, x0 = recon.image_grid(C, beds[0], xy=args.xy)
    vox = [float(v) for v in x0.voxel_sizes()]           # (z, y, x) mm
    # dimensions() is (tof, axial, view, tangential) and costs nothing --
    # as_array() on TOF prompts would materialise gigabytes just to read a 0.
    n_tof = int(y0.dimensions()[0])
    print(f"image {x0.as_array().shape}  "
          f"voxel {vox[2]:.4f} × {vox[1]:.4f} × {vox[0]:.4f} mm")
    if n_tof > 1:
        # Which scatter TOF source each bed will actually use is decided per bed
        # inside terms.expand_to_tof (it depends on whether that bed was
        # estimated with reconMethod 3), so only the override is known here.
        src = args.tof_scatter or "per bed: GE's own, else measured in place"
        print(f"TOF: {n_tof} bins, mash {y0.get_tof_mash_factor()}, "
              f"scatter {src}")
    print()

    ct_dir = args.ct or terms.ct_dir(C, beds[0])
    at = attn.Attenuation(C, ct_dir, x0, y0)
    print(at.describe())
    print("\nattenuation per bed (water at 511 keV ≈ 0.096 1/cm):")
    af = at.all(beds)

    # Resolved once, before any bed runs: a bad path should fail now, not after
    # the first bed has spent minutes reconstructing.
    tof_scatter = np.load(args.tof_scatter) if args.tof_scatter else None

    print()
    img, sens = recon.reconstruct_all(C, beds, af, x0,
                                      n_sub=args.subsets, n_it=args.iters,
                                      tof_scatter=tof_scatter,
                                      tangential_lors=args.lors,
                                      projector=args.projector)

    print()
    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    # BEFORE the post-filter: the seam check compares two independent recons of
    # the same planes, and smoothing them towards each other is exactly the way
    # to make a misplaced bed stop showing up.
    stitch.overlap_report(C, beds, img, factors)

    vol = stitch.post_filter(vol, vox, args.post_filter, args.z_ratio)

    C.root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.recon, vol=vol, z0=z0, vox=np.array(vox),
        beds=np.array(beds), decay=np.array([factors[n] for n in beds]),
        n_subsets=args.subsets, n_iterations=args.iters, ct=ct_dir,
        # Part of the model, not a speed setting: a volume made with --lors 1 is
        # a different reconstruction and must not be compared with a default one.
        tangential_lors=args.lors,
        post_filter_fwhm_mm=args.post_filter, post_filter_z_ratio=args.z_ratio,
        # Provenance: where this volume's scatter TOF distribution came from, so
        # an image built on a measured stand-in can never be mistaken for one
        # built on GE's. It travels with the volume, not with the run log.
        n_tof=n_tof,
        tof_scatter=((args.tof_scatter or "per-bed")
                     if n_tof > 1 else "n/a"))
    print(f"\nwrote {C.recon}  ({vol.shape}, count/voxel referred to the injection time)")
    print(f"next:  d710 export --case {C.name} --format nifti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
