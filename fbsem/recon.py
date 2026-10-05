from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from utils.paths import case as get_case
from utils.scanner import (DR_MM, GEOMETRY, N_ITERATIONS, N_SUBSETS, PLANE_MM,
                           POST_FILTER_Z_RATIO, PSF_FWHM_MM, XY)

ENGINE = "FBSEM-Net (Mehranian & Reader 2019), parallelproj joseph3d, sinogram non-TOF"


def parser():
    ap = argparse.ArgumentParser(prog="fbsem recon")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", help="output root; defaults to $D710_OUT")
    ap.add_argument("--model", help="checkpoint (.pt) from fbsem train")
    ap.add_argument("--beds", type=int, nargs="+")
    ap.add_argument("--n-it", type=int)
    ap.add_argument("--n-sub", type=int)
    ap.add_argument("--xy", type=int)
    ap.add_argument("--check-osem", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--max-rays", type=int)
    ap.add_argument("--ct")
    ap.add_argument("--device")
    return ap


def file_sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.model and not args.check_osem:
        raise SystemExit("error: --model is required (or --check-osem)")

    import pytomography
    import torch

    from osem import stitch
    from sino import terms
    from utils import terms as sidecar

    from . import bed as B
    from . import model as M

    dev = torch.device(args.device or pytomography.device)
    C = get_case(args.case, args.out)
    cache = Path(args.cache) if args.cache else None
    beds = args.beds or [n for n in C.decoded_beds()
                         if all(p.exists() for p in terms.paths(C, n))]
    if not beds:
        raise SystemExit(f"error: no bed of {C.root} has prompts, normdt, background "
                         f"and attn")

    if args.check_osem:
        net, ck = M.FBSEMNet().to(dev), {}
        with torch.no_grad():
            net.gamma.zero_()
        psf = list(PSF_FWHM_MM)
        n_it, n_sub = args.n_it or N_ITERATIONS, args.n_sub or N_SUBSETS
        xy = args.xy or XY
        tag = "gamma=0"
    else:
        net, ck = M.load(args.model, dev)
        psf = ck.get("psf", list(PSF_FWHM_MM))
        n_it, n_sub = args.n_it or ck["n_it"], args.n_sub or ck["n_sub"]
        xy = args.xy or ck.get("xy", XY)
        tag = f"{Path(args.model).name}:{file_sha(args.model)}"
    net.eval()
    print(f"case {C.root}: beds {beds}, {n_it}x{n_sub}, {tag}")

    img, sens = {}, {}
    for n in beds:
        t0 = time.time()
        a = B.load_arrays(C, n, n_sub, psf, cache, xy=xy)
        bed = B.build(a, n_sub, psf, dev, args.max_rays, xy,
                      init="mask" if args.check_osem else "counts")
        with torch.no_grad():
            x = net(bed, n_it)
        img[n] = np.ascontiguousarray(x.cpu().numpy().transpose(2, 1, 0))
        sens[n] = np.ascontiguousarray(
            bed.sens_full().cpu().numpy().transpose(2, 1, 0))
        el = time.time() - t0
        del bed, x
        if args.check_osem:
            ref = C.work_bed(n) / "sino.npz"
            if not ref.exists():
                print(f"  bed {n}: no {ref}, nothing to compare with")
                continue
            r = np.load(ref)["img"]
            diff = float(np.abs(img[n] - r).max() / np.abs(r).max())
            print(f"  bed {n}: max|fbsem(gamma=0) - sino.npz| / max = {diff:.2e}"
                  f"   ({el:.0f} s)")
            continue
        np.savez_compressed(C.work_bed(n) / "fbsem.npz", img=img[n], sens=sens[n],
                            key=tag, bed=n, seconds=el, n_it=n_it, n_sub=n_sub)
        print(f"  bed {n}: {el:.0f} s  -> work/bed{n}/fbsem.npz", flush=True)

    if args.check_osem:
        return 0

    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    stitch.overlap_report(C, beds, img, factors)
    try:
        ct = args.ct or sidecar.ct_dir(C, beds[0])
    except (SystemExit, OSError, KeyError):
        ct = ""
    np.savez_compressed(C.recon_fbsem, vol=vol, z0=z0,
                        vox=np.array([PLANE_MM, DR_MM, DR_MM]), beds=np.array(beds),
                        decay=np.array([factors[n] for n in beds]),
                        n_subsets=n_sub, n_iterations=n_it, ct=ct, n_tof=1,
                        tangential_lors=0, psf_fwhm_mm=np.array(psf, float),
                        post_filter_fwhm_mm=0.0,
                        post_filter_z_ratio=POST_FILTER_Z_RATIO, engine=ENGINE,
                        geometry=GEOMETRY, model=tag)
    print(f"\nwrote {C.recon_fbsem}  ({vol.shape})")
    print(f"next:  d710 export --out {C.root.parent} --case {C.name} --fbsem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
