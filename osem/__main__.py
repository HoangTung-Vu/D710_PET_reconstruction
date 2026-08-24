"""`d710 osem` — tái tạo mọi bed của một ca rồi ghép, không cần notebook.

    python3 -m osem --case ped [--beds 1 2 3] [--iters 3] [--subsets 12]

Cần môi trường project (`conda activate petct_reconstruction`): SIRF không có
trong image `d710:full`.

Ghi `<ca>/recon.npz` — count/voxel đã quy về thời điểm tiêm, cộng hình học đủ
để `d710 export` đổi sang Bq/mL mà không phải chạy lại gì.
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
    ap.add_argument("--out", help="gốc đầu ra; mặc định $D710_OUT")
    ap.add_argument("--ct", help="series CT; mặc định lấy từ sidecar của bed đầu")
    ap.add_argument("--beds", type=int, nargs="+",
                    help="mặc định: mọi bed đã xong cả ba bước")
    ap.add_argument("--iters", type=int, default=recon.N_ITERATIONS)
    ap.add_argument("--subsets", type=int, default=recon.N_SUBSETS)
    ap.add_argument("--xy", type=int, default=recon.XY,
                    help="SỐ VOXEL ngang -> quyết định FOV, không phải độ phân giải")
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    beds = args.beds or C.beds()
    if not beds:
        raise SystemExit(
            "error: ca %r chưa có bed nào xong bước 2–3.\n"
            "  chạy: d710 exam --raw <...> --ct <...> --case %s"
            % (args.case, args.case))

    missing = [n for n in beds if not (C.work_bed(n) / "normdt.hs").exists()]
    if missing:
        raise SystemExit("error: bed %s chưa có số hạng ở %s"
                         % (missing, C.work))

    sirf_env.setup(C)
    print(f"ca {C.name!r}: {len(beds)} bed  ->  {beds}\n")

    y0, x0 = recon.image_grid(C, beds[0], xy=args.xy)
    vox = [float(v) for v in x0.voxel_sizes()]           # (z, y, x) mm
    print(f"ảnh {x0.as_array().shape}  "
          f"voxel {vox[2]:.4f} × {vox[1]:.4f} × {vox[0]:.4f} mm\n")

    ct_dir = args.ct or terms.ct_dir(C, beds[0])
    at = attn.Attenuation(C, ct_dir, x0, y0)
    print(at.describe())
    print("\nsuy giảm theo bed (nước ở 511 keV ≈ 0.096 1/cm):")
    af = at.all(beds)

    print()
    img, sens = recon.reconstruct_all(C, beds, af, x0,
                                      n_sub=args.subsets, n_it=args.iters)

    print()
    vol, z0, factors = stitch.stitch(C, beds, img, sens)
    stitch.overlap_report(C, beds, img, factors)

    C.root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.recon, vol=vol, z0=z0, vox=np.array(vox),
        beds=np.array(beds), decay=np.array([factors[n] for n in beds]),
        n_subsets=args.subsets, n_iterations=args.iters, ct=ct_dir)
    print(f"\nghi {C.recon}  ({vol.shape}, count/voxel quy về thời điểm tiêm)")
    print(f"tiếp:  d710 export --case {C.name} --format nifti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
