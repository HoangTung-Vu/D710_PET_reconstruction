#!/usr/bin/env python3
"""Compare our images with GE's on the SUV scale, every case in one table."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.paths import out_root

BODY_SUV = 0.5

NOISE_BOX = (7, 2)


def resample_onto(src, ref):
    """Resample `src` onto `ref`'s grid, using the affine of each."""
    from scipy.ndimage import map_coordinates

    m = np.linalg.inv(src.affine) @ ref.affine
    g = np.indices(ref.shape).reshape(3, -1)
    idx = m[:3, :3] @ g + m[:3, 3:4]
    return map_coordinates(src.get_fdata(), idx, order=1,
                           mode="constant", cval=0.0).reshape(ref.shape)


def noise_box(ge_vol):
    """A box at the centre of the warmest region, identical in both images."""
    z = int(np.argmax([(ge_vol[:, :, k] > 1.0).sum()
                       for k in range(ge_vol.shape[2])]))
    yy, xx = np.where(ge_vol[:, :, z] > 1.0)
    if yy.size == 0:
        return None
    cy, cx = int(np.median(yy)), int(np.median(xx))
    r, dz = NOISE_BOX
    return (slice(max(0, cy - r), cy + r + 1),
            slice(max(0, cx - r), cx + r + 1),
            slice(max(0, z - dz), z + dz + 1))


def cov(vol, box) -> float:
    v = vol[box]
    return float(np.std(v) / max(np.mean(v), 1e-9))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_suv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", help="gốc đầu ra; mặc định $D710_OUT")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--md", metavar="PATH", help="ghi bảng ra Markdown")
    args = ap.parse_args(argv)

    import nibabel as nib

    root = out_root(args.out_root)
    cases = args.cases or sorted(
        os.path.basename(os.path.dirname(os.path.dirname(p)))
        for p in glob.glob(str(root / "*" / "export" / "*_ge_suvbw.nii.gz")))
    if not cases:
        raise SystemExit(
            f"error: không ca nào dưới {root} có <case>_ge_suvbw.nii.gz\n"
            f"  dựng trước bằng: python3 -m tools.dicom_suv --all")

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    hdr = (f"{'ca':<14}{'đường':<7}{'r':>7}{'SUV mine':>9}{'SUV GE':>8}"
           f"{'tỉ số':>7}{'p95 mine':>9}{'p95 GE':>8}{'CoV mine':>9}{'CoV GE':>8}")
    say(hdr)
    say("-" * len(hdr))

    agg = {}
    for c in cases:
        e = root / c / "export"
        ge_p = e / f"{c}_ge_suvbw.nii.gz"
        if not ge_p.exists():
            say(f"{c:<14}(chưa có {ge_p.name})")
            continue
        ge = nib.load(str(ge_p))
        G = ge.get_fdata()
        body = G > BODY_SUV
        box = noise_box(G)

        for tag, name in (("", "sino"), ("_lm", "lm")):
            f = e / f"{c}{tag}_suvbw.nii.gz"
            if not f.exists():
                continue
            A = resample_onto(nib.load(str(f)), ge)
            m = body & (A > 0)
            if m.sum() < 1000:
                say(f"{c:<14}{name:<7}(chỉ {m.sum()} voxel chồng nhau)")
                continue
            r = float(np.corrcoef(A[m], G[m])[0, 1])
            a_med, g_med = float(np.median(A[m])), float(np.median(G[m]))
            ca, cg = (cov(A, box), cov(G, box)) if box else (np.nan, np.nan)
            say(f"{c:<14}{name:<7}{r:>7.3f}{a_med:>9.3f}{g_med:>8.3f}"
                f"{a_med / g_med:>7.3f}{np.percentile(A[m], 95):>9.2f}"
                f"{np.percentile(G[m], 95):>8.2f}{ca:>9.3f}{cg:>8.3f}")
            agg.setdefault(name, []).append((r, a_med / g_med, ca / cg))

    say()
    for name, v in agg.items():
        a = np.array(v)
        say(f"{name:<7} trung vị {len(a)} ca:  r {np.median(a[:, 0]):.3f}   "
            f"tỉ số SUV {np.median(a[:, 1]):.3f}   "
            f"nhiễu so với GE {np.median(a[:, 2]):.2f}x")

    if args.md:
        with open(args.md, "w") as fh:
            fh.write("# Ours vs GE, SUVbw\n\n```\n" + "\n".join(lines) + "\n```\n")
        print(f"\nđã ghi {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
