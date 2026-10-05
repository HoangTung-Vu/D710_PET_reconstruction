from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils.interfile import Header
from utils.paths import out_root
from utils.scanner import (DR_MM, PLANE_MM, POST_FILTER_FWHM_MM,
                           POST_FILTER_Z_RATIO, fov_mask)

METHODS = ("fbsem", "osem", "osem_pf")


def parser():
    ap = argparse.ArgumentParser(prog="fbsem eval")
    ap.add_argument("--root", help="default $D710_OUT")
    ap.add_argument("--sets", nargs="+", default=["thyr_testset"])
    ap.add_argument("--sim", default="sim_an_s1")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--csv", help="default <root>/fbsem/eval_<sets>.csv")
    return ap


def metrics(x, xt, b) -> dict:
    d = x[b] - xt[b]
    return {"nrmse": float(np.linalg.norm(d) / np.linalg.norm(xt[b])),
            "ratio": float(np.median(x[b] / xt[b]))}


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    from osem.stitch import post_filter

    from . import data

    root = out_root(args.root)
    cases = data.sim_cases(root, args.sets, args.sim, args.cases)
    out = Path(args.csv) if args.csv else (
        root / "fbsem" / f"eval_{'_'.join(args.sets)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    vox = [PLANE_MM, DR_MM, DR_MM]

    rows = []
    for C, beds in cases:
        for n in beds:
            f, s = C.work_bed(n) / "fbsem.npz", C.work_bed(n) / "sino.npz"
            if not (f.exists() and s.exists()):
                continue
            zf, zs = np.load(f), np.load(s)
            fb, os_ = zf["img"], zs["img"]
            fov = fov_mask(os_.shape[1], Header(C.prompt(n)).n_tang)
            xt = np.load(C.raw_sim / f"bed{n}_x_true.npy") * fov
            b = data.body_mask(xt)
            imgs = {"fbsem": fb, "osem": os_,
                    "osem_pf": post_filter(os_, vox, POST_FILTER_FWHM_MM,
                                           POST_FILTER_Z_RATIO, verbose=False)}
            row = {"case": C.root.parent.name, "bed": n, "model": str(zf["key"])}
            for k in METHODS:
                for name, v in metrics(imgs[k], xt, b).items():
                    row[f"{k}_{name}"] = v
            rows.append(row)
    if not rows:
        raise SystemExit(f"error: no bed under {args.sets} has both fbsem.npz and "
                         f"sino.npz; run `d710 fbsem recon` first")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"n_cases": len({r["case"] for r in rows}), "n_beds": len(rows),
               "models": sorted({r["model"] for r in rows})}
    print(f"{summary['n_cases']} cases, {summary['n_beds']} beds  ->  {out}")
    print(f"{'method':>8} {'NRMSE mean':>11} {'sd':>7} {'median ratio':>13}")
    for k in METHODS:
        e = np.array([r[f"{k}_nrmse"] for r in rows])
        q = np.array([r[f"{k}_ratio"] for r in rows])
        summary[k] = {"nrmse_mean": float(e.mean()), "nrmse_sd": float(e.std()),
                      "ratio_median": float(np.median(q))}
        print(f"{k:>8} {e.mean():>11.4f} {e.std():>7.4f} {np.median(q):>13.4f}")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
