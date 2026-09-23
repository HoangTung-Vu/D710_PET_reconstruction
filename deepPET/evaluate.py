"""DeepPET against 2D OSEM on the test patients, at fixed count levels.

    python -m deepPET.evaluate --name g128 [--counts 1e5 1e6 1e7] [--limit 300]

Every test slice is simulated once per count level, deterministically. DeepPET
reads the precorrected sinogram; OSEM (5 it x 16 subsets, 6.4 mm post-filter,
the paper's comparison) reads the raw counts with the same attenuation, randoms
and scatter in its model. Metrics are inside the bore: rRMSE (the paper's),
PSNR, SSIM. Writes `eval_<ckpt>.csv`, `eval_<ckpt>.json` and a Fig. 6-style
`eval_<ckpt>.png` into the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import metrics as M


def _osem_job(args):
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .osem2d import osem
    from .scanner2d import Scanner2D

    y, mult, gamma, grid, it, sub = args
    return osem(y, mult, gamma, Scanner2D(grid), it, sub)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="best", choices=("best", "last"))
    ap.add_argument("--out", default=None, help="runs root; default $D710_OUT/deeppet/runs")
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--counts", type=float, nargs="+", default=[1e5, 1e6, 1e7])
    ap.add_argument("--limit", type=int, default=300, help="test slices per count level")
    ap.add_argument("--osem-iters", type=int, default=5)
    ap.add_argument("--osem-subsets", type=int, default=16)
    ap.add_argument("--no-osem", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from .dataset import SinoDataset, default_data
    from .model import DeepPET
    from .scanner2d import fov_mask
    from .train import pick_device, run_dir

    rd = run_dir(a.name, a.out)
    device = pick_device(a.device)
    ck = torch.load(rd / f"{a.ckpt}.pt", map_location=device, weights_only=False)
    targs = ck["args"]
    grid, mode = targs["grid"], targs["mode"]
    model = DeepPET(grid, targs["bn_momentum"]).to(device).eval()
    model.load_state_dict(ck["model"])
    mask = fov_mask(grid)
    data = a.data or targs.get("data") or str(default_data())
    print(f"{a.name}/{a.ckpt}.pt (epoch {ck['epoch']}), grid {grid}, mode {mode}, "
          f"{a.split} split, {a.limit} slices per level")

    rows, examples = [], []
    import multiprocessing as mp

    # spawn: this process has run parallelproj (OpenMP) already, and a forked
    # child of a libgomp process can hang in its first parallel region
    ex = None if a.no_osem else ProcessPoolExecutor(max_workers=a.workers,
                                                    mp_context=mp.get_context("spawn"))
    for c in a.counts:
        ds = SinoDataset(data, a.split, grid, mode, train=False, counts=c,
                         max_items=a.limit, seed=1, return_raw=True,
                         limit_studies=targs.get("limit_studies"))
        t0 = time.time()
        items = [ds.sample(j) for j in range(len(ds))]
        xs = torch.from_numpy(np.stack([x for x, _, _ in items]))[:, None]
        preds = []
        with torch.no_grad():
            for b in range(0, len(xs), 16):
                preds.append(model(xs[b:b + 16].to(device)).float().cpu().numpy()[:, 0])
        pred = np.concatenate(preds) * mask
        osems = [None] * len(items)
        if ex is not None:
            jobs = [(i["y"], i["mult"], i["gamma"], grid, a.osem_iters, a.osem_subsets)
                    for _, _, i in items]
            osems = list(ex.map(_osem_job, jobs, chunksize=4))
        for j, (x, t, info) in enumerate(items):
            if info["empty"]:
                continue
            r = {"counts": c, "study": info["study"], "slice": info["slice"]}
            for name, img in (("deeppet", pred[j]), ("osem", osems[j])):
                if img is None:
                    continue
                r[f"{name}_rrmse"] = M.rrmse(img, t, mask)
                r[f"{name}_psnr"] = M.psnr(img, t, mask)
                r[f"{name}_ssim"] = M.ssim(img, t, mask)
            rows.append(r)
        order = np.argsort([-t.sum() for _, t, _ in items])
        j = int(order[len(order) // 4])
        examples.append((c, items[j][0], items[j][1], osems[j], pred[j]))
        print(f"  {c:.0e} prompts: {len(items)} slices in {time.time() - t0:.0f} s")
    if ex is not None:
        ex.shutdown()

    keys = [k for k in rows[0] if k.endswith(("_rrmse", "_psnr", "_ssim"))]
    summary = {}
    print(f"\n{'prompts':>8} " + " ".join(f"{k:>16}" for k in keys))
    for c in a.counts:
        sel = [r for r in rows if r["counts"] == c]
        summary[f"{c:.0e}"] = {k: [float(np.mean([r[k] for r in sel])),
                                   float(np.std([r[k] for r in sel]))] for k in keys}
        print(f"{c:8.0e} " + " ".join(f"{summary[f'{c:.0e}'][k][0]:9.4f}+-"
                                     f"{summary[f'{c:.0e}'][k][1]:<5.3f}" for k in keys))

    tag = f"eval_{a.ckpt}_{a.split}"
    with open(rd / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (rd / f"{tag}.json").write_text(json.dumps(summary, indent=1))

    cols = ["sinogram", "truth", "OSEM 5x16", "DeepPET"]
    fig, ax = plt.subplots(len(examples), 4, figsize=(14, 3.4 * len(examples)), squeeze=False)
    for r, (c, x, t, o, p) in enumerate(examples):
        vmax = float(np.percentile(t[mask], 99.9))
        for k, img in enumerate((x, t, o, p)):
            if img is None:
                ax[r, k].axis("off")
                continue
            kw = {"aspect": "auto"} if k == 0 else {"vmin": 0, "vmax": vmax}
            ax[r, k].imshow(img, cmap="hot" if k else "gray_r", **kw)
            label = f"{c:.0e} prompts" if k == 0 else \
                (f"rRMSE {M.rrmse(img, t, mask):.2f}" if k > 1 else "SUV")
            ax[r, k].set_title(f"{cols[k]}  {label}", fontsize=9)
            ax[r, k].axis("off")
    fig.tight_layout()
    fig.savefig(rd / f"{tag}.png", dpi=90)
    print(f"\nwrote {rd / tag}.csv/.json/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
