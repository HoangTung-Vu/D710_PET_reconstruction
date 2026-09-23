"""Turn an nnU-Net style PET/CT dataset into DeepPET training slices.

    python -m deepPET.prepare --data ~/Downloads/H108/H108_PETCT_Thyroid/Dataset4000_PETCT_H108
                              [--out $D710_OUT/deeppet/data] [--limit 3] [--workers 4]

Each study is `images{Tr,Ts}/<pid>_<date>_0000.nii.gz` (CT, HU) and `_0001`
(PET, SUVbw) on one shared grid. Per study this writes, under `<out>/slices/`:

    <sid>_suv.npy   (n, 256, 256) float16  SUV, clipped at 0
    <sid>_mu.npy    (n, 256, 256) float16  mu at 511 keV, 1/mm
    <sid>_z.npy     (n,)          float32  patient z of each slice, mm

on the 256 x 2.734375 mm grid centred on the gantry axis (GE's clinical grid;
the 128 grid is its 2 x 2 mean, taken on the fly). mu is made from HU at the
native resolution *before* resampling: it is mu that gets line-integrated, so
averaging mu keeps the integral, where averaging HU across the bone kink would
not. Slices holding less than `--min-activity` SUV.cm^2 are dropped.

`<out>/index.json` lists the studies; `<out>/split.json` splits them 70/10/20
by patient (`<pid>`, so a patient's repeat scans never straddle two splits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from utils.attenuation import hu_to_mu

from . import nifti
from .scanner2d import FOV_MM

STORE_GRID = 256

SPLIT = (0.7, 0.1, 0.2)


def find_studies(root: Path) -> list[dict]:
    """Every `(CT, PET)` pair under `images{Tr,Ts}`, sorted by study id."""
    out = []
    for sub in ("imagesTr", "imagesTs"):
        for pet in sorted((root / sub).glob("*_0001.nii.gz")):
            sid = pet.name[: -len("_0001.nii.gz")]
            ct = pet.with_name(f"{sid}_0000.nii.gz")
            if not ct.exists():
                print(f"  skip {sid}: no {ct.name}", file=sys.stderr)
                continue
            out.append({"sid": sid, "pid": sid.split("_")[0], "ct": str(ct),
                        "pet": str(pet), "subset": sub})
    return sorted(out, key=lambda s: s["sid"])


def prepare_study(st: dict, out_dir: str, min_activity: float, kvp: float) -> dict:
    """Resample one study and write its three arrays; returns its index row."""
    ct = nifti.load(st["ct"])
    pet = nifti.load(st["pet"])
    if ct.data.shape != pet.data.shape or abs(ct.pixel_mm - pet.pixel_mm) > 1e-4 \
            or abs(ct.x0 - pet.x0) > 1e-2 or abs(ct.y0 - pet.y0) > 1e-2 \
            or not np.allclose(ct.z, pet.z, atol=1e-2):
        raise ValueError(f"{st['sid']}: CT and PET are not on one grid "
                         f"({ct.data.shape} @ {ct.pixel_mm:.4f} vs "
                         f"{pet.data.shape} @ {pet.pixel_mm:.4f})")
    mu = hu_to_mu(ct.data, kvp)
    suv = np.clip(pet.data, 0.0, None)
    suv_g = nifti.to_grid(suv, pet.x0, pet.y0, pet.pixel_mm, STORE_GRID, 0.0)
    mu_g = nifti.to_grid(mu, pet.x0, pet.y0, pet.pixel_mm, STORE_GRID, 0.0)
    np.clip(suv_g, 0.0, None, out=suv_g)
    np.clip(mu_g, 0.0, None, out=mu_g)

    v = FOV_MM / STORE_GRID
    act = suv_g.sum(axis=(1, 2), dtype=np.float64) * v * v / 100.0
    act_src = suv.sum(axis=(1, 2), dtype=np.float64) * pet.pixel_mm ** 2 / 100.0
    keep = act >= min_activity
    if not keep.any():
        raise ValueError(f"{st['sid']}: no slice holds {min_activity} SUV.cm^2")

    base = Path(out_dir) / "slices" / st["sid"]
    np.save(f"{base}_suv.npy", suv_g[keep].astype(np.float16))
    np.save(f"{base}_mu.npy", mu_g[keep].astype(np.float16))
    np.save(f"{base}_z.npy", pet.z[keep].astype(np.float32))
    kept = act_src[keep].sum()
    return {**st, "n": int(keep.sum()), "n_source": int(len(keep)),
            "pixel_mm": pet.pixel_mm, "source_fov_mm": pet.pixel_mm * pet.data.shape[2],
            "activity_ratio": float(act[keep].sum() / kept) if kept > 0 else float("nan"),
            "suv_max": float(suv_g.max())}


def split(studies: list[dict], seed: int = 0) -> dict:
    """Patient-level train/val/test lists of study ids."""
    pids = sorted({s["pid"] for s in studies})
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_tr = int(round(SPLIT[0] * len(pids)))
    n_va = int(round(SPLIT[1] * len(pids)))
    parts = {"train": set(pids[:n_tr]), "val": set(pids[n_tr:n_tr + n_va]),
             "test": set(pids[n_tr + n_va:])}
    return {k: [s["sid"] for s in studies if s["pid"] in v] for k, v in parts.items()}


def default_out() -> Path:
    from utils.paths import out_root

    return out_root() / "deeppet" / "data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="dataset root holding imagesTr/ imagesTs/")
    ap.add_argument("--out", default=None, help="default: $D710_OUT/deeppet/data")
    ap.add_argument("--limit", type=int, default=None, help="first N studies only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-activity", type=float, default=20.0,
                    help="drop slices below this many SUV.cm^2 (default 20)")
    ap.add_argument("--kvp", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    root = Path(os.path.expanduser(a.data))
    out = Path(os.path.expanduser(a.out)) if a.out else default_out()
    (out / "slices").mkdir(parents=True, exist_ok=True)
    studies = find_studies(root)
    if not studies:
        raise SystemExit(f"error: no *_0001.nii.gz under {root}/imagesTr or imagesTs")
    if a.limit:
        studies = studies[: a.limit]
    print(f"{len(studies)} studies, {len({s['pid'] for s in studies})} patients -> {out}")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(prepare_study, s, str(out), a.min_activity, a.kvp) for s in studies]
        for i, (s, f) in enumerate(zip(studies, futs)):
            try:
                r = f.result()
            except Exception as e:  # one bad study must not stop the other 232
                print(f"  [{i + 1}/{len(studies)}] {s['sid']}: FAILED {e}", file=sys.stderr)
                continue
            rows.append(r)
            print(f"  [{i + 1}/{len(studies)}] {r['sid']}: {r['n']}/{r['n_source']} slices, "
                  f"{r['pixel_mm']:.3f} mm ({r['source_fov_mm']:.0f} mm FOV), "
                  f"activity kept {r['activity_ratio']:.3f}, SUVmax {r['suv_max']:.1f}  "
                  f"{time.time() - t0:.0f} s")

    (out / "index.json").write_text(json.dumps(rows, indent=1))
    sp = split(rows, a.seed)
    (out / "split.json").write_text(json.dumps(sp, indent=1))
    n = {k: sum(r["n"] for r in rows if r["sid"] in set(v)) for k, v in sp.items()}
    print(f"split by patient: " + ", ".join(f"{k} {len(v)} studies / {n[k]} slices"
                                            for k, v in sp.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
