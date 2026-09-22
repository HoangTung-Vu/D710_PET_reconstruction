"""Simulated raw data against the real raw data of the same exam and bed.

"Real" is what `d710 decode` made of `data/cases/<case>/raw`: GE's RDF can only
be read through the vendor decoder, so `$D710_OUT/<case>/decoded/` is the
reference. A simulation may cover a shorter time than the real frame (GATE
often does); every count is then scaled to the real frame by the ratio of the
two decay integrals, and that factor is reported.

Per bed and per simulation:

  counts     prompts, delays, delays/prompts; scatter fraction (GATE or pp
             truth against GE's own scatter estimate)
  singles    per crystal, real `bed<n>.singles.npy` against GATE's rates
  shapes     axial (per plane), radial (per tangential bin) and angular (per
             view) profiles of the prompt sinogram, and the view-summed
             sinogram (plane x tang): Pearson r, and the relative L2 once
             both totals are equal
  TOF        the TOF bin of every event, oriented to its sinogram bin (det1 to
             det2), so that the shape does not depend on which crystal the
             decoder happens to list first
  crystals   hits per crystal (24 x 576): per ring and per position in a block
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from utils.binmap import BinMap
from utils.paths import Case

from . import events_io as eio

TARGETS = {"prompts_ratio": (0.9, 1.1), "delays_fraction_diff": (-0.05, 0.05),
           "profile_r": 0.98, "singles_ratio": (0.9, 1.1)}


def decay_integral(t_s: float, half_life_s: float) -> float:
    lam = math.log(2.0) / half_life_s
    return (1.0 - math.exp(-lam * t_s)) / lam


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rel_l2(a, b) -> float:
    """`|a' - b| / |b|` with `a'` scaled to `b`'s total."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    a = a * (b.sum() / max(a.sum(), 1e-30))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def sinogram(C: Case, bed: int, binmap: BinMap) -> np.ndarray:
    a = np.fromfile(C.decoded / f"bed{bed}.s", "<i2").astype(np.float32)
    return a.reshape(-1, *binmap.shape).sum(axis=0)


def profiles(s) -> dict:
    return {"axial": s.sum(axis=(1, 2)), "angular": s.sum(axis=(0, 2)),
            "radial": s.sum(axis=(0, 1)), "mashed": s.sum(axis=1)}


def oriented_tof(e, binmap: BinMap, n_max: int = 5_000_000, seed: int = 0):
    """`(hist over -27..27, mean oriented bin per tangential bin)` from up to `n_max` events."""
    if len(e) > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(len(e), n_max, replace=False))
        e = e[idx]
    b, swap = binmap.flat(np.asarray(e["xtal_a"]), np.asarray(e["xtal_b"]),
                          with_swap=True)
    ok = b >= 0
    t = np.where(swap, -np.asarray(e["tof_bin"], np.int64),
                 np.asarray(e["tof_bin"], np.int64))[ok]
    h = np.bincount(t + eio.TOF_HALF, minlength=2 * eio.TOF_HALF + 1).astype(np.float64)
    u = (b[ok] % binmap.n_tang).astype(np.int64)
    n = np.bincount(u, minlength=binmap.n_tang)
    m = np.bincount(u, weights=t, minlength=binmap.n_tang) / np.maximum(n, 1)
    return h, m


def crystal_hits(e) -> np.ndarray:
    from utils.scanner import NXTAL

    return (np.bincount(np.asarray(e["xtal_a"]), minlength=NXTAL)
            + np.bincount(np.asarray(e["xtal_b"]), minlength=NXTAL)).astype(np.float64)


def block_profile(per_crystal) -> np.ndarray:
    """Mean over crystals of each of the 9 transaxial positions inside a block."""
    from utils.scanner import NDET, NRINGS

    return per_crystal.reshape(NRINGS, NDET // 9, 9).mean(axis=(0, 1))


def compare_bed(real: Case, sims: dict, bed: int, out_dir: Path, out=print) -> dict:
    """`sims` maps a label to a simulated `Case`; writes `bed<n>.json` and `bed<n>.png`."""
    binmap = BinMap(real.prompt(bed))
    hr = real.header(bed)
    half = float(hr["half_life_s"])
    fr = float(hr["frame_duration_ms"]) / 1000.0
    s_real = sinogram(real, bed, binmap)
    p_real = profiles(s_real)
    e_real = np.load(real.decoded / f"bed{bed}.lm.npy", mmap_mode="r")
    tof_real = oriented_tof(e_real, binmap)
    hits_real = crystal_hits(e_real)
    sing_p = real.decoded / f"bed{bed}.singles.npy"
    singles_real = np.load(sing_p).astype(np.float64) / fr if sing_p.exists() else None
    ge_sc = float(np.fromfile(real.work_bed(bed) / "scatter.s", "<f4").sum(dtype=np.float64))
    ge_rn = float(np.fromfile(real.work_bed(bed) / "randoms.s", "<f4").sum(dtype=np.float64))

    report = {"case": real.name, "bed": bed, "real": {
        "prompts": int(hr["prompts"]), "delays": int(hr["delays"]),
        "delays_fraction": hr["delays"] / hr["prompts"], "frame_s": fr,
        "scatter_fraction_ge": ge_sc / max(hr["prompts"] - ge_rn, 1),
        "singles_mcps": None if singles_real is None else float(singles_real.sum() / 1e6)},
        "sims": {}, "targets": TARGETS}
    curves = {"real": (p_real, tof_real, hits_real, singles_real)}

    for label, S in sims.items():
        hs = S.header(bed)
        fs = float(hs["frame_duration_ms"]) / 1000.0
        k = decay_integral(fr, half) / decay_integral(fs, half)
        s_sim = sinogram(S, bed, binmap)
        p_sim = profiles(s_sim)
        e_sim = np.load(S.decoded / f"bed{bed}.lm.npy", mmap_mode="r")
        tof_sim = oriented_tof(e_sim, binmap)
        hits_sim = crystal_hits(e_sim)
        sim_meta = hs.get("simulated", {})
        singles_sim = None
        gsum = S.raw_sim / "gate" / f"bed{bed}" / "summary.npz"
        if gsum.exists():
            singles_sim = np.load(gsum)["singles_rate"].astype(np.float64)
        truth = S.raw_sim / f"bed{bed}_truth.npz"
        sf, by_label = None, {}
        if truth.exists():
            z = np.load(truth)
            if "is_scatter" in z.files:
                sc, rn = z["is_scatter"], z["is_random"]
                lab = np.where(rn, 2, np.where(sc, 1, 0))
            else:
                lab = z["label"]
            sf = float((lab == 1).sum() / max((lab != 2).sum(), 1))
            for i, name in enumerate(("true", "scatter", "random")):
                h = crystal_hits(e_sim[lab == i]).reshape(24, -1).sum(1)
                by_label[name] = (h / max(h.sum(), 1)).tolist()
        row = {
            "frame_s": fs, "scale_to_real_frame": k,
            "prompts_scaled": hs["prompts"] * k,
            "prompts_ratio": hs["prompts"] * k / hr["prompts"],
            "delays_scaled": hs.get("delays", 0) * k,
            "delays_fraction": hs.get("delays", 0) / max(hs["prompts"], 1),
            "delays_fraction_diff": hs.get("delays", 0) / max(hs["prompts"], 1)
            - hr["delays"] / hr["prompts"],
            "scatter_fraction_truth": sf,
            "singles_mcps": None if singles_sim is None else float(singles_sim.sum() / 1e6),
            "singles_ratio": None if singles_sim is None or singles_real is None
            else float(singles_sim.sum() / singles_real.sum()),
            "singles_per_crystal_r": None if singles_sim is None or singles_real is None
            else pearson(singles_sim, singles_real),
            "profile_r": {n: pearson(p_sim[n], p_real[n]) for n in p_sim},
            "profile_rel_l2": {n: rel_l2(p_sim[n], p_real[n]) for n in p_sim},
            "tof_hist_r": pearson(tof_sim[0], tof_real[0]),
            "tof_mean_by_tang_r": pearson(tof_sim[1], tof_real[1]),
            "crystal_hits_r": pearson(hits_sim, hits_real),
            "ring_profile_r": pearson(hits_sim.reshape(24, -1).sum(1),
                                      hits_real.reshape(24, -1).sum(1)),
            "block_profile_r": pearson(block_profile(hits_sim), block_profile(hits_real)),
            "ring_profile_real": (hits_real.reshape(24, -1).sum(1)
                                  / hits_real.sum()).tolist(),
            "ring_profile_by_label": by_label,
            "simulated": sim_meta,
        }
        row["verdict"] = verdict(row)
        report["sims"][label] = row
        curves[label] = (p_sim, tof_sim, hits_sim, singles_sim)
        out(f"  {label}: prompts x{row['prompts_ratio']:.3f} of real "
            f"(scaled x{k:.2f}), delays/prompts {row['delays_fraction']:.3f} vs "
            f"{report['real']['delays_fraction']:.3f}, axial r "
            f"{row['profile_r']['axial']:.4f}, radial r {row['profile_r']['radial']:.4f}, "
            f"singles {row['singles_mcps']} vs {report['real']['singles_mcps']} Mcps")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"bed{bed}.json").write_text(json.dumps(report, indent=2, default=float))
    figure(curves, out_dir / f"bed{bed}.png", f"{real.name} bed {bed}")
    return report


def verdict(row) -> dict:
    t = TARGETS
    v = {"prompts": t["prompts_ratio"][0] <= row["prompts_ratio"] <= t["prompts_ratio"][1],
         "delays_fraction": t["delays_fraction_diff"][0] <= row["delays_fraction_diff"]
         <= t["delays_fraction_diff"][1],
         "profiles": all(r >= t["profile_r"] for n, r in row["profile_r"].items()
                         if n in ("axial", "radial"))}
    if row["singles_ratio"] is not None:
        v["singles"] = t["singles_ratio"][0] <= row["singles_ratio"] <= t["singles_ratio"][1]
    return v


def figure(curves: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for label, (p, tof, hits, singles) in curves.items():
        kw = {"lw": 1.2 if label == "real" else 0.9, "label": label}
        for a, n in zip(ax[0], ("axial", "radial", "angular")):
            a.plot(p[n] / p[n].sum(), **kw)
            a.set_title(f"{n} profile (normalised)")
        ax[1, 0].plot(np.arange(-eio.TOF_HALF, eio.TOF_HALF + 1), tof[0] / tof[0].sum(), **kw)
        ax[1, 0].set_title("TOF bin, oriented det1 to det2")
        ax[1, 1].plot(tof[1], **kw)
        ax[1, 1].set_title("mean oriented TOF bin per tangential bin")
        ring = hits.reshape(24, -1).sum(1)
        ax[1, 2].plot(ring / ring.sum(), marker=".", **kw)
        ax[1, 2].set_title("crystal hits per ring (normalised)")
    for a in ax.ravel():
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
