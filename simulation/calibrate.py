from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from . import analytic as an
from . import compare, sss
from . import events_io as eio
from . import singles as sg

TRAIN_BEDS = (2, 3, 5, 6)
WEIGHTS = ("1", "normdt")
SMOOTH_TANG = 5
EFF_FILE = "crystal_eff.npy"


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def smooth_tang(a, shape):
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(np.asarray(a, np.float32).reshape(shape), SMOOTH_TANG, axis=2)


def planes(a, shape):
    return np.asarray(a, np.float64).reshape(shape).sum((1, 2))


def tang(a, shape):
    return np.asarray(a, np.float64).reshape(shape).sum((0, 1))


def term(case, bed, name) -> np.ndarray:
    return np.fromfile(case.work_bed(bed) / f"{name}.s", "<f4").astype(np.float64)


def compare_terms(model, ge, shape) -> dict:
    return {"total_model": float(model.sum()), "total_ge": float(ge.sum()),
            "ratio": float(model.sum() / ge.sum()),
            "r_bin_smoothed": pearson(smooth_tang(model, shape), smooth_tang(ge, shape)),
            "r_plane": pearson(planes(model, shape), planes(ge, shape)),
            "r_tang": pearson(tang(model, shape), tang(ge, shape))}


def per_ring(rate) -> list:
    return np.asarray(rate, np.float64).reshape(24, -1).sum(1).tolist()


def fit_singles(G: dict, S: dict, beds, iters: int = 200):
    eff = np.ones_like(next(iter(G.values())))
    c_s, c_0 = 1.0, 0.0
    for _ in range(iters):
        X = np.concatenate([np.stack([eff * G[b], eff], 1) for b in beds])
        y = np.concatenate([S[b] for b in beds])
        (c_s, c_0), *_ = np.linalg.lstsq(X, y, rcond=None)
        if c_0 < 0:
            c_0 = 0.0
            c_s = float((X[:, 0] * y).sum() / (X[:, 0] ** 2).sum())
        m = {b: c_s * G[b] + c_0 for b in beds}
        eff = sum(S[b] * m[b] for b in beds) / sum(m[b] ** 2 for b in beds)
        k = eff.mean()
        eff, c_s, c_0 = eff / k, c_s * k, c_0 * k
    return eff, float(c_s), float(c_0)


def run(case, beds=None, train=TRAIN_BEDS, image_step: int = sss.IMAGE_STEP,
        crystal_step: int = sss.CRYSTAL_STEP, ring_step: int = sss.RING_STEP,
        margin_planes: int = an.SSS_MARGIN_PLANES, use_shield: bool = True,
        out_json: Path = an.CALIB_JSON, out=print, device=None) -> dict:
    beds = list(beds or case.decoded_beds())
    train = [b for b in train if b in beds]
    test = [b for b in beds if b not in train]
    rep_dir = eio.sim_root(case) / "calib"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rows, G, S, prof = {}, {}, {}, {}
    for bed in beds:
        t0 = time.time()
        out(f"calibrate bed {bed}")
        b = an.Bed(case, bed)
        shape = b.binmap.shape
        prompts = compare.sinogram(case, bed, b.binmap).reshape(-1).astype(np.float64)
        r_ge, s_ge = term(case, bed, "randoms"), term(case, bed, "scatter")
        t_bin, _ = an.trues(b, out)
        f, info = an.sss_sparse(b, image_step, crystal_step, ring_step,
                                margin_planes=margin_planes, use_shield=use_shield,
                                device=device, out=out)
        sss_bin = an.sss_bins(b, f, info)
        rate = an.measured_rate(b)
        r_meas = an.randoms_bins(b, rate)
        G[bed], S[bed] = an.geometric(b, use_shield, out, device), rate
        real_trues = prompts - r_ge - s_ge
        row = {"prompts": float(prompts.sum()), "randoms_ge": float(r_ge.sum()),
               "scatter_ge": float(s_ge.sum()), "trues_real": float(real_trues.sum()),
               "trues_model_unscaled": float(t_bin.sum()),
               "kappa_bed": float(real_trues.sum() / t_bin.sum()),
               "trues_r_plane": pearson(planes(real_trues, shape), planes(t_bin, shape)),
               "randoms_tripwire": compare_terms(r_meas, r_ge, shape),
               "sss_points": info["scatter_points"], "sss_seconds": info["seconds"],
               "scatter": {}}
        prof[bed] = {"real_trues": planes(real_trues, shape), "trues": planes(t_bin, shape),
                     "scatter_ge": planes(s_ge, shape), "randoms_ge": planes(r_ge, shape),
                     "randoms_meas": planes(r_meas, shape)}
        for w in WEIGHTS:
            m = sss_bin * an.scatter_weight(b, w)
            c = compare_terms(m, s_ge, shape)
            c["unscaled_sum"] = float(m.sum())
            row["scatter"][w] = c
            prof[bed][f"scatter_{w}"] = planes(m, shape)
        row["wall_s"] = round(time.time() - t0, 1)
        rows[bed] = row
        tw = row["randoms_tripwire"]
        out(f"  bed {bed}: kappa_bed {row['kappa_bed']:.6g}  randoms tripwire ratio "
            f"{tw['ratio']:.4f} r {tw['r_bin_smoothed']:.4f}  ({row['wall_s']:.0f} s)")

    kappa = (sum(rows[b]["trues_real"] for b in train)
             / sum(rows[b]["trues_model_unscaled"] for b in train))
    ks = {w: sum(rows[b]["scatter_ge"] for b in train)
          / sum(rows[b]["scatter"][w]["unscaled_sum"] for b in train) for w in WEIGHTS}
    mean_r = {w: float(np.mean([rows[b]["scatter"][w]["r_plane"] for b in train]))
              for w in WEIGHTS}
    w_best = max(WEIGHTS, key=lambda w: mean_r[w])
    eff, c_s, c_0 = fit_singles(G, S, train)
    np.save(out_json.with_name(EFF_FILE), eff.astype(np.float32))

    calib = {"case": case.name, "made": date.today().isoformat(), "beds": beds,
             "train_beds": train, "test_beds": test, "kappa": float(kappa),
             "scatter": {"w": w_best, "k_s": float(ks[w_best]), "k_s_by_w": ks,
                         "mean_r_plane_train": mean_r, "image_step": image_step,
                         "crystal_step": crystal_step, "ring_step": ring_step,
                         "e_res": sss.E_RES, "cutoff": sss.MU_CUTOFF,
                         "margin_planes": int(margin_planes), "shield": bool(use_shield)},
             "singles": {"c_s": c_s, "c_0": c_0, "eff_file": EFF_FILE, "shield": bool(use_shield),
                         "down_zyx": list(sg.DOWN_ZYX), "crystal_step": sg.CRYSTAL_STEP},
             "window_ns": an.WINDOW_NS,
             "shield_mm": {"z": an.SHIELD_Z_MM, "gap": an.SHIELD_GAP_MM,
                           "radius": "patient_port_mm / 2"}}
    calib_rt = dict(calib, eff=eff)

    for bed in beds:
        b = an.Bed(case, bed)
        shape = b.binmap.shape
        row = rows[bed]
        row["kappa_ratio"] = row["kappa_bed"] / kappa
        m_rate = an.model_rate(b, calib_rt, out)
        row["singles"] = {"model_over_measured": float(m_rate.sum() / S[bed].sum()),
                          "r_crystal": pearson(m_rate, S[bed]),
                          "ring_ratio": (np.array(per_ring(m_rate))
                                         / np.array(per_ring(S[bed]))).tolist()}
        r_model = an.randoms_bins(b, m_rate)
        r_ge = term(case, bed, "randoms")
        row["randoms_model"] = compare_terms(r_model, r_ge, shape)
        prof[bed]["randoms_model"] = planes(r_model, shape)
        s_model = rows[bed]["scatter"][w_best]["unscaled_sum"] * ks[w_best]
        pred = kappa * row["trues_model_unscaled"] + s_model + float(r_model.sum())
        row["prompts_pred_over_real"] = pred / row["prompts"]
        row["scatter_fraction"] = {"ge": row["scatter_ge"] / (row["prompts"] - row["randoms_ge"]),
                                   "model": s_model / (pred - float(r_model.sum()))}
        row["randoms_fraction"] = {"ge": row["randoms_ge"] / row["prompts"],
                                   "model": float(r_model.sum()) / pred}
        out(f"  bed {bed}: kappa {row['kappa_ratio']:.4f}  prompts pred/real "
            f"{row['prompts_pred_over_real']:.4f}  singles {row['singles']['model_over_measured']:.4f}"
            f"  randoms model/GE {row['randoms_model']['ratio']:.4f}")

    eff_r = {}
    for i, a in enumerate(train):
        for c in train[i + 1:]:
            ea = S[a] / (c_s * G[a] + c_0)
            ec = S[c] / (c_s * G[c] + c_0)
            eff_r[f"{a}-{c}"] = pearson(ea, ec)
    calib["singles"]["eff_r_between_train_beds"] = eff_r
    out_json.write_text(json.dumps(calib, indent=1))
    (rep_dir / "report.json").write_text(json.dumps({"calib": calib, "beds": rows},
                                                    indent=1, default=float))
    np.savez_compressed(rep_dir / "profiles.npz",
                        **{f"bed{b}_{k}": v for b, d in prof.items() for k, v in d.items()})
    figure(prof, rows, calib, rep_dir / "calib.png")
    out(f"-> {out_json}\n-> {rep_dir}")
    return calib


def figure(prof, rows, calib, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beds = sorted(prof)
    fig, ax = plt.subplots(3, len(beds), figsize=(3.2 * len(beds), 8), squeeze=False)
    w, ks, kappa = calib["scatter"]["w"], calib["scatter"]["k_s"], calib["kappa"]
    for j, bed in enumerate(beds):
        p = prof[bed]
        tag = "train" if bed in calib["train_beds"] else "test"
        ax[0, j].plot(p["real_trues"], lw=0.8, label="prompts-R-S (GE)")
        ax[0, j].plot(kappa * p["trues"], lw=0.8, label="kappa n AF Px")
        ax[0, j].set_title(f"bed {bed} ({tag})")
        ax[1, j].plot(p["scatter_ge"], lw=0.8, label="GE scatter")
        ax[1, j].plot(ks * p[f"scatter_{w}"], lw=0.8, label=f"SSS x k_s (w={w})")
        ax[2, j].plot(p["randoms_ge"], lw=0.8, label="GE randoms")
        ax[2, j].plot(p["randoms_meas"], lw=0.8, ls="--", label="2w S S (measured)")
        ax[2, j].plot(p["randoms_model"], lw=0.8, label="2w S S (model)")
        for i in range(3):
            ax[i, j].set_xlabel("plane")
    for i in range(3):
        ax[i, 0].legend(fontsize=7)
    ax[0, 0].set_ylabel("trues per plane")
    ax[1, 0].set_ylabel("scatter per plane")
    ax[2, 0].set_ylabel("randoms per plane")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
