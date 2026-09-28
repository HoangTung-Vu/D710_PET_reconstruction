from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.binmap import BinMap
from utils.paths import Case

from . import analytic as an
from . import calibrate as cal
from . import compare
from . import events_io as eio

TAIL_AF = 0.995
POISSON_SD = 5.0


def _read(C: Case, bed: int, name: str) -> np.ndarray:
    p = C.work_bed(bed) / f"{name}.s"
    if not p.exists():
        raise SystemExit(f"error: no {p}")
    return np.fromfile(p, "<f4").astype(np.float64)


def ring_ratio(sim, ge, binmap: BinMap) -> list:
    r1, r2, pl = binmap.ring_pairs_by_plane()
    shape = binmap.shape
    s = np.asarray(sim).reshape(shape).sum((1, 2))
    g = np.asarray(ge).reshape(shape).sum((1, 2))
    out = []
    for r in range(binmap.nrings):
        p = int(pl[(r1 == r) & (r2 == r)][0])
        out.append(float(s[p] / g[p]) if g[p] > 0 else float("nan"))
    return out


def plane_ratio_range(sim, ge, shape) -> list:
    q = cal.planes(sim, shape) / np.maximum(cal.planes(ge, shape), 1e-12)
    return [float(np.percentile(q, 5)), float(np.median(q)), float(np.percentile(q, 95))]


def check_bed(real: Case, sim: Case, bed: int, calib_beds: set, out=print):
    binmap = BinMap(real.prompt(bed))
    shape = binmap.shape
    rs, ss = _read(sim, bed, "randoms"), _read(sim, bed, "scatter")
    rg, sg = _read(real, bed, "randoms"), _read(real, bed, "scatter")
    af = _read(sim, bed, "attn").reshape(shape)
    prompts = compare.sinogram(real, bed, binmap).astype(np.float64).reshape(shape)
    tail = af > TAIL_AF

    def tail_planes(a):
        return np.where(tail, np.asarray(a).reshape(shape), 0.0).sum((1, 2))

    pt = tail_planes(prompts)
    tail_sim = pt / np.maximum(tail_planes(rs + ss), 1e-12)
    tail_ge = pt / np.maximum(tail_planes(rg + sg), 1e-12)
    row = {"bed": bed, "role": "calibration" if bed in calib_beds else "held-out",
           "randoms": {**cal.compare_terms(rs, rg, shape),
                       "plane_ratio_p5_p50_p95": plane_ratio_range(rs, rg, shape),
                       "direct_plane_ratio_by_ring": ring_ratio(rs, rg, binmap)},
           "scatter": {**cal.compare_terms(ss, sg, shape),
                       "plane_ratio_p5_p50_p95": plane_ratio_range(ss, sg, shape)},
           "tail": {"bins": int(tail.sum()), "prompts": float(pt.sum()),
                    "prompts_over_sim_RS": float(pt.sum() / tail_planes(rs + ss).sum()),
                    "prompts_over_ge_RS": float(pt.sum() / tail_planes(rg + sg).sum()),
                    "plane_p5_p50_p95_sim": np.percentile(tail_sim, [5, 50, 95]).tolist(),
                    "plane_p5_p50_p95_ge": np.percentile(tail_ge, [5, 50, 95]).tolist()}}
    hs = sim.header(bed)
    exp = hs.get("simulated", {}).get("expected", {})
    truth = sim.raw_sim / f"bed{bed}_truth.npz"
    if truth.exists() and exp:
        lab = np.load(truth)["label"]
        labels = {}
        for i, name in enumerate(an.LABELS):
            n, mu = int((lab == i).sum()), float(exp[name])
            labels[name] = {"events": n, "expected": mu,
                            "z": (n - mu) / np.sqrt(mu) if mu > 0 else 0.0}
        row["labels"] = labels
        row["labels_within_poisson"] = bool(all(abs(v["z"]) < POISSON_SD
                                                for v in labels.values()))
    row["prompts_sim_over_real"] = float(hs["prompts"]) / float(real.header(bed)["prompts"])
    prof = {"randoms_sim": cal.planes(rs, shape), "randoms_ge": cal.planes(rg, shape),
            "scatter_sim": cal.planes(ss, shape), "scatter_ge": cal.planes(sg, shape),
            "tail_prompts": pt, "tail_sim": tail_planes(rs + ss), "tail_ge": tail_planes(rg + sg)}
    r, s = row["randoms"], row["scatter"]
    out(f"  bed {bed} ({row['role']}): randoms sim/GE {r['ratio']:.4f} r_plane {r['r_plane']:.4f}"
        f" | scatter sim/GE {s['ratio']:.4f} r_plane {s['r_plane']:.4f}"
        f" | tail prompts/(R+S) sim {row['tail']['prompts_over_sim_RS']:.4f}"
        f" GE {row['tail']['prompts_over_ge_RS']:.4f}")
    return row, prof


def run(real: Case, label: str = "an_s1", beds=None, out=print) -> dict:
    sim = Case(f"{real.name}_sim_{label}", real.root.parent)
    if not sim.decoded.is_dir():
        raise SystemExit(f"error: no simulated case {sim.root}\n"
                         f"  run: d710 simulate analytic --case {real.name}")
    calib_beds = set()
    cj = an.CALIB_JSON
    if cj.exists():
        c = json.loads(cj.read_text())
        if c.get("case") == real.name:
            calib_beds = set(c.get("train_beds", []))
    beds = beds or [b for b in sim.decoded_beds() if b in real.decoded_beds()]
    out_dir = eio.sim_root(real) / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for bed in beds:
        row, prof = check_bed(real, sim, bed, calib_beds, out)
        rows[bed] = row
        figure(prof, row, out_dir / f"terms_bed{bed}.png")
    rep = {"case": real.name, "sim": sim.name, "calibration_beds": sorted(calib_beds),
           "beds": rows}
    p = out_dir / f"terms_{label}.json"
    p.write_text(json.dumps(rep, indent=1, default=float))
    out(f"-> {p}")
    return rep


def figure(prof, row, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(prof["randoms_ge"], lw=0.8, label="GE")
    ax[0].plot(prof["randoms_sim"], lw=0.8, label="simulated")
    ax[0].set_title(f"randoms per plane, sim/GE {row['randoms']['ratio']:.3f}")
    ax[1].plot(prof["scatter_ge"], lw=0.8, label="GE")
    ax[1].plot(prof["scatter_sim"], lw=0.8, label="simulated")
    ax[1].set_title(f"scatter per plane, sim/GE {row['scatter']['ratio']:.3f}")
    ax[2].plot(prof["tail_prompts"], lw=0.8, label="real prompts")
    ax[2].plot(prof["tail_ge"], lw=0.8, label="GE R+S")
    ax[2].plot(prof["tail_sim"], lw=0.8, label="simulated R+S")
    ax[2].set_title(f"outside the body (AF > {TAIL_AF})")
    for a in ax:
        a.set_xlabel("plane")
        a.legend(fontsize=7)
    fig.suptitle(f"bed {row['bed']} ({row['role']})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
