from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from utils.attenuation import hu_to_mu
from utils.binmap import BinMap
from utils.scanner import NRINGS, NSEG0, RING_PITCH_MM

from . import events_io as eio
from . import phantom as ph
from . import pp, singles, sss
from .gate.run import WINDOW_NS

CALIB_JSON = Path(__file__).with_name("analytic_calib.json")
METHOD = "an"
SINGLES_MODES = ("model", "measured")
LABELS = pp.LABELS
SSS_MARGIN_PLANES = 24
SHIELD_GAP_MM = 30.0
SHIELD_Z_MM = NRINGS * RING_PITCH_MM / 2 + SHIELD_GAP_MM


def load_calib(path=CALIB_JSON) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: no {p}\n  run: d710 simulate calibrate --case fdg26081901")
    c = json.loads(p.read_text())
    c["eff"] = np.load(p.with_name(c["singles"]["eff_file"])).astype(np.float64)
    return c


class Bed:
    def __init__(self, case, bed: int, seconds: float | None = None):
        self.case, self.bed = case, bed
        self.pdir = ph.directory(eio.sim_root(case), bed)
        self.meta = ph.load(self.pdir)
        self.half = float(self.meta["timing"]["half_life_s"])
        self.frame = float(seconds or self.meta["frame_s"])
        self.k_frame = ph.frame_integral_s(self.frame, self.half) / self.meta["frame_integral_s"]
        native = ph.decays_per_voxel(self.pdir, self.meta) / np.float32(ph.VOXEL_ML * 1000.0)
        self.x_native = pp.projector_image(native)
        self.x = pp.projector_image(native * np.float32(self.k_frame))
        self.mu = pp.projector_image(np.load(self.pdir / "mu_bed.npy"))
        self.binmap = BinMap(case.prompt(bed))
        self.pairs = pp.RingPairs(self.binmap)
        self.lut = pp.crystal_lut()
        self.normdt = np.fromfile(case.work_bed(bed) / "normdt.s", "<f4")
        self.normdt_lor = pp.per_lor(self.normdt, self.binmap)
        self.i2 = ph.decay_integral(0.0, self.frame, self.half, power=2)
        self.e_low = float(case.header(bed).get("energy_window_low_kev") or sss.E_LOW_KEV)

    def key(self, **params) -> str:
        h = hashlib.sha1((self.pdir / "phantom.json").read_bytes())
        h.update(json.dumps(params, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def per_bin(self, flat):
        return np.asarray(flat, np.float32).reshape(self.binmap.shape)


def trues(b: Bed, out=print):
    import parallelproj

    acc_t = np.zeros(b.binmap.n_bin, np.float64)
    acc_af = np.zeros(b.binmap.n_bin, np.float64)
    t0 = time.time()
    for i, (_p, a, c, bins) in enumerate(b.pairs):
        xs, xe = b.lut[a], b.lut[c]
        af = np.exp(-parallelproj.joseph3d_fwd(xs, xe, *b.mu))
        proj = np.maximum(parallelproj.joseph3d_fwd(xs, xe, *b.x), 0.0)
        acc_af[bins] += af
        acc_t[bins] += b.normdt_lor[bins] * af * proj
    out(f"    trues: {len(b.pairs)} ring pairs in {time.time() - t0:.0f} s")
    af_bin = (acc_af.reshape(b.binmap.shape) / b.binmap.mult[:, None, None]).reshape(-1)
    return acc_t, af_bin.astype(np.float32)


def shield(b: Bed):
    port = b.case.header(b.bed).get("patient_port_mm")
    return (SHIELD_Z_MM, float(port) / 2.0) if port else None


def sss_images(b: Bed, margin_planes: int):
    if margin_planes <= 0:
        return b.x_native, b.mu
    first = -ph.grid_planes(b.meta["margin_mm"])[0]
    m = min(int(margin_planes), first)
    sl = slice(first - m, first + NSEG0 + m)
    act = ph.read_mhd(b.pdir / "act_bqml.mhd")[sl]
    hu = ph.read_mhd(b.pdir / "ct_hu.mhd")[sl]
    k = (b.meta["decay_scan_to_bed"] * b.meta["timing"]["positron_fraction"]
         * b.meta["frame_integral_s"] / 1000.0)
    mu = hu_to_mu(hu, b.meta.get("ct_kvp", 120.0)).astype(np.float32)
    return pp.projector_image(act * np.float32(k)), pp.projector_image(mu)


def sss_sparse(b: Bed, image_step: int = sss.IMAGE_STEP, crystal_step: int = sss.CRYSTAL_STEP,
               ring_step: int = sss.RING_STEP, seed: int = 0,
               margin_planes: int = SSS_MARGIN_PLANES, use_shield: bool = False, out=print):
    sh = shield(b) if use_shield else None
    params = {"image_step": image_step, "crystal_step": crystal_step, "ring_step": ring_step,
              "seed": seed, "e_low": b.e_low, "e_res": sss.E_RES, "cutoff": sss.MU_CUTOFF}
    if margin_planes > 0 or sh is not None:
        params.update(margin_planes=int(margin_planes), shield=sh)
    d = eio.sim_root(b.case) / "sss"
    p = d / f"bed{b.bed}_{b.key(**params)}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return z["f"], json.loads(str(z["info"]))
    x, mu = sss_images(b, margin_planes)
    f, info = sss.simulate_sparse(x, mu, b.lut, sss.sample_rings(ring_step),
                                  sss.sample_crystals(crystal_step), image_step,
                                  sss.MU_CUTOFF, b.e_low, sss.E_RES, seed, shield=sh, out=out)
    info["margin_planes"] = int(margin_planes)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(p, f=f, info=json.dumps(info))
    out(f"    sss: {info['scatter_points']} scatter points in {info['seconds']:.0f} s -> {p.name}")
    return f, info


def sss_bins(b: Bed, f, info) -> np.ndarray:
    return sss.to_bins(f, info, b.pairs).astype(np.float64) * b.k_frame


def scatter_weight(b: Bed, w: str) -> np.ndarray:
    if w == "normdt":
        return b.normdt_lor.astype(np.float64)
    return np.ones(b.binmap.n_bin, np.float64)


def geometric(b: Bed, use_shield: bool = True, out=print) -> np.ndarray:
    sh = shield(b) if use_shield else None
    params = {"down": singles.DOWN_ZYX, "crystal_step": singles.CRYSTAL_STEP,
              "keep": singles.KEEP_FRACTION}
    if sh is not None:
        params["shield"] = sh
    d = eio.sim_root(b.case) / "singles"
    p = d / f"bed{b.bed}_{b.key(**params)}.npy"
    if p.exists():
        return np.load(p)
    A, mu = singles.coarse_phantom(b.pdir, b.meta)
    G = singles.geometric_singles(A, mu, b.lut, shield=sh, out=out)
    d.mkdir(parents=True, exist_ok=True)
    np.save(p, G)
    return G


def randoms_bins(b: Bed, rate) -> np.ndarray:
    return pp.randoms_per_bin(b.pairs, pp.singles_randoms(rate, WINDOW_NS, b.i2, None)).reshape(-1)


def measured_rate(b: Bed) -> np.ndarray:
    fr = float(b.case.header(b.bed)["frame_duration_ms"]) / 1000.0
    return singles.rate_at_bed_start(singles.measured(b.case, b.bed), fr, b.half)


def model_rate(b: Bed, calib: dict, out=print) -> np.ndarray:
    s = calib["singles"]
    G = geometric(b, s.get("shield", False), out)
    return singles.model(G, calib["eff"], s["c_s"], s["c_0"])


def run_bed(case, dst, bed: int, seconds: float | None, seed: int, calib: dict,
            singles_mode: str = "model", out=print) -> dict:
    t0 = time.time()
    b = Bed(case, bed, seconds)
    rng = np.random.default_rng(seed)
    sc = calib["scatter"]
    f, info = sss_sparse(b, sc["image_step"], sc["crystal_step"], sc["ring_step"],
                         margin_planes=sc.get("margin_planes", 0),
                         use_shield=sc.get("shield", False), out=out)
    scatter_bin = sc["k_s"] * scatter_weight(b, sc["w"]) * sss_bins(b, f, info)
    rate = measured_rate(b) if singles_mode == "measured" else model_rate(b, calib, out)
    randoms_fn = pp.singles_randoms(rate, WINDOW_NS, b.i2, None)
    randoms_bin = pp.randoms_per_bin(b.pairs, randoms_fn)
    af_acc = np.zeros(b.binmap.n_bin, np.float64)
    xa, xb, tof, lab, expected = pp.simulate(
        b.pairs, b.lut, b.mu, b.x, b.normdt_lor, calib["kappa"], rng, randoms_fn,
        pp.per_lor(scatter_bin.astype(np.float32), b.binmap), None, 1, out, af_out=af_acc)
    af_bin = (af_acc.reshape(b.binmap.shape) / b.binmap.mult[:, None, None]).astype(np.float32)
    xa, xb, tof = eio.swap_randomly(xa, xb, tof.astype(np.int64), rng)
    t_ms = int(b.meta["bed_start_ticks"]) + pp.sample_times(len(xa), b.frame, b.half, rng)
    ev = eio.events(xa, xb, tof, t_ms)
    counts = {n: int((lab == i).sum()) for i, n in enumerate(LABELS)}
    header = {"frame_duration_ms": b.frame * 1000.0, "delays": int(rng.poisson(expected[2])),
              "simulated": {"method": "analytic", "seconds": b.frame, "seed": seed,
                            "kappa": calib["kappa"], "scatter": {k: sc[k] for k in ("w", "k_s")},
                            "singles": singles_mode, "sss": info,
                            "expected": dict(zip(LABELS, map(float, expected))),
                            "counts": counts, "tof_bins": 1,
                            "calib_case": calib.get("case"), "calib_made": calib.get("made")}}
    terms = {"attn": af_bin, "randoms": randoms_bin,
             "scatter": b.per_bin(scatter_bin)}
    row = eio.write_bed(case, dst, bed, ev, terms, header, {"label": lab},
                        README.format(case=case.name, seed=seed, seconds=b.frame,
                                      singles=singles_mode))
    np.save(dst.raw_sim / f"bed{bed}_singles_rate.npy", rate.astype(np.float32))
    row.update(header["simulated"])
    row["wall_s"] = round(time.time() - t0, 1)
    out(f"  bed {bed}: {row['prompts']:,} prompts in {b.frame:g} s -- {counts}  "
        f"({row['wall_s']:.0f} s)")
    eio.manifest(dst, {"source_case": case.name, "method": "analytic", "seed": seed,
                       "beds": [row]})
    return row


README = """\
Simulated raw data -- written by `d710 simulate analytic`. NOT measured data.
Analytic model of the D710 over the CT and PET of case {case!r}; seed {seed},
{seconds:g} s simulated, non-TOF, singles: {singles}. Trues are parallelproj
line integrals, scatter is single-scatter simulation, randoms are 2w S_a S_b;
the constants come from simulation/analytic_calib.json.

  ../decoded/bed<n>.lm.npy        events (tof_bin 0): trues, scatter, randoms
  ../decoded/bed<n>.s             histogrammed from those events
  ../work/bed<n>/normdt           copied from the source case
  ../work/bed<n>/attn             the attenuation factors the model used
  ../work/bed<n>/randoms          the randoms mean the events were drawn from
  ../work/bed<n>/scatter          the scatter mean the events were drawn from
  bed<n>_truth.npz                per event: label 0 true, 1 scatter, 2 random
  bed<n>_singles_rate.npy         singles per crystal at bed start, cps
"""
