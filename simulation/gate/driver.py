"""`d710 simulate gate`: run GATE for one bed in resumable chunks, then convert.

The frame is simulated in chunks of `chunk_s` seconds, each its own process
with its own seed and its own start time, so that decay across the frame is
right and an interrupted run resumes at the first missing chunk. A separate
short run (`singles_s`) also writes every single, for the per-crystal singles
rates; writing them for the whole frame would cost ~25 GB per bed.

Measured on this laptop (16 threads), fdg26081008 bed 1 with the whole GE
image as the source (136 MBq of positrons) and the CT at 6.4 x 6.4 x 9.8 mm:
81,000 decays/s, so one simulated second takes ~28 minutes of wall time and
the full 90 s frame ~42 hours.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from utils.binmap import BinMap
from utils.paths import Case

from .. import events_io as eio
from .. import phantom as ph
from .. import pp
from . import coinc
from .geometry import ShieldSpec
from .run import WINDOW_NS, RunConfig

HERE = Path(__file__).resolve().parents[2]


def raw_dir(dst: Case, bed: int) -> Path:
    return dst.raw_sim / "gate" / f"bed{bed}"


def gate_python() -> list[str]:
    """How to start the interpreter that runs Geant4.

    `D710_GATE_RUNNER` replaces it with a command prefix, for a host whose
    glibc is too old for the opengate wheels: `./d710_apptainer` sets it to
    `apptainer exec --bind ... d710_gate.sif python` (see
    `simulation/gate/Dockerfile`). Everything else -- the conversion, the
    reconstruction, the comparison -- stays on the host.
    """
    runner = os.environ.get("D710_GATE_RUNNER", "").strip()
    return shlex.split(runner) if runner else [sys.executable]


def _launch(cfg: RunConfig, out=print) -> None:
    d = Path(cfg.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg.save(d / "cfg.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    cmd = gate_python() + ["-u", "-m", "simulation.gate.run", str(d / "cfg.json")]
    if os.environ.get("D710_GATE_RUNNER"):
        out(f"      {' '.join(cmd[:-3])} ...")
    with open(d / "log.txt", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           env=env, cwd=str(HERE))
    tail = (d / "log.txt").read_text().splitlines()[-3:]
    for ln in tail:
        out(f"      {ln}")
    if r.returncode != 0 or not (d / "run.json").exists():
        raise SystemExit(f"error: GATE run failed, see {d / 'log.txt'}")


def _done(d: Path, cfg: RunConfig) -> bool:
    p = d / "run.json"
    if not p.exists():
        return False
    return json.loads(p.read_text()).get("config") == json.loads(
        json.dumps(asdict(cfg)))


def run_bed(real: Case, dst: Case, bed: int, seconds: float | None,
            chunk_s: float, singles_s: float, threads: int, seed: int,
            shield: ShieldSpec, positron: bool, ct_step, out=print) -> Path:
    phantom_dir = ph.directory(eio.sim_root(real), bed)
    meta = ph.load(phantom_dir)
    total = float(seconds or meta["frame_s"])
    raw = raw_dir(dst, bed)
    key = (f"{meta['margin_mm']}:{meta['shape_zyx']}:"
           f"{meta['grid_positron_activity_at_bed_start_bq']:.6e}:{meta['pet']}")
    common = dict(phantom_dir=str(phantom_dir), threads=threads,
                  positron=positron, shield=asdict(shield), ct_step=list(ct_step),
                  phantom_key=key)

    cfg = RunConfig(out_dir=str(raw / "singles"), seconds=singles_s,
                    seed=seed * 1000 + 999, write_singles=True, **common)
    if _done(raw / "singles", cfg):
        out(f"  bed {bed}: singles run already done")
    else:
        out(f"  bed {bed}: singles run, {singles_s:g} s simulated")
        _launch(cfg, out)

    n = max(1, math.ceil(total / chunk_s - 1e-9))
    for k in range(n):
        t0 = k * chunk_s
        cfg = RunConfig(out_dir=str(raw / f"chunk_{k:03d}"), t0=t0,
                        seconds=min(chunk_s, total - t0), seed=seed * 1000 + k,
                        **common)
        if _done(raw / f"chunk_{k:03d}", cfg):
            out(f"  bed {bed}: chunk {k + 1}/{n} already done")
            continue
        out(f"  bed {bed}: chunk {k + 1}/{n}, t = {t0:g}..{t0 + cfg.seconds:g} s")
        _launch(cfg, out)
    for extra in sorted(raw.glob("chunk_*"))[n:]:
        out(f"  note: {extra.name} lies beyond --seconds {total:g} and is ignored")
    return raw


def randoms_per_bin(pairs: pp.RingPairs, f) -> np.ndarray:
    acc = np.zeros(pairs.binmap.n_bin, np.float64)
    for _p, a, c, bins in pairs:
        acc[bins] += f(a, c, bins)
    return acc.reshape(pairs.binmap.shape).astype(np.float32)


def convert_bed(real: Case, dst: Case, bed: int, seconds: float | None,
                seed: int, out=print) -> dict:
    """Chunks to `decoded/bed<n>.*` and `work/bed<n>/*`, plus `summary.npz` for `pp`."""
    phantom_dir = ph.directory(eio.sim_root(real), bed)
    meta = ph.load(phantom_dir)
    raw = raw_dir(dst, bed)
    total = float(seconds or meta["frame_s"])
    n = max(1, math.ceil(total / json.loads(
        (raw / "chunk_000" / "cfg.json").read_text())["seconds"] - 1e-9))
    chunks = [raw / f"chunk_{k:03d}" for k in range(n)]
    missing = [c.name for c in chunks if not (c / "run.json").exists()]
    if missing:
        raise SystemExit(f"error: chunks not simulated yet: {missing}")
    t_sim = sum(coinc.read_run(c)["config"]["seconds"] for c in chunks)
    half = meta["timing"]["half_life_s"]
    rng = np.random.default_rng(seed)

    binmap = BinMap(real.prompt(bed))
    normdt = np.fromfile(real.work_bed(bed) / "normdt.s", "<f4")
    accept = coinc.norm_acceptance(normdt, binmap)
    cols, stats = coinc.convert(chunks, binmap, accept,
                                int(meta["bed_start_ticks"]), rng, out)
    xa, xb, tof = eio.swap_randomly(cols["xa"], cols["xb"], cols["tof"], rng)
    ev = eio.events(xa, xb, tof, cols["t"])
    rnd, sc = cols["rnd"].astype(bool), cols["sc"].astype(bool)
    true = ~rnd & ~sc

    run_s = coinc.read_run(raw / "singles")["config"]["seconds"]
    rate = coinc.singles_rate(raw / "singles", run_s)
    i2 = coinc.decay_integral(0.0, t_sim, half, power=2)
    pairs = pp.RingPairs(binmap)
    randoms = randoms_per_bin(pairs, pp.singles_randoms(rate, WINDOW_NS, i2, accept))
    scatter = coinc.smooth_scatter(xa[sc], xb[sc], binmap, 1.0)
    phi = coinc.tof_profile(xa[sc], xb[sc], tof[sc], binmap)

    mu = pp.projector_image(np.load(phantom_dir / "mu_bed.npy"))
    af = attenuation_per_bin(pairs, pp.crystal_lut(), mu)

    header = {"frame_duration_ms": t_sim * 1000.0, "delays": stats["delays"],
              "simulated": {"method": "gate", "seconds": t_sim, **stats,
                            "trues": int(true.sum()), "scatter": int(sc.sum()),
                            "randoms_true": int(rnd.sum()),
                            "singles_cps": float(rate.sum()),
                            "randoms_from_singles": float(randoms.sum())}}
    readme = README.format(case=real.name, seed=seed, seconds=t_sim)
    row = eio.write_bed(real, dst, bed, ev,
                        {"randoms": randoms, "scatter": scatter, "attn": af},
                        header, {"is_random": rnd, "is_scatter": sc}, readme)
    np.savez_compressed(raw / "summary.npz", singles_rate=rate,
                        scatter_bin=scatter.reshape(-1), phi=phi,
                        n_true=int(true.sum()), n_scatter=int(sc.sum()),
                        n_random=int(rnd.sum()), delays=stats["delays"],
                        seconds=t_sim, accept=accept)
    row.update(header["simulated"])
    out(f"  bed {bed}: {row['prompts']:,} prompts in {t_sim:g} s -- trues "
        f"{int(true.sum()):,}, scatter {int(sc.sum()):,}, randoms {int(rnd.sum()):,}; "
        f"delays {stats['delays']:,}; singles {rate.sum() / 1e6:.2f} Mcps")
    eio.manifest(dst, {"source_case": real.name, "method": "gate", "seed": seed,
                       "beds": [row]})
    return row


def attenuation_per_bin(pairs: pp.RingPairs, lut, mu) -> np.ndarray:
    import parallelproj

    b = pairs.binmap
    acc = np.zeros(b.n_bin, np.float64)
    for _p, a, c, bins in pairs:
        acc[bins] += np.exp(-parallelproj.joseph3d_fwd(lut[a], lut[c], *mu))
    return (acc.reshape(b.shape) / b.mult[:, None, None]).astype(np.float32)


README = """\
Simulated raw data -- written by `d710 simulate gate`. NOT measured data.
GATE 10 Monte Carlo of the D710 around the CT of case {case!r}, with its PET as
the source; seed {seed}, {seconds:g} s simulated.

  ../decoded/bed<n>.lm.npy   the coincidences as the decoder's event table
  ../decoded/bed<n>.s        histogrammed from those events, real layout
  ../work/bed<n>/normdt      GE's, copied (it also set the norm acceptance)
  ../work/bed<n>/attn        parallelproj line integrals through the CT mu-map
  ../work/bed<n>/randoms     2 w S_a S_b from GATE's own singles rates
  ../work/bed<n>/scatter     GATE's scatter-flagged coincidences, smoothed
  bed<n>_truth.npz           per event (in lm.npy order): is_random, is_scatter
  gate/bed<n>/               the GATE runs: configs, logs, ROOT files

See simulation/README.md.
"""
