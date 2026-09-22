"""The parallelproj model of one bed: expected counts per crystal pair and TOF bin.

For the LOR between crystals `a` and `b`, which falls in sinogram bin `beta`,
and TOF bin `t`:

    y(a,b,t) = kappa * n_beta * exp(-int mu) * P_t[x](a,b)
             + R_ab / 55 + S_beta * phi(u, t)

  P_t      `parallelproj.joseph3d_fwd_tof_sino` of `x`, the positron decays
           per mm^3 over the frame, from `a` to `b` -- the LOR ends are
           `utils.geometry.crystal_positions(stir_frame=True)`, the frame the
           reconstruction uses; sigma = c * 675 ps / 2 / 2.355, 55 bins of
           89.2459 ps, cut at 3 sigma
  mu       `joseph3d_fwd` through the CT's mu-map, 1/mm
  n        GE's `normdt` per LOR (the bin's value over its ring-pair count)
  kappa    the one absolute constant a line integral cannot supply: fitted so
           that the trues equal GATE's on the same bed. Never fitted to the
           real data, which are what the result is compared with
  R        `2w S_a S_b`, singles-rate randoms with GATE's singles and window
  S, phi   GATE's scatter-flagged coincidences, smoothed, and their TOF shape

Each part is drawn from its own Poisson distribution, so every event carries a
truth label (true, scatter, random) and the sum is Poisson in `y`.
"""

from __future__ import annotations

import math
import time

import numpy as np

from utils.binmap import BinMap
from utils.geometry import crystal_positions, det_pair_map
from utils.scanner import C_MM_PS, N_TOF_RAW, TIMING_PS

from . import events_io as eio
from . import phantom as ph

SIGMA_TOF_MM = C_MM_PS * TIMING_PS / 2.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

N_SIGMAS = 3.0

LABELS = ("true", "scatter", "random")


class RingPairs:
    """The crystal ids and bin of every LOR, one ring pair at a time."""

    def __init__(self, binmap: BinMap):
        self.binmap = binmap
        d1, d2 = det_pair_map(binmap.n_view, binmap.n_tang, binmap.ndet)
        det2xtal = np.argsort(binmap.xtal2det).astype(np.int64)
        self.t1, self.t2 = det2xtal[d1].ravel(), det2xtal[d2].ravel()
        self.nvt = binmap.n_view * binmap.n_tang
        self.k = np.arange(self.nvt, dtype=np.int64)
        self.u = self.k % binmap.n_tang
        self.r1, self.r2, self.plane = binmap.ring_pairs_by_plane()

    def __len__(self) -> int:
        return len(self.r1)

    def __iter__(self):
        nd = self.binmap.ndet
        for r1, r2, p in zip(self.r1, self.r2, self.plane):
            yield (int(p), int(r1) * nd + self.t1, int(r2) * nd + self.t2,
                   int(p) * self.nvt + self.k)


def projector_image(arr_zyx, voxel_xyz=ph.VOXEL_XYZ):
    """`(img, origin, voxel)` for parallelproj from a `(z, y, x)` array on the bed grid."""
    img = np.ascontiguousarray(np.asarray(arr_zyx).transpose(2, 1, 0), np.float32)
    return img, ph.world_origin(arr_zyx.shape), np.asarray(voxel_xyz, np.float32)


def tof_kernel(n_tof: int = N_TOF_RAW):
    """`(bin width, sigma, centre offset)` as parallelproj takes them."""
    width = eio.TOF_BIN_MM * (N_TOF_RAW / n_tof)
    return (float(width), np.array([SIGMA_TOF_MM], np.float32),
            np.array([0.0], np.float32))


def sample_times(n: int, frame_s: float, half_life_s: float, rng) -> np.ndarray:
    """Arrival times in ms over `[0, frame)`, with density decaying at `half_life_s`."""
    lam = math.log(2.0) / half_life_s
    u = rng.random(n)
    t = -np.log1p(-u * (1.0 - math.exp(-lam * frame_s))) / lam
    return np.minimum((t * 1000.0).astype(np.int64), int(frame_s * 1000) - 1)


def per_lor(term_flat, binmap: BinMap) -> np.ndarray:
    """A per-bin term divided by its bin's ring-pair count."""
    return (term_flat.reshape(binmap.shape) / binmap.mult[:, None, None]).reshape(-1)


def attenuation_and_sensitivity(pairs: RingPairs, lut, mu, x, normdt_lor, out=print):
    """Pass 1, non-TOF: `(af per bin, sum of n af P[x] over every LOR)`."""
    import parallelproj

    b = pairs.binmap
    acc = np.zeros(b.n_bin, np.float64)
    total = 0.0
    t0 = time.time()
    for i, (p, a, c, bins) in enumerate(pairs):
        xs, xe = lut[a], lut[c]
        af = np.exp(-parallelproj.joseph3d_fwd(xs, xe, *mu))
        proj = parallelproj.joseph3d_fwd(xs, xe, *x)
        acc[bins] += af
        total += float(np.dot(normdt_lor[bins] * af, proj))
        if i % 96 == 0:
            out(f"    pass 1: ring pair {i}/{len(pairs)}  {time.time() - t0:.0f} s")
    af_bin = (acc.reshape(b.shape) / b.mult[:, None, None]).astype(np.float32)
    return af_bin, total


def simulate(pairs: RingPairs, lut, mu, x, normdt_lor, kappa, rng,
             randoms_lor=None, scatter_lor=None, phi=None, n_tof=N_TOF_RAW, out=print):
    """Pass 2, TOF: Poisson counts for every LOR and TOF bin, as events.

    Returns `(xtal_a, xtal_b, tof_bin, label, expected)`, where `expected` is
    the sum of the means of the three parts.
    """
    import parallelproj

    width, sigma, offset = tof_kernel(n_tof)
    half = n_tof // 2
    out_a, out_b, out_t, out_l = [], [], [], []
    expected = np.zeros(3)
    t0 = time.time()
    for i, (p, a, c, bins) in enumerate(pairs):
        xs, xe = lut[a], lut[c]
        af = np.exp(-parallelproj.joseph3d_fwd(xs, xe, *mu))
        if n_tof > 1:
            proj = parallelproj.joseph3d_fwd_tof_sino(xs, xe, *x, width, sigma,
                                                      offset, N_SIGMAS, n_tof)
        else:
            proj = parallelproj.joseph3d_fwd(xs, xe, *x)[:, None]
        np.maximum(proj, 0.0, out=proj)
        parts = [np.float32(kappa) * (normdt_lor[bins] * af)[:, None] * proj]
        if scatter_lor is not None:
            s = scatter_lor[bins][:, None]
            parts.append(s * (phi[pairs.u] if phi is not None else 1.0 / n_tof))
        else:
            parts.append(None)
        if randoms_lor is not None:
            parts.append(np.broadcast_to(
                (randoms_lor(a, c, bins) / n_tof)[:, None], proj.shape))
        else:
            parts.append(None)
        for lab, mean in enumerate(parts):
            if mean is None:
                continue
            if not np.isfinite(mean).all() or (mean < 0).any():
                raise ValueError(
                    f"ring pair {i} (plane {p}), part {LABELS[lab]}: mean has "
                    f"{int((~np.isfinite(mean)).sum())} non-finite and "
                    f"{int((mean < 0).sum())} negative values (min {np.nanmin(mean)})")
            # joseph3d interpolation leaves values like -1.5e-21 where the
            # image is zero; they are clipped above, before this check.
            expected[lab] += float(mean.sum(dtype=np.float64))
            n = rng.poisson(mean)
            li, ti = np.nonzero(n)
            if li.size == 0:
                continue
            reps = n[li, ti]
            li, ti = np.repeat(li, reps), np.repeat(ti, reps)
            out_a.append(a[li])
            out_b.append(c[li])
            out_t.append((half - ti).astype(np.int8) if n_tof > 1
                         else np.zeros(li.size, np.int8))
            out_l.append(np.full(li.size, lab, np.int8))
        if i % 48 == 0:
            out(f"    pass 2: ring pair {i}/{len(pairs)}  "
                f"{sum(map(len, out_a)):,} events  {time.time() - t0:.0f} s")
    cat = (lambda v, d: np.concatenate(v) if v else np.zeros(0, d))
    return (cat(out_a, np.int64), cat(out_b, np.int64), cat(out_t, np.int8),
            cat(out_l, np.int8), expected)


def singles_randoms(rate_cps, window_ns: float, time_s: float, accept_flat):
    """`f(a, b, bins)`: randoms per LOR, `2 w S_a S_b` times `time_s` and the norm acceptance."""
    r = np.asarray(rate_cps, np.float64)
    k = 2.0 * window_ns * 1e-9 * time_s

    def f(a, b, bins):
        v = k * r[a] * r[b]
        if accept_flat is not None:
            v = v * accept_flat[bins]
        return v.astype(np.float32)

    return f


def crystal_lut() -> np.ndarray:
    return crystal_positions().astype(np.float32)


def run_bed(real, dst, bed: int, gate_case, seconds: float | None, seed: int,
            n_tof: int = N_TOF_RAW, kappa: float | None = None, out=print) -> dict:
    """Simulate one bed with parallelproj and write it as a case.

    Without a GATE run of the same bed (`gate_case`), only the trues can be
    drawn, and `kappa` must be given.
    """
    from .gate import coinc
    from .gate.driver import attenuation_per_bin, randoms_per_bin, raw_dir
    from .gate.run import WINDOW_NS

    phantom_dir = ph.directory(eio.sim_root(real), bed)
    meta = ph.load(phantom_dir)
    frame = float(seconds or meta["frame_s"])
    half = meta["timing"]["half_life_s"]
    rng = np.random.default_rng(seed)

    g = None
    if gate_case is not None:
        p = raw_dir(gate_case, bed) / "summary.npz"
        if not p.exists():
            raise SystemExit(f"error: no {p}\n  run: d710 simulate gate --case "
                             f"{real.name} --beds {bed} first")
        g = coinc.load_summary(p)
    elif kappa is None:
        raise SystemExit("error: without a GATE run, give --kappa (trues only)")

    binmap = BinMap(real.prompt(bed))
    pairs = RingPairs(binmap)
    lut = crystal_lut()
    normdt_lor = per_lor(np.fromfile(real.work_bed(bed) / "normdt.s", "<f4"), binmap)

    k_frame = (ph.frame_integral_s(frame, half) / meta["frame_integral_s"])
    dec = ph.decays_per_voxel(phantom_dir, meta) * np.float32(
        k_frame / (ph.VOXEL_ML * 1000.0))
    x = projector_image(dec)
    mu = projector_image(np.load(phantom_dir / "mu_bed.npy"))
    out(f"  bed {bed}: {float(dec.sum()) * ph.VOXEL_ML * 1000:.4g} positron decays in "
        f"the FOV over {frame:g} s")

    af_bin, sens = attenuation_and_sensitivity(pairs, lut, mu, x, normdt_lor, out)
    randoms_fn = scatter_lor = phi = randoms_bin = scatter_bin = None
    if g is not None:
        g_s = float(g["seconds"])
        i1 = ph.frame_integral_s(frame, half) / coinc.decay_integral(0, g_s, half)
        kappa = float(g["n_true"]) * i1 / sens
        i2 = coinc.decay_integral(0.0, frame, half, power=2)
        randoms_fn = singles_randoms(g["singles_rate"], WINDOW_NS, i2, g["accept"])
        scatter_bin = (g["scatter_bin"] * np.float32(i1)).astype(np.float32)
        scatter_lor = per_lor(scatter_bin, binmap)
        phi = g["phi"] if n_tof == N_TOF_RAW else None
        randoms_bin = randoms_per_bin(pairs, randoms_fn)
        out(f"  kappa {kappa:.6g}: GATE's {int(g['n_true']):,} trues in {g_s:g} s "
            f"scaled to {frame:g} s")

    xa, xb, tof, lab, expected = simulate(pairs, lut, mu, x, normdt_lor, kappa, rng,
                                          randoms_fn, scatter_lor, phi, n_tof, out)
    xa, xb, tof = eio.swap_randomly(xa, xb, tof.astype(np.int64), rng)
    t_ms = int(meta["bed_start_ticks"]) + sample_times(len(xa), frame, half, rng)
    ev = eio.events(xa, xb, tof, t_ms)

    delays = int(rng.poisson(expected[2])) if randoms_fn is not None else 0
    counts = {n: int((lab == i).sum()) for i, n in enumerate(LABELS)}
    header = {"frame_duration_ms": frame * 1000.0, "delays": delays,
              "simulated": {"method": "pp", "seconds": frame, "kappa": kappa,
                            "expected": dict(zip(LABELS, map(float, expected))),
                            "counts": counts, "tof_bins": n_tof}}
    terms = {"attn": af_bin, "randoms": randoms_bin,
             "scatter": None if scatter_bin is None else scatter_bin.reshape(binmap.shape)}
    row = eio.write_bed(real, dst, bed, ev, terms, header,
                        {"label": lab}, README.format(case=real.name, seed=seed,
                                                      seconds=frame))
    row.update(header["simulated"])
    out(f"  bed {bed}: {row['prompts']:,} prompts in {frame:g} s -- {counts}")
    eio.manifest(dst, {"source_case": real.name, "method": "pp", "seed": seed,
                       "beds": [row]})
    return row


README = """\
Simulated raw data -- written by `d710 simulate pp`. NOT measured data.
parallelproj model of the D710 over the CT and PET of case {case!r}; seed
{seed}, {seconds:g} s simulated. Randoms, scatter and the absolute scale come
from the GATE run of the same bed (see simulation/README.md).

  ../decoded/bed<n>.lm.npy   events: trues, scatter and randoms, each Poisson
  ../decoded/bed<n>.s        histogrammed from those events, real layout
  ../work/bed<n>/normdt      GE's, copied -- the model multiplies by it
  ../work/bed<n>/attn        the attenuation factors the model used
  ../work/bed<n>/randoms     the randoms mean the events were drawn from
  ../work/bed<n>/scatter     the scatter mean the events were drawn from
  bed<n>_truth.npz           per event (in lm.npy order): label 0 true,
                             1 scatter, 2 random
"""
