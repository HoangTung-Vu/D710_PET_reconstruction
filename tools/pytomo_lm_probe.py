#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from utils.scanner import (C_MM_PS as C_MM_PER_PS,
                           NXTAL, RING_PITCH_MM, R_MM, TIMING_PS as RES_PS,
                           TOF_RANGE_MM)
from utils.scanner import NDET as NDET_RING
from utils.scanner import NRINGS, PSF_FWHM_MM
from utils.scanner import N_TOF_RAW as N_TOF
N_VALID_LORS = 60_679_584


def d710_scanner_lut() -> torch.Tensor:
    """`(13824, 3)` crystal coordinates."""
    pitch = RING_PITCH_MM
    i = torch.arange(NXTAL)
    ring, trans = i // NDET_RING, i % NDET_RING
    ang = 2 * np.pi * trans.double() / NDET_RING
    return torch.stack([R_MM * torch.cos(ang),
                        R_MM * torch.sin(ang),
                        (ring.double() - (NRINGS - 1) / 2) * pitch], dim=1).float()


def d710_tof_meta(n_sigmas: float = 3.0):
    from pytomography.metadata.PET import PETTOFMeta
    return PETTOFMeta(num_bins=N_TOF,
                      tof_range=TOF_RANGE_MM,
                      fwhm=C_MM_PER_PS * RES_PS / 2,
                      n_sigmas=n_sigmas)


def load_events(path: str | None, n: int, seed: int = 0) -> torch.Tensor:
    """`(N, 3)` int32 `[xtal_a, xtal_b, tof_idx]`, with `tof_idx` 0-based."""
    if path:
        e = np.load(path)
        return torch.from_numpy(np.stack([
            e["xtal_a"].astype(np.int32),
            e["xtal_b"].astype(np.int32),
            (e["tof_bin"].astype(np.int32) + N_TOF // 2)], axis=1))
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, NXTAL, (n,), generator=g)
    opp = (a % NDET_RING + NDET_RING // 2
           + torch.randint(-120, 121, (n,), generator=g)) % NDET_RING
    b = torch.randint(0, NRINGS, (n,), generator=g) * NDET_RING + opp
    return torch.stack([a, b, torch.randint(0, N_TOF, (n,), generator=g)],
                       dim=1).to(torch.int32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", help="a real EVENT_DTYPE .npy; default synthetic")
    ap.add_argument("--n-events", type=int, default=14_809_731,
                    help="synthetic event count (default = ped2 bed 1's real count)")
    ap.add_argument("--n-sens", type=int, default=12_000_000, metavar="N",
                    help="sensitivity LORs to actually run (default %(default)s); "
                         f"the real set is {N_VALID_LORS:,}, and the cost is "
                         "linear, so this is extrapolated rather than allocated")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--subsets", type=int, default=24)
    ap.add_argument("--psf", type=float, nargs="+", default=list(PSF_FWHM_MM),
                    metavar="MM",
                    help="XY [Z] mm FWHM; one value = isotropic; 0 disables")
    ap.add_argument("--xy", type=int, default=128)
    args = ap.parse_args(argv)

    import pytomography
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.metadata import ObjectMeta
    from pytomography.metadata.PET import PETLMProjMeta
    from pytomography.projectors.PET import PETLMSystemMatrix
    from pytomography.transforms.shared import GaussianFilter

    pitch = RING_PITCH_MM
    print(f"pytomography device: {pytomography.device}")
    tof = d710_tof_meta()
    print(f"D710: {NXTAL} crystals, R={R_MM} mm, pitch={pitch:.3f} mm | "
          f"TOF {N_TOF} bins, range {float(tof.bin_edges[-1] - tof.bin_edges[0]):.1f} mm, "
          f"fwhm {C_MM_PER_PS * RES_PS / 2:.1f} mm")

    lut = d710_scanner_lut()
    ev = load_events(args.events, args.n_events)
    print(f"scanner_LUT {tuple(lut.shape)}   detector_ids {tuple(ev.shape)} {ev.dtype}")

    g = torch.Generator().manual_seed(1)
    sa = torch.randint(0, NXTAL, (args.n_sens,), generator=g)
    sb = torch.randint(0, NRINGS, (args.n_sens,), generator=g) * NDET_RING + \
        (sa % NDET_RING + NDET_RING // 2
         + torch.randint(-190, 191, (args.n_sens,), generator=g)) % NDET_RING
    sens_ids = torch.stack([sa, sb], dim=1).to(torch.int32)

    t = time.time()
    proj_meta = PETLMProjMeta(ev, info=None, scanner_LUT=lut, tof_meta=tof,
                              detector_ids_sensitivity=sens_ids)
    print(f"PETLMProjMeta built in {time.time() - t:.1f}s   info=None accepted")

    from lm.recon import psf_fwhm

    f = psf_fwhm(args.psf)
    tr = [GaussianFilter(f)] if f else []
    object_meta = ObjectMeta(dr=(2.13, 2.13, pitch / 2),
                             shape=(args.xy, args.xy, 47))
    t = time.time()
    sm = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=tr, N_splits=8)
    t_sens = time.time() - t
    print(f"sensitivity over {args.n_sens:,} LORs: {t_sens:.1f}s  "
          f"-> {t_sens / args.n_sens * N_VALID_LORS:.0f}s extrapolated to the "
          f"real {N_VALID_LORS:,}", flush=True)

    t = time.time()
    rec = OSEM(PoissonLogLikelihood(sm))(n_iters=args.iters, n_subsets=args.subsets)
    t_osem = time.time() - t
    print(f"OSEM {args.iters} iters x {args.subsets} subsets on {len(ev):,} events: "
          f"{t_osem:.1f}s")
    print(f"TOTAL for one bed: ~{t_sens / args.n_sens * N_VALID_LORS + t_osem:.0f}s")
    print(f"recon {tuple(rec.shape)}  min {float(rec.min()):.4g}  max {float(rec.max()):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
