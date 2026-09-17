"""PyTomography geometry for the D710: crystal positions, TOF metadata, bin map.

`BinMap` is re-exported from `utils.binmap`, which is where the
detector-to-ring pairing is stated and where the measurement that fixes it
is recorded. It lives under `utils/` because `utils.attn_proj` needs the
same map and `utils/` must not import `lm/`.
"""

from __future__ import annotations

import numpy as np

from utils.binmap import BinMap  # noqa: F401  (re-exported: callers say geom.BinMap)
from utils.geometry import crystal_positions
from utils.scanner import (C_MM_PS, NDET, NRINGS, NXTAL,
                           N_TOF_RAW, RING_PITCH_MM, R_EFF_MM, R_MM, TIMING_PS,
                           TOF_LSB_PS, TOF_RANGE_MM, VIEW_OFFSET_DEG)


def scanner_lut(nrings=NRINGS, ndet=NDET, r_mm=R_EFF_MM, pitch_mm=RING_PITCH_MM,
                offset_deg=VIEW_OFFSET_DEG, stir_frame=True):
    """`(nrings*ndet, 3)` crystal centres in mm, as PyTomography's `scanner_LUT`."""
    return crystal_positions(nrings, ndet, r_mm, pitch_mm, offset_deg, stir_frame)


def tof_meta(n_bins=N_TOF_RAW, n_sigmas=3.0):
    """PyTomography's TOF metadata for `n_bins` mashed bins."""
    from pytomography.metadata.PET import PETTOFMeta

    return PETTOFMeta(num_bins=n_bins, tof_range=TOF_RANGE_MM,
                      fwhm=C_MM_PS * TIMING_PS / 2, n_sigmas=n_sigmas)


def tof_to_stir(tof_bin, n_out=N_TOF_RAW, n_raw=N_TOF_RAW):
    """GE's signed bin (-27..+27) to a 0-based STIR timing position, rebinned to `n_out`."""
    if n_raw % n_out:
        raise ValueError(f"{n_out} TOF bins does not divide {n_raw}")
    ge = np.asarray(tof_bin, np.int32) + n_raw // 2
    return (n_raw - 1 - ge) // (n_raw // n_out)
