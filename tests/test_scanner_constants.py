"""`utils/scanner.py` against itself, and against every copy of it in the tree.

Two things, neither of which needs the vendor tree or SIRF, so they run anywhere:

1. **The derived identities.** Several constants are defined in terms of others
   and then quoted as literals in the comments. If one is edited and the other
   is not, nothing else notices.
2. **The copies.** `scanner.py` is meant to be the one place a machine constant
   lives, but two more copies exist and cannot be removed: `tests/synth_hs.py`
   keeps them spelled out so it does not import `utils`, and
   `custom_tool/gerdf/interfile.py` lives in a different container image
   entirely. Both say in a comment that they must match. This is that comment,
   enforced.

The Interfile header is the ONLY channel by which `osem/` learns the geometry:
the decoder writes it from `gerdf/interfile.py`'s constants and STIR reads
everything back out of it. A drift there is silent and total, which is why the
`gerdf` half is the most valuable test in this file.

What the values *should be* is not asked here -- that is `test_petsw_config.py`,
against GE's own configuration.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import synth_hs
from conftest import ROOT
from utils import scanner

GERDF = ROOT.parent / "custom_tool" / "gerdf" / "interfile.py"


def gerdf_interfile():
    """`custom_tool/gerdf/interfile.py` as a module, or a skip.

    `gerdf` normally lives inside the `d710:full` image rather than on this
    interpreter's path, so `custom_tool/` is put on the path here. It has to be
    imported as part of its package -- the module uses relative imports, so
    loading the single file by path fails.
    """
    if not GERDF.exists():
        pytest.skip(f"no {GERDF}; the decoder source is not in this checkout")
    root = str(GERDF.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module("gerdf.interfile")
    except Exception as e:                       # pragma: no cover - import-time
        pytest.skip(f"cannot import gerdf.interfile ({e})")


# --------------------------------------------------------- derived identities

def test_the_view_offset_is_derived_from_the_crystal_offset():
    """`VIEW_OFFSET_DEG = -(XTAL0 + pitch)`, and the comment says +4.3960."""
    assert scanner.VIEW_OFFSET_DEG == pytest.approx(
        -(scanner.XTAL0_OFFSET_DEG + scanner.XTAL_PITCH_DEG), abs=1e-12)
    assert scanner.VIEW_OFFSET_DEG == pytest.approx(4.3960, abs=1e-4)


def test_the_effective_radius_is_the_ring_plus_the_doi():
    assert scanner.R_EFF_MM == pytest.approx(scanner.R_MM + scanner.DOI_MM, abs=1e-12)
    assert scanner.R_EFF_MM == pytest.approx(413.50, abs=1e-9)


def test_the_crystal_pitch_closes_the_ring():
    assert scanner.XTAL_PITCH_DEG * scanner.NDET == pytest.approx(360.0, abs=1e-12)
    assert scanner.NXTAL == scanner.NRINGS * scanner.NDET


def test_segment_zero_holds_two_planes_per_ring_less_one():
    assert scanner.NSEG0 == 2 * scanner.NRINGS - 1


def test_the_ring_pitch_is_two_planes():
    assert scanner.RING_PITCH_MM == pytest.approx(2 * scanner.PLANE_MM, abs=1e-12)


def test_the_tof_range_is_the_bins_times_half_a_bin_of_flight():
    want = (scanner.N_TOF_RAW * scanner.C_MM_PS * scanner.TOF_LSB_PS) / 2
    assert scanner.TOF_RANGE_MM == pytest.approx(want, abs=1e-9)


def test_the_transverse_voxel_is_the_bin_at_zoom_one():
    assert scanner.DR_MM == scanner.BIN_MM
    assert scanner.FOV_MM == pytest.approx(scanner.XY * scanner.DR_MM, abs=1e-9)


def test_the_subsets_divide_the_views():
    """`scanner.py` states it in a comment; STIR raises much later if it fails."""
    assert 288 % scanner.N_SUBSETS == 0


def test_the_image_grid_reaches_past_the_lors():
    """The square grid must cover the round FOV, or counts land outside it."""
    assert scanner.FOV_MM / 2 >= scanner.fov_radius_mm(381)


# ------------------------------------------------- the copies must not drift

def test_synth_hs_matches_the_scanner():
    """`tests/synth_hs.py` keeps four constants spelled out; they must agree.

    Its own comment says "Must equal `utils.scanner.VIEW_OFFSET_DEG`". Without
    this, a header built for a test would describe a different scanner from the
    one the code reconstructs, and the geometry tests would pass on both.
    """
    assert synth_hs.PLANE_MM == scanner.PLANE_MM
    assert synth_hs.BIN_SIZE_CM * 10 == pytest.approx(scanner.BIN_MM, abs=1e-9)
    assert synth_hs.AVG_DOI_CM * 10 == pytest.approx(scanner.DOI_MM, abs=1e-9)
    assert synth_hs.VIEW_OFFSET_DEG == pytest.approx(
        scanner.VIEW_OFFSET_DEG, abs=1e-4)


@pytest.mark.parametrize(
    "gerdf_name, scanner_value, scale",
    [("XTAL0_OFFSET_DEG", "XTAL0_OFFSET_DEG", 1.0),
     ("VIEW_OFFSET_DEG", "VIEW_OFFSET_DEG", 1.0),
     ("AVG_DOI_CM", "DOI_MM", 10.0),
     ("DEFAULT_BIN_SIZE_CM", "BIN_MM", 10.0),
     ("RING_SPACING_CM", "RING_PITCH_MM", 10.0),
     ("TOF_BIN_SIZE_PS", "TOF_LSB_PS", 1.0),
     ("TOF_RESOLUTION_PS", "TIMING_PS", 1.0)])
def test_the_decoder_writes_the_geometry_the_reconstructor_expects(
        gerdf_name, scanner_value, scale):
    """`gerdf/interfile.py` stamps these into every header `osem/` then reads.

    The decoder runs inside the `d710:full` image and cannot import
    `utils.scanner`, so the two sets of constants are maintained by hand. This
    is the only thing standing between that and a silent geometry mismatch --
    `interfile.py:40` even says "Must stay identical" and nothing enforced it.
    """
    mod = gerdf_interfile()
    got = getattr(mod, gerdf_name) * scale
    # float32, not float64: these constants meet as decimal text in an Interfile
    # header that STIR parses into float32, so agreement closer than ~1e-7
    # relative is not observable by anything downstream. `PLANE_MM 3.2699997` is
    # exactly the float32 round-trip of the vendor's `3.27`, and the decoder
    # writes the `3.27` -- at rel=1e-9 that reads as a drift when it is not.
    # A real drift is far bigger: GE's own other axial pitch is 6.529, 0.17 % off.
    assert got == pytest.approx(getattr(scanner, scanner_value), rel=1e-6), (
        f"gerdf/interfile.py:{gerdf_name} x {scale:g} = {got} against "
        f"utils/scanner.py:{scanner_value} = {getattr(scanner, scanner_value)}")
