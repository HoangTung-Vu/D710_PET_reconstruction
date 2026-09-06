"""Every constant that claims a source in GE's console tree, against that source.

`utils/scanner.py` cites a file and often a line number for most of its values
(`cmcfg.XR.xml:677`, `sharcAp.cfg:46`, ...). Nothing checked those citations, and
four constants were once found to be STIR's **Discovery 690** defaults rather
than this scanner's -- a class of error that produces a plausible image, not an
error message.

The console tree is 18 GB, lives outside this repo at `custom_tool/petsw/`, and
is not redistributable, so **every test here skips when it is absent**. That is
the normal state on a machine that has the code but not the vendor software.

Values are looked up **by name**, then the cited line number is checked
separately: a vendor-tree update that moves a line should say so plainly rather
than silently reading the wrong element.

Deliberate deviations from GE are asserted **as deviations**. They are choices,
and pinning them here is what stops one being "fixed" by accident or drifting
without anyone noticing.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from conftest import ROOT
from utils import scanner

PETSW = ROOT.parent / "custom_tool" / "petsw" / "usr" / "PET" / "systemConfig"
LOCAL = PETSW / "local"


def _need(p):
    if not p.exists():
        pytest.skip(f"no {p}; the 18 GB console tree is not in this checkout")
    return p


@pytest.fixture(scope="module")
def cmcfg():
    """`{tag: text}` from `cmcfg.XR.xml`, the machine's configuration manager."""
    root = ET.parse(_need(LOCAL / "cmcfg.XR.xml")).getroot()
    return {el.tag: (el.text or "").strip() for el in root.iter()}


@pytest.fixture(scope="module")
def sharc():
    """`{key: [tokens]}` from `sharcAp.cfg.XR`, the recon engine's parameters."""
    out = {}
    for line in _need(LOCAL / "sharcAp.cfg.XR").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        out[parts[0]] = parts[1:]
    return out


def line_of(path, pattern) -> int:
    """1-based line number of the first line matching `pattern`, or 0."""
    for i, line in enumerate(_need(path).read_text().splitlines(), 1):
        if re.search(pattern, line):
            return i
    return 0


# ------------------------------------------------------------------ geometry

@pytest.mark.parametrize("tag, const, scale", [
    ("SYS_DETECTOR_RADIAL_SIZE", "R_MM", 1.0),
    ("SYS_TRANSAXIAL_CRYSTAL_0_OFFSET", "XTAL0_OFFSET_DEG", 1.0),
    ("SYS_RAD_CRYSTALS_PER_SYSTEM", "NDET", 1.0),
    ("SYS_AXI_CRYSTALS_PER_SYSTEM", "NRINGS", 1.0),
    ("SYS_NUMBER_SLICES", "NSEG0", 1.0),
    ("SYS_SLICE_THICKNESS", "PLANE_MM", 1.0),
    ("deltaAngle", "XTAL_PITCH_DEG", 1.0),
    ("CMP_CFG_IR_Z_AXIS_STANDARD_FILTER_RATIO_DEFAULT", "POST_FILTER_Z_RATIO", 1.0),
    # /cm in the file, 1/mm in the code.
    ("CTAC_DEFAULT_MU_WATER_511", "MU_WATER_511", 0.1),
    ("CTAC_DEFAULT_MU_BONE_511", "MU_BONE_511", 0.1),
])
def test_the_scanner_constant_is_what_the_console_declares(
        cmcfg, tag, const, scale):
    """One row per constant that names `cmcfg.XR.xml` as its source.

    float32 tolerance: `PLANE_MM` is the float32 round-trip of the vendor's
    exact `3.27`, and nothing downstream can carry more precision than that.
    """
    assert tag in cmcfg, f"{tag} is not in cmcfg.XR.xml any more"
    assert float(cmcfg[tag]) * scale == pytest.approx(
        getattr(scanner, const), rel=1e-6), (
        f"cmcfg.XR.xml:{tag} = {cmcfg[tag]} (x {scale:g}) against "
        f"utils/scanner.py:{const} = {getattr(scanner, const)}")


def test_the_doi_is_the_effective_ring_diameter_minus_the_ring(cmcfg):
    """`DOI_MM` is not measured here, it is derived -- so derive it and compare.

    NOT 9.4: that is STIR's Discovery 690 default, which this once was.
    """
    eff = float(cmcfg["SYS_EFF_RING_DIAMETER"])
    assert eff == pytest.approx(827.0, abs=1e-9)
    assert eff / 2 - float(cmcfg["SYS_DETECTOR_RADIAL_SIZE"]) == pytest.approx(
        scanner.DOI_MM, abs=1e-6)
    assert eff / 2 == pytest.approx(scanner.R_EFF_MM, abs=1e-6)


def test_the_cited_line_numbers_still_point_at_the_right_elements():
    """`scanner.py` cites `cmcfg.XR.xml:721` and `:677`; check they are still there."""
    p = LOCAL / "cmcfg.XR.xml"
    assert line_of(p, r"<SYS_TRANSAXIAL_CRYSTAL_0_OFFSET\b") == 721
    assert line_of(p, r"<SYS_EFF_RING_DIAMETER\b") == 677


# ----------------------------------------------------------------- TOF / PSF

def test_the_timing_resolution_is_the_machines_own(sharc):
    """NOT 550: that is STIR's Discovery 690 placeholder, 23 % too narrow."""
    assert float(sharc["TIMING_RESOLUTION"][0]) == pytest.approx(
        scanner.TIMING_PS, abs=1e-9)
    assert line_of(LOCAL / "sharcAp.cfg.XR", r"^TIMING_RESOLUTION\b") == 46


def test_the_unsuffixed_config_names_are_this_scanners(cmcfg):
    """`sharcAp.cfg` and `cmcfg.xml` must resolve to the `.XR` variants.

    `scanner.py` cites both `sharcAp.cfg:46` and `sharcAp.cfg.XR`. They are the
    same file only because the console symlinks the bare name onto the XR one;
    eight other variants sit beside it, each a different scanner. If the link
    ever pointed elsewhere the citation would silently name another machine.
    """
    for bare, want in (("sharcAp.cfg", "sharcAp.cfg.XR"),
                       ("cmcfg.xml", "cmcfg.XR.xml")):
        p = _need(LOCAL / bare)
        assert p.is_symlink(), f"{bare} is no longer a symlink"
        assert p.readlink().name == want, f"{bare} -> {p.readlink().name}, not {want}"


def test_the_axial_post_filter_kernel_is_ges_own(sharc):
    """`[1, ratio, 1]` normalised must equal `PSF_AXIAL_KERNEL[]` character for character.

    The `[1, ratio, 1]` shape is inferred; this is the evidence for it. Note the
    semantic difference that remains: GE applies this kernel INSIDE the system
    model as an axial PSF, `osem/` applies it as a post-filter.
    """
    got = [float(x) for x in sharc["PSF_AXIAL_KERNEL[]"]]
    r = scanner.POST_FILTER_Z_RATIO
    want = [1 / (2 + r), r / (2 + r), 1 / (2 + r)]
    assert got == pytest.approx(want, abs=5e-8)
    assert int(sharc["PSF_AXIAL_WINDOW_WIDTH"][0]) == len(got)


def test_the_post_filter_ratio_is_corroborated_by_the_config(cmcfg):
    """The DICOM private tag is the source; the config agrees, so pin both."""
    assert float(cmcfg["CMP_CFG_IR_Z_AXIS_STANDARD_FILTER_RATIO_DEFAULT"]) == (
        pytest.approx(scanner.POST_FILTER_Z_RATIO, abs=1e-9))


def test_the_private_tags_scanner_cites_are_defined(cmcfg):
    """`(0009,10BA/10BB/10DB/10DC)` -- the tag names `PSF_MM` and the filters cite."""
    tags = _need(LOCAL / "tagDetails.xml").read_text()
    for element, name, vr in (("0x10BA", "post_filter", "SS"),
                              ("0x10BB", "post_filt_parm", "FL"),
                              ("0x10DB", "ir_z_filter_flag", "SL"),
                              ("0x10DC", "ir_z_filter_ratio", "FL")):
        assert re.search(
            rf'VR="{vr}"\s+element="{element}"\s+group="0x0009"\s+name="{name}"',
            tags), f"(0009,{element[2:]}) {name} is not defined as {vr}"


# ----------------------------------------------- deliberate deviations from GE

def test_the_transaxial_psf_is_a_scalar_where_ge_uses_a_lut():
    """`PSF_MM` is one Gaussian; GE's is a 381 x 32 float32 LUT that varies with radius.

    Asserted so the simplification stays visible. Replacing the scalar with
    `psfLUT.XR` is an open item, not a bug.
    """
    lut = _need(LOCAL / "psfLUT.XR")
    assert lut.stat().st_size == 381 * 32 * 4
    assert isinstance(scanner.PSF_MM, float)


def test_the_reconstructed_fov_deliberately_exceeds_ges_maximum(sharc):
    """`FOV_MM` 718.01 against GE's `CMP_MAX_DFOV` 700.0 -- 18 mm wider, on purpose.

    The grid has to cover the LORs (`fov_radius_mm` x 2 = 711.6 mm), which GE's
    own display FOV does not.
    """
    ge = float(sharc["CMP_MAX_DFOV"][0])
    assert ge == pytest.approx(700.0, abs=1e-9)
    assert scanner.FOV_MM > ge
    assert scanner.FOV_MM >= 2 * scanner.fov_radius_mm(381)


def test_the_bin_size_is_stirs_not_ges(cmcfg):
    """`BIN_MM` 2.1306 is STIR's D690 value; GE's own CTAC default is 2.11480.

    A 0.74 % difference, carried knowingly: `BIN_MM` is what the decoder stamps
    into every header, so changing it changes the geometry of every existing
    sinogram. Pinned so the gap cannot widen unnoticed.
    """
    ge = float(cmcfg["CTAC_DEFAULT_RSINO_STEP"])
    assert ge == pytest.approx(2.11480, abs=1e-9)
    assert scanner.BIN_MM != pytest.approx(ge, rel=1e-4)
    assert abs(scanner.BIN_MM / ge - 1) < 0.01      # within 1 %, not a typo


def test_our_mu_is_no_longer_ges_scatter_model_mu(cmcfg):
    """The specific confusion that was fixed: CTAC mu, not scatter-model mu.

    `MU_WATER_511` was 0.0096, which is exactly
    `CMP_CFG_MODEL_SCATTER_MU_DEFAULT` -- GE's mu for the SCATTER model, used by
    mistake in the attenuation path and 3.2 % high there.
    """
    scatter_mu = float(cmcfg["CMP_CFG_MODEL_SCATTER_MU_DEFAULT"])
    assert scatter_mu == pytest.approx(0.0096, abs=1e-9)
    assert scanner.MU_WATER_511 != pytest.approx(scatter_mu, rel=1e-6)


# ------------------------------------------------------------- calibration

def test_the_bundled_cal_files_are_the_consoles_own():
    """`vendor/cal/*` must be byte-identical to the console records they copy."""
    import hashlib

    ours = sorted((ROOT / "vendor" / "cal").glob("*.3dnorm")) + \
        sorted((ROOT / "vendor" / "cal").glob("*.3dwcc"))
    if not ours:
        pytest.skip("no bundled cal files")
    _need(PETSW / "cal")
    for p in ours:
        theirs = PETSW / "cal" / p.name
        assert theirs.exists(), f"{p.name} is not in the console's cal directory"
        assert (hashlib.sha256(p.read_bytes()).hexdigest()
                == hashlib.sha256(theirs.read_bytes()).hexdigest()), (
            f"{p.name} differs from the console's copy")


def test_the_default_registry_names_the_bundled_calibration():
    """`cal/.default` says which cal set is live; the bundle must be that one."""
    text = _need(PETSW / "cal" / ".default").read_text()
    for suffix in ("3dnorm", "3dwcc"):
        ours = sorted((ROOT / "vendor" / "cal").glob(f"*.{suffix}"))
        if not ours:
            continue
        uid = ours[0].name[: -len(suffix) - 1]
        assert uid in text, (
            f"vendor/cal holds {uid}.{suffix} but cal/.default names a "
            f"different {suffix} record")
