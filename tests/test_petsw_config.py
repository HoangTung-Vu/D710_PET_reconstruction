from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np
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
    root = ET.parse(_need(LOCAL / "cmcfg.XR.xml")).getroot()
    return {el.tag: (el.text or "").strip() for el in root.iter()}


@pytest.fixture(scope="module")
def sharc():
    out = {}
    for line in _need(LOCAL / "sharcAp.cfg.XR").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        out[parts[0]] = parts[1:]
    return out


def line_of(path, pattern) -> int:
    for i, line in enumerate(_need(path).read_text().splitlines(), 1):
        if re.search(pattern, line):
            return i
    return 0


@pytest.mark.parametrize("tag, const, scale", [
    ("SYS_DETECTOR_RADIAL_SIZE", "R_MM", 1.0),
    ("SYS_TRANSAXIAL_CRYSTAL_0_OFFSET", "XTAL0_OFFSET_DEG", 1.0),
    ("SYS_RAD_CRYSTALS_PER_SYSTEM", "NDET", 1.0),
    ("SYS_AXI_CRYSTALS_PER_SYSTEM", "NRINGS", 1.0),
    ("SYS_NUMBER_SLICES", "NSEG0", 1.0),
    ("SYS_SLICE_THICKNESS", "PLANE_MM", 1.0),
    ("deltaAngle", "XTAL_PITCH_DEG", 1.0),
    ("CMP_CFG_IR_Z_AXIS_STANDARD_FILTER_RATIO_DEFAULT", "POST_FILTER_Z_RATIO", 1.0),
    ("CTAC_DEFAULT_MU_WATER_511", "MU_WATER_511", 0.1),
    ("CTAC_DEFAULT_MU_BONE_511", "MU_BONE_511", 0.1),
])
def test_the_scanner_constant_is_what_the_console_declares(
        cmcfg, tag, const, scale):
    assert tag in cmcfg, f"{tag} is not in cmcfg.XR.xml any more"
    assert float(cmcfg[tag]) * scale == pytest.approx(
        getattr(scanner, const), rel=1e-6), (
        f"cmcfg.XR.xml:{tag} = {cmcfg[tag]} (x {scale:g}) against "
        f"utils/scanner.py:{const} = {getattr(scanner, const)}")


def test_the_doi_is_the_effective_ring_diameter_minus_the_ring(cmcfg):
    eff = float(cmcfg["SYS_EFF_RING_DIAMETER"])
    assert eff == pytest.approx(827.0, abs=1e-9)
    assert eff / 2 - float(cmcfg["SYS_DETECTOR_RADIAL_SIZE"]) == pytest.approx(
        scanner.DOI_MM, abs=1e-6)
    assert eff / 2 == pytest.approx(scanner.R_EFF_MM, abs=1e-6)


def test_the_cited_line_numbers_still_point_at_the_right_elements():
    p = LOCAL / "cmcfg.XR.xml"
    assert line_of(p, r"<SYS_TRANSAXIAL_CRYSTAL_0_OFFSET\b") == 721
    assert line_of(p, r"<SYS_EFF_RING_DIAMETER\b") == 677


def test_the_timing_resolution_is_the_machines_own(sharc):
    assert float(sharc["TIMING_RESOLUTION"][0]) == pytest.approx(
        scanner.TIMING_PS, abs=1e-9)
    assert line_of(LOCAL / "sharcAp.cfg.XR", r"^TIMING_RESOLUTION\b") == 46


def test_the_unsuffixed_config_names_are_this_scanners(cmcfg):
    for bare, want in (("sharcAp.cfg", "sharcAp.cfg.XR"),
                       ("cmcfg.xml", "cmcfg.XR.xml")):
        p = _need(LOCAL / bare)
        assert p.is_symlink(), f"{bare} is no longer a symlink"
        assert p.readlink().name == want, f"{bare} -> {p.readlink().name}, not {want}"


def test_the_axial_post_filter_kernel_is_ges_own(sharc):
    got = [float(x) for x in sharc["PSF_AXIAL_KERNEL[]"]]
    r = scanner.POST_FILTER_Z_RATIO
    want = [1 / (2 + r), r / (2 + r), 1 / (2 + r)]
    assert got == pytest.approx(want, abs=5e-8)
    assert int(sharc["PSF_AXIAL_WINDOW_WIDTH"][0]) == len(got)


def test_the_post_filter_ratio_is_corroborated_by_the_config(cmcfg):
    assert float(cmcfg["CMP_CFG_IR_Z_AXIS_STANDARD_FILTER_RATIO_DEFAULT"]) == (
        pytest.approx(scanner.POST_FILTER_Z_RATIO, abs=1e-9))


def test_the_private_tags_scanner_cites_are_defined(cmcfg):
    tags = _need(LOCAL / "tagDetails.xml").read_text()
    for element, name, vr in (("0x10BA", "post_filter", "SS"),
                              ("0x10BB", "post_filt_parm", "FL"),
                              ("0x10DB", "ir_z_filter_flag", "SL"),
                              ("0x10DC", "ir_z_filter_ratio", "FL")):
        assert re.search(
            rf'VR="{vr}"\s+element="{element}"\s+group="0x0009"\s+name="{name}"',
            tags), f"(0009,{element[2:]}) {name} is not defined as {vr}"


PSF_LUT = np.dtype([("n", "<i4"), ("off", "<i4"), ("k", "<f4", 30)])

FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def _tangential_mm(n_tang: int):
    from utils.geometry import det_pair_map, detector_xy_mm

    d1, d2 = det_pair_map(scanner.NDET // 2, n_tang, scanner.NDET)
    xy = detector_xy_mm().astype(np.float64)
    a, b = xy[d1[0]], xy[d2[0]]
    return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.linalg.norm(b - a, axis=1)


def test_the_radial_psf_is_the_body_mean_of_ges_psf_lut():
    opt = pytest.importorskip("scipy.optimize")
    lut = _need(LOCAL / "psfLUT.XR")
    assert lut.stat().st_size == 381 * PSF_LUT.itemsize
    r = np.fromfile(lut, PSF_LUT)
    s = _tangential_mm(len(r))
    assert int((r["off"] + r["n"]).max()) <= len(s)

    def gauss(x, a, m, sg):
        return a * np.exp(-(x - m) ** 2 / (2 * sg ** 2))

    fwhm = {}
    for b in np.nonzero(np.abs(s) < 195.0)[0]:
        n, off = int(r["n"][b]), int(r["off"][b])
        x = s[off:off + n]
        dens = r["k"][b][:n].astype(np.float64) / np.abs(np.gradient(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", opt.OptimizeWarning)
            (_a, _m, sg), _ = opt.curve_fit(gauss, x, dens,
                                            p0=(dens.max(), s[b], 2.0))
        fwhm[int(b)] = FWHM_PER_SIGMA * abs(sg)

    centre = int(np.argmin(np.abs(s)))
    assert fwhm[centre] == pytest.approx(3.56, abs=0.02)
    got = float(np.mean(list(fwhm.values())))
    assert got == pytest.approx(scanner.PSF_XY_MM, abs=0.05), got
    assert list(scanner.PSF_FWHM_MM[:2]) == [scanner.PSF_XY_MM] * 2


def test_the_axial_psf_is_ges_psf_axial_kernel(sharc):
    k = np.array([float(x) for x in sharc["PSF_AXIAL_KERNEL[]"]])
    d = (np.arange(len(k)) - (len(k) - 1) / 2) * scanner.PLANE_MM
    var = float((k * d ** 2).sum() / k.sum())
    assert FWHM_PER_SIGMA * np.sqrt(var) == pytest.approx(scanner.PSF_Z_MM, abs=0.02)
    assert scanner.PSF_FWHM_MM[2] == scanner.PSF_Z_MM


def test_the_reconstructed_fov_deliberately_exceeds_ges_maximum(sharc):
    ge = float(sharc["CMP_MAX_DFOV"][0])
    assert ge == pytest.approx(700.0, abs=1e-9)
    assert scanner.FOV_MM > ge
    assert scanner.FOV_MM >= 2 * scanner.fov_radius_mm(381)


def test_the_bin_size_is_stirs_not_ges(cmcfg):
    ge = float(cmcfg["CTAC_DEFAULT_RSINO_STEP"])
    assert ge == pytest.approx(2.11480, abs=1e-9)
    assert scanner.BIN_MM != pytest.approx(ge, rel=1e-4)
    assert abs(scanner.BIN_MM / ge - 1) < 0.01


def test_our_mu_is_no_longer_ges_scatter_model_mu(cmcfg):
    scatter_mu = float(cmcfg["CMP_CFG_MODEL_SCATTER_MU_DEFAULT"])
    assert scatter_mu == pytest.approx(0.0096, abs=1e-9)
    assert scanner.MU_WATER_511 != pytest.approx(scatter_mu, rel=1e-6)


def test_the_bundled_cal_files_are_the_consoles_own():
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
    text = _need(PETSW / "cal" / ".default").read_text()
    for suffix in ("3dnorm", "3dwcc"):
        ours = sorted((ROOT / "vendor" / "cal").glob(f"*.{suffix}"))
        if not ours:
            continue
        uid = ours[0].name[: -len(suffix) - 1]
        assert uid in text, (
            f"vendor/cal holds {uid}.{suffix} but cal/.default names a "
            f"different {suffix} record")
