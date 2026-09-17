"""`vendor/estimate.py`: the single command that drives GE's kernel."""

from __future__ import annotations

import ast
import os
import sys

import pytest

import estimate
from utils import container

INFO = """\
file            : /drop/petRDFS/AAA/BBB/CCC/SINO0001
size            : 27,930,476 bytes
data type       : 7 (sinogram (block-compressed))

-- scan --
  frame_duration_ms     : 300,000
  prompts               : 13,954,115
  delays                : 1,726,089
  bed_number            : 2
  table_position_mm     : -125.17

-- geometry --
  num_tof_bins          : 55
  axial_fov_mm          : 156.70

-- UIDs / calibration --
  study_instance_uid    : 1.2.840.113619.2.290.663120.1697775916.195435
  series_instance_uid   : 1.2.840.113619.2.290.663120.1697776129.151600
"""

NORM_UID = "1.2.840.113619.2.290.663120.1697775916.195435"
BUNDLED_SRC = "/petRDFS/JFEJGPAB/SYSNDZAQ/PQRPXCCJ/SINO0001"


def test_estimate_needs_nothing_but_the_standard_library():
    tree = ast.parse(open(estimate.__file__).read())
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    allowed = set(sys.stdlib_module_names) | {"utils", "__future__"}
    assert not (imported - allowed), (
        "estimate.py imports something outside the standard library: %s"
        % sorted(imported - allowed))


def test_the_only_way_out_to_a_vendor_file_is_utils_container():
    src = open(estimate.__file__).read()
    assert "custom_tool" not in src.replace("/opt/custom_tool", "")
    assert "REPO" not in src


def stub_info(monkeypatch, text=INFO):
    """Stand in for `ge_rdf_tool.py info` running in the container."""
    monkeypatch.setattr(container, "rdf_info", lambda raw, **kw: text)


def test_raw_header_reads_every_field(monkeypatch):
    stub_info(monkeypatch)
    info = estimate.raw_header("/drop/petRDFS/AAA/BBB/CCC/SINO0001")
    assert info["table_position_mm"] == pytest.approx(-125.17)
    assert info["bed_number"] == 2
    assert info["prompts"] == 13954115
    assert info["frame_duration_ms"] == 300000
    assert info["num_tof_bins"] == 55
    assert info["axial_fov_mm"] == pytest.approx(156.70)


def test_raw_header_relabels_the_two_mislabelled_uids(monkeypatch):
    stub_info(monkeypatch)
    info = estimate.raw_header("/x/SINO0001")
    assert info["norm_cal_uid"] == NORM_UID
    assert info["wcc_cal_uid"] == "1.2.840.113619.2.290.663120.1697776129.151600"


def test_raw_header_keeps_a_negative_table_position(monkeypatch):
    stub_info(monkeypatch)
    assert estimate.raw_header("/x/SINO0001")["table_position_mm"] < 0


def test_raw_header_refuses_when_the_table_position_is_absent(monkeypatch):
    stub_info(monkeypatch, "-- scan --\n  prompts : 5\n")
    with pytest.raises(SystemExit, match="table_position_mm"):
        estimate.raw_header("/x/SINO0001")


def test_raw_header_shows_why_the_tool_failed(monkeypatch):
    stub_info(monkeypatch, "\nerror: cannot open /x/SINO0001\n")
    with pytest.raises(SystemExit) as e:
        estimate.raw_header("/x/SINO0001")
    assert "cannot open" in str(e.value)


@pytest.fixture
def cal(monkeypatch, tmp_path):
    """Stub the container's view of `/usr/PET/systemConfig/cal/`."""
    records: dict[str, dict] = {}
    vendor_cal = tmp_path / "vendor" / "cal"
    vendor_cal.mkdir(parents=True)
    monkeypatch.setattr(estimate, "HERE", str(tmp_path / "vendor"))
    monkeypatch.setattr(container, "cal_tags",
                        lambda uid, suffix, tags, **kw: records.get(uid))

    class Cal:
        tmp = tmp_path
        dir = str(vendor_cal)

        @staticmethod
        def record(uid, kind="PET 3D Normalization", src=BUNDLED_SRC):
            records[uid] = {"kind": kind, "src": src}

        @staticmethod
        def bundle(src=BUNDLED_SRC):
            open(os.path.join(str(vendor_cal),
                              "norm_DXRM3_20231020.rdf"), "wb").write(b"norm")
            for uid in (NORM_UID,):
                open(os.path.join(str(vendor_cal), uid + ".3dnorm"), "wb").write(b"x")
            monkeypatch.setattr(estimate, "bundled_source", lambda record: src)
            return os.path.join(str(vendor_cal), "norm_DXRM3_20231020.rdf")

    return Cal


def test_no_uid_means_no_norm(cal):
    assert estimate.resolve_norm("", "/x/SINO0001") is None
    assert estimate.resolve_norm(None, "/x/SINO0001") is None


def test_a_missing_cal_record_is_reported_not_guessed(cal, capsys):
    assert estimate.resolve_norm(NORM_UID, "/x/SINO0001") is None
    assert ".3dnorm" in capsys.readouterr().err


def test_a_wcc_record_is_refused(cal, capsys):
    cal.record(NORM_UID, kind="3DWCC Annulus")
    assert estimate.resolve_norm(NORM_UID, "/x/SINO0001") is None
    assert "not a 3D normalisation" in capsys.readouterr().err


def test_the_norm_is_found_inside_the_exams_own_drop(cal, capsys):
    cal.record(NORM_UID)
    drop = cal.tmp / "12082026"
    (drop / "petRDFS" / "JFEJGPAB" / "SYSNDZAQ" / "PQRPXCCJ").mkdir(parents=True)
    want = drop / "petRDFS" / "JFEJGPAB" / "SYSNDZAQ" / "PQRPXCCJ" / "SINO0001"
    want.write_bytes(b"norm")
    raw = drop / "petRDFS" / "NQULSZKB" / "SYCQQWRX" / "PDKQJRIW" / "SINO0001"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"emission")

    assert estimate.resolve_norm(NORM_UID, str(raw)) == str(want)
    capsys.readouterr()


def test_the_bundled_copy_is_used_for_the_exam_it_belongs_to(cal, capsys):
    cal.record(NORM_UID)
    bundled = cal.bundle(BUNDLED_SRC)
    raw = cal.tmp / "drop" / "petRDFS" / "A" / "B" / "C" / "SINO0001"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"emission")

    assert estimate.resolve_norm(NORM_UID, str(raw)) == bundled
    capsys.readouterr()


def test_the_bundled_copy_is_refused_for_a_different_scan(cal, capsys):
    other_uid = "1.2.840.113619.2.290.663120.9999999999.000001"
    cal.record(other_uid, src="/petRDFS/NQNGGPAB/SYUKRVRX/ZVOAMELQ/SINO0001")
    cal.bundle(BUNDLED_SRC)
    raw = cal.tmp / "drop" / "petRDFS" / "A" / "B" / "C" / "SINO0001"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"emission")

    assert estimate.resolve_norm(other_uid, str(raw)) is None
    assert "not in this drop" in capsys.readouterr().err


JOB_HEAD = """\
python _s("IgJobReq.inputEmissionFileName[0]", "/old/em.rdf")
set var IgJobReq.reconType = 3
python _s("IgJobReq.normalizationSinogramFile", "/old/norm.rdf")
set var IgJobReq.emissionScatterFlag = 2
python _s("IgJobReq.inputTransmissionFileName[0]", "/old/pifa.dat")
set var IgJobReq.hrActivityFactor = 4.113164
"""


def test_write_job_swaps_exactly_the_three_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(estimate, "HERE", str(tmp_path))
    (tmp_path / "job.gdb").write_text(JOB_HEAD)
    out = tmp_path / "new.gdb"
    estimate.write_job(str(out), "/data/emission.rdf", "/data/mu.pifa",
                       "/data/norm.rdf")

    got = out.read_text().splitlines()
    src = JOB_HEAD.splitlines()
    assert len(got) == len(src)
    changed = [i for i, (a, b) in enumerate(zip(src, got)) if a != b]
    assert changed == [0, 2, 4]
    assert '"/data/emission.rdf"' in got[0]
    assert '"/data/norm.rdf"' in got[2]
    assert '"/data/mu.pifa"' in got[4]
    assert got[5] == "set var IgJobReq.hrActivityFactor = 4.113164"


def test_write_job_refuses_a_template_missing_a_field(tmp_path, monkeypatch):
    monkeypatch.setattr(estimate, "HERE", str(tmp_path))
    (tmp_path / "job.gdb").write_text(
        'python _s("IgJobReq.inputEmissionFileName[0]", "/old/em.rdf")\n')
    with pytest.raises(SystemExit, match="normalizationSinogramFile"):
        estimate.write_job(str(tmp_path / "n.gdb"), "a", "b", "c")


def test_the_real_job_gdb_carries_all_three_fields(tmp_path):
    real = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(estimate.__file__))), "vendor", "job.gdb")
    if not os.path.exists(real):
        pytest.skip("vendor/job.gdb not present")
    out = tmp_path / "job.gdb"
    estimate.write_job(str(out), "/data/emission.rdf", "/data/mu.pifa",
                       "/data/norm.rdf")
    src = open(real).read().splitlines()
    got = out.read_text().splitlines()
    assert len(got) == len(src)
    assert len([1 for a, b in zip(src, got) if a != b]) == 3
