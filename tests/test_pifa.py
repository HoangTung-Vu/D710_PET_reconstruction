"""`vendor/make_pifa.py` and `vendor/ct_to_pifa.py` -- GE's mu-map container.

The PIFA is the one file `pet_recon` reads for the scatter model, and two of
its properties cannot be checked by looking at the output sinogram: the header
byte layout (a wrong offset makes `ValidateCTAC` reject the file with a message
about a different field) and the units (mm^-1, not STIR's cm^-1, which is a
silent factor of ten in the scatter estimate).
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from utils import attenuation
import ct_to_pifa
import make_pifa

FORF = "1.2.840.113619.2.290.3.663120.905.1784281107.116"


def header_fields(path):
    """Decode the 164-byte header the same way `pet_recon` does."""
    with open(path, "rb") as f:
        h = f.read(make_pifa.HEADER_BYTES)
    ver, job = struct.unpack_from("<fI", h, 0)
    x, y, z = struct.unpack_from("<HHH", h, 8)
    dfov, = struct.unpack_from("<f", h, 16)
    entry, pos = struct.unpack_from("<II", h, 20)
    table, = struct.unpack_from("<f", h, 28)
    off, = struct.unpack_from("<I", h, 160)
    return dict(version=ver, job=job, matrix=(x, y, z), dfov=dfov,
                entry=entry, position=pos, table=table,
                frame_of_reference=h[32:96].split(b"\0")[0].decode(),
                spare=bytes(h[96:160]), offset=off)


# ------------------------------------------------------------ pack_header

def test_header_is_164_bytes_and_says_so():
    h = make_pifa.pack_header(128, 128, 47, 700.0, -125.17, FORF)
    assert len(h) == make_pifa.HEADER_BYTES == 164
    assert struct.unpack_from("<I", h, 160)[0] == 164


def test_every_field_lands_at_the_dwarf_offset(tmp_path):
    p = tmp_path / "a.pifa"
    p.write_bytes(make_pifa.pack_header(128, 128, 47, 700.0, -125.17, FORF,
                                        version=1.0, job_id=7,
                                        patient_entry=2, patient_position=3))
    f = header_fields(p)
    assert f == dict(version=1.0, job=7, matrix=(128, 128, 47), dfov=700.0,
                     entry=2, position=3, table=pytest.approx(-125.17, rel=1e-6),
                     frame_of_reference=FORF, spare=b"\0" * 64, offset=164)


def test_frame_of_reference_stays_nul_terminated(tmp_path):
    """`ValidateCTAC` strcmp's it; an unterminated 64th byte runs into spare."""
    p = tmp_path / "a.pifa"
    p.write_bytes(make_pifa.pack_header(8, 8, 2, 700.0, 0.0, "X" * 200))
    with open(p, "rb") as fh:
        h = fh.read(164)
    assert h[32:95] == b"X" * 63
    assert h[95] == 0


def test_spare_fields_are_copied_verbatim(tmp_path):
    spare = bytes(range(64))
    h = make_pifa.pack_header(8, 8, 2, 700.0, 0.0, FORF, spare=spare)
    assert h[96:160] == spare


def test_a_short_spare_is_padded_not_shifted(tmp_path):
    h = make_pifa.pack_header(8, 8, 2, 700.0, 0.0, FORF, spare=b"ab")
    assert h[96:160] == b"ab" + b"\0" * 62
    assert struct.unpack_from("<I", h, 160)[0] == 164     # not overwritten


# ------------------------------------------------------------ the CLI

def _cli(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["make_pifa.py", *[str(a) for a in argv]])
    make_pifa.main()


def test_npy_input_reads_the_shape_from_the_array(tmp_path, monkeypatch, capsys):
    mu = np.full((4, 8, 8), 0.0096, dtype=np.float32)     # z, y, x
    np.save(tmp_path / "mu.npy", mu)
    out = tmp_path / "out.pifa"
    _cli(monkeypatch, tmp_path / "mu.npy", out, "--dfov", 700,
         "--table-location", -47.92, "--frame-of-reference", FORF)
    capsys.readouterr()
    f = header_fields(out)
    assert f["matrix"] == (8, 8, 4)                       # x, y, z in the header
    assert os.path.getsize(out) == 164 + mu.size * 4


def test_cm_input_is_divided_by_ten(tmp_path, monkeypatch, capsys):
    """STIR's convention is cm^-1; writing it straight through is a 10x error."""
    np.save(tmp_path / "mu.npy", np.full((2, 4, 4), 0.096, dtype=np.float32))
    out = tmp_path / "out.pifa"
    _cli(monkeypatch, tmp_path / "mu.npy", out, "--units", "cm-1")
    capsys.readouterr()
    a = np.fromfile(out, dtype="<f4", offset=164)
    assert a == pytest.approx(0.0096, rel=1e-6)


def test_raw_input_needs_a_matrix_and_honours_cm(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "mu.raw"
    np.full(2 * 4 * 4, 0.096, dtype="<f4").tofile(raw)
    out = tmp_path / "out.pifa"
    with pytest.raises(SystemExit):
        _cli(monkeypatch, raw, out)                       # no --matrix
    _cli(monkeypatch, raw, out, "--matrix", 4, 4, 2, "--units", "cm-1")
    capsys.readouterr()
    assert np.fromfile(out, dtype="<f4", offset=164) == pytest.approx(0.0096, rel=1e-6)


def test_a_matrix_that_contradicts_the_array_is_refused(tmp_path, monkeypatch, capsys):
    np.save(tmp_path / "mu.npy", np.zeros((2, 4, 4), dtype=np.float32))
    with pytest.raises(SystemExit, match="contradicts"):
        _cli(monkeypatch, tmp_path / "mu.npy", tmp_path / "o.pifa",
             "--matrix", 4, 4, 3)


def test_raw_input_of_the_wrong_length_is_refused(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "mu.raw"
    np.zeros(7, dtype="<f4").tofile(raw)
    with pytest.raises(SystemExit, match="expected"):
        _cli(monkeypatch, raw, tmp_path / "o.pifa", "--matrix", 4, 4, 2)


def test_inspect_reports_what_was_written(tmp_path, monkeypatch, capsys):
    np.save(tmp_path / "mu.npy", np.full((3, 8, 8), 0.0093, dtype=np.float32))
    out = tmp_path / "out.pifa"
    _cli(monkeypatch, tmp_path / "mu.npy", out, "--table-location", -125.17,
         "--frame-of-reference", FORF)
    capsys.readouterr()
    _cli(monkeypatch, "--inspect", out)
    text = capsys.readouterr().out
    assert "8x8x3" in text
    assert "-125.17" in text
    assert FORF in text
    assert "4.000 B/voxel" in text                        # float32, no padding


# --------------------------------------------------------- ct_to_pifa

def test_resample_keeps_mu_in_per_mm(ct_dir, capsys):
    ct = attenuation.load(ct_dir)
    mu = ct_to_pifa.resample_to_pifa(ct, float(ct.z[0]), 128, 700.0, 47, 3.264583)
    capsys.readouterr()
    assert mu.shape == (47, 128, 128)
    water = mu[(mu > 0.005) & (mu < 0.012)]
    assert water.size and np.median(water) == pytest.approx(0.0096, rel=0.05)
    # GE's own selftest PIFA peaks at 0.0093 mm^-1; anything near 0.096 is cm^-1.
    assert mu.max() < 0.05


def test_resample_does_not_flip_y(ct_dir):
    """The PIFA is plain DICOM LPS: `mu_image`'s y-flip must NOT be applied.

    Flipping it makes GE's table mask remove tissue instead of the table, and
    the SSS estimate inflates (32.98 % -> 49.28 % measured on NEMA bed 2).
    """
    ct = attenuation.load(ct_dir)
    mu = ct_to_pifa.resample_to_pifa(ct, float(ct.z[0]), 64, 200.0, 47, 3.264583)
    rows = mu[mu.shape[0] // 2].sum(axis=1)
    half = len(rows) // 2
    assert rows[half:].sum() > rows[:half].sum(), "the insert moved to -y"

    # And it is exactly the y-mirror of what `attenuation` hands STIR.
    stir_side = attenuation.to_radiological(mu)
    assert (stir_side[:, ::-1] == mu).all()


def test_resample_grid_is_centred_on_the_scanner_axis(ct_dir):
    """`sx = dfov / matrix`: 700/128 = 5.46875 mm, centred on (0, 0)."""
    ct = attenuation.load(ct_dir)
    mu = ct_to_pifa.resample_to_pifa(ct, float(ct.z[0]), 32, 32 * ct.pixel_mm,
                                     47, 3.264583)
    # A cylinder centred on the axis stays centred after resampling.
    prof = mu[mu.shape[0] // 2].sum(axis=0)
    com = float((prof * np.arange(prof.size)).sum() / prof.sum())
    assert com == pytest.approx(32 // 2, abs=1.0)


def test_resample_warns_when_the_bed_leaves_the_ct(ct_dir, capsys):
    """Unlike `mu_image`, this fills air -- so it must at least say so."""
    ct = attenuation.load(ct_dir)
    ct_to_pifa.resample_to_pifa(ct, float(ct.z[-1]), 32, 700.0, 47, 3.264583)
    assert "NOT fully inside the CT" in capsys.readouterr().err


def test_the_pifa_plane_pitch_is_what_pet_recon_reported():
    """3.264583 = axial FOV / 48, which is not `geometry.PLANE_MM` (3.27).

    The two differ by 0.17 %, i.e. 0.25 mm across a 47-plane bed.  Keep them
    apart on purpose: the PIFA has to match what the vendor's kernel expects,
    and the STIR side has to match the Interfile's ring spacing.
    """
    from utils import geometry

    assert 3.264583 == pytest.approx(156.7 / 48.0, rel=1e-5)   # axial FOV / 48
    assert abs(3.264583 - geometry.PLANE_MM) / geometry.PLANE_MM < 2e-3
