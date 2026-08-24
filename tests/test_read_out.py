"""`vendor/read_out.py` -- summarise what `extract.gdb` dumped.

Its one hard constraint is that it must run **inside the container**, which has
no numpy, so the stdlib fallback is not decoration.  These tests exercise both
paths.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import read_out

SHAPE = (4, 6, 5)          # view x v x u, the sidecar's own axis names


def dump(tmp_path, name="normdt.f32", dtype="<f4", meta=None, data=None):
    p = tmp_path / name
    a = (np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
         if data is None else data)
    a.astype(dtype).tofile(p)
    side = {"produced_by": "extract.gdb", "axes": "phi, v_theta, u",
            "dtype": dtype.lstrip("<"), "wcc_applied": False,
            "number_phi": SHAPE[0], "number_v_theta": SHAPE[1],
            "number_u": SHAPE[2]}
    side.update(meta or {})
    with open(str(p) + ".json", "w") as f:
        json.dump(side, f)
    return str(p)


def test_shape_comes_from_the_sidecar_not_from_a_constant(tmp_path):
    meta = read_out.sidecar(dump(tmp_path))
    assert read_out.shape_of(meta) == SHAPE


def test_a_sidecar_without_the_axis_keys_has_no_shape():
    assert read_out.shape_of({"produced_by": "x"}) is None


def test_summary_reports_shape_and_provenance(tmp_path, monkeypatch, capsys):
    p = dump(tmp_path)
    monkeypatch.setattr("sys.argv", ["read_out.py", p])
    read_out.main()
    out = capsys.readouterr().out
    assert "extract.gdb" in out
    assert "(4, 6, 5)" in out
    assert "no WCC applied" in out


def test_a_note_in_the_sidecar_is_shown(tmp_path, monkeypatch, capsys):
    p = dump(tmp_path, meta={"note": "randoms, from singles"})
    monkeypatch.setattr("sys.argv", ["read_out.py", p])
    read_out.main()
    assert "randoms, from singles" in capsys.readouterr().out


def test_a_single_view_can_be_summarised(tmp_path, monkeypatch, capsys):
    p = dump(tmp_path)
    monkeypatch.setattr("sys.argv", ["read_out.py", p, "--view", "1"])
    read_out.main()
    out = capsys.readouterr().out
    a = np.arange(np.prod(SHAPE), dtype="<f4").reshape(SHAPE)
    assert "view 1" in out
    assert f"max {a[1].max():g}" in out


def test_npy_export_round_trips(tmp_path, monkeypatch, capsys):
    p = dump(tmp_path)
    monkeypatch.setattr("sys.argv", ["read_out.py", p, "--npy"])
    read_out.main()
    capsys.readouterr()
    a = np.load(str(tmp_path / "normdt.npy"))
    assert a.shape == SHAPE
    assert a.dtype == np.dtype("<f4")


def test_an_int_dump_is_read_with_its_own_dtype(tmp_path, monkeypatch, capsys):
    """`singles.i32` is 576 x 24 int32, not a sinogram; the sidecar says so."""
    data = np.arange(24, dtype=np.int64).reshape(4, 6)
    p = dump(tmp_path, name="singles.i32", dtype="<i4", data=data,
             meta={"number_phi": 4, "number_v_theta": 6, "number_u": 1})
    monkeypatch.setattr("sys.argv", ["read_out.py", p])
    read_out.main()
    out = capsys.readouterr().out
    assert "96 bytes, 24 elements of i4" in out
    assert "max 23" in out


def test_it_works_without_numpy(tmp_path, monkeypatch, capsys):
    """The container that produces these files has no numpy."""
    p = dump(tmp_path)
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr("sys.argv", ["read_out.py", p])
    read_out.main()
    out = capsys.readouterr().out
    assert "extract.gdb" in out
    assert "nonzero" in out
    assert "shape" not in out          # the fallback stays 1-D on purpose


def test_npy_without_numpy_fails_loudly(tmp_path, monkeypatch, capsys):
    p = dump(tmp_path)
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr("sys.argv", ["read_out.py", p, "--npy"])
    with pytest.raises(SystemExit, match="needs numpy"):
        read_out.main()
