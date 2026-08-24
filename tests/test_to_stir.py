"""`vendor/to_stir.py` -- GE's flat `.f32` arrays -> Interfile STIR.

The converter hard-codes the real bed shape (288 x 553 x 381 = 243 MB per
float32 term), so the tests run it against the miniature scanner by pointing
`GE_SHAPE` at that instead.  Nothing else is patched: the mapping, the header
clone and the refusals are the shipped code.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import synth_hs
import to_stir

VIEWS, TANG, RINGS = 8, 9, 6
PLANES = synth_hs.num_planes(RINGS)
MINI_SHAPE = (VIEWS, PLANES, TANG)                 # view x plane x u, GE order


@pytest.fixture(autouse=True)
def _mini_shape(monkeypatch):
    monkeypatch.setattr(to_stir, "GE_SHAPE", MINI_SHAPE)


def ge_array(seed, dtype="<f4", scale=100.0):
    rng = np.random.default_rng(seed)
    a = rng.random(MINI_SHAPE) * scale
    return (np.rint(a).astype(dtype) if dtype != "<f4" else a.astype("<f4"))


def make_case(tmp_path, prompts=None, template_data=None):
    """A vendor output directory + the matching decoded template."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    prompts = ge_array(0, "<u2", 300.0) if prompts is None else prompts
    prompts.astype("<u2").tofile(vendor / "prompts.u16")
    for i, name in enumerate(("randoms", "scatter", "normdt", "norm_only"), 1):
        ge_array(i).tofile(vendor / f"{name}.f32")
    with open(vendor / "estimate.json", "w") as f:
        json.dump({"wcc_applied": False, "table_position_mm": -125.17}, f)

    decoded = tmp_path / "decoded"
    decoded.mkdir()
    data = (to_stir.ge_to_stir(prompts).astype("<i2") if template_data is None
            else template_data)
    synth_hs.write(str(decoded / "bed1"), RINGS, VIEWS * 2, TANG, data=data)
    return str(vendor), str(decoded / "bed1.hs")


# ------------------------------------------------------------- the mapping

def test_ge_to_stir_is_the_documented_index_formula():
    """`stir[0, plane, 287 - ge_view, u] = ge[ge_view, plane, u]`, and only that."""
    ge = ge_array(7)
    st = to_stir.ge_to_stir(ge)
    assert st.shape == (1, PLANES, VIEWS, TANG)
    for v in range(VIEWS):
        for p in range(PLANES):
            assert (st[0, p, VIEWS - 1 - v] == ge[v, p]).all()


def test_the_plane_axis_is_untouched():
    """No interpolation anywhere: the two michelograms are the same one."""
    ge = ge_array(8)
    st = to_stir.ge_to_stir(ge)
    assert (st[0].sum(axis=(1, 2)) == pytest.approx(ge.sum(axis=(0, 2)), rel=1e-6))


def test_ge_to_stir_preserves_every_count():
    ge = ge_array(9)
    assert to_stir.ge_to_stir(ge).sum() == pytest.approx(ge.sum(), rel=1e-6)


def test_ge_to_stir_output_is_contiguous():
    """`.tofile` on a view would silently write the wrong byte order out."""
    assert to_stir.ge_to_stir(ge_array(10)).flags["C_CONTIGUOUS"]


# -------------------------------------------------------------- read_ge

def test_read_ge_refuses_the_wrong_element_count(tmp_path):
    p = tmp_path / "short.f32"
    np.zeros(10, "<f4").tofile(p)
    with pytest.raises(SystemExit, match="expected"):
        to_stir.read_ge(str(p), "<f4")


def test_read_ge_returns_ge_order(tmp_path):
    p = tmp_path / "a.f32"
    a = ge_array(11)
    a.tofile(p)
    assert to_stir.read_ge(str(p), "<f4").shape == MINI_SHAPE


# ------------------------------------------------------- template_data_file

def test_template_data_file_resolves_beside_the_header(tmp_path):
    _vendor, template = make_case(tmp_path)
    got = to_stir.template_data_file(template)
    assert os.path.isabs(got) and got.endswith("bed1.s") and os.path.exists(got)


def test_template_data_file_refuses_a_header_without_the_key(tmp_path):
    p = tmp_path / "bad.hs"
    p.write_text("!INTERFILE :=\n")
    with pytest.raises(SystemExit, match="name of data file"):
        to_stir.template_data_file(str(p))


# --------------------------------------------------------------- verify

def test_verify_passes_when_the_two_paths_agree(tmp_path):
    vendor, template = make_case(tmp_path)
    checks = to_stir.verify(vendor, template)
    assert checks["bit_exact_vs_decoded"] is True
    assert checks["prompts"] > 0


def test_verify_refuses_a_bin_mapping_that_does_not_reproduce(tmp_path):
    """One flipped bin must stop the run; a wrong bin order is unrecoverable."""
    prompts = ge_array(0, "<u2", 300.0)
    wrong = to_stir.ge_to_stir(prompts).astype("<i2").copy()
    wrong[0, 0, 0, 0] += 1
    vendor, template = make_case(tmp_path, prompts=prompts, template_data=wrong)
    with pytest.raises(SystemExit, match="does not reproduce"):
        to_stir.verify(vendor, template)


def test_verify_refuses_a_view_axis_written_straight_through(tmp_path):
    """The whole point of the check: a mirrored reconstruction is caught here."""
    prompts = ge_array(0, "<u2", 300.0)
    straight = np.ascontiguousarray(prompts.transpose(1, 0, 2))[None].astype("<i2")
    vendor, template = make_case(tmp_path, prompts=prompts, template_data=straight)
    with pytest.raises(SystemExit, match="does not reproduce"):
        to_stir.verify(vendor, template)


def test_verify_refuses_a_template_from_another_bed(tmp_path):
    vendor, _template = make_case(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    synth_hs.write(str(other / "bed9"), RINGS, VIEWS * 2, TANG + 2)
    with pytest.raises(SystemExit, match="not the same bed"):
        to_stir.verify(vendor, str(other / "bed9.hs"))


def test_verify_refuses_when_prompts_are_missing(tmp_path):
    vendor, template = make_case(tmp_path)
    os.remove(os.path.join(vendor, "prompts.u16"))
    with pytest.raises(SystemExit, match="cannot verify"):
        to_stir.verify(vendor, template)


# ------------------------------------------------------------- write_term

def test_write_term_clones_the_template_header(tmp_path):
    """Same ExamInfo by construction -- that is what killed `same_bins()`."""
    _vendor, template = make_case(tmp_path)
    out = tmp_path / "work"
    out.mkdir()
    to_stir.write_term(ge_array(3), "randoms", str(out), template)

    src = open(template).read().splitlines()
    got = open(out / "randoms.hs").read().splitlines()
    assert len(src) == len(got)
    changed = {a for a, b in zip(src, got) if a != b}
    assert len(changed) == 3, f"more than the three intended keys changed: {changed}"
    body = "\n".join(got)
    assert "name of data file := randoms.s" in body
    assert "!number format := float" in body
    assert "!number of bytes per pixel := 4" in body


def test_write_term_writes_float32_in_stir_order(tmp_path):
    _vendor, template = make_case(tmp_path)
    out = tmp_path / "work"
    out.mkdir()
    ge = ge_array(4)
    stats = to_stir.write_term(ge, "scatter", str(out), template)
    a = np.fromfile(out / "scatter.s", dtype="<f4")
    assert a.size == ge.size
    assert stats["sum"] == pytest.approx(ge.sum(), rel=1e-5)
    assert (a.reshape(1, PLANES, VIEWS, TANG) == to_stir.ge_to_stir(ge)).all()


def test_the_written_terms_read_back_through_stir(tmp_path, sirf):
    """The header clone has to survive STIR's own parser, not just a regex."""
    _vendor, template = make_case(tmp_path)
    out = tmp_path / "work"
    out.mkdir()
    ge = ge_array(5)
    to_stir.write_term(ge, "normdt", str(out), template)
    ad = sirf.AcquisitionData(str(out / "normdt.hs"))
    assert ad.as_array().shape == (1, PLANES, VIEWS, TANG)
    assert ad.as_array().sum() == pytest.approx(ge.sum(), rel=1e-4)


# ------------------------------------------------------------------ main

def test_main_writes_five_terms_and_a_sidecar(tmp_path, monkeypatch, capsys):
    vendor, template = make_case(tmp_path)
    out = tmp_path / "work"
    monkeypatch.setattr("sys.argv",
                        ["to_stir.py", "--vendor", vendor,
                         "--template", template, "--out", str(out)])
    assert to_stir.main() == 0
    capsys.readouterr()

    for stem in ("randoms", "scatter", "normdt", "norm_only", "background"):
        assert (out / f"{stem}.hs").exists() and (out / f"{stem}.s").exists()

    meta = json.loads((out / "to_stir.json").read_text())
    assert meta["verified"]["bit_exact_vs_decoded"] is True
    assert meta["wcc_applied"] is False
    assert meta["estimate"]["wcc_applied"] is False
    assert meta["mapping"] == "stir[0, plane, 287 - ge_view, u] = ge[ge_view, plane, u]"


def test_background_is_exactly_randoms_plus_scatter(tmp_path, monkeypatch, capsys):
    """`b` has to exist as one file; the notebook must not re-add the two."""
    vendor, template = make_case(tmp_path)
    out = tmp_path / "work"
    monkeypatch.setattr("sys.argv",
                        ["to_stir.py", "--vendor", vendor,
                         "--template", template, "--out", str(out)])
    to_stir.main()
    capsys.readouterr()
    r = np.fromfile(out / "randoms.s", "<f4")
    s = np.fromfile(out / "scatter.s", "<f4")
    b = np.fromfile(out / "background.s", "<f4")
    assert b == pytest.approx(r + s, rel=1e-6)


def test_main_writes_nothing_when_the_mapping_is_unproven(tmp_path, monkeypatch):
    prompts = ge_array(0, "<u2", 300.0)
    wrong = to_stir.ge_to_stir(prompts).astype("<i2").copy()
    wrong[0, 0, 0, 0] += 1
    vendor, template = make_case(tmp_path, prompts=prompts, template_data=wrong)
    out = tmp_path / "work"
    monkeypatch.setattr("sys.argv",
                        ["to_stir.py", "--vendor", vendor,
                         "--template", template, "--out", str(out)])
    with pytest.raises(SystemExit):
        to_stir.main()
    assert not list(out.glob("*.s"))


def test_a_missing_term_is_reported_as_a_failure(tmp_path, monkeypatch, capsys):
    """No scatter means no `b`, and the caller has to hear about it.

    Writing randoms alone under the name `background` would be a silent
    under-subtraction, so the term is skipped -- but exiting 0 would let
    `run_exam.sh` mark the bed finished while `work/<bed>/` has no `b` in it.
    """
    vendor, template = make_case(tmp_path)
    os.remove(os.path.join(vendor, "scatter.f32"))
    out = tmp_path / "work"
    monkeypatch.setattr("sys.argv",
                        ["to_stir.py", "--vendor", vendor,
                         "--template", template, "--out", str(out)])
    assert to_stir.main() == 1
    err = capsys.readouterr().err
    assert "scatter" in err and "background" in err
    assert (out / "randoms.hs").exists()
    assert not (out / "scatter.hs").exists()
    assert not (out / "background.hs").exists()
