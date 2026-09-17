"""GE's TOF scatter distribution, from `scatter_tof.f32` to the weights OSEM uses."""

from __future__ import annotations

import json

import numpy as np
import pytest

import to_stir
from utils import terms

VIEWS, NTOF, DSNU, TANG = 8, 12, 5, 21


def write_vendor(tmp_path, buf=None, ntof=NTOF, dsnu=DSNU, nview=VIEWS):
    """A vendor directory holding only `scatter_tof.f32` and its sidecar."""
    d = tmp_path / "vendor"
    d.mkdir(exist_ok=True)
    if buf is None:
        rng = np.random.default_rng(0)
        buf = rng.random((nview, ntof, dsnu) + to_stir.TOF_AXIAL_SHAPE)
    buf.astype("<f4").tofile(d / "scatter_tof.f32")
    with open(d / "scatter_tof.f32.json", "w") as f:
        json.dump({"number_phi": nview, "numTOF_bins": ntof, "ds_nu": dsnu}, f)
    return d


class FakeCase:
    """Just enough of `utils.paths.Case` for `vendor_tof_weights`."""

    name = "fake"

    def __init__(self, work):
        self.work = work

    def work_bed(self, n):
        return self.work


def test_absent_scatter_tof_is_not_an_error(tmp_path):
    d = tmp_path / "vendor"
    d.mkdir()
    assert to_stir.convert_scatter_tof(str(d), str(tmp_path)) is None


def test_the_sidecar_is_required(tmp_path):
    d = write_vendor(tmp_path)
    (d / "scatter_tof.f32.json").unlink()
    with pytest.raises(SystemExit, match="sidecar"):
        to_stir.convert_scatter_tof(str(d), str(tmp_path))


def test_reshape_and_axis_reversals(tmp_path):
    rng = np.random.default_rng(1)
    buf = rng.random((VIEWS, NTOF, DSNU) + to_stir.TOF_AXIAL_SHAPE)
    d = write_vendor(tmp_path, buf)
    stats = to_stir.convert_scatter_tof(str(d), str(tmp_path))
    assert stats["shape"] == [NTOF, VIEWS, DSNU]
    assert stats["tof_axis"] == "stir"

    w = np.load(tmp_path / "scatter_tof.npy")
    want = buf.sum(axis=(3, 4))
    for v in range(VIEWS):
        for t in range(NTOF):
            assert w[NTOF - 1 - t, VIEWS - 1 - v] == pytest.approx(want[v, t],
                                                                   rel=1e-6)


def test_the_reported_peak_bin_is_on_the_axis_as_written(tmp_path):
    buf = np.zeros((VIEWS, NTOF, DSNU) + to_stir.TOF_AXIAL_SHAPE)
    buf[:, 2] = 1.0
    d = write_vendor(tmp_path, buf)
    stats = to_stir.convert_scatter_tof(str(d), str(tmp_path))
    assert stats["peak_tof_bin"] == NTOF - 1 - 2
    w = np.load(tmp_path / "scatter_tof.npy")
    assert int(w.sum(axis=(1, 2)).argmax()) == stats["peak_tof_bin"]


def full_scatter_from(prof):
    """A `(view, plane, tangential)` scatter whose coarse binning is `prof`."""
    edges = np.linspace(0, TANG, DSNU + 1).astype(int)
    full = np.zeros((prof.shape[0], TANG))
    for j in range(DSNU):
        full[:, edges[j]:edges[j + 1]] = (prof[:, j:j + 1]
                                          / (edges[j + 1] - edges[j]))
    return np.broadcast_to(full[:, None, :], (prof.shape[0], 3, TANG)).copy()


def test_the_cross_check_accepts_a_consistent_scatter(tmp_path):
    rng = np.random.default_rng(2)
    prof = rng.random((VIEWS, DSNU))
    buf = (prof[:, None, :, None, None]
           * rng.random((1, NTOF, 1) + to_stir.TOF_AXIAL_SHAPE))
    d = write_vendor(tmp_path, buf)
    stats = to_stir.convert_scatter_tof(str(d), str(tmp_path),
                                        scatter_ge=full_scatter_from(prof))
    assert stats["corr_vs_scatter"] > 0.95
    assert (tmp_path / "scatter_tof.npy").exists()


def test_a_flipped_tangential_axis_is_refused_and_writes_nothing(tmp_path):
    rng = np.random.default_rng(3)
    prof = np.linspace(1.0, 8.0, DSNU) * rng.random((VIEWS, 1))
    buf = (prof[:, None, :, None, None]
           * np.ones((1, NTOF, 1) + to_stir.TOF_AXIAL_SHAPE))
    d = write_vendor(tmp_path, buf)
    full = full_scatter_from(prof)

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit, match="correlates"):
        to_stir.convert_scatter_tof(str(d), str(out), scatter_ge=full[:, :, ::-1])
    assert not (out / "scatter_tof.npy").exists()


def test_strip_tof_turns_a_5d_header_into_a_4d_one():
    hdr = ("number of dimensions := 5\n"
           "matrix axis label [5] := timing positions\n"
           "!matrix size [5] := 11\n"
           "matrix axis label [4] := segment\n"
           "TOF mashing factor := 5\n"
           "Maximum number of (unmashed) TOF time bins := 55\n")
    out = to_stir.strip_tof(hdr)
    assert "number of dimensions := 4" in out
    assert "[5]" not in out
    assert "TOF mashing factor" not in out
    assert "Maximum number of (unmashed) TOF time bins := 55" in out


def test_template_tof_bins_reads_axis_5(tmp_path):
    p = tmp_path / "t.hs"
    p.write_text("!matrix size [4] := 23\n!matrix size [5] := 11\n")
    assert to_stir.template_tof_bins(str(p)) == 11
    q = tmp_path / "n.hs"
    q.write_text("!matrix size [4] := 23\n")
    assert to_stir.template_tof_bins(str(q)) == 1


def test_upsample_matrix_is_a_partition_of_unity():
    M = terms._upsample_matrix(DSNU, TANG)
    assert M.shape == (DSNU, TANG)
    assert M.min() >= 0
    np.testing.assert_allclose(M.sum(axis=0), 1.0, atol=1e-12)


def test_upsample_matrix_reproduces_a_constant():
    M = terms._upsample_matrix(DSNU, TANG)
    np.testing.assert_allclose(np.ones(DSNU) @ M, np.ones(TANG), atol=1e-12)


def test_weights_sum_to_one_along_tof(tmp_path):
    d = write_vendor(tmp_path)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    w, note = terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS, TANG)
    assert w.shape == (NTOF, VIEWS, TANG)
    np.testing.assert_allclose(w.sum(axis=0), 1.0, rtol=1e-5)
    assert "mash 1" in note


def test_mashing_groups_adjacent_bins(tmp_path):
    rng = np.random.default_rng(4)
    buf = rng.random((VIEWS, NTOF, DSNU) + to_stir.TOF_AXIAL_SHAPE)
    d = write_vendor(tmp_path, buf)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))

    fine, _ = terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS, TANG)
    coarse, note = terms.vendor_tof_weights(FakeCase(tmp_path), 1, 4, VIEWS, TANG)
    assert coarse.shape == (4, VIEWS, TANG)
    assert "mash 3" in note
    np.testing.assert_allclose(fine.reshape(4, 3, VIEWS, TANG).sum(axis=1),
                               coarse, rtol=1e-5)


def test_a_mash_that_does_not_divide_is_refused(tmp_path):
    d = write_vendor(tmp_path)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    with pytest.raises(SystemExit, match="does not divide"):
        terms.vendor_tof_weights(FakeCase(tmp_path), 1, 5, VIEWS, TANG)


def test_a_view_count_mismatch_is_refused(tmp_path):
    d = write_vendor(tmp_path)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    with pytest.raises(SystemExit, match="views"):
        terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS + 1, TANG)


def test_an_empty_buffer_is_refused(tmp_path):
    zeros = np.zeros((VIEWS, NTOF, DSNU) + to_stir.TOF_AXIAL_SHAPE)
    d = write_vendor(tmp_path, zeros)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    with pytest.raises(SystemExit, match="empty"):
        terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS, TANG)


def test_missing_weights_return_none_so_the_caller_can_fall_back(tmp_path):
    assert terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS, TANG) is None


def test_weights_from_before_the_axis_fix_are_refused(tmp_path):
    d = write_vendor(tmp_path)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    (tmp_path / "to_stir.json").write_text(json.dumps({"scatter_tof": {}}))
    with pytest.raises(SystemExit, match="2026-08-29"):
        terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS, TANG)


class FakeCaseWithPrompt(FakeCase):
    """A `FakeCase` that also carries the prompts header `check_tof_axis` reads."""

    def prompt(self, n):
        return self.work / f"bed{n}.hs"


def test_tof_prompts_without_the_axis_stamp_are_refused(tmp_path):
    c = FakeCaseWithPrompt(tmp_path)
    c.prompt(1).write_text("!matrix size [5] := 11\nTOF mashing factor := 5\n")
    with pytest.raises(SystemExit, match="MIRRORED"):
        terms.check_tof_axis(c, 1)


def test_stamped_tof_prompts_are_accepted(tmp_path):
    c = FakeCaseWithPrompt(tmp_path)
    c.prompt(1).write_text("!matrix size [5] := 11\n"
                           f"{terms.TOF_AXIS_KEY} := STIR timing positions\n")
    terms.check_tof_axis(c, 1)


class FakeAcq:
    """The `get_uniform_copy` and `fill` methods `expand_to_tof` requires of the prompts."""

    def __init__(self, shape):
        self.shape = shape
        self.filled = None

    def get_uniform_copy(self, _v):
        return FakeAcq(self.shape)

    def fill(self, a):
        self.filled = a


def tiny_terms(n_tof):
    """`(objs, A)` holding one bed's worth of the arrays `expand_to_tof` needs."""
    shape = (1, 4, VIEWS, TANG)
    rng = np.random.default_rng(6)
    A = {"prompts": rng.random((n_tof,) + shape[1:]).astype(np.float32),
         "randoms": rng.random(shape).astype(np.float32),
         "scatter": rng.random(shape).astype(np.float32)}
    A["background"] = A["randoms"] + A["scatter"]
    return {"prompts": FakeAcq(shape)}, A


def test_an_unmashed_override_profile_is_mashed_to_fit(tmp_path):
    objs, A = tiny_terms(4)
    prof = np.arange(1, 13, dtype=float)
    terms.expand_to_tof(FakeCase(tmp_path), 1, objs, A, 4, tof_scatter=prof)
    assert A["background"].shape == (4,) + A["scatter"].shape[1:]
    assert terms.total(A["background"]) == pytest.approx(
        terms.total(A["randoms"]) + terms.total(A["scatter"]), rel=1e-5)


def test_an_override_profile_that_does_not_divide_is_refused(tmp_path):
    objs, A = tiny_terms(4)
    with pytest.raises(SystemExit, match="does not divide"):
        terms.expand_to_tof(FakeCase(tmp_path), 1, objs, A, 4,
                            tof_scatter=np.ones(7))


def test_a_negative_override_profile_is_refused(tmp_path):
    objs, A = tiny_terms(4)
    with pytest.raises(SystemExit, match="non-negative"):
        terms.expand_to_tof(FakeCase(tmp_path), 1, objs, A, 4,
                            tof_scatter=np.array([1.0, -1.0, 1.0, 1.0]))


def test_the_vendor_weights_conserve_the_background_total(tmp_path):
    d = write_vendor(tmp_path, nview=VIEWS, ntof=4, dsnu=DSNU)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    objs, A = tiny_terms(4)
    terms.expand_to_tof(FakeCase(tmp_path), 1, objs, A, 4)
    assert terms.total(A["background"]) == pytest.approx(
        terms.total(A["randoms"]) + terms.total(A["scatter"]), rel=1e-5)
    assert objs["background"].filled.shape == (4, 4, VIEWS, TANG)


def test_lors_with_no_scatter_get_zero_not_a_made_up_profile(tmp_path):
    rng = np.random.default_rng(5)
    buf = rng.random((VIEWS, NTOF, DSNU) + to_stir.TOF_AXIAL_SHAPE)
    buf[:, :, 0] = 0.0
    d = write_vendor(tmp_path, buf)
    to_stir.convert_scatter_tof(str(d), str(tmp_path))
    w, _ = terms.vendor_tof_weights(FakeCase(tmp_path), 1, NTOF, VIEWS, TANG)
    tot = w.sum(axis=0)
    assert (tot[:, 0] == 0).all()
    assert np.isfinite(w).all()
