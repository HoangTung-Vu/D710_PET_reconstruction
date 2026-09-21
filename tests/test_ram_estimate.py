from __future__ import annotations

import pytest

from tools import ram_estimate as R

FULL = dict(xy=337, n_plane=47, n_phase=8)


def test_primal_per_phase_matches_the_measured_anchor():
    e = R.estimate(events=1, ckpt_phase=True, amp=False, **FULL)
    assert e["primal_saved_channels"] == 1540
    assert e["primal_per_phase_GiB"] == pytest.approx(30.6, abs=0.1)


def test_dual_per_1e6_events_matches_the_measured_anchor():
    e = R.estimate(events=1_000_000, **FULL)
    assert e["dual_saved_floats"] == 291
    assert e["dual_per_phase_GiB"] == pytest.approx(1.08, abs=0.01)


def test_amp_halves_the_primal():
    a = R.estimate(events=1, amp=False, **FULL)["primal_per_phase_GiB"]
    b = R.estimate(events=1, amp=True, **FULL)["primal_per_phase_GiB"]
    assert b == pytest.approx(a / 2, rel=1e-9)


def test_checkpointing_removes_the_per_phase_multiplier():
    on = R.estimate(events=9_000_000, ckpt_phase=True, amp=False, **FULL)
    off = R.estimate(events=9_000_000, ckpt_phase=False, amp=False, **FULL)
    assert off["activations_GiB"] == pytest.approx(
        8 * on["activations_GiB"], rel=1e-9)
    assert off["peak_GiB"] > 300
    assert on["peak_GiB"] < 45


def test_ckpt_block_cuts_the_primal():
    a = R.estimate(events=1, ckpt_block=False, **FULL)["primal_per_phase_GiB"]
    b = R.estimate(events=1, ckpt_block=True, **FULL)["primal_per_phase_GiB"]
    assert b < a


def test_parameter_count_is_about_12M():
    n = R.estimate(events=1, **FULL)["parameters"]
    assert 11_000_000 < n < 14_000_000


def test_peak_grows_with_events_and_voxels():
    base = R.estimate(events=1_000_000, **FULL)["peak_GiB"]
    more = R.estimate(events=9_000_000, **FULL)["peak_GiB"]
    smaller = R.estimate(xy=97, n_plane=47, n_phase=8,
                         events=1_000_000)["peak_GiB"]
    assert more > base > smaller


def test_check_reports_whether_it_fits():
    tight = R.check(budget_gib=8.0, xy=337, n_plane=47, events=9_000_000,
                    n_phase=8)
    roomy = R.check(budget_gib=96.0, xy=337, n_plane=47, events=9_000_000,
                    n_phase=8, amp=True, ckpt_block=True)
    assert not tight["fits"]
    assert roomy["fits"]


def test_require_refuses_a_configuration_that_cannot_fit():
    with pytest.raises(SystemExit):
        R.require(budget_gib=4.0, xy=337, n_plane=47, events=9_000_000,
                  n_phase=8)


def test_require_passes_with_force():
    e = R.require(budget_gib=4.0, force=True, xy=337, n_plane=47,
                  events=9_000_000, n_phase=8)
    assert not e["fits"]
