"""`lowdose`: the decimator's statistics and the case it writes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lm import events as ev
from lm import geom
from lowdose import thin, verify, write
from utils.paths import Case

RINGS, NDET, NTANG = 6, 16, 9
N_EVENTS = 60_000


@pytest.fixture(scope="module")
def binmap(mini_hs):
    return geom.BinMap(mini_hs)


@pytest.fixture(scope="module")
def events(binmap):
    rng = np.random.default_rng(7)
    ids, bins = binmap.lor_table()
    k = rng.integers(0, len(ids), N_EVENTS)
    e = np.zeros(N_EVENTS, dtype=[("xtal_a", "<u2"), ("xtal_b", "<u2"),
                                  ("tof_bin", "i1"), ("t_ms", "<u4")])
    e["xtal_a"], e["xtal_b"] = ids[k, 0], ids[k, 1]
    e["tof_bin"] = rng.integers(-27, 28, N_EVENTS)
    return e


@pytest.mark.parametrize("spelling,want,power",
                         [("low-count", "low-count", 1), ("uniform", "low-count", 1),
                          ("low-dose", "low-dose", 2), ("randoms", "low-dose", 2)])
def test_both_spellings_of_a_mode_mean_the_same_thing(spelling, want, power):
    assert thin.canonical(spelling) == want
    assert thin.randoms_power(spelling) == power


@pytest.mark.parametrize("junk", ["lowcount", "low dose", "", "f2", None])
def test_an_unknown_mode_is_refused(junk):
    with pytest.raises(ValueError):
        thin.canonical(junk)


def test_f_equals_one_is_the_identity(events):
    m = thin.keep(events, 1.0, rng=np.random.default_rng(0))
    assert m.all()


@pytest.mark.parametrize("f", [0.5, 0.1, 0.01])
def test_uniform_thinning_is_binomial(events, f):
    n = len(events)
    kept = [int(thin.keep(events, f, rng=np.random.default_rng(s)).sum())
            for s in range(30)]
    mu, sd = f * n, np.sqrt(n * f * (1 - f))
    assert abs(np.mean(kept) - mu) < 3 * sd / np.sqrt(len(kept))
    assert 0.4 < np.var(kept) / (n * f * (1 - f)) < 2.5


def test_thinning_is_a_subset(events):
    a = thin.keep(events, 0.5, rng=np.random.default_rng(1))
    b = thin.keep(events, 0.25, rng=np.random.default_rng(1))
    assert b.sum() < a.sum()


def test_bad_dose_fraction_is_refused(events):
    for f in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError):
            thin.keep(events, f)


def test_low_dose_keep_probability_is_below_f(events, binmap):
    f = 0.2
    b = ev.bins(events, binmap)
    rho = np.full(binmap.n_bin, 0.5, np.float32)
    m = thin.keep(events, f, "low-dose", np.random.default_rng(0), bins=b, rho=rho)
    q = f * 0.5 + f * f * 0.5
    assert q < f
    assert abs(m.mean() - q) < 5 * np.sqrt(q * (1 - q) / len(events))


def test_low_dose_sends_pure_randoms_to_f_squared(events, binmap):
    f = 0.25
    b = ev.bins(events, binmap)
    m = thin.keep(events, f, "low-dose", np.random.default_rng(0), bins=b,
                  rho=np.ones(binmap.n_bin, np.float32))
    assert abs(m.mean() - f * f) < 5 * np.sqrt(f * f / len(events))


def test_rho_is_clipped_and_broadcast(binmap):
    p = np.array([10.0] * binmap.n_plane)
    r = np.array([25.0] * binmap.n_plane)
    rho = thin.rho_per_plane(p, r, binmap)
    assert rho.shape == (binmap.n_bin,)
    assert rho.min() == 1.0 and rho.max() == 1.0
    assert thin.rho_per_plane(np.zeros(binmap.n_plane), r, binmap).max() == 0.0


def test_rho_per_tangential_bin_varies_across_u_not_views(binmap):
    shape = (binmap.n_plane, binmap.n_tang)
    P = np.full(shape, 100.0)
    R = np.tile(np.linspace(0.0, 100.0, binmap.n_tang), (binmap.n_plane, 1))
    rho = thin.rho_bins(P, R, binmap, "tangential")
    assert rho.shape == (binmap.n_bin,)
    cube = rho.reshape(binmap.n_plane, binmap.n_view, binmap.n_tang)
    assert np.allclose(cube[:, 0, :], cube[:, -1, :])
    assert np.allclose(cube[0, 0, :], R[0] / 100.0)
    assert cube[0, 0, 0] == 0.0 and cube[0, 0, -1] == 1.0


def test_rho_bins_refuses_the_wrong_shape(binmap):
    with pytest.raises(ValueError):
        thin.rho_bins(np.ones(binmap.n_plane), np.ones(binmap.n_plane),
                      binmap, "tangential")
    with pytest.raises(ValueError):
        thin.rho_bins(np.ones((binmap.n_plane, binmap.n_tang)),
                      np.ones((binmap.n_plane, binmap.n_tang)), binmap, "plane")
    with pytest.raises(ValueError):
        thin.rho_bins(np.ones(binmap.n_plane), np.ones(binmap.n_plane),
                      binmap, "bin")


def test_per_plane_rho_keeps_far_too_much_in_a_randoms_tail(binmap):
    f, n_tang = 0.1, binmap.n_tang
    tail = np.arange(n_tang) >= n_tang // 2
    P = np.full((binmap.n_plane, n_tang), 100.0)
    R = np.where(tail, 100.0, 0.0)[None, :] * np.ones((binmap.n_plane, 1))

    fine = thin.rho_bins(P, R, binmap, "tangential")
    coarse = thin.rho_bins(P.sum(1), R.sum(1), binmap, "plane")
    q = lambda rho: f * (1 - rho) + f * f * rho

    cube = q(fine).reshape(binmap.n_plane, binmap.n_view, n_tang)[0, 0]
    assert cube[-1] == pytest.approx(f * f)
    assert cube[0] == pytest.approx(f)
    q_plane = float(q(coarse)[0])
    assert q_plane / (f * f) == pytest.approx(5.0, rel=1e-5)
    assert q_plane < f


def test_low_dose_needs_its_inputs(events):
    with pytest.raises(ValueError):
        thin.keep(events, 0.5, "low-dose", np.random.default_rng(0))


N_TOF = 55


def _peaked_phi(n_tang, n_tof=N_TOF, width=6.0):
    t = np.arange(n_tof) - (n_tof - 1) / 2.0
    g = np.exp(-0.5 * (t / width) ** 2)
    return np.tile(g, (n_tang, 1))


def _phi_of(b, t_idx, binmap, n_tof=N_TOF):
    ok = b >= 0
    u = (b[ok] % binmap.n_tang).astype(np.int64)
    return np.bincount(u * n_tof + t_idx[ok].astype(np.int64),
                       minlength=binmap.n_tang * n_tof
                       ).reshape(binmap.n_tang, n_tof).astype(float)


@pytest.fixture(scope="module")
def tof_events(binmap):
    """As `events`, but with the peaked TOF distribution real prompts have."""
    rng = np.random.default_rng(23)
    ids, _ = binmap.lor_table()
    k = rng.integers(0, len(ids), N_EVENTS)
    e = np.zeros(N_EVENTS, dtype=[("xtal_a", "<u2"), ("xtal_b", "<u2"),
                                  ("tof_bin", "i1"), ("t_ms", "<u4")])
    e["xtal_a"], e["xtal_b"] = ids[k, 0], ids[k, 1]
    e["tof_bin"] = np.clip(np.round(rng.normal(0, 6.0, N_EVENTS)), -27, 27)
    return e


def test_a_flat_tof_profile_leaves_rho_untouched(binmap):
    flat = np.full((binmap.n_tang, N_TOF), 3.7)
    assert np.allclose(thin.tof_rho_factor(flat, N_TOF), 1.0)


def test_tof_factor_lowers_rho_at_the_peak_and_raises_it_in_the_tails(binmap):
    fac = thin.tof_rho_factor(_peaked_phi(binmap.n_tang), N_TOF)
    peak, tail = fac[0, N_TOF // 2], fac[0, 0]
    assert peak < 1.0 < tail
    assert tail / peak > 50


def test_tof_factor_reproduces_a_directly_measured_rho(binmap):
    n_tof, R_b, y_b = N_TOF, 300.0, 1000.0
    phi = _peaked_phi(1, n_tof)[0]
    phi = phi / phi.sum()
    y_bt = y_b * phi
    direct = np.clip((R_b / n_tof) / y_bt, 0.0, 1.0)
    fac = thin.tof_rho_factor(phi[None, :], n_tof)[0]
    modelled = np.clip((R_b / y_b) * fac, 0.0, 1.0)
    assert np.allclose(direct, modelled)


def test_pure_randoms_stay_at_f_squared_under_the_tof_model(binmap, events):
    f = 0.1
    b, swap = ev.bins(events, binmap, with_swap=True)
    t_idx = ev.tof_index(events, binmap, N_TOF, swap)
    flat = np.ones((binmap.n_tang, N_TOF))
    m = thin.keep(events, f, "low-dose", np.random.default_rng(0), bins=b,
                  rho=np.ones(binmap.n_bin, np.float32),
                  tof=(t_idx, thin.tof_rho_factor(flat, N_TOF)))
    assert abs(m.mean() - f * f) < 5 * np.sqrt(f * f / len(events))


def test_the_tof_model_splits_one_lor_across_its_tof_bins(binmap, tof_events):
    f = 0.1
    b, swap = ev.bins(tof_events, binmap, with_swap=True)
    t_idx = ev.tof_index(tof_events, binmap, N_TOF, swap)
    phi = _phi_of(b, t_idx, binmap)
    fac = thin.tof_rho_factor(phi, N_TOF)

    ok = b >= 0
    u = b[ok] % binmap.n_tang
    r = np.clip(0.5 * fac[u, t_idx[ok]], 0.0, 1.0)
    q = f * (1 - r) + f * f * r
    peak = np.abs(t_idx[ok] - N_TOF // 2) <= 2
    deep = np.abs(t_idx[ok] - N_TOF // 2) >= 20
    assert 0.8 * f < q[peak].mean() < f
    assert q[deep].mean() < 1.5 * f * f
    assert q[peak].mean() / q[deep].mean() > 5
    assert q.max() <= f + 1e-12 and q.min() >= f * f - 1e-12


def test_expectation_is_the_exact_bernoulli_sum(binmap):
    q = np.array([0.1, 0.2, 0.3, 0.4])
    planes = np.array([0, 0, 1, 1])
    mu, var = thin.expectation(q, planes, binmap.n_plane)
    assert mu[0] == pytest.approx(0.3) and mu[1] == pytest.approx(0.7)
    assert var[0] == pytest.approx(0.1 * 0.9 + 0.2 * 0.8)
    assert var[1] == pytest.approx(0.3 * 0.7 + 0.4 * 0.6)
    assert mu[2:].sum() == 0


def test_event_q_and_keep_agree_on_which_events_can_survive(binmap, tof_events):
    f = 0.2
    b, swap = ev.bins(tof_events, binmap, with_swap=True)
    rho = np.full(binmap.n_bin, 0.5, np.float32)
    q, ok = thin.event_q(f, b, rho)
    assert len(q) == int(ok.sum()) and ok.sum() == int((b >= 0).sum())
    assert np.allclose(q, f * 0.5 + f * f * 0.5)
    m = thin.keep(tof_events, f, "low-dose", np.random.default_rng(0), bins=b, rho=rho)
    assert not m[~ok].any()


def test_a_tof_resolved_draw_lands_inside_its_own_exact_band(tmp_path, binmap,
                                                             tof_events):
    C = _mini_case(tmp_path, "src", binmap, tof_events)
    D = Case("src_pair", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    f, nvt = 0.25, binmap.n_view * binmap.n_tang
    b, swap = ev.bins(tof_events, binmap, with_swap=True)
    t_idx = ev.tof_index(tof_events, binmap, N_TOF, swap)
    tof = (t_idx, thin.tof_rho_factor(_phi_of(b, t_idx, binmap), N_TOF))
    rho = thin.rho_bins(np.full((binmap.n_plane, binmap.n_tang), 1.0),
                        np.full((binmap.n_plane, binmap.n_tang), 0.3),
                        binmap, "tangential")

    mask = thin.keep(tof_events, f, "low-dose", np.random.default_rng(3), bins=b,
                     rho=rho, tof=tof)
    write.bed(C, D, 1, tof_events, mask, binmap, f, "low-dose")

    q, ok = thin.event_q(f, b, rho, tof)
    pair = thin.expectation(q, b[ok] // nvt, binmap.n_plane)
    assert pair[0].sum() == pytest.approx(q.sum())
    lines = []
    bad = verify.binomial(C, D, [1], {1: pair}, binmap.n_plane, nvt, out=lines.append)
    assert bad <= max(1, binmap.n_plane // 20)
    assert lines


def test_tof_factor_checks_its_shape(binmap):
    with pytest.raises(ValueError, match="TOF bins"):
        thin.tof_rho_factor(np.ones((binmap.n_tang, 11)), N_TOF)


@pytest.mark.parametrize("q", [0.5, 0.1])
def test_binomial_sinogram_thins_to_q_times_y(binmap, events, q):
    h, _ = ev.histogram(events, binmap, 1)
    y = h.reshape(1, binmap.n_plane, -1)
    out = thin.binomial_sinogram(y, np.full(binmap.n_plane, q),
                                 np.random.default_rng(4))
    assert out.dtype == y.dtype
    n = y.sum(dtype=np.int64)
    mu, sd = q * n, np.sqrt(n * q * (1 - q))
    assert abs(out.sum(dtype=np.int64) - mu) < 4 * sd
    assert (out <= y).all()


def test_binomial_sinogram_at_q_one_is_the_identity(binmap, events):
    h, _ = ev.histogram(events, binmap, 1)
    y = h.reshape(1, binmap.n_plane, -1)
    out = thin.binomial_sinogram(y, np.ones(binmap.n_plane), np.random.default_rng(0))
    assert np.array_equal(out, y)


def test_binomial_sinogram_checks_the_plane_count(binmap, events):
    h, _ = ev.histogram(events, binmap, 1)
    with pytest.raises(ValueError):
        thin.binomial_sinogram(h.reshape(1, binmap.n_plane, -1),
                               np.full(binmap.n_plane + 1, 0.5))


FRAME_MS, HL_S = 90_000.0, 6586.2


def test_a_time_window_keeps_the_first_f_of_the_frame():
    t = np.arange(0, int(FRAME_MS), dtype=np.int64) + 521_831_027
    m = thin.time_window(t, 0.1, FRAME_MS)
    assert m[:9000].all() and not m[9000:].any()
    assert thin.time_window(t, 1.0, FRAME_MS).all()


def test_a_time_window_is_deterministic():
    t = np.sort(np.random.default_rng(5).integers(0, int(FRAME_MS), 10_000))
    a, b = (thin.time_window(t, 0.25, FRAME_MS) for _ in range(2))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("f", [0.5, 0.1, 0.01])
def test_decay_scales_exceed_f_and_randoms_exceed_trues(f):
    lin, quad = thin.decay_scales(HL_S, FRAME_MS, f)
    assert f < lin < quad
    assert (quad / f - 1) == pytest.approx(2 * (lin / f - 1), rel=0.02)


def test_decay_scales_match_the_closed_form():
    f = 0.1
    mu = np.log(2) / HL_S
    T, tau = FRAME_MS / 1000.0, 0.1 * FRAME_MS / 1000.0
    lin, quad = thin.decay_scales(HL_S, FRAME_MS, f)
    assert lin == pytest.approx((1 - np.exp(-mu * tau)) / (1 - np.exp(-mu * T)))
    assert quad == pytest.approx((1 - np.exp(-2 * mu * tau))
                                 / (1 - np.exp(-2 * mu * T)))
    assert lin == pytest.approx(0.100427, abs=1e-6)
    assert quad == pytest.approx(0.100855, abs=1e-6)


def test_decay_scales_blow_up_for_a_short_lived_tracer():
    _, f18 = thin.decay_scales(6586.2, FRAME_MS, 0.1)
    _, o15 = thin.decay_scales(122.2, FRAME_MS, 0.1)
    assert f18 / 0.1 - 1 < 0.01
    assert o15 / 0.1 - 1 > 0.5


def test_decay_scales_are_the_identity_at_f_one():
    lin, quad = thin.decay_scales(HL_S, FRAME_MS, 1.0)
    assert lin == pytest.approx(1.0) and quad == pytest.approx(1.0)


def test_uniform_scales_terms_by_f_and_time_by_the_decay_factors(binmap):
    hdr = {"half_life_s": HL_S, "frame_duration_ms": FRAME_MS}
    f = 0.1
    assert write.term_scales(hdr, f, "low-count", "uniform") == (f, f, 1.0)
    assert write.term_scales(hdr, f, "low-dose", "uniform") == (f * f, f, 1.0)
    r, s, fr = write.term_scales(hdr, f, "low-count", "time")
    lin, quad = thin.decay_scales(HL_S, FRAME_MS, f)
    assert (r, s, fr) == (quad, lin, f)
    assert r > s > f


def test_a_time_window_shortens_the_frame_in_the_sidecar(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    hdr = json.loads((C.decoded / "bed1.json").read_text())
    hdr.update(half_life_s=HL_S, frame_duration_ms=FRAME_MS)
    (C.decoded / "bed1.json").write_text(json.dumps(hdr))

    D = Case("src_win", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    row = write.bed(C, D, 1, events, np.ones(len(events), bool), binmap, 0.1,
                    "low-count", window="time")
    assert D.header(1)["frame_duration_ms"] == pytest.approx(9_000.0)
    assert row["randoms_scale"] > row["scatter_scale"] > 0.1
    assert D.header(1)["delays"] == int(round(20_000 * row["randoms_scale"]))


def test_split_is_a_disjoint_partition():
    k, n = 4, 40_000
    lab = thin.split(n, k, np.random.default_rng(3))
    assert set(np.unique(lab)) == set(range(k))
    assert sum((lab == i).sum() for i in range(k)) == n
    sizes = np.array([(lab == i).sum() for i in range(k)])
    assert np.abs(sizes - n / k).max() < 5 * np.sqrt(n / k)


def test_split_is_reproducible_so_replicates_stay_disjoint():
    a = thin.split(1000, 3, np.random.default_rng(0))
    b = thin.split(1000, 3, np.random.default_rng(0))
    assert np.array_equal(a, b)


def _mini_case(root, name, binmap, events):
    import synth_hs

    C = Case(name, root)
    C.decoded.mkdir(parents=True, exist_ok=True)
    h, _ = ev.histogram(events, binmap, 1)
    synth_hs.write(str(C.decoded / "bed1"), RINGS, NDET, NTANG, data=h)
    np.save(C.decoded / "bed1.lm.npy", events)
    (C.decoded / "bed1.json").write_text(json.dumps(
        {"prompts": int(h.sum()), "delays": 20_000, "bed_number": 1,
         "table_position_mm": 0.0, "frame_duration_ms": 90_000}))

    w = C.work_bed(1)
    w.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    shape = (1,) + binmap.shape
    terms = {"randoms": rng.random(shape) * 0.20 * h.mean(),
             "scatter": rng.random(shape) * 0.05 * h.mean(),
             "normdt": np.full(shape, 0.9), "norm_only": np.full(shape, 0.95),
             "attn": np.full(shape, 0.4)}
    for stem, a in terms.items():
        write.clone_header(C.decoded / "bed1.hs", w / f"{stem}.hs", f"{stem}.s")
        a.astype("<f4").tofile(w / f"{stem}.s")
    (w / "to_stir.json").write_text('{"verified": {"bit_exact_vs_decoded": true}}')
    np.savez(w / "lm.npz", key=np.array([0]))
    return C


def test_clone_header_changes_only_the_data_file(tmp_path, mini_hs):
    src = __import__("pathlib").Path(mini_hs)
    dst = tmp_path / "x.hs"
    write.clone_header(src, dst, "x.s")
    a, b = src.read_text().splitlines(), dst.read_text().splitlines()
    diff = [(i, j) for i, j in zip(a, b) if i != j]
    assert len(diff) == 1 and diff[0][1].endswith("x.s")


def test_scale_term_scales_and_keeps_the_geometry(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    out = tmp_path / "out"
    out.mkdir()
    before = np.fromfile(C.work_bed(1) / "randoms.s", "<f4").sum(dtype=np.float64)
    after = write.scale_term(C.work_bed(1), out, "randoms", 0.25)
    assert after == pytest.approx(0.25 * before, rel=1e-5)
    assert (out / "randoms.hs").read_text().count("name of data file") == 1


def _thin_one_bed(tmp_path, binmap, events, mode, f=0.25, name="src_drf4"):
    C = _mini_case(tmp_path, "src", binmap, events)
    D = Case(name, tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    b = ev.bins(events, binmap)
    rho = thin.rho_per_plane(np.full(binmap.n_plane, 100.0),
                             np.full(binmap.n_plane, 40.0), binmap)
    low_dose = thin.canonical(mode) == "low-dose"
    mask = thin.keep(events, f, mode, np.random.default_rng(5),
                     bins=b, rho=rho if low_dose else None)
    row = write.bed(C, D, 1, events, mask, binmap, f, mode)
    write.manifest(D, C.name, f, mode, 5, [row])
    write.readme(D, C.name, f, mode, 5, "derived")
    return C, D, row


@pytest.mark.parametrize("mode,power",
                         [("low-count", 1), ("uniform", 1),
                          ("low-dose", 2), ("randoms", 2)])
def test_written_case_is_a_complete_case(tmp_path, binmap, events, mode, power):
    f = 0.25
    C, D, row = _thin_one_bed(tmp_path, binmap, events, mode, f)

    for p in (D.prompt(1), D.decoded / "bed1.s", D.decoded / "bed1.lm.npy",
              D.decoded / "bed1.json", D.work_bed(1) / "background.hs",
              D.work_bed(1) / "normdt.s", D.work_bed(1) / "attn.s"):
        assert p.exists(), p
    assert D.beds() == [1]

    assert row["events"] == len(np.load(D.decoded / "bed1.lm.npy"))
    assert row["prompts"] == row["events"]
    assert D.header(1)["prompts"] == row["prompts"]
    assert 0.5 * f * len(events) < row["events"] < 1.5 * f * len(events)

    sr = np.fromfile(C.work_bed(1) / "randoms.s", "<f4").sum(dtype=np.float64)
    ss = np.fromfile(C.work_bed(1) / "scatter.s", "<f4").sum(dtype=np.float64)
    assert row["randoms"] == pytest.approx(sr * f ** power, rel=1e-5)
    assert row["scatter"] == pytest.approx(ss * f, rel=1e-5)
    assert np.allclose(np.fromfile(D.work_bed(1) / "background.s", "<f4"),
                       np.fromfile(D.work_bed(1) / "randoms.s", "<f4")
                       + np.fromfile(D.work_bed(1) / "scatter.s", "<f4"))

    assert np.array_equal(np.fromfile(C.work_bed(1) / "normdt.s", "<f4"),
                          np.fromfile(D.work_bed(1) / "normdt.s", "<f4"))

    m = json.loads((D.root / "lowdose.json").read_text())
    assert m["k_scale"] == 1 / f and m["randoms_power"] == power
    assert m["mode"] == thin.canonical(mode)


@pytest.mark.parametrize("mode", ["low-count", "low-dose"])
def test_everything_thinned_lives_in_raw_simulation(tmp_path, binmap, events, mode):
    _, D, _ = _thin_one_bed(tmp_path, binmap, events, mode)

    for p in (D.raw_sim / "bed1.hs", D.raw_sim / "bed1.s",
              D.raw_sim / "bed1.lm.npy", D.raw_sim / "README.txt",
              D.raw_sim_bed(1) / "randoms.s", D.raw_sim_bed(1) / "scatter.s",
              D.raw_sim_bed(1) / "background.s"):
        assert p.is_file() and not p.is_symlink(), p

    for p in (D.decoded / "bed1.hs", D.decoded / "bed1.s",
              D.decoded / "bed1.lm.npy", D.work_bed(1) / "randoms.hs",
              D.work_bed(1) / "scatter.s", D.work_bed(1) / "background.s"):
        assert p.is_symlink(), f"{p} should be a link into raw_simulation/"

    for stem in write.COPY:
        assert not (D.work_bed(1) / f"{stem}.s").is_symlink()


@pytest.mark.parametrize("mode", ["low-count", "low-dose"])
def test_every_link_is_relative_resolves_and_reads_back(tmp_path, binmap, events, mode):
    import os

    _, D, _ = _thin_one_bed(tmp_path, binmap, events, mode)
    links = [p for p in D.root.rglob("*") if p.is_symlink()]
    assert len(links) == 9

    for p in links:
        target = os.readlink(p)
        assert not os.path.isabs(target), f"{p} -> {target} is absolute"
        assert p.resolve().is_file(), f"{p} -> {target} dangles"
        assert D.root.resolve() in p.resolve().parents
        assert p.read_bytes() == p.resolve().read_bytes()


def test_the_source_list_mode_cache_never_travels(tmp_path, binmap, events):
    C, D, _ = _thin_one_bed(tmp_path, binmap, events, "low-count")
    assert (C.work_bed(1) / "lm.npz").exists()
    assert not (D.work_bed(1) / "lm.npz").exists()


def test_readme_names_the_source_and_the_settings(tmp_path, binmap, events):
    _, D, _ = _thin_one_bed(tmp_path, binmap, events, "low-dose")
    text = (D.raw_sim / "README.txt").read_text()
    assert "NOT measured data" in text
    assert "'src'" in text and "low-dose" in text and "f^2" in text


def test_binomial_sinogram_builds_a_case_without_an_event_table(tmp_path, binmap,
                                                                events):
    C = _mini_case(tmp_path, "src", binmap, events)
    (C.decoded / "bed1.lm.npy").unlink()
    D = Case("src_drf2_sino", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    f = 0.5
    row = write.bed(C, D, 1, None, None, binmap, f, "low-count",
                    sinogram="binomial", q=np.full(binmap.n_plane, f),
                    rng=np.random.default_rng(6))

    assert row["events"] is None
    assert not (D.decoded / "bed1.lm.npy").exists()
    assert (D.raw_sim / "bed1.s").is_file()
    assert D.beds() == [1]
    y = np.fromfile(C.decoded / "bed1.s", "<i2").sum(dtype=np.int64)
    assert abs(row["prompts"] - f * y) < 4 * np.sqrt(y * f * (1 - f))


def test_measured_tof_weights_is_absent_until_written(tmp_path, binmap, events):
    from utils import terms as sino

    C = _mini_case(tmp_path, "src", binmap, events)
    assert sino.measured_tof_weights(C, 1, 55) is None


@pytest.mark.parametrize("n_tof", [55, 11, 5, 1])
def test_measured_tof_weights_mashes_55_down_to_any_divisor(tmp_path, binmap,
                                                            events, n_tof):
    from utils import terms as sino

    C = _mini_case(tmp_path, "src", binmap, events)
    full = np.arange(1.0, 56.0)
    np.save(C.work_bed(1) / sino.TOF_PROFILE_FILE, full)

    w, note = sino.measured_tof_weights(C, 1, n_tof)
    assert w.shape == (n_tof,)
    assert w.sum() == pytest.approx(1.0)
    want = full.reshape(n_tof, 55 // n_tof).sum(axis=1)
    assert np.allclose(w, want / want.sum())
    assert "full-count source" in note


def test_measured_tof_weights_refuses_a_profile_that_does_not_divide(tmp_path,
                                                                     binmap, events):
    from utils import terms as sino

    C = _mini_case(tmp_path, "src", binmap, events)
    np.save(C.work_bed(1) / sino.TOF_PROFILE_FILE, np.ones(55))
    with pytest.raises(SystemExit, match="does not divide"):
        sino.measured_tof_weights(C, 1, 7)


@pytest.mark.parametrize("bad", [np.zeros(55), -np.ones(55)])
def test_measured_tof_weights_refuses_a_degenerate_profile(tmp_path, binmap,
                                                           events, bad):
    from utils import terms as sino

    C = _mini_case(tmp_path, "src", binmap, events)
    np.save(C.work_bed(1) / sino.TOF_PROFILE_FILE, bad)
    with pytest.raises(SystemExit, match="non-negative"):
        sino.measured_tof_weights(C, 1, 55)


def test_carry_tof_profile_reports_rather_than_dies_when_it_cannot_measure(
        tmp_path, binmap, events):
    from utils import terms as sino

    _, D, row = _thin_one_bed(tmp_path, binmap, events, "low-count")
    assert not (D.work_bed(1) / sino.TOF_PROFILE_FILE).exists()
    assert row["scatter_tof_profile"].startswith("NOT carried")


def test_carry_tof_profile_defers_to_ges_own_weights(tmp_path, binmap, events):
    from utils import terms as sino

    C = _mini_case(tmp_path, "src", binmap, events)
    np.save(C.work_bed(1) / "scatter_tof.npy", np.ones((55, binmap.n_view, 4)))
    D = Case("src_drf4_vendor", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    mask = thin.keep(events, 0.25, rng=np.random.default_rng(0))
    row = write.bed(C, D, 1, events, mask, binmap, 0.25, "low-count")
    assert row["scatter_tof_profile"] is None
    assert (D.work_bed(1) / "scatter_tof.npy").exists()
    assert not (D.work_bed(1) / sino.TOF_PROFILE_FILE).exists()


def test_lm_matches_sinogram_passes_on_a_faithful_pair(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    assert verify.lm_matches_sinogram(C, [1], out=lambda *_: None) == []


def test_lm_matches_sinogram_catches_a_short_event_table(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    np.save(C.decoded / "bed1.lm.npy", events[:-1])
    assert verify.lm_matches_sinogram(C, [1], out=lambda *_: None) == [1]


def test_lm_matches_sinogram_reports_a_missing_event_table(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    (C.decoded / "bed1.lm.npy").unlink()
    assert verify.lm_matches_sinogram(C, [1], out=lambda *_: None) == [1]


def test_written_case_passes_the_binomial_check(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    D = Case("src_drf2", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    f = 0.5
    mask = thin.keep(events, f, rng=np.random.default_rng(2))
    write.bed(C, D, 1, events, mask, binmap, f, "low-count")

    nvt = binmap.n_view * binmap.n_tang
    lines = []
    bad = verify.binomial(C, D, [1], f, binmap.n_plane, nvt, out=lines.append)
    assert bad <= max(1, binmap.n_plane // 50)
    assert lines


def test_k_scale_reaches_export(tmp_path, binmap, events):
    from utils import quant

    C = _mini_case(tmp_path, "src", binmap, events)
    assert quant.lowdose_k_scale(C) == 1.0
    write.manifest(C, "src", 0.1, "low-count", 0, [])
    assert quant.lowdose_k_scale(C) == 10.0
