"""`lowdose` -- the decimator's statistics, and the case it writes.

Everything here runs on the miniature scanner, so it needs no data and no SIRF.
The statistical claims are the ones the simulator stands on: `f = 1` is the
identity, thinning is binomial, replicates are disjoint, and the randoms-aware
mode really does send randoms to `f^2`.
"""

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


# ------------------------------------------------------------------ thin
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
    # thinned counts are Poisson-like: variance tracks the binomial one
    assert 0.4 < np.var(kept) / (n * f * (1 - f)) < 2.5


def test_thinning_is_a_subset(events):
    a = thin.keep(events, 0.5, rng=np.random.default_rng(1))
    b = thin.keep(events, 0.25, rng=np.random.default_rng(1))
    assert b.sum() < a.sum()               # same stream, smaller draw


def test_bad_dose_fraction_is_refused(events):
    for f in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError):
            thin.keep(events, f)


# ----------------------------------------------------------- randoms mode
def test_randoms_mode_keep_probability_is_below_f(events, binmap):
    f = 0.2
    b = ev.bins(events, binmap)
    rho = np.full(binmap.n_bin, 0.5, np.float32)
    m = thin.keep(events, f, "randoms", np.random.default_rng(0), bins=b, rho=rho)
    q = f * 0.5 + f * f * 0.5
    assert q < f
    assert abs(m.mean() - q) < 5 * np.sqrt(q * (1 - q) / len(events))


def test_randoms_mode_sends_pure_randoms_to_f_squared(events, binmap):
    f = 0.25
    b = ev.bins(events, binmap)
    m = thin.keep(events, f, "randoms", np.random.default_rng(0), bins=b,
                  rho=np.ones(binmap.n_bin, np.float32))
    assert abs(m.mean() - f * f) < 5 * np.sqrt(f * f / len(events))


def test_rho_is_clipped_and_broadcast(binmap):
    p = np.array([10.0] * binmap.n_plane)
    r = np.array([25.0] * binmap.n_plane)          # nonsense: r > p
    rho = thin.rho_per_plane(p, r, binmap)
    assert rho.shape == (binmap.n_bin,)
    assert rho.min() == 1.0 and rho.max() == 1.0   # clipped, not > 1
    assert thin.rho_per_plane(np.zeros(binmap.n_plane), r, binmap).max() == 0.0


def test_randoms_mode_needs_its_inputs(events):
    with pytest.raises(ValueError):
        thin.keep(events, 0.5, "randoms", np.random.default_rng(0))


# ------------------------------------------------------------ replicates
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


# ----------------------------------------------------------------- write
def _mini_case(root, name, binmap, events):
    """A source case on disk: decoded prompts + the five vendor terms."""
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


@pytest.mark.parametrize("mode,power", [("uniform", 1), ("randoms", 2)])
def test_written_case_is_a_complete_case(tmp_path, binmap, events, mode, power):
    C = _mini_case(tmp_path, "src", binmap, events)
    D = Case("src_drf4", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    f = 0.25
    b = ev.bins(events, binmap)
    rho = thin.rho_per_plane(np.full(binmap.n_plane, 100.0),
                             np.full(binmap.n_plane, 40.0), binmap)
    mask = thin.keep(events, f, mode, np.random.default_rng(5),
                     bins=b, rho=rho if mode == "randoms" else None)
    row = write.bed(C, D, 1, events, mask, binmap, f, randoms_power=power)
    write.manifest(D, C.name, f, mode, 5, [row])

    for p in (D.prompt(1), D.decoded / "bed1.s", D.decoded / "bed1.lm.npy",
              D.decoded / "bed1.json", D.work_bed(1) / "background.hs",
              D.work_bed(1) / "normdt.s", D.work_bed(1) / "attn.s"):
        assert p.exists(), p
    assert D.beds() == [1]

    # counts really were thinned, and the sidecar says the same number
    assert row["events"] == int(mask.sum())
    assert row["prompts"] == row["events"]
    assert D.header(1)["prompts"] == row["prompts"]
    assert 0.5 * f * len(events) < row["events"] < 1.5 * f * len(events)

    # randoms go as f^power, scatter always as f
    sr = np.fromfile(C.work_bed(1) / "randoms.s", "<f4").sum(dtype=np.float64)
    ss = np.fromfile(C.work_bed(1) / "scatter.s", "<f4").sum(dtype=np.float64)
    assert row["randoms"] == pytest.approx(sr * f ** power, rel=1e-5)
    assert row["scatter"] == pytest.approx(ss * f, rel=1e-5)
    assert np.allclose(np.fromfile(D.work_bed(1) / "background.s", "<f4"),
                       np.fromfile(D.work_bed(1) / "randoms.s", "<f4")
                       + np.fromfile(D.work_bed(1) / "scatter.s", "<f4"))

    # sensitivity is untouched -- it does not depend on the dose
    assert np.array_equal(np.fromfile(C.work_bed(1) / "normdt.s", "<f4"),
                          np.fromfile(D.work_bed(1) / "normdt.s", "<f4"))

    m = json.loads((D.root / "lowdose.json").read_text())
    assert m["k_scale"] == 1 / f and m["randoms_power"] == power


def test_written_case_passes_the_binomial_check(tmp_path, binmap, events):
    C = _mini_case(tmp_path, "src", binmap, events)
    D = Case("src_drf2", tmp_path)
    D.root.mkdir(parents=True, exist_ok=True)
    f = 0.5
    mask = thin.keep(events, f, rng=np.random.default_rng(2))
    write.bed(C, D, 1, events, mask, binmap, f, randoms_power=1)

    nvt = binmap.n_view * binmap.n_tang
    lines = []
    bad = verify.binomial(C, D, [1], f, binmap.n_plane, nvt, out=lines.append)
    assert bad <= max(1, binmap.n_plane // 50)      # ~0.3 % expected by chance
    assert lines


def test_k_scale_reaches_export(tmp_path, binmap, events):
    from utils import quant

    C = _mini_case(tmp_path, "src", binmap, events)
    assert quant.lowdose_k_scale(C) == 1.0
    write.manifest(C, "src", 0.1, "uniform", 0, [])
    assert quant.lowdose_k_scale(C) == 10.0
