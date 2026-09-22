from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

import lmmini as M


@pytest.fixture
def torch():
    return M.torch()


def _bed(name, n_all, seed):
    from lmnet import sens, train

    raw = M.build(200).norm_BP.cpu().numpy().astype(np.float32)
    s = sens.image(raw)
    lab = (M.blob(seed=seed, scale=5.0).numpy() * (s > 0)).astype(np.float32)
    return train.Bed(name, n_all, M.events(n_all, seed=seed).astype(np.int16),
                     np.full(n_all, 0.05, np.float32), raw, s, lab, 281,
                     dict(xy=M.XY, n_plane=M.NPLANE, psf=M.PSF, lut=M.lut(),
                          tof=M.tof_meta()))


def _args(**kw):
    a = dict(phases=2, dual_feature="scaled", block=False, no_amp=False,
             lr=3e-3, warmup=1, clip=1.0, epochs=2, samples_per_bed=1,
             min_events=500, max_events=2000, n_splits=2, seed=0,
             resume=False, val_every=1, loss="mse", name="t")
    a.update(kw)
    return argparse.Namespace(**a)


def test_thinning_scales_the_target_by_f_over_kappa(torch):
    from lmnet import train

    b = _bed("b", 8000, 1)
    smp = train.make_sample(b, 2000, np.random.default_rng(0), 2)
    assert smp["f"] == pytest.approx(0.25)
    assert abs(smp["n"] - 2000) < 200
    assert smp["kappa"] == pytest.approx(smp["n"] / b.sum_s)
    np.testing.assert_allclose(smp["target"],
                               b.label * smp["f"] / smp["kappa"], rtol=1e-6)
    np.testing.assert_allclose(smp["a"], 0.05 * smp["f"], rtol=1e-6)
    assert smp["sm"].proj_meta.detector_ids.shape[0] == smp["n"]


def test_more_events_than_the_bed_uses_all_of_them(torch):
    from lmnet import train

    b = _bed("b", 1000, 2)
    smp = train.make_sample(b, 10 ** 9, np.random.default_rng(0), 2)
    assert smp["f"] == 1.0 and smp["n"] == 1000


def test_run_trains_validates_checkpoints_and_resumes(torch, tmp_path):
    from lmnet import train

    tr = [_bed(f"b{i}", 4000, i) for i in range(2)]
    vals = train.val_samples([_bed("v", 4000, 9)], [1000], 0, 2, osem=True)
    rd = tmp_path / "run"
    train.run(_args(), tr, vals, rd, torch.device("cpu"))

    names = {p.name for p in rd.iterdir()}
    assert {"last.pt", "best.pt", "config.json", "train.jsonl", "val.jsonl",
            "val_last.npz", "val_best.npz"} <= names
    rows = [json.loads(x) for x in open(rd / "train.jsonl")]
    assert len(rows) == 4
    assert all(np.isfinite(r["loss"]) and r["g_dual"] > 0 for r in rows)
    val = [json.loads(x) for x in open(rd / "val.jsonl")]
    assert [v["epoch"] for v in val] == [0, 1]
    assert "nmse_osem" in val[0]["rows"][0]

    train.run(_args(resume=True, epochs=3), tr, vals, rd, torch.device("cpu"))
    assert sum(1 for _ in open(rd / "train.jsonl")) == 6
    assert torch.load(rd / "last.pt", weights_only=False)["epoch"] == 2


def test_frozen_bn_leaves_the_running_stats_alone(torch):
    from lmnet import train
    from lmnet.model import LMPDNet3D

    m = LMPDNet3D(n_phase=1)
    bn = m.primal[0].blocks[0][1]
    before = bn.running_mean.clone()
    m.train()
    with train._FrozenBN(m), torch.no_grad():
        bn(torch.randn(1, bn.num_features, 3, 3, 3) + 5.0)
    assert torch.equal(bn.running_mean, before)
    assert bn.momentum == 0.1
