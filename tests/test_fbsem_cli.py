from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pytest

pytest.importorskip("pytomography")
pytest.importorskip("parallelproj")
pytest.importorskip("nibabel")

import pytomography  # noqa: E402
import torch  # noqa: E402

import synth_hs  # noqa: E402
from utils.binmap import BinMap  # noqa: E402
from utils.paths import case as get_case  # noqa: E402
from utils.scanner import DR_MM, PLANE_MM  # noqa: E402

XY, RINGS, NSUB = 48, 24, 4
PSF = [4.87, 4.87, 4.45]
INJ = dt.datetime(2026, 7, 28, 2, 45, tzinfo=dt.timezone.utc).timestamp()
HDR = {"radiopharm_start_datetime": "20260728024500.00", "frame_duration_ms": 90000,
       "half_life_s": 6586.2002, "positron_fraction": 0.967, "dose_mbq": 185.0,
       "residual_dose_mbq": 0.0, "patient_weight_kg": 60.0,
       "patient_height_m": 1.6, "radiopharmaceutical": "FDG",
       "manufacturer": "GE MEDICAL SYSTEMS", "model_name": "Discovery 710",
       "study_instance_uid": "1.2.3", "sop_instance_uid": "1.2.3.4"}
CASES = ("thyr_trainset_0001_20260101", "thyr_trainset_0003_20260101",
         "thyr_testset_0002_20260101")


def phantom(seed):
    rng = np.random.default_rng(seed)
    c = (np.arange(XY) - (XY - 1) / 2) * DR_MM
    xx, yy = np.meshgrid(c, c, indexing="ij")
    x = np.where(np.hypot(xx, yy) < 40.0, 1.0, 0.0)
    cx, cy = rng.uniform(-5, 5, 2)
    x = x + np.where(np.hypot(xx - cx, yy - cy) < 8.0, 3.0, 0.0)
    return np.repeat(x[:, :, None], 2 * RINGS - 1, 2).astype(np.float32)


def write_term(path, a):
    path.with_suffix(".hs").write_text(synth_hs.header(
        RINGS, data_file=path.name + ".s", number_format="float",
        bytes_per_pixel=4))
    a.astype("<f4")[None].tofile(path.with_suffix(".s"))


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    from sino.projector import SinogramSystemMatrix

    root = tmp_path_factory.mktemp("fbsem_out")
    for i, name in enumerate(CASES):
        C = get_case("sim_an_s1", root / name).mkdirs()
        C.raw_sim.mkdir(parents=True, exist_ok=True)
        for n in (1, 2):
            rng = np.random.default_rng(10 * i + n)
            hs = synth_hs.write(str(C.decoded / f"bed{n}"), num_rings=RINGS)
            bm = BinMap(hs)
            sm = SinogramSystemMatrix(bm, xy=XY, psf=PSF, device="cpu")
            x = phantom(10 * i + n)
            S = rng.uniform(0.4, 1.0, bm.shape).astype(np.float32)
            fx = sm.forward(torch.from_numpy(x)).numpy()
            b = np.full(bm.shape, 0.3 * fx.mean(), np.float32)
            y = rng.poisson(20 * (S * fx + b)).astype(np.int16)
            synth_hs.write(str(C.decoded / f"bed{n}"), num_rings=RINGS, data=y[None])
            C.work_bed(n).mkdir(parents=True, exist_ok=True)
            write_term(C.work_bed(n) / "normdt", 20 * S)
            write_term(C.work_bed(n) / "attn", np.ones(bm.shape, np.float32))
            write_term(C.work_bed(n) / "background", 20 * b)
            np.save(C.raw_sim / f"bed{n}_x_true.npy", x.transpose(2, 1, 0).copy())
            hdr = dict(HDR, bed_start_time=INJ + 3600 + 100 * n,
                       table_position_mm=(n - 1) * 38 * PLANE_MM)
            (C.decoded / f"bed{n}.json").write_text(json.dumps(hdr))
    return root


@pytest.fixture(autouse=True)
def env(monkeypatch, root):
    monkeypatch.setattr(pytomography, "device", torch.device("cpu"))
    monkeypatch.setenv("D710_OUT", str(root))


def test_end_to_end(root, capsys):
    from fbsem import evaluate, recon, train
    from sino.__main__ import main as sino_main
    from utils import export

    for name in CASES:
        assert sino_main(["--case", "sim_an_s1", "--out", str(root / name),
                          "--xy", str(XY), "--subsets", str(NSUB), "--ct", "x"]) == 0

    common = ["--xy", str(XY), "--n-sub", str(NSUB), "--device", "cpu"]
    assert train.main(["--run", "smoke", "--epochs", "2", "--beds-per-case", "2",
                       "--depth", "3", "--kernels", "4", "--max-steps", "3",
                       *common]) == 0
    run = root / "fbsem" / "runs" / "smoke"
    rows = (run / "log.csv").read_text().strip().splitlines()
    assert len(rows) == 4
    args = json.loads((run / "args.json").read_text())
    assert args["u"] > 0 and args["k"] > 0 and args["xy"] == XY
    assert (run / "last.pt").exists() and (run / "epoch_00.pt").exists()
    val = (run / "val.csv").read_text().strip().splitlines()
    assert len(val) == 2 and (run / "best.pt").exists()
    v = dict(zip(val[0].split(","), val[1].split(",")))
    assert float(v["val_nrmse"]) > 0 and float(v["osem_nrmse"]) > 0
    assert train.main(["--run", "smoke", "--epochs", "2", "--beds-per-case", "2",
                       "--depth", "3", "--kernels", "4", "--max-steps", "5",
                       "--val-every", "1", "--resume", *common]) == 0
    log = [r.split(",") for r in (run / "log.csv").read_text().strip().splitlines()]
    assert [int(r[1]) for r in log[1:]] == [1, 2, 3, 4, 5]
    assert len((run / "val.csv").read_text().strip().splitlines()) == 4
    assert len((run / "val_beds.csv").read_text().strip().splitlines()) == 4
    assert (run / "epoch_01.pt").exists()
    with pytest.raises(SystemExit):
        train.main(["--run", "smoke", *common])
    cache = list((root / "fbsem" / "cache").rglob("*.npy"))
    assert cache and all(np.load(p).shape == (NSUB, XY, XY, 2 * RINGS - 1)
                         for p in cache)

    test = root / CASES[2]
    capsys.readouterr()
    assert recon.main(["--case", "sim_an_s1", "--out", str(test), "--check-osem",
                       *common]) == 0
    diffs = [float(v) for v in re.findall(r"= ([0-9.e+-]+)", capsys.readouterr().out)]
    assert len(diffs) == 2 and max(diffs) < 1e-5

    assert recon.main(["--case", "sim_an_s1", "--out", str(test), "--model",
                       str(run / "last.pt"), "--device", "cpu"]) == 0
    C = get_case("sim_an_s1", test)
    z = np.load(C.recon_fbsem)
    assert z["vol"].shape == (47 + 38, XY, XY)
    assert float(z["post_filter_fwhm_mm"]) == 0.0
    assert np.load(C.work_bed(1) / "fbsem.npz")["img"].shape == (47, XY, XY)

    assert export.main(["--case", "sim_an_s1", "--out", str(test), "--fbsem",
                        "--format", "nifti"]) == 0
    assert (C.export / "sim_an_s1_fbsem_suvbw.nii.gz").exists()

    out = root / "eval.csv"
    assert evaluate.main(["--sets", "thyr_testset", "--csv", str(out)]) == 0
    s = json.loads(out.with_suffix(".json").read_text())
    assert s["n_beds"] == 2
    assert all(np.isfinite(s[k]["nrmse_mean"]) for k in ("fbsem", "osem", "osem_pf"))
